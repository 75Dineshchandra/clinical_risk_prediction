"""
List every MLflow run with the group it belongs to, so you can tell the original runs from the later attempts.
Read-only unless you add --delete-v2 --apply.

Run from the project root, in the project environment (.venv):
    python tools/mlflow_runs.py                       # list all runs, grouped
    python tools/mlflow_runs.py --delete-v2           # show which runs of the rejected v2 attempt WOULD be deleted
    python tools/mlflow_runs.py --delete-v2 --apply   # soft-delete them (they disappear from the UI; files are not removed)

Groups
    training runs carry the tag data_tag (added to train.py for the v2 attempt). A run with no tag is "original".
    evaluation runs (made by test.py) carry no data_tag, so they join the group of the most recent training run before them.
    Examples: "original", "v2-clean-train-plus-holdout70" (the rejected attempt), "retrain-retrain-2026..." (the Prefect flow).

Safety
    - Runs that back a registered model version are NEVER deleted (the registry still needs them).
    - Only groups whose tag starts with "v2-" can be deleted with this script.
    - Deletion is soft (MLflow marks the run deleted). Nothing in mlruns/ is removed, and `mlflow gc` is not run.
    - Take a copy of mlflow.db first:  Copy-Item mlflow.db mlflow_backup.db
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_NAME = "readmission"
EXPERIMENT = "readmission-30d"


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir() or (candidate / "mlflow.db").exists():
            return candidate
    raise FileNotFoundError(f"Could not find the project root next to or above {here}")


def main() -> None:
    ap = argparse.ArgumentParser(description="List MLflow runs by group; optionally soft-delete the v2 attempt.")
    ap.add_argument("--delete-v2", action="store_true", help="select the runs of the rejected v2 attempt")
    ap.add_argument("--apply", action="store_true", help="with --delete-v2: really soft-delete them")
    args = ap.parse_args()
    if args.apply and not args.delete_v2:
        raise SystemExit("--apply only makes sense together with --delete-v2.")

    root = find_project_root()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(root / 'mlflow.db').as_posix()}"))
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise SystemExit(f"Experiment '{EXPERIMENT}' not found in {mlflow.get_tracking_uri()}")
    runs = client.search_runs([exp.experiment_id], max_results=5000, order_by=["attributes.start_time ASC"])
    versions = {}
    try:
        for v in client.search_model_versions(f"name='{REGISTERED_NAME}'"):
            versions.setdefault(v.run_id, []).append(int(v.version))
    except Exception:
        pass

    rows, current = [], "original"
    for r in runs:
        tags = r.data.tags
        if tags.get("stage") in ("baseline", "candidate"):                  # a training run
            current = tags.get("data_tag") or "original"
        group = current
        started = datetime.fromtimestamp(r.info.start_time / 1000)
        rows.append({"id": r.info.run_id, "name": r.info.run_name or "", "kind": tags.get("stage", "?"),
                     "family": tags.get("model_family", ""), "group": group, "start": started,
                     "versions": versions.get(r.info.run_id, [])})

    print(f"{len(rows)} runs in experiment '{EXPERIMENT}' ({mlflow.get_tracking_uri()})\n")
    print(f"{'started':<17}{'kind':<11}{'model':<10}{'registry':<10}{'group':<42}run id")
    for x in rows:
        reg = ",".join(f"v{v}" for v in x["versions"]) or "-"
        print(f"{x['start']:%Y-%m-%d %H:%M}  {x['kind']:<11}{x['family']:<10}{reg:<10}{x['group'][:40]:<42}{x['id']}")
    print("\nRuns per group:")
    for g in dict.fromkeys(x["group"] for x in rows):
        n = sum(x["group"] == g for x in rows)
        reg = sum(bool(x["versions"]) for x in rows if x["group"] == g)
        print(f"  {g}: {n} runs, {reg} of them registered in the registry")

    if args.delete_v2:
        target = [x for x in rows if x["group"].startswith("v2-") and not x["versions"]]
        protected = [x for x in rows if x["group"].startswith("v2-") and x["versions"]]
        print(f"\nRuns of the v2 attempt that {'will be' if args.apply else 'would be'} soft-deleted: {len(target)}")
        for x in target:
            print(f"  {x['start']:%Y-%m-%d %H:%M}  {x['kind']:<11}{x['family']:<10}{x['id']}")
        if protected:
            print(f"Kept (registered in the registry): {[x['id'] for x in protected]}")
        if args.apply:
            for x in target:
                client.delete_run(x["id"])
            print("Done. They no longer show in the MLflow UI. Files in mlruns/ were not removed.")
        else:
            print("Dry run: nothing deleted. Add --apply to delete.")


if __name__ == "__main__":
    main()
