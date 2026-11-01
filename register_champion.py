"""
Register models in the MLflow Model Registry and set aliases (step 9).

Run from the project root, after test.py:
    python register_champion.py --champion lightgbm --reason "why you chose it"
    python register_champion.py --challenger lightgbm --reason "retrained on new data"
    python register_champion.py --no-verify          # skip the load-and-compare check

What it does
- Refuses to promote a champion that did not PASS the gate in models/evaluation.json.
- Registers one registered model, "readmission", with one version per model family:
      alias "champion"   -> the model that passed the gate and was chosen
      alias "challenger" -> other models that passed the gate
      alias "baseline"   -> logistic regression (always registered)
- With --challenger it registers ONLY that model, as a new version with alias @challenger. The
  champion and baseline are untouched; promote.py then decides whether it replaces the champion.
  Use this for every retrained model, so a new model never skips the promotion rule.
- Tags each version with the evidence: gate result, CV and test scores, why it was chosen.
- Verifies the registered champion loads from models:/readmission@champion and gives the
  same predictions as the saved joblib file.

Writes models/champion.json (machine-readable) and models/champion_report.md (plain language).

Downstream code should load:  mlflow.sklearn.load_model("models:/readmission@champion")
Re-running is safe: a run that is already registered is reused, not duplicated.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

EXPERIMENT = "readmission-30d"
REGISTERED_NAME = "readmission"
TARGET, PATIENT = "readmitted", "patient_nbr"
NOT_FEATURES = ["encounter_id", PATIENT, "readmitted_raw", TARGET, "diag_1", "diag_2", "diag_3"]
CV_MATCH_TOLERANCE = 0.0006   # evaluation.json stores CV PR-AUC rounded to 3 decimals


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find a data/ folder next to or above {here}")


ROOT = find_project_root()
EVAL_JSON = ROOT / "models" / "evaluation.json"
MODELS_DIR = ROOT / "models"
TEST_FILE = ROOT / "data" / "splits" / "test.csv"
CHAMPION_JSON = MODELS_DIR / "champion.json"
CHAMPION_REPORT = MODELS_DIR / "champion_report.md"
LOG_FILE = MODELS_DIR / "promotion_log.jsonl"


def find_training_run(client: MlflowClient, exp_id: str, family: str, cv_expected: float):
    """Latest finished TRAINING run of this model family whose CV score matches the evaluated file."""
    runs = client.search_runs(
        [exp_id],
        filter_string=f"tags.model_family = '{family}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"], max_results=50)
    for r in runs:
        if r.data.tags.get("stage") not in ("baseline", "candidate"):
            continue                                   # skip evaluation runs
        cv = r.data.metrics.get("cv_pr_auc_mean")
        if cv is not None and abs(cv - cv_expected) <= CV_MATCH_TOLERANCE:
            return r
    raise RuntimeError(
        f"No finished training run for '{family}' matches CV PR-AUC {cv_expected:.3f}. "
        f"Re-run training, or look the run up in the MLflow UI.")


def model_uri_for(client: MlflowClient, exp_id: str, run_id: str, family: str) -> str:
    """Artifact path used by train_models.py is readmission_<family>. MLflow 3 may store it as a logged model."""
    uri = f"runs:/{run_id}/readmission_{family}"
    if hasattr(mlflow, "search_logged_models"):
        try:
            found = mlflow.search_logged_models(
                experiment_ids=[exp_id], filter_string=f"name = 'readmission_{family}'",
                order_by=[{"field_name": "creation_time", "ascending": False}], output_format="list")
            for m in found:
                if getattr(m, "source_run_id", None) == run_id:
                    return f"models:/{m.model_id}"
        except Exception:
            pass
    return uri


def register_version(client: MlflowClient, uri: str, run_id: str):
    for v in client.search_model_versions(f"name='{REGISTERED_NAME}'"):
        if v.run_id == run_id:
            return v, False
    try:
        client.create_registered_model(REGISTERED_NAME)
    except MlflowException:
        pass                                           # already exists
    mv = mlflow.register_model(uri, REGISTERED_NAME)
    return client.get_model_version(REGISTERED_NAME, mv.version), True


def verify_against_joblib(uri: str, joblib_name: str, label: str) -> bool:
    """The registered model must give the same predictions as the saved joblib file."""
    loaded = mlflow.sklearn.load_model(uri)
    saved = joblib.load(MODELS_DIR / joblib_name)
    df = pd.read_csv(TEST_FILE, dtype={"encounter_id": str, PATIENT: str}).head(500)
    X = df[list(saved.named_steps["prep"].feature_names_in_)]
    p_reg, p_file = loaded.predict_proba(X)[:, 1], saved.predict_proba(X)[:, 1]
    if not np.allclose(p_reg, p_file, atol=1e-9):
        raise SystemExit(f"VERIFY FAILED: {label} differs from {joblib_name} "
                         f"(max diff {np.abs(p_reg - p_file).max():.2e}).")
    print(f"OK: {label} loads and matches {joblib_name} on 500 rows.")
    return True


def register_challenger(client, exp_id, results, family, reason, verify) -> None:
    """Register one model as @challenger. Champion and baseline are not touched."""
    r = results[family]
    run = find_training_run(client, exp_id, family, r["cv_pr_auc"])
    uri = model_uri_for(client, exp_id, run.info.run_id, family)
    mv, created = register_version(client, uri, run.info.run_id)
    try:
        champ = client.get_model_version_by_alias(REGISTERED_NAME, "champion")
        if champ.version == mv.version:
            raise SystemExit(f"REFUSED: this exact training run is already the champion (version {mv.version}). "
                             f"Retrain first, then register the new model as a challenger.")
    except MlflowException:
        pass                                           # no champion yet
    tags = {"model_family": family, "role": "challenger", "gate": "PASS",
            "cv_pr_auc": f"{r['cv_pr_auc']:.3f}", "test_pr_auc": f"{r['test_pr_auc']:.3f}",
            "test_roc_auc": f"{r['test_roc_auc']:.3f}", "test_lift": f"{r['lift']:.2f}",
            "evaluated_on": "test split", "training_run_id": run.info.run_id, "registration_reason": reason}
    for k, v in tags.items():
        client.set_model_version_tag(REGISTERED_NAME, mv.version, k, v)
    try:                                               # the previous challenger is retired, not deleted
        old = client.get_model_version_by_alias(REGISTERED_NAME, "challenger")
        if old.version != mv.version:
            client.set_model_version_tag(REGISTERED_NAME, old.version, "role", "retired_challenger")
    except MlflowException:
        pass
    client.set_registered_model_alias(REGISTERED_NAME, "challenger", mv.version)
    print(f"[{family}] {'registered' if created else 'already registered'} as version {mv.version}, alias 'challenger'")
    if verify:
        verify_against_joblib(f"models:/{REGISTERED_NAME}/{mv.version}", r["file"], f"version {mv.version}")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "decision": "challenger_registered",
                            "model_family": family, "version": int(mv.version), "reason": reason,
                            "test_pr_auc": round(r["test_pr_auc"], 4)}) + "\n")
    print("The champion was not changed. Next: python promote.py  (it decides whether the challenger replaces it)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--champion", help="model family to register as champion (your decision)")
    grp.add_argument("--challenger", help="model family to register as a CHALLENGER; the champion is not touched")
    ap.add_argument("--reason", required=True, help="why this model was chosen (stored as a tag and in the report)")
    ap.add_argument("--force", action="store_true", help="re-register even if promote.py has changed the champion")
    args = ap.parse_args()

    if not EVAL_JSON.exists():
        raise FileNotFoundError(f"Missing {EVAL_JSON}. Run test.py first.")
    ev = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    results = ev["results"]
    target = args.champion or args.challenger
    champion = args.champion                           # None when registering a challenger
    if target not in results:
        raise ValueError(f"'{target}' not found in evaluation.json")
    if not results[target]["passed"]:
        raise SystemExit(f"REFUSED: {target} did not pass the evaluation gate. Nothing was registered.")
    reason = args.reason

    if champion and CHAMPION_JSON.exists() and not args.force:
        cur = json.loads(CHAMPION_JSON.read_text(encoding="utf-8"))
        if cur.get("promoted_at") or cur.get("rolled_back_at"):
            raise SystemExit("REFUSED: promote.py has changed the champion since the first registration. "
                             "Re-running this script would reset the aliases to the old champion. "
                             "Use --force only if you really mean to.")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"))
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT}' not found in {mlflow.get_tracking_uri()}")

    if args.challenger:
        register_challenger(client, exp.experiment_id, results, target, reason, not args.no_verify)
        return

    versions = {}
    for family, r in results.items():
        role = "champion" if family == champion else ("baseline" if family == "logistic" else "challenger")
        if role == "challenger" and not r["passed"]:
            print(f"[{family}] skipped: did not pass the gate, so it is not registered as a challenger.")
            continue
        run = find_training_run(client, exp.experiment_id, family, r["cv_pr_auc"])
        uri = model_uri_for(client, exp.experiment_id, run.info.run_id, family)
        mv, created = register_version(client, uri, run.info.run_id)
        tags = {"model_family": family, "role": role, "gate": "PASS" if r["passed"] else "FAIL",
                "cv_pr_auc": f"{r['cv_pr_auc']:.3f}", "test_pr_auc": f"{r['test_pr_auc']:.3f}",
                "test_roc_auc": f"{r['test_roc_auc']:.3f}", "test_lift": f"{r['lift']:.2f}",
                "evaluated_on": "test split", "training_run_id": run.info.run_id}
        if role == "champion":
            tags["selection_reason"] = reason
        for k, v in tags.items():
            client.set_model_version_tag(REGISTERED_NAME, mv.version, k, v)
        client.set_registered_model_alias(REGISTERED_NAME, role, mv.version)
        versions[family] = (mv.version, role, run.info.run_id)
        print(f"[{family}] {'registered' if created else 'already registered'} as version {mv.version}, alias '{role}'")

    print(f"\nRegistry '{REGISTERED_NAME}':")
    for family, (ver, role, rid) in versions.items():
        print(f"  v{ver}  {family:<9} @{role:<10} run {rid}")

    verified = None
    if not args.no_verify:
        print("\nVerifying champion ...")
        verified = verify_against_joblib(f"models:/{REGISTERED_NAME}@champion", results[champion]["file"],
                                         f"models:/{REGISTERED_NAME}@champion")

    write_champion_files(champion, results, versions, reason, verified, ev.get("created"))
    print(f"\nSaved: {CHAMPION_JSON}\n       {CHAMPION_REPORT}")


def write_champion_files(champion, results, versions, reason, verified, evaluated_at) -> None:
    label = {"logistic": "Logistic regression", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    nm = lambda n: label.get(n, n)
    r = results[champion]
    ver, _, run_id = versions[champion]
    now = f"{datetime.now():%Y-%m-%d %H:%M}"
    CHAMPION_JSON.write_text(json.dumps({
        "registered_model": REGISTERED_NAME, "alias": "champion", "model_uri": f"models:/{REGISTERED_NAME}@champion",
        "version": int(ver), "model_family": champion, "training_run_id": run_id, "file": r["file"],
        "gate": "PASS" if r["passed"] else "FAIL", "gate_checks": r["checks"],
        "cv_pr_auc": r["cv_pr_auc"], "test_pr_auc": round(r["test_pr_auc"], 4),
        "test_pr_auc_ci95": [round(x, 4) for x in r["pr_auc_ci95"]], "test_roc_auc": round(r["test_roc_auc"], 4),
        "test_lift": round(r["lift"], 3), "params": r["params"], "selection_reason": reason,
        "verified_against_joblib": verified, "evaluated_at": evaluated_at, "registered_at": now,
        "other_versions": {f: {"version": int(v), "alias": role, "training_run_id": rid}
                           for f, (v, role, rid) in versions.items() if f != champion},
    }, indent=2, default=str), encoding="utf-8")

    out = ["# Champion model", "",
           f"Created {now}. Generated automatically by `register_champion.py` and replaced on every run.", "",
           "## Current champion", "",
           f"- **Model:** {nm(champion)} (registry `{REGISTERED_NAME}`, version {ver}, alias `champion`).",
           f"- **Load it with:** `mlflow.sklearn.load_model(\"models:/{REGISTERED_NAME}@champion\")`",
           f"- **Training run:** `{run_id}`; saved file `models/{r['file']}`.",
           f"- **Registered:** {now}. " + ("Registry copy checked against the saved file: predictions match."
                                         if verified else "Registry copy was not checked against the saved file."), "",
           "## Why it is the champion", "", reason, "",
           "## Evidence (test split, never used for training)", "",
           "| Measure | Value |", "|---|---|",
           f"| Test PR-AUC | {r['test_pr_auc']:.3f} (95% range {r['pr_auc_ci95'][0]:.3f} to {r['pr_auc_ci95'][1]:.3f}) |",
           f"| CV PR-AUC at training | {r['cv_pr_auc']:.3f} |",
           f"| Test ROC-AUC | {r['test_roc_auc']:.3f} |",
           f"| Lift over chance | {r['lift']:.2f}x |", "",
           "Gate checks:", ""]
    out += [f"- {'Pass' if ok else 'FAIL'}: {c}" for c, ok in r["checks"].items()]
    out += ["", "## Other registered versions", "", "| Version | Model | Alias |", "|---|---|---|"]
    out += [f"| {v} | {nm(f)} | `{role}` |" for f, (v, role, _) in versions.items()]
    out += ["", "## Settings it was trained with", "", f"`{json.dumps(r['params'], default=str)}`", "",
            "## Replacing the champion", "",
            "A challenger replaces this model only if it passes the same evaluation gate and beats it on the test split "
            "by a margin that is clearly more than noise (the 95% range of the paired PR-AUC difference must be entirely "
            "above zero). A tie is not a reason to switch.", "",
            "## Limits", "",
            "- Scores rank patients by risk. They are not calibrated probabilities (class weights inflate them).",
            "- Performance is weakest for the oldest patients; see the fairness section of `models/test_report.md`.",
            "- Not validated for clinical use.", ""]
    CHAMPION_REPORT.write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()