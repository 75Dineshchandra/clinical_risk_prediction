"""
Champion / challenger promotion (supports plan steps 9 and 15: the alias, and the retrain loop).

Run from the project root:
    python promote.py                       # challenger (alias @challenger) vs champion, apply if it wins
    python promote.py --dry-run             # show the decision, change nothing
    python promote.py --candidate-version 4 # use a specific registered version instead of @challenger
    python promote.py --rollback            # put @previous_champion back as champion

The rule (same thresholds as test.py, which is the single source for them):
    A candidate replaces the champion only if
      1. it passes every evaluation-gate check (lift, CV-to-test drop, fairness gaps, beats baseline), AND
      2. its PR-AUC beats the champion's with the paired patient-level bootstrap 95% range
         ENTIRELY above zero (a tie is not a reason to switch).
    Otherwise the champion stays.

What it writes
    models/promotion_log.jsonl    one line per decision (promote, reject, rollback): the audit trail
    models/promotion_report.md    plain-language report of the latest decision
    models/champion.json and models/champion_report.md   updated only when the champion changes

Warning: the default data is the test split, which has already been used to choose between models.
Re-using it for many challengers inflates results. Prefer fresh data (--data path/to/file.csv with the
same columns) once you have it; the holdout (step 11) is the right reserve for the final check.
"""

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

REGISTERED_NAME = "readmission"
EXPERIMENT = "readmission-30d"
REQUIRE_CI_ABOVE = 0.0    # lower end of the paired 95% range must exceed this


def load_gate():
    """Load test.py by path (its name clashes with Python's own 'test' package)."""
    path = Path(__file__).resolve().parent / "test.py"
    spec = importlib.util.spec_from_file_location("evaluation_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g = load_gate()
ROOT = g.ROOT
MODELS_DIR = ROOT / "models"
LOG_FILE = MODELS_DIR / "promotion_log.jsonl"
PROMO_REPORT = MODELS_DIR / "promotion_report.md"
CHAMPION_JSON = MODELS_DIR / "champion.json"
CHAMPION_REPORT = MODELS_DIR / "champion_report.md"
NAMES = {"logistic": "Logistic regression", "xgboost": "XGBoost", "lightgbm": "LightGBM"}


def get_version(client, ref):
    if str(ref).isdigit():
        return client.get_model_version(REGISTERED_NAME, str(ref))
    return client.get_model_version_by_alias(REGISTERED_NAME, ref)


def score(version, df):
    pipe = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}/{version.version}")
    cols = list(pipe.named_steps["prep"].feature_names_in_)
    return pipe.predict_proba(df[cols])[:, 1]


def family(v) -> str:
    return v.tags.get("model_family", f"v{v.version}")


def cv_score(client, v):
    run = client.get_run(v.run_id)
    return run.data.metrics.get("cv_pr_auc_mean")


def append_log(entry: dict) -> None:
    entry = {"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", **entry}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def refresh_champion_files(client, v, extra: dict) -> None:
    """Rewrite champion.json and champion_report.md so they describe the CURRENT champion."""
    t = v.tags
    cur = json.loads(CHAMPION_JSON.read_text(encoding="utf-8")) if CHAMPION_JSON.exists() else {}
    for stale in ("file", "params", "test_pr_auc_ci95", "gate_checks", "selection_reason",
                  "verified_against_joblib", "other_versions", "evaluated_at"):
        cur.pop(stale, None)
    num = lambda k: float(t[k]) if k in t else None
    cur.update({"registered_model": REGISTERED_NAME, "alias": "champion",
                "model_uri": f"models:/{REGISTERED_NAME}@champion", "version": int(v.version),
                "model_family": family(v), "training_run_id": t.get("training_run_id", v.run_id),
                "gate": t.get("gate"), "cv_pr_auc": num("cv_pr_auc"), "test_pr_auc": num("test_pr_auc"),
                "test_roc_auc": num("test_roc_auc"), "test_lift": num("test_lift"), **extra})
    CHAMPION_JSON.write_text(json.dumps(cur, indent=2, default=str), encoding="utf-8")
    CHAMPION_REPORT.write_text("\n".join([
        "# Champion model", "",
        f"Updated {datetime.now():%Y-%m-%d %H:%M} by `promote.py`. Replaced on every promotion or rollback.", "",
        f"- **Model:** {NAMES.get(family(v), family(v))} (registry `{REGISTERED_NAME}`, version {v.version}, alias `champion`).",
        f"- **Load it with:** `mlflow.sklearn.load_model(\"models:/{REGISTERED_NAME}@champion\")`",
        f"- **Training run:** `{cur['training_run_id']}`",
        f"- **Scores:** CV PR-AUC {cur['cv_pr_auc']}, test PR-AUC {cur['test_pr_auc']}, "
        f"test ROC-AUC {cur['test_roc_auc']}, lift {cur['test_lift']}x.",
        f"- **Why it is champion:** {extra.get('reason', 'see models/promotion_report.md')}", "",
        "Full evidence for the latest decision is in `models/promotion_report.md`; every decision is in "
        "`models/promotion_log.jsonl`. Scores rank patients by risk and are not calibrated probabilities.", ""]),
        encoding="utf-8")


def swap_aliases(client, new_v, old_v, new_role_tag="champion"):
    client.set_registered_model_alias(REGISTERED_NAME, "previous_champion", old_v.version)
    client.set_registered_model_alias(REGISTERED_NAME, "champion", new_v.version)
    client.set_model_version_tag(REGISTERED_NAME, new_v.version, "role", new_role_tag)
    client.set_model_version_tag(REGISTERED_NAME, old_v.version, "role", "previous_champion")
    try:  # the candidate is no longer a challenger
        if client.get_model_version_by_alias(REGISTERED_NAME, "challenger").version == new_v.version:
            client.delete_registered_model_alias(REGISTERED_NAME, "challenger")
    except Exception:
        pass


def do_rollback(client, dry_run: bool) -> None:
    cur = get_version(client, "champion")
    try:
        prev = get_version(client, "previous_champion")
    except Exception:
        raise SystemExit("No @previous_champion alias exists, so there is nothing to roll back to.")
    print(f"Rollback: champion v{cur.version} ({family(cur)}) -> v{prev.version} ({family(prev)})")
    if dry_run:
        print("Dry run: nothing changed.")
        return
    swap_aliases(client, prev, cur)
    reason = f"Rolled back from version {cur.version}."
    refresh_champion_files(client, prev, {"rolled_back_at": f"{datetime.now():%Y-%m-%d %H:%M}", "reason": reason})
    append_log({"decision": "rollback", "from_version": int(cur.version), "to_version": int(prev.version)})
    print("Done. champion.json, champion_report.md and promotion_log.jsonl updated.")
    print("The API still serves the old bundle. Refresh it: python serving/export_champion.py, then POST http://localhost:8000/reload")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-version", default=None, help="registered version number (default: @challenger)")
    ap.add_argument("--data", default=None, help="CSV to compare on (default: the test split)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"))
    client = MlflowClient()
    if args.rollback:
        return do_rollback(client, args.dry_run)

    try:
        champ = get_version(client, "champion")
    except MlflowException:
        raise SystemExit('No @champion in the registry. Register one first: '
                         'python register_champion.py --champion <model> --reason "..."')
    cand_ref = args.candidate_version or "challenger"
    try:
        cand = get_version(client, cand_ref)
    except MlflowException:
        raise SystemExit("No @challenger in the registry. Register one first: "
                         'python register_champion.py --challenger <model> --reason "..."'
                         if not args.candidate_version else
                         f"Version {cand_ref} was not found. Check the version number in the MLflow Models tab.")
    if cand.version == champ.version:
        raise SystemExit("Candidate and champion are the same version. Nothing to compare.")
    try:
        base = get_version(client, "baseline")
    except Exception:
        base = None

    data_path = Path(args.data) if args.data else g.TEST_FILE
    df = pd.read_csv(data_path, dtype={"encounter_id": str, g.PATIENT: str})
    y = df[g.TARGET].astype(int).to_numpy()
    prevalence = float(y.mean())
    s_champ, s_cand = score(champ, df), score(cand, df)

    pr_c, pr_n = average_precision_score(y, s_champ), average_precision_score(y, s_cand)
    roc_n = roc_auc_score(y, s_cand)
    lift = pr_n / prevalence
    cv = cv_score(client, cand)
    cutoff = np.quantile(s_cand, 1 - g.FLAG_FRACTION)
    recall = float(y[s_cand >= cutoff].sum() / y.sum())
    fair = g.fairness(df, y, s_cand, cutoff, roc_n, recall)
    gated = fair[fair["gated"]] if len(fair) else fair
    auc_gap = float(gated["auc_gap"].max()) if len(gated) else 0.0
    rec_gap = float(gated["recall_gap"].max()) if len(gated) else 0.0

    boot = g.patient_bootstrap(y, {"champ": s_champ, "cand": s_cand}, df[g.PATIENT], g.N_BOOT, g.SEED)
    k = min(len(boot["champ"]), len(boot["cand"]))
    diff = boot["cand"][:k] - boot["champ"][:k]
    d_lo, d_hi = (float(x) for x in np.percentile(diff, [2.5, 97.5]))

    checks = {f"lift >= {g.MIN_LIFT}": lift >= g.MIN_LIFT,
              f"ROC-AUC gap by group <= {g.MAX_GROUP_AUC_GAP}": auc_gap <= g.MAX_GROUP_AUC_GAP,
              f"recall gap by group <= {g.MAX_RECALL_GAP}": rec_gap <= g.MAX_RECALL_GAP}
    if cv is not None:
        checks[f"CV-to-test drop <= {g.MAX_CV_DROP}"] = (cv - pr_n) <= g.MAX_CV_DROP
    if g.MUST_BEAT_BASELINE and base is not None and family(cand) != "logistic":
        checks["beats logistic baseline"] = pr_n >= average_precision_score(y, score(base, df))
    checks[f"beats champion (95% range of PR-AUC gain above {REQUIRE_CI_ABOVE:+.3f})"] = d_lo > REQUIRE_CI_ABOVE
    checks = {c: bool(ok) for c, ok in checks.items()}   # numpy bools would be logged as strings
    promote = all(checks.values())

    cn, nn = NAMES.get(family(champ), family(champ)), NAMES.get(family(cand), family(cand))
    print(f"Champion  v{champ.version} {cn}: PR-AUC {pr_c:.3f}")
    print(f"Candidate v{cand.version} {nn}: PR-AUC {pr_n:.3f}  ROC-AUC {roc_n:.3f}  lift {lift:.2f}x")
    print(f"Paired PR-AUC gain (candidate - champion): {diff.mean():+.3f}  95% range [{d_lo:+.3f}, {d_hi:+.3f}]")
    for c, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {c}")
    print(f"\nDECISION: {'PROMOTE' if promote else 'KEEP CHAMPION'}" + ("  (dry run)" if args.dry_run else ""))
    if data_path == g.TEST_FILE:
        print("Note: compared on the test split, which has already been used for model selection.")

    reason = (f"{nn} beat {cn} on PR-AUC with the paired 95% range above zero and passed every gate check."
              if promote else "Candidate did not meet every condition; champion kept.")
    entry = {"decision": "promote" if promote else "reject", "dry_run": args.dry_run,
             "champion_version": int(champ.version), "candidate_version": int(cand.version),
             "champion_pr_auc": round(pr_c, 4), "candidate_pr_auc": round(pr_n, 4),
             "paired_gain_ci95": [round(d_lo, 4), round(d_hi, 4)], "checks": checks, "data": str(data_path)}
    if not args.dry_run:                                  # a dry run is not a decision
        append_log(entry)

    PROMO_REPORT.write_text("\n".join([
        "# Promotion report", "",
        f"Created {datetime.now():%Y-%m-%d %H:%M}. Generated by `promote.py`; replaced on every run.", "",
        f"## Decision: {'candidate promoted' if promote else 'champion kept'}" + (" (dry run, nothing changed)" if args.dry_run else ""), "",
        f"- Champion: {cn}, version {champ.version}, PR-AUC {pr_c:.3f}.",
        f"- Candidate: {nn}, version {cand.version}, PR-AUC {pr_n:.3f}, ROC-AUC {roc_n:.3f}, lift {lift:.2f}x.",
        f"- Difference in PR-AUC (candidate minus champion): {diff.mean():+.3f}, 95% range {d_lo:+.3f} to {d_hi:+.3f} "
        "(patients resampled 200 times). If the range includes zero, the data cannot tell the models apart.", "",
        "## Rule", "",
        "A candidate replaces the champion only if it passes every gate check and its improvement over the champion is "
        "clearly more than noise. A tie keeps the champion.", ""] +
        [f"- {'Pass' if ok else 'FAIL'}: {c}" for c, ok in checks.items()] + ["",
        "## Caution", "",
        "Compared on " + ("the test split, which has already been used to choose between models, so each extra comparison "
        "makes the result slightly more optimistic. Use fresh data or the holdout for the final check."
        if data_path == g.TEST_FILE else f"`{data_path.name}`."), ""]), encoding="utf-8")
    print(f"\nSaved: {PROMO_REPORT}" + ("" if args.dry_run else f"\n       {LOG_FILE}"))

    if promote and not args.dry_run:
        for k_, v_ in {"cv_pr_auc": f"{cv:.3f}" if cv is not None else "", "test_pr_auc": f"{pr_n:.3f}",
                       "test_roc_auc": f"{roc_n:.3f}", "test_lift": f"{lift:.2f}", "gate": "PASS",
                       "promotion_data": data_path.name, "promoted_at": f"{datetime.now():%Y-%m-%d %H:%M}",
                       "replaced_version": str(champ.version), "selection_reason": reason}.items():
            if v_ != "":
                client.set_model_version_tag(REGISTERED_NAME, cand.version, k_, v_)
        swap_aliases(client, cand, champ)
        cand = client.get_model_version(REGISTERED_NAME, cand.version)
        refresh_champion_files(client, cand, {"promoted_at": f"{datetime.now():%Y-%m-%d %H:%M}",
                                              "previous_version": int(champ.version), "reason": reason})
        print(f"Promoted: version {cand.version} is now @champion; version {champ.version} is @previous_champion.")
        print("The API still serves the old bundle. Refresh it: python serving/export_champion.py, then POST http://localhost:8000/reload")


if __name__ == "__main__":
    main()