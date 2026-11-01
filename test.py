"""
Evaluation gate (step 8): score every saved model ONCE on the test split.

Run from the project root:
    python test.py                      # gate every saved model
    python test.py --reference xgboost  # model used for the paired comparison and fairness detail
    python test.py --data data/splits/holdout.csv --no-mlflow   # same gate on other data, e.g. the holdout
    python test.py --no-mlflow          # skip logging to MLflow

Reads: data/splits/test.csv and models/NN_<model>_prauc<cv>.joblib
Writes: models/test_report.md (plain language), data/test_report.txt (technical),
        models/evaluation.json, and one MLflow run per model.

This script gates every model but does NOT choose the champion. Choosing one is a human decision,
recorded with a reason by register_champion.py. The "reference model" below is only the model the
others are compared against: --reference if given, else the current champion in models/champion.json,
else the model with the highest test PR-AUC.

Other data (--data): the same rules and the same models, but results go to SEPARATE files named after the
data (evaluation_<name>.json, <name>_report.md, data/<name>_report.txt), so models/evaluation.json, which
register_champion.py reads, is never overwritten. Every use of any dataset is appended to
models/data_use_log.jsonl. Treat the holdout as single-use: look, then decide; do not iterate on it.

Nothing is retrained: the saved pipelines are loaded as-is, so parameters and preprocessing
are exactly those used in training. Feature columns are checked against each pipeline.

The pass/fail thresholds below are FIXED BEFORE looking at test results.
Do not edit them after seeing the numbers. Do not tune models on the test split.
Holdout (step 11) is never loaded here.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# ---------------- Gate criteria (pre-registered) ----------------
MIN_LIFT = 1.5            # test PR-AUC / test positive rate must be at least this
MAX_CV_DROP = 0.02        # CV PR-AUC minus test PR-AUC must be at most this
MUST_BEAT_BASELINE = True # non-baseline models must score >= logistic baseline on test
MAX_GROUP_AUC_GAP = 0.05  # subgroup ROC-AUC may be at most this far below overall
MIN_GROUP_ROWS = 500      # subgroups smaller than this are listed but not gated
MIN_GROUP_POSITIVES = 50
FLAG_FRACTION = 0.20      # "flag the top 20% highest-risk stays" for recall checks
MAX_RECALL_GAP = 0.10     # subgroup recall at that cutoff may be at most this far below overall
FAIRNESS_COLUMNS = ["gender", "race", "age"]
N_BOOT = 200
# Calibration is reported but NOT gated: models were trained with class weights,
# so raw probabilities are expected to be inflated (needs a calibration step later).
# ------------------------------------------------------------------

TARGET = "readmitted"
PATIENT = "patient_nbr"
SEED = 42
NOT_FEATURES = ["encounter_id", PATIENT, "readmitted_raw", TARGET, "diag_1", "diag_2", "diag_3"]
EXPERIMENT = "readmission-30d"


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find a data/ folder next to or above {here}")


ROOT = find_project_root()
TEST_FILE = ROOT / "data" / "splits" / "test.csv"
MODELS_DIR = ROOT / "models"
REPORT_FILE = ROOT / "data" / "test_report.txt"
EVAL_JSON = MODELS_DIR / "evaluation.json"
PLAIN_REPORT = MODELS_DIR / "test_report.md"
DATA_USE_LOG = MODELS_DIR / "data_use_log.jsonl"
KEY_PARAMS = {"n_estimators", "learning_rate", "max_depth", "num_leaves", "subsample", "subsample_freq",
              "colsample_bytree", "min_child_weight", "min_child_samples", "scale_pos_weight",
              "C", "class_weight", "max_iter", "random_state"}


def load_models() -> dict:
    found = {}
    for p in sorted(MODELS_DIR.glob("[0-9][0-9]_*_prauc*.joblib")):
        m = re.match(r"\d+_(.+)_prauc([\d.]+)\.joblib", p.name)
        if m:
            found[m.group(1)] = {"file": p, "cv_pr_auc": float(m.group(2)), "pipe": joblib.load(p)}
    if not found:
        raise FileNotFoundError(f"No models found in {MODELS_DIR}. Run training first.")
    return found


def ece(y, p, bins=10) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    total = 0.0
    for chunk in np.array_split(order, bins):
        total += len(chunk) / len(y) * abs(p[chunk].mean() - y[chunk].mean())
    return float(total)


def recall_at_top(y, s, frac) -> float:
    """Recall if we flag the top `frac` of this array by score (cutoff supplied separately)."""
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-s)[:k]
    return float(y[top].sum() / max(1, y.sum()))


def patient_bootstrap(y, scores, groups, n_boot, seed):
    """Resample whole patients. Returns {model: array of PR-AUC} so differences can be paired."""
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    idx_by_pat = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)
    out = {m: [] for m in scores}
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_pat), len(idx_by_pat))
        idx = np.concatenate([idx_by_pat[i] for i in pick])
        if y[idx].sum() == 0:
            continue
        for m, s in scores.items():
            out[m].append(average_precision_score(y[idx], s[idx]))
    return {m: np.array(v) for m, v in out.items()}


def fairness(df, y, s, cutoff, overall_auc, overall_recall):
    rows = []
    for col in FAIRNESS_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].astype(str)
        for level in sorted(vals.unique()):
            mask = (vals == level).to_numpy()
            n, pos = int(mask.sum()), int(y[mask].sum())
            gated = n >= MIN_GROUP_ROWS and pos >= MIN_GROUP_POSITIVES and pos < n
            auc = float(roc_auc_score(y[mask], s[mask])) if 0 < pos < n else float("nan")
            flagged = s[mask] >= cutoff
            rec = float(y[mask][flagged].sum() / pos) if pos else float("nan")
            rows.append({"column": col, "group": level, "n": n, "positives": pos, "roc_auc": auc,
                         "recall_at_cutoff": rec, "flag_rate": float(flagged.mean()), "gated": gated,
                         "auc_gap": overall_auc - auc if gated else float("nan"),
                         "recall_gap": overall_recall - rec if gated else float("nan")})
    return pd.DataFrame(rows)


def choose_reference(requested, results):
    """Reference model for paired comparisons. Not a champion decision."""
    if requested:
        if requested not in results:
            raise ValueError(f"Reference '{requested}' not in saved models {sorted(results)}")
        return requested, "chosen with --reference"
    cj = MODELS_DIR / "champion.json"
    if cj.exists():
        try:
            fam = json.loads(cj.read_text(encoding="utf-8")).get("model_family")
        except ValueError:
            fam = None
        if fam in results:
            return fam, "current champion in models/champion.json"
    best = max(results, key=lambda n: results[n]["test_pr_auc"])
    return best, "highest test PR-AUC (no champion registered yet)"


def relabel(text: str, label: str) -> str:
    """Reports are written for the test split; reword them when another dataset is evaluated."""
    if label == "test":
        return text
    short = (label[:4] + " PR").rjust(7)
    swaps = [
        ("# Test evaluation report", f"# Evaluation report on {label} data"),
        ("on the test split they have never seen", f"on the {label} data they have never seen"),
        ("before the test results were seen", f"before the {label} results were seen"),
        ("before the test data was scored", f"before the {label} data was scored"),
        ("## The test data in brief", f"## The {label} data in brief"),
        ("Test PR-AUC", f"{label.capitalize()} PR-AUC"),
        ("a big drop on test would suggest", f"a big drop on {label} data would suggest"),
        ("`data/test_report.txt`", f"`data/{label}_report.txt`"),
        ("EVALUATION GATE (test split)", f"EVALUATION GATE ({label} data)"),
        ("Test rows:", f"Rows ({label}):"),
        ("Test results (PR-AUC", f"Results on {label} data (PR-AUC"),
        ("  test PR", "  " + short),
        ("CV-to-test drop", f"CV-to-{label} drop"),
        ("- The test split has now been used to choose between models; do not tune on it. The holdout (step 11) is the next untouched check.",
         f"- Results on {label} data are for confirmation. Once they have informed a decision, this data is no longer a clean "
         f"final check: do not tune on it, and every use is logged in models/data_use_log.jsonl."),
        ("Reminder: the test split has now been used for selection. Do not tune on it;",
         f"Reminder: {label} results are for confirmation. Do not tune on {label} data;"),
        ("any new challenger needs the same gate plus a fresh check against the holdout later.",
         "every use of this data is logged; once it has informed a decision it is no longer a clean final check."),
    ]
    for old, new in swaps:
        text = text.replace(old, new)
    return text


def write_plain_report(results, boot, champion, prevalence, n_rows, n_patients, versions, basis="",
                       data_label="test", out_path=None) -> None:
    """Write models/test_report.md: the gate results explained for people who did not run the code."""
    label = {"logistic": "Logistic regression", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    nm = lambda n: label.get(n, n)
    ch = results[champion]
    out = ["# Test evaluation report", "",
           "Created {:%Y-%m-%d %H:%M}. Generated automatically by `test.py` and replaced on every run.".format(datetime.now()), "",
           "## What this report shows", "",
           "The models were trained earlier on the training data only. This report scores the saved models, "
           "unchanged, on the test split they have never seen. Nothing was retrained or tuned here. "
           "This is the evaluation gate (step 8): a model is promoted only if it passes rules that were "
           "written down before the test results were seen.", "",
           "## The test data in brief", "",
           "- {:,} hospital stays from {:,} patients.".format(n_rows, n_patients),
           "- About {:.1%} ended in a readmission within 30 days, so plain accuracy would be misleading.".format(prevalence), "",
           "## Results", "",
           "Best first. The 95% range comes from resampling whole patients 200 times; "
           "if two models' ranges overlap heavily, treat them as tied.", "",
           "| Model | CV PR-AUC | Test PR-AUC | 95% range | ROC-AUC | Lift over chance | Gate |",
           "|---|---|---|---|---|---|---|"]
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["test_pr_auc"]):
        lo, hi = r["pr_auc_ci95"]
        out.append("| {} | {:.3f} | {:.3f} | {:.3f} to {:.3f} | {:.3f} | {:.2f}x | {} |".format(
            nm(n), r["cv_pr_auc"], r["test_pr_auc"], lo, hi, r["test_roc_auc"], r["lift"],
            "PASS" if r["passed"] else "FAIL"))
    out += ["", "No-skill PR-AUC is about {:.3f} (the positive rate). Lift = PR-AUC / positive rate. "
            "CV PR-AUC is the training-time estimate; a big drop on test would suggest overfitting.".format(prevalence), "",
            "## Gate rules and outcome", "",
            "Rules were fixed in `test.py` before the test data was scored.", ""]
    for n, r in results.items():
        out.append("**{}**: {}".format(nm(n), "passes all checks" if r["passed"] else "FAILS"))
        out.append("")
        for c, ok in r["checks"].items():
            out.append("- {} {}".format("Pass:" if ok else "FAIL:", c))
        out.append("")
    if len(results) > 1:
        out += ["## {} against the other models".format(nm(champion)), "",
                "(Reference model: {}.)".format(basis), "",
                "Difference in PR-AUC, resampled by patient. If the range includes zero, the data cannot tell the models apart.", ""]
        for n in results:
            if n == champion:
                continue
            k = min(len(boot[n]), len(boot[champion]))
            d = boot[champion][:k] - boot[n][:k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            v = "reference clearly better" if lo > 0 else ("other model clearly better" if hi < 0 else "no clear difference")
            out.append("- vs {}: {:+.3f} (range {:+.3f} to {:+.3f}): {}.".format(nm(n), d.mean(), lo, hi, v))
        out.append("")
    passed = [nm(n) for n, r in results.items() if r["passed"]]
    failed = [nm(n) for n, r in results.items() if not r["passed"]]
    out += ["## Decision", "",
            "Passed the gate (eligible to be registered): " + (", ".join(passed) or "none") + ".",
            "Failed the gate: " + (", ".join(failed) or "none") + ".", "",
            "The gate does not choose a champion. Choosing one is a human decision, recorded with a reason "
            "by `register_champion.py`.", "",
            "## Fairness", "",
            "Flagging the top {:.0%} of stays by risk catches {:.1%} of readmissions overall. "
            "Worst subgroup shortfall for the reference model ({}): ROC-AUC {:.3f} below overall, recall {:.3f} below overall "
            "(groups under {} stays or {} readmissions are listed in `data/test_report.txt` but not judged)."
            .format(FLAG_FRACTION, ch["recall_at_cutoff"], nm(champion), ch["worst_auc_gap"],
                    ch["worst_recall_gap"], MIN_GROUP_ROWS, MIN_GROUP_POSITIVES), "",
            "## Calibration", "",
            "Mean predicted risk for the reference model ({}) is {:.3f} against an actual rate of {:.3f} (Brier {:.3f}, ECE {:.3f}). "
            "The models were trained with class weights, so raw scores run higher than real probabilities. "
            "They rank patients well but should not be read as 'a 30% chance' until a calibration step is added. "
            "This is reported, not gated.".format(nm(champion), ch["mean_pred"], prevalence, ch["brier"], ch["ece"]), "",
            "## Parameters and versions used", "",
            "Models were loaded from their saved files, not rebuilt, so these are the training-time settings.", ""]
    for n, r in results.items():
        out.append("- **{}** (`{}`): `{}`".format(nm(n), r["file"], json.dumps(r["params"], default=str)))
    out += ["", "Packages at test time: " + ", ".join("{} {}".format(k, v) for k, v in versions.items()) +
            ". Compare with the versions used for training.", "",
            "## What these results do not tell you", "",
            "- The test split has now been used to choose between models; do not tune on it. The holdout (step 11) is the next untouched check.",
            "- The data has no dates, so behaviour over time is untested.",
            "- This is not a clinical tool and has not been validated for use with real patients.", ""]
    (out_path or PLAIN_REPORT).write_text(relabel("\n".join(out), data_label), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", "--champion", dest="reference", default=None,
                    help="model for paired comparison and fairness detail (default: current champion, else best test PR-AUC)")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--data", default=None, help="CSV to evaluate on (default: the test split)")
    ap.add_argument("--name", default=None, help="short name for this data in reports (default: the file name)")
    args = ap.parse_args()

    data_path = Path(args.data).resolve() if args.data else TEST_FILE.resolve()
    is_test_file = data_path == TEST_FILE.resolve()
    label = args.name or ("test" if is_test_file else data_path.stem)
    if label == "test" and not is_test_file:
        raise SystemExit("The name 'test' is reserved for the real test split. Choose another --name.")
    is_test = label == "test"
    report_file = REPORT_FILE if is_test else ROOT / "data" / f"{label}_report.txt"
    eval_json = EVAL_JSON if is_test else MODELS_DIR / f"evaluation_{label}.json"
    plain_report = PLAIN_REPORT if is_test else MODELS_DIR / f"{label}_report.md"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}." + (" Run split_data.py first." if is_test_file else ""))
    df = pd.read_csv(data_path, dtype={"encounter_id": str, PATIENT: str})
    for needed in (TARGET, PATIENT):
        if needed not in df.columns:
            raise ValueError(f"{data_path.name} has no '{needed}' column. It must be cleaned like the training data and include the outcome.")
    if not is_test and DATA_USE_LOG.exists():
        prior = [json.loads(l) for l in DATA_USE_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
        prior = [e for e in prior if e.get("data_name") == label]
        if prior:
            print(f"NOTE: '{label}' data has already been used {len(prior)} time(s); the last time was {prior[-1]['time']}. "
                  f"Each further look makes it a less clean final check.")
    if not is_test:
        print(f"NOTE: results on '{label}' data are for confirmation only. Do not tune on them.\n")
    y = df[TARGET].astype(int).to_numpy()
    X = df[[c for c in df.columns if c not in NOT_FEATURES]]
    prevalence = float(y.mean())
    models = load_models()

    for n, m in models.items():
        expected = list(m["pipe"].named_steps["prep"].feature_names_in_)
        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise ValueError(f"{n}: test data lacks columns the model was trained on: {missing}")
        extra = [c for c in X.columns if c not in expected]
        if extra:
            print(f"[{n}] note: ignoring test columns not used in training: {extra}")
    scores = {n: m["pipe"].predict_proba(X[list(m["pipe"].named_steps["prep"].feature_names_in_)])[:, 1]
              for n, m in models.items()}
    boot = patient_bootstrap(y, scores, df[PATIENT], N_BOOT, SEED)

    results = {}
    for name, m in models.items():
        s = scores[name]
        pr, roc = float(average_precision_score(y, s)), float(roc_auc_score(y, s))
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        cutoff = np.quantile(s, 1 - FLAG_FRACTION)
        overall_recall = float(y[s >= cutoff].sum() / y.sum())
        fair = fairness(df, y, s, cutoff, roc, overall_recall)
        gated = fair[fair["gated"]] if len(fair) else fair
        results[name] = {
            "file": m["file"].name, "cv_pr_auc": m["cv_pr_auc"], "test_pr_auc": pr,
            "pr_auc_ci95": [float(lo), float(hi)], "test_roc_auc": roc, "lift": pr / prevalence,
            "cv_drop": m["cv_pr_auc"] - pr, "brier": float(brier_score_loss(y, s)),
            "ece": ece(y, s), "mean_pred": float(s.mean()), "recall_at_cutoff": overall_recall,
            "worst_auc_gap": float(gated["auc_gap"].max()) if len(gated) else 0.0,
            "worst_recall_gap": float(gated["recall_gap"].max()) if len(gated) else 0.0,
            "fairness": fair,
            "params": {k: v for k, v in m["pipe"].named_steps["model"].get_params().items()
                       if k in KEY_PARAMS and (v is None or isinstance(v, (int, float, str, bool)))},
        }

    base_pr = results["logistic"]["test_pr_auc"] if "logistic" in results else None
    for name, r in results.items():
        checks = {
            f"lift >= {MIN_LIFT}": r["lift"] >= MIN_LIFT,
            f"CV-to-test drop <= {MAX_CV_DROP}": r["cv_drop"] <= MAX_CV_DROP,
            f"ROC-AUC gap by group <= {MAX_GROUP_AUC_GAP}": r["worst_auc_gap"] <= MAX_GROUP_AUC_GAP,
            f"recall gap by group <= {MAX_RECALL_GAP}": r["worst_recall_gap"] <= MAX_RECALL_GAP,
        }
        if MUST_BEAT_BASELINE and base_pr is not None and name != "logistic":
            checks["beats logistic baseline"] = r["test_pr_auc"] >= base_pr
        r["checks"] = {c: bool(ok) for c, ok in checks.items()}   # plain booleans, never numpy
        r["passed"] = all(r["checks"].values())
    ref, basis = choose_reference(args.reference, results)

    # ---------- report ----------
    L = ["EVALUATION GATE (test split)", "=" * 60,
         f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         f"Test rows: {len(df):,}   Patients: {df[PATIENT].nunique():,}   Positive rate: {prevalence:.2%}",
         f"Reference model (for comparisons and fairness detail): {ref}, {basis}", "",
         "Test results (PR-AUC with patient-level bootstrap 95% CI)",
         f"{'model':<10}{'CV PR':>8}{'test PR':>9}{'95% CI':>18}{'ROC':>7}{'lift':>7}{'drop':>8}{'gate':>7}"]
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["test_pr_auc"]):
        ci = "{:.3f}-{:.3f}".format(*r["pr_auc_ci95"])
        L.append(f"{n:<10}{r['cv_pr_auc']:>8.3f}{r['test_pr_auc']:>9.3f}{ci:>18}{r['test_roc_auc']:>7.3f}"
                 f"{r['lift']:>6.2f}x{r['cv_drop']:>+8.3f}{'PASS' if r['passed'] else 'FAIL':>7}")
    L += ["", "Gate checks"]
    for n, r in results.items():
        L.append(f"  {n}:")
        L += [f"    [{'x' if ok else ' '}] {c}" for c, ok in r["checks"].items()]
    if len(results) > 1:
        L += ["", "Paired bootstrap: PR-AUC difference vs the reference model (positive = reference better)"]
        for n in results:
            if n == ref:
                continue
            k = min(len(boot[n]), len(boot[ref]))
            d = boot[ref][:k] - boot[n][:k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            verdict = "reference clearly better" if lo > 0 else ("other model clearly better" if hi < 0 else "no clear difference")
            L.append(f"  {ref} - {n}: {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
    L += ["", "Calibration (informational, not gated: class weights inflate probabilities)",
          f"  actual positive rate: {prevalence:.3f}"]
    for n, r in results.items():
        L.append(f"  {n:<10} mean predicted {r['mean_pred']:.3f}   Brier {r['brier']:.3f}   ECE {r['ece']:.3f}")
    ch = results[ref]
    L += ["", f"Fairness detail for {ref} (flag top {FLAG_FRACTION:.0%} by score; overall recall {ch['recall_at_cutoff']:.3f})",
          f"{'column':<8}{'group':<16}{'n':>7}{'pos':>6}{'ROC':>7}{'recall':>8}{'flag%':>7}  gated"]
    for _, f in ch["fairness"].iterrows():
        L.append(f"{f['column']:<8}{f['group']:<16}{f['n']:>7}{f['positives']:>6}{f['roc_auc']:>7.3f}"
                 f"{f['recall_at_cutoff']:>8.3f}{f['flag_rate']:>7.1%}  {'yes' if f['gated'] else 'no (small)'}")
    L += ["", "GATE OUTCOME (the gate does not choose a champion; that is a human decision):",
          *[f"  {n}: {'PASS (eligible to register)' if r['passed'] else 'FAIL (do not promote)'}" for n, r in results.items()],
          "", "Reminder: the test split has now been used for selection. Do not tune on it;",
          "any new challenger needs the same gate plus a fresh check against the holdout later."]
    text = relabel("\n".join(L), label)
    report_file.write_text(text, encoding="utf-8")
    slim = {n: {k: v for k, v in r.items() if k not in ("fairness",)} for n, r in results.items()}
    eval_json.write_text(json.dumps({"created": f"{datetime.now():%Y-%m-%d %H:%M}", "data_name": label,
                                     "data_file": str(data_path), "reference_model": ref, "reference_basis": basis,
                                     "results": slim}, indent=2, default=str), encoding="utf-8")
    with open(DATA_USE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "data_name": label, "data_file": str(data_path),
                            "rows": int(len(df)), "models": sorted(results), "reference_model": ref,
                            "gate": {n: r["passed"] for n, r in results.items()}}) + "\n")
    import sklearn, lightgbm, xgboost
    versions = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__}
    write_plain_report(results, boot, ref, prevalence, len(df), int(df[PATIENT].nunique()), versions, basis, label, plain_report)
    print(text)
    print(f"\nPlain-language report: {plain_report}\nTechnical report:      {report_file}\nJSON summary:          {eval_json}")

    if not args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"))
        mlflow.set_experiment(EXPERIMENT)
        for n, r in results.items():
            with mlflow.start_run(run_name=f"eval-{n}-" + ("" if is_test else f"{label}-") + f"{datetime.now():%m%d-%H%M}"):
                mlflow.set_tags({"stage": "evaluation", "model_family": n, "evaluated_on": "test split" if is_test else f"{label} data",
                                 "gate": "PASS" if r["passed"] else "FAIL"})
                mlflow.log_metrics({"test_pr_auc": r["test_pr_auc"], "test_roc_auc": r["test_roc_auc"],
                                    "test_lift": r["lift"], "cv_to_test_drop": r["cv_drop"],
                                    "test_brier": r["brier"], "test_ece": r["ece"],
                                    "worst_group_auc_gap": r["worst_auc_gap"],
                                    "worst_group_recall_gap": r["worst_recall_gap"]})
                mlflow.log_artifact(str(report_file))
                mlflow.log_artifact(str(plain_report))
                mlflow.log_params({f"model_{k}": v for k, v in r["params"].items()})


if __name__ == "__main__":
    main()