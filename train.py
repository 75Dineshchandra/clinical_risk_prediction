"""
Train models and log every run to MLflow (steps 6 and 7).

Run from the project root (this file can sit in the main folder, in code/, or in data/):
    python train.py
    python train.py --models logistic                 # one model only (do not use: it deletes the other ranked files)
    python train.py --cv-folds 3                      # quicker check
    python train.py --data-tag v2-clean-train-plus-holdout70    # label recorded in MLflow for this data version
    python train.py --train-file path/to/other_train.csv        # train on another file (never test.csv or holdout.csv)

Reads ONLY data/splits/train.csv (or --train-file). Test and holdout are never loaded here:
the test set is for the evaluation gate (step 8), the holdout for the replay (step 11).

Data version (new): every MLflow run gets the tags `data_tag` and `train_file` and the parameter
`train_file_sha256`, so you can always tell which data trained which model. Without --data-tag the tag is "default".

For every model it
- builds one pipeline: preprocessing + model,
- scores it with patient-grouped, stratified cross-validation inside train,
- fits the final pipeline on all of train and logs it to MLflow,
- logs parameters, CV metrics, feature list, split report, cleaning config,
- logs explanations: SHAP (XGBoost, LightGBM) or coefficients (logistic regression).
A model that crashes is recorded as FAILED and the others still run.

Readable outputs (MLflow's own folders have random IDs by design; use these instead):
    models/01_xgboost_prauc0.231.joblib   models ranked by CV PR-AUC, score in the name
    models/02_logistic_prauc0.220.joblib
    models/best_model.json                which model ranked first, its run ID and file
    models/model_report.md                plain-language report for people who did not run the code
    data/training_report.txt              technical summary: settings, results, best model, failures

The models/ files are replaced on every run; the history of all runs stays in MLflow.
"Best" means best cross-validated PR-AUC on train. It is a candidate until the
evaluation gate (step 8) confirms it on the test split.

Tracking: MLFLOW_TRACKING_URI if set, else a SQLite file at <project root>/mlflow.db.
View results with:   mlflow ui --backend-store-uri sqlite:///mlflow.db

Needs: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, matplotlib, mlflow, joblib.
"""

import argparse
import hashlib
import json
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier



def find_project_root() -> Path:
    """The folder that contains data/. Works if this file is in the main folder, in code/, or in data/."""
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find a data/ folder next to or above {here}")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_FILE = DATA_DIR / "splits" / "train.csv"
REPORT_FILE = DATA_DIR / "training_report.txt"
MODEL_REPORT_FILE = MODELS_DIR / "model_report.md"
EXTRA_ARTIFACTS = [DATA_DIR / "splits" / "split_report.txt", DATA_DIR / "cleaning_config.json"]

EXPERIMENT = "readmission-30d"
TARGET = "readmitted"
PATIENT = "patient_nbr"
SEED = 42
MIN_CATEGORY_ROWS = 50          # rarer categories go into one "infrequent" bucket
SHAP_SAMPLE_ROWS = 2000

# Not features: identifiers, the target in raw form, and raw diagnosis codes
# (hundreds of values each; the diag_*_group columns stand in for them).
NOT_FEATURES = ["encounter_id", PATIENT, "readmitted_raw", TARGET, "diag_1", "diag_2", "diag_3"]
NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "n_drugs_taken", "n_drugs_changed",
]
LOG_COUNTS = ["number_outpatient", "number_emergency", "number_inpatient"]  # heavily skewed

MODEL_PARAMS = {
    "logistic": {"C": 1.0, "class_weight": "balanced", "max_iter": 2000},
    "xgboost": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8,
                "colsample_bytree": 0.8, "min_child_weight": 5},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
                 "subsample_freq": 1, "colsample_bytree": 0.8, "min_child_samples": 40},
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_preprocessor(categorical: list) -> ColumnTransformer:
    other_numeric = [c for c in NUMERIC if c not in LOG_COUNTS]
    log_step = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("log_counts", log_step, LOG_COUNTS),
        ("numeric", StandardScaler(), other_numeric),
        ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=MIN_CATEGORY_ROWS, sparse_output=True), categorical),
    ])


def build_model(name: str, pos_weight: float):
    p = dict(MODEL_PARAMS[name])
    if name == "logistic":
        return LogisticRegression(random_state=SEED, **p), p
    if name == "xgboost":
        p["scale_pos_weight"] = round(pos_weight, 3)
        return XGBClassifier(eval_metric="aucpr", tree_method="hist", random_state=SEED, n_jobs=-1, **p), p
    p["scale_pos_weight"] = round(pos_weight, 3)
    return LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1, **p), p


def log_explanations(name: str, pipe: Pipeline, X: pd.DataFrame, tmp: Path) -> pd.Series:
    """Logs explanations and returns the feature ranking (|coefficient| or mean |SHAP|)."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    if name == "logistic":
        coefs = pd.Series(model.coef_[0], index=names)
        out = coefs.reindex(coefs.abs().sort_values(ascending=False).index).rename("coefficient")
        out.to_csv(tmp / "logistic_coefficients.csv")
        mlflow.log_artifact(str(tmp / "logistic_coefficients.csv"), "explanations")
        return out
    sample = X.sample(min(SHAP_SAMPLE_ROWS, len(X)), random_state=SEED)
    Xs = prep.transform(sample)
    Xs = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    mean_abs.rename("mean_abs_shap").to_csv(tmp / "shap_importance.csv")
    shap.summary_plot(sv, Xs, feature_names=names, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(tmp / "shap_summary.png", dpi=130)
    plt.close()
    mlflow.log_artifact(str(tmp / "shap_importance.csv"), "explanations")
    mlflow.log_artifact(str(tmp / "shap_summary.png"), "explanations")
    return mean_abs


def log_pipeline(pipe: Pipeline, name: str, example: pd.DataFrame) -> None:
    """Log the fitted pipeline with a readable name (cloudpickle: skops rejects XGBoost/LightGBM)."""
    kwargs = dict(input_example=example,
                  serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
    try:
        mlflow.sklearn.log_model(pipe, name=f"readmission_{name}", **kwargs)
    except TypeError:  # MLflow 2.x has no `name` argument
        mlflow.sklearn.log_model(pipe, artifact_path=f"readmission_{name}", **kwargs)


def export_ranked_models(ok: list) -> list:
    """Save fitted pipelines as readable files ranked by CV PR-AUC. Returns the ranking."""
    MODELS_DIR.mkdir(exist_ok=True)
    for old in MODELS_DIR.glob("[0-9][0-9]_*_prauc*.joblib"):   # only files this script made
        old.unlink()
    ranked = sorted(ok, key=lambda r: r["pr_mean"], reverse=True)
    for i, r in enumerate(ranked, start=1):
        path = MODELS_DIR / f"{i:02d}_{r['name']}_prauc{r['pr_mean']:.3f}.joblib"
        joblib.dump(r["pipe"], path)
        r["file"] = path
    best = ranked[0]
    (MODELS_DIR / "best_model.json").write_text(json.dumps({
        "model": best["name"], "file": best["file"].name, "mlflow_run_id": best["run_id"],
        "cv_pr_auc_mean": round(best["pr_mean"], 4), "cv_roc_auc_mean": round(best["roc_mean"], 4),
        "status": "candidate: chosen by cross-validation on train only; confirm on test (step 8)",
        "created": f"{datetime.now():%Y-%m-%d %H:%M}",
    }, indent=2), encoding="utf-8")
    return ranked


MODEL_BLURBS = {
    "logistic": ("Logistic regression",
                 "A simple model that adds up weighted evidence from each feature. Easy to explain. "
                 "It is the baseline: any more complex model has to beat it to be worth using."),
    "xgboost": ("XGBoost",
                "Many small decision trees built one after another, each fixing the mistakes of the "
                "previous ones. It can pick up interactions between features."),
    "lightgbm": ("LightGBM",
                 "Another tree-boosting model, built differently from XGBoost. Included as a second opinion."),
}


def original_column(feature_name: str, categorical: list) -> str:
    """Map a one-hot column such as 'age_[70-80)' back to its source column 'age'."""
    best = ""
    for col in categorical:
        if feature_name.startswith(col + "_") and len(col) > len(best):
            best = col
    return best or feature_name


def write_model_report(ranked: list, failed: list, prevalence: float, n_rows: int, n_patients: int,
                       n_features: int, cv_folds: int, categorical: list, data_tag: str = "default") -> None:
    """Write models/model_report.md: the results explained for people who did not run the code."""
    label = lambda n: MODEL_BLURBS[n][0]
    out = [
        "# Readmission model report", "",
        "Created {:%Y-%m-%d %H:%M}. Generated automatically by `train_models.py` and replaced on every run.".format(datetime.now()),
        "", "## What this report shows", "",
        "We trained models that give each diabetic hospital patient a risk score for being readmitted "
        "within 30 days of discharge. This report compares the models and explains the numbers. "
        "Every result here comes from the training data only (cross-validation). "
        "The test data has not been used yet, so these are **not final results**.", "",
        "## The data in brief", "",
        "- {:,} hospital stays from {:,} patients (the training file, data version `{}`).".format(n_rows, n_patients, data_tag),
        "- About {:.1%} of stays ended in a readmission within 30 days. This is the minority outcome, "
        "so the data is imbalanced and plain accuracy would be misleading.".format(prevalence),
        "- {} pieces of information per stay: age, length of stay, prior visits, lab tests, medications, "
        "diagnosis groups, and so on.".format(n_features),
        "- Stays from the same patient are always kept together when checking the model, "
        "so each model is judged on patients it has not seen.", "",
        "## Models compared", "", "| Model | What it is |", "|---|---|",
    ]
    out += ["| {} | {} |".format(*MODEL_BLURBS[n]) for n in MODEL_BLURBS if any(r["name"] == n for r in ranked + failed)]

    if not ranked:
        out += ["", "## Results", "", "No model finished training. See the failures below and `data/training_report.txt`."]
    else:
        out += ["", "## Results", "",
                "Best first. Scores are averages over {} validation rounds; the number after the plus-minus sign "
                "shows how much the score moved between rounds.".format(cv_folds), "",
                "| Rank | Model | PR-AUC | ROC-AUC | Lift over chance |", "|---|---|---|---|---|"]
        for i, r in enumerate(ranked, start=1):
            out.append("| {} | {} | {:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} | {:.2f}x |".format(
                i, label(r["name"]), r["pr_mean"], r["pr_std"], r["roc_mean"], r["roc_std"], r["pr_mean"] / prevalence))
        out += ["", "### How to read these numbers", "",
                "- **PR-AUC** (precision-recall area under the curve) measures how well the model puts the patients "
                "who really were readmitted near the top of its risk ranking without flagging too many who were not. "
                "A model with no skill scores about {:.3f} (the share of readmitted patients). 1.0 is perfect.".format(prevalence),
                "- **Lift over chance** is the PR-AUC divided by that no-skill value. A lift of 1.0x means no better than guessing; "
                "1.5x means the model finds readmitted patients about one and a half times as well as chance.",
                "- **ROC-AUC** is the chance that a randomly chosen readmitted patient gets a higher risk score than a "
                "randomly chosen patient who was not readmitted. 0.5 is a coin flip and 1.0 is perfect.",
                "- **The plus-minus value** shows stability. If two models differ by less than this, treat them as tied.", ""]

        best = ranked[0]
        out += ["## Verdict", "",
                "- Best model by PR-AUC: **{}**, with {:.3f} (lift {:.2f}x over chance).".format(
                    label(best["name"]), best["pr_mean"], best["pr_mean"] / prevalence)]
        lift = best["pr_mean"] / prevalence
        if lift < 1.1:
            strength = "barely better than guessing"
        elif lift < 1.5:
            strength = "a modest signal"
        else:
            strength = "a useful signal"
        out.append("- Rule of thumb used here: lift under 1.1x is barely better than guessing, 1.1x to 1.5x is a modest signal, "
                   "above 1.5x is useful. This model shows {}. Readmission is hard to predict from hospital records alone, "
                   "so modest scores are not a sign of a bug.".format(strength))
        base = next((r for r in ranked if r["name"] == "logistic"), None)
        if base is not None and best is not base:
            gain = best["pr_mean"] - base["pr_mean"]
            noise = max(best["pr_std"], base["pr_std"])
            if gain > 2 * noise:
                text = "clearly better than the simple baseline (gain {:+.3f}, more than twice the round-to-round variation).".format(gain)
            elif gain > noise:
                text = "somewhat better than the simple baseline (gain {:+.3f}), but the gap is only a little larger than the round-to-round variation.".format(gain)
            else:
                text = "not meaningfully better than the simple baseline (gain {:+.3f}, within the round-to-round variation). The simpler model is the safer choice.".format(gain)
            out.append("- {} is {}".format(label(best["name"]), text))
        elif base is not None and len(ranked) > 1:
            out.append("- No tree model beat the simple logistic baseline, so the simpler, easier-to-explain model is preferred.")
        out.append("- The ranking is a shortlist. The test data decides the final choice in the evaluation gate (step 8).")
        out.append("")

        out += ["## What drives the predictions", "",
                "Names look like `column_value`, for example `age_[70-80)` means the age group 70 to 80, and "
                "`_infrequent_sklearn` means rare values of that column grouped together. "
                "Influence is not the same as cause: these are patterns in the data, not proof of why patients return.", ""]
        for r in ranked:
            out += ["### {}".format(label(r["name"])), ""]
            if r["name"] == "logistic":
                out.append("Largest coefficients. Positive raises the predicted risk, negative lowers it (features are standardized, so sizes are comparable).")
                out.append("")
                for n, v in r["top"].items():
                    out.append("- `{}`: {} ({:+.2f})".format(n, "raises risk" if v > 0 else "lowers risk", v))
            else:
                out.append("Average influence on predictions (mean absolute SHAP value). Bigger means more influence; it does not show direction.")
                out.append("")
                for n, v in r["top"].items():
                    out.append("- `{}` ({:.3f})".format(n, v))
            out.append("")
        if len(ranked) > 1:
            sets = [{original_column(n, categorical) for n in r["top"].index} for r in ranked]
            common = sorted(set.intersection(*sets))
            if common:
                out.append("Columns in the top 10 of every model: {}. When several different models lean on the same columns, "
                           "those columns are the more reliable signals.".format(", ".join("`{}`".format(c) for c in common)))
            else:
                out.append("No column appears in the top 10 of every model, so the models lean on different signals.")
            out.append("")

    if failed:
        out += ["## Models that failed", "", "These did not finish. The others were not affected.", ""]
        out += ["- **{}**: `{}`".format(label(r["name"]), r["error"]) for r in failed]
        out.append("")

    out += ["## What these results do not tell you", "",
            "- They are not accuracy. With {:.0%} positives, a model that always says no would be right {:.0%} of the time and still useless.".format(prevalence, 1 - prevalence),
            "- They come from cross-validation on training data. Performance on the untouched test data may differ.",
            "- The data has no dates, so we could not check how the model holds up over time.",
            "- Fairness across age, race and gender, and whether the scores are well calibrated as probabilities, are checked later (step 8).",
            "- This is not a clinical tool. It has not been validated for use with real patients.", "",
            "## Where to find things", "",
            "| What | Where |", "|---|---|",
            "| Saved models, ranked, score in the file name | `models/01_...joblib`, `models/02_...joblib` |",
            "| Which model ranked first | `models/best_model.json` |",
            "| Technical summary of this run | `data/training_report.txt` |",
            "| All runs, metrics, SHAP charts | MLflow: run `mlflow ui --backend-store-uri sqlite:///mlflow.db` in the project folder and open http://127.0.0.1:5000 |",
            "", "In MLflow, open the `readmission-30d` experiment, click **Columns** and tick `cv_pr_auc_mean` and `cv_roc_auc_mean` "
            "to see the scores. Click a run, then **Artifacts**, then `explanations` for the SHAP chart or coefficients. "
            "The folders with random names are MLflow's internal storage; use the names above instead.", "",
            "## Glossary", "",
            "- **Cross-validation**: splitting the training data into parts, training on some and scoring on the held-out part, and repeating, so every score comes from data the model did not train on.",
            "- **Baseline**: a simple model used as the bar every other model must clear.",
            "- **Class weights**: telling the model that a missed readmission costs more than a false alarm, because readmissions are rare.",
            "- **SHAP**: a method that measures how much each feature pushed a prediction up or down.",
            "- **Calibration**: whether a predicted risk of 20% really corresponds to about 20% of such patients being readmitted.", ""]
    MODELS_DIR.mkdir(exist_ok=True)
    MODEL_REPORT_FILE.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="logistic,xgboost,lightgbm")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--train-file", default=None, help="CSV to train on (default: data/splits/train.csv)")
    parser.add_argument("--data-tag", default="", help="label for this data version, recorded in MLflow (default: 'default')")
    args = parser.parse_args()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(model_names) - set(MODEL_PARAMS)
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}. Choose from {sorted(MODEL_PARAMS)}.")
    train_path = Path(args.train_file).resolve() if args.train_file else TRAIN_FILE
    if train_path.name in ("test.csv", "holdout.csv"):
        raise ValueError(f"Refusing to train on {train_path.name}: test and holdout are never used for training.")
    if not train_path.exists():
        raise FileNotFoundError(f"Run data_split.py first. Missing: {train_path}")
    data_tag = args.data_tag.strip() or "default"
    train_sha = file_sha256(train_path)

    df = pd.read_csv(train_path, dtype={"encounter_id": str, PATIENT: str})
    y = df[TARGET].astype(int)
    groups = df[PATIENT]
    features = [c for c in df.columns if c not in NOT_FEATURES]
    categorical = [c for c in features if c not in NUMERIC]
    X = df[features]
    prevalence = float(y.mean())
    pos_weight = float((y == 0).sum() / (y == 1).sum())

    mlflow.set_tracking_uri(os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"))
    mlflow.set_experiment(EXPERIMENT)
    cv = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)
    stamp = datetime.now().strftime("%m%d-%H%M")
    results = []

    for name in model_names:
        try:
            model, params = build_model(name, pos_weight)
            pipe = Pipeline([("prep", build_preprocessor(categorical)), ("model", model)])
            with mlflow.start_run(run_name=f"{name}-{stamp}"), tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                mlflow.set_tags({"stage": "baseline" if name == "logistic" else "candidate",
                                 "model_family": name, "trained_on": "train split only",
                                 "data_tag": data_tag, "train_file": train_path.name})
                mlflow.log_params({f"model_{k}": v for k, v in params.items()})
                mlflow.log_params({
                    "cv_folds": args.cv_folds, "seed": SEED, "min_category_rows": MIN_CATEGORY_ROWS,
                    "n_features_in": len(features), "n_train_rows": len(df),
                    "n_train_patients": int(groups.nunique()), "train_file_sha256": train_sha[:16],
                })
                mlflow.log_metric("train_positive_rate", prevalence)

                # error_score="raise": a failing fold must stop the model with the REAL error,
                # not return NaN scores (which MLflow then fails to store, hiding the cause).
                res = cross_validate(pipe, X, y, groups=groups, cv=cv, n_jobs=1, error_score="raise",
                                     scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"})
                for metric in ("pr_auc", "roc_auc"):
                    if not np.isfinite(res[f"test_{metric}"]).all():
                        raise ValueError(f"Non-finite cross-validation {metric} scores: {res[f'test_{metric}']}")
                for metric in ("pr_auc", "roc_auc"):
                    vals = res[f"test_{metric}"]
                    mlflow.log_metric(f"cv_{metric}_mean", float(vals.mean()))
                    mlflow.log_metric(f"cv_{metric}_std", float(vals.std()))
                    for i, v in enumerate(vals):
                        mlflow.log_metric(f"cv_{metric}_fold", float(v), step=i)
                mlflow.log_metric("cv_pr_auc_lift_over_base_rate", float(res["test_pr_auc"].mean() / prevalence))

                pipe.fit(X, y)
                (tmp / "features.json").write_text(
                    json.dumps({"numeric": NUMERIC, "log_counts": LOG_COUNTS, "categorical": categorical}, indent=2))
                mlflow.log_artifact(str(tmp / "features.json"))
                for extra in EXTRA_ARTIFACTS:
                    if extra.exists():
                        mlflow.log_artifact(str(extra), "data_context")
                top = log_explanations(name, pipe, X, tmp)
                log_pipeline(pipe, name, X.head(5))

                results.append({
                    "name": name, "status": "ok", "run_id": mlflow.active_run().info.run_id,
                    "params": params, "pipe": pipe, "top": top.head(10),
                    "top_label": "coefficient" if name == "logistic" else "mean |SHAP|",
                    "pr_mean": float(res["test_pr_auc"].mean()), "pr_std": float(res["test_pr_auc"].std()),
                    "roc_mean": float(res["test_roc_auc"].mean()), "roc_std": float(res["test_roc_auc"].std()),
                    "folds": [float(v) for v in res["test_pr_auc"]],
                })
        except Exception as exc:  # one failing model must not stop the others
            results.append({"name": name, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()})
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]
    ranked = export_ranked_models(ok) if ok else []

    lines = [
        "TRAINING REPORT", "=" * 60,
        f"Created: {datetime.now():%Y-%m-%d %H:%M}",
        f"Data: {train_path.name} only. Test and holdout were not loaded.",
        f"Data version: tag '{data_tag}', file sha256 {train_sha[:12]}",
        f"Train rows: {len(df):,}   Patients: {groups.nunique():,}   Positive rate: {prevalence:.2%}",
        f"Features: {len(features)} ({len(NUMERIC)} numeric, {len(categorical)} categorical)",
        f"Cross-validation: {args.cv_folds}-fold, stratified, grouped by {PATIENT}, seed {SEED}",
        f"Rare categories (under {MIN_CATEGORY_ROWS} rows) are grouped into one bucket.",
        f"MLflow: {mlflow.get_tracking_uri()}   Experiment: {EXPERIMENT}", "",
    ]
    if ranked:
        best = ranked[0]
        lines += [f"BEST MODEL (by CV PR-AUC, among models that finished): {best['name']}",
                  f"  PR-AUC {best['pr_mean']:.3f}   file: models/{best['file'].name}   run: {best['run_id']}",
                  "  A candidate only: confirm on the test split in the evaluation gate (step 8).", ""]
    lines += ["CV results, best first (mean +/- std across folds)",
              f"{'rank':<6}{'model':<10}{'PR-AUC':>18}{'ROC-AUC':>18}{'lift':>8}   file"]
    for i, r in enumerate(ranked, start=1):
        pr_txt = "{:.3f} +/- {:.3f}".format(r["pr_mean"], r["pr_std"])
        roc_txt = "{:.3f} +/- {:.3f}".format(r["roc_mean"], r["roc_std"])
        lift_txt = "{:.2f}x".format(r["pr_mean"] / prevalence)
        lines.append(f"{i:<6}{r['name']:<10}{pr_txt:>18}{roc_txt:>18}{lift_txt:>8}   {r['file'].name}")
    lines += [f"No-skill PR-AUC is about {prevalence:.3f} (the positive rate). Lift = PR-AUC / positive rate.", ""]
    if failed:
        lines += ["FAILED MODELS (the others were not affected)"]
        for r in failed:
            lines += [f"- {r['name']}: {r['error']}", *["    " + t for t in r["trace"].strip().splitlines()[-8:]]]
        lines.append("")
    for r in ranked:
        folds_txt = ", ".join("{:.3f}".format(v) for v in r["folds"])
        lines += [f"{r['name']}  (run {r['run_id']})",
                  f"  params: {json.dumps(r['params'])}",
                  f"  PR-AUC by fold: {folds_txt}",
                  f"  top 10 features by {r['top_label']}:",
                  *[f"    {n}: {v:.4f}" for n, v in r["top"].items()], ""]
    lines += ["Notes",
              "- Compare models on these CV numbers. The test split is reserved for the evaluation gate (step 8).",
              "- Only the logistic model is tagged baseline; the others are tagged candidate.",
              "- SHAP / coefficients are on one-hot columns, so names look like column_value.",
              "- Load a saved model: joblib.load('models/<file>'), same package versions as training."]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    write_model_report(ranked, failed, prevalence, len(df), int(groups.nunique()), len(features),
                       args.cv_folds, categorical, data_tag)
    print("\n".join(lines))
    print("\nPlain-language report: {}".format(MODEL_REPORT_FILE))
    print("\nCompare runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()