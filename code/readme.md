# Code walkthrough

A reference for every Python script in the project: what it does, what each function does, which packages it imports and why,
what it accepts, what it produces, which other scripts it depends on, and what happens when something fails.

Related documents: [README](../README.md) (how to run it), [ARCHITECTURE](ARCHITECTURE.md) (the MLOps life cycle and design decisions),
[DEPLOYMENT_AND_MONITORING](DEPLOYMENT_AND_MONITORING.md) (Docker, Render, Prometheus, Grafana, runbooks).

Each script also starts with a docstring that says the same things in the author's words; the docstring is the first place to look when the
code and this document disagree.

---

## Contents

1. [Conventions used across the code](#1-conventions-used-across-the-code)
2. [Package reference](#2-package-reference)
3. [How the scripts depend on each other](#3-how-the-scripts-depend-on-each-other)
4. [Data preparation (`code/`)](#4-data-preparation-code)
5. [`train.py`](#5-trainpy)
6. [`test.py`](#6-testpy)
7. [`register_champion.py`](#7-register_championpy)
8. [`promote.py`](#8-promotepy)
9. [`retrain.py`](#9-retrainpy)
10. [`orchestration/retrain_flow.py`](#10-orchestrationretrain_flowpy)
11. [`serving/export_champion.py`](#11-servingexport_championpy)
12. [`serving/app.py`](#12-servingapppy)
13. [`serving/ui/app.js`](#13-servinguiappjs)
14. [`serving/send_sample.py`](#14-servingsend_samplepy)
15. [`serving/replay.py`](#15-servingreplaypy)
16. [`serving/rollback_drill.py`](#16-servingrollback_drillpy)
17. [`monitoring/drift_lib.py`](#17-monitoringdrift_libpy)
18. [`monitoring/drift_report.py`](#18-monitoringdrift_reportpy)
19. [`monitoring/drift_exporter.py`](#19-monitoringdrift_exporterpy)
20. [`monitoring/performance_report.py`](#20-monitoringperformance_reportpy)
21. [`deploy/prepare_bundle.py`](#21-deployprepare_bundlepy)
22. [`tools/mlflow_runs.py`](#22-toolsmlflow_runspy)
23. [Tests](#23-tests)
24. [Failure handling cheat sheet](#24-failure-handling-cheat-sheet)
25. [Known quirks](#25-known-quirks)

---

## 1. Conventions used across the code

These patterns repeat in almost every script, so they are explained once here.

| Convention | What it is | Why |
|---|---|---|
| `find_project_root()` | Walks up from the script's folder to the first folder that contains `data/` (or `mlflow.db`, in `tools/`) and treats it as the project root. | Scripts work from the project root, from `code/`, `serving/` or `monitoring/`, and `tests/conftest.py` creates an empty `data/` so CI can find a root. |
| Fixed seed `SEED = 42` | Used by CV folds, the bootstrap, week assignment, the noise-floor sample. | A rerun on the same data and packages reproduces the same numbers. A different number means something changed. |
| `SystemExit("message")` for expected user errors | A missing prerequisite or a refused action ends the script with a readable message and exit code 1 (no traceback). | The message tells the person what to run next. |
| Exceptions (`FileNotFoundError`, `ValueError`, `RuntimeError`) for real faults | Data problems, schema mismatches, MLflow problems. | They show a traceback and a non-zero exit, which the retrain flow treats as a real failure. |
| `assert` for invariants | "No patient in two splits", "every encounter assigned once". | A broken invariant must stop the pipeline, not warn. |
| Atomic writes | `export_champion.py` and `prepare_bundle.py` write a temp file then `os.replace`; `meta.json` is written **last**. | A half-finished export fails the checksum instead of being loaded. |
| ID columns read as strings | `dtype={"encounter_id": str, "patient_nbr": str}` everywhere. | Large integer ids must not turn into floats or lose leading digits, and the join between predictions and labels needs identical text. |
| One `Pipeline(prep, model)` | Preprocessing and model live in a single scikit-learn object. | Training, the gate, the registry copy and the API apply exactly the same transformations. The API reads its input contract from the fitted pipeline itself. |
| Append-only logs | `promotion_log.jsonl`, `data_use_log.jsonl`, `predictions.jsonl` are JSON lines, appended and never rewritten. | An audit trail that scripts add to cannot be silently edited by a later run. |
| Dry run before change | `promote.py --dry-run`, the retrain flow's default mode, `replay.py --dry-run`, `mlflow_runs.py` without `--apply`. | Every script that can change state has a way to show what it would do. |
| Lazy imports | `test.py` imports `mlflow` only when logging; `app.py` imports `shap` only when building an explainer. | Faster start, and `--no-mlflow` works without touching the tracking store. |

---

## 2. Package reference

Pinned versions are in `requirements.txt` (pipeline), `serving/requirements-serve.txt` (container), `monitoring/requirements-drift.txt`
(drift exporter) and `requirements-dev.txt` (tests, lint). Prefect is installed only in the separate `.venv-prefect`.

| Package | Used by | Why it is used |
|---|---|---|
| `pandas` | nearly everything | CSV reading, grouping, merging predictions with labels, building frames for the pipeline |
| `numpy` | training, gate, drift, performance, replay | Quantiles (flag cutoff), resampling, vector math, `searchsorted` for PSI bins |
| `scikit-learn` (1.5.2) | train, test, promote, performance, leak review, app | `Pipeline`, `ColumnTransformer`, `OneHotEncoder` (rare and unseen categories), `StandardScaler`, `LogisticRegression`, `StratifiedGroupKFold`, `cross_validate`, metrics (`average_precision_score`, `roc_auc_score`, `brier_score_loss`) |
| `xgboost` (2.0.3) | train | The XGBoost model. Version is tied to scikit-learn 1.5.2: a newer scikit-learn breaks it |
| `lightgbm` (4.6.0) | train | The LightGBM model (the champion). Needs the OpenMP runtime `libgomp1` in Docker images |
| `shap` | train, app | Feature explanations: mean absolute SHAP for reports, per-stay drivers in the API |
| `matplotlib` (Agg backend) | train, drift_report, performance_report | Charts written to files with no display (`matplotlib.use("Agg")`) |
| `joblib` | train, test, register, export, app | Saving and loading fitted pipelines as single files |
| `cloudpickle` | via MLflow, app | Serialisation format MLflow uses for the logged pipeline (skops rejects XGBoost and LightGBM) |
| `mlflow` (3.1.4) | train, test, register, promote, export, mlflow_runs | Run tracking and the model registry (aliases, tags). SQLite backend `sqlite:///mlflow.db` |
| `fastapi`, `starlette`, `pydantic`, `uvicorn` | app | The HTTP API, request validation of the outer envelope, the ASGI server |
| `prometheus-client` | app, drift_exporter | Counters, histograms and gauges exposed on `/metrics` |
| `prefect` | retrain_flow only | Task and flow logging and failure tracking. Kept in its own environment so it cannot disturb the pinned MLflow, pydantic and scikit-learn |
| `pytest`, `ruff` | tests, CI | Unit tests; lint for real errors only (syntax errors, undefined names) |
| Standard library | all | `argparse` (CLI), `json` (reports, logs), `pathlib`, `hashlib` (checksums), `subprocess` (flow runs scripts), `urllib` (HTTP calls without extra packages), `importlib.util` (load `test.py` by path), `threading` (locks in the API), `datetime`, `re`, `os`, `shutil`, `tempfile`, `traceback`, `copy`, `time`, `uuid`, `warnings`, `logging`, `dataclasses`, `contextlib` |

---

## 3. How the scripts depend on each other

Three kinds of dependency exist. Knowing which kind you are looking at tells you how a change propagates.

### 3.1 Python imports (code is shared)

```
test.py  <-- loaded by path (importlib) --  promote.py          (thresholds, fairness(), patient_bootstrap(), TEST_FILE, ROOT)
promote.py <-- imported as P ---------  serving/rollback_drill.py  (get_version, swap_aliases, refresh_champion_files, append_log, family)
monitoring/drift_lib.py <-- imported --  drift_report.py, drift_exporter.py
monitoring/drift_lib.py <-- imported (sys.path insert) -- serving/export_champion.py   (build_reference)
serving/app.py <-- imported -------------  tests/test_build_frame.py
retrain.py <-- imported ------------------ tests/test_retrain_decide.py
code/data_cleaning.py <-- imported ------  tests/test_icd9_group.py
```

`promote.py` loads `test.py` by file path because a module called `test` collides with Python's built-in `test` package. Loading it executes the
top of `test.py` (constants, `find_project_root`, `ROOT`) at import time.

### 3.2 Files (one script writes, another reads)

| Written by | File | Read by |
|---|---|---|
| `code/data_cleaning.py` | `data/diabetic_data_cleaned.csv`, `cleaning_config.json` | `leak_review.py`, `data_split.py`, `train.py` (config as MLflow artifact) |
| `code/data_split.py` | `data/splits/{train,test,holdout}.csv`, `split_report.txt` | `train.py`, `test.py`, `promote.py`, `export_champion.py`, `replay.py` |
| `train.py` | `models/NN_<model>_prauc<cv>.joblib`, MLflow runs | `test.py` (loads joblibs), `register_champion.py` (finds runs by CV score and verifies against joblib) |
| `test.py` | `models/evaluation.json` | `register_champion.py`, `performance_report.py` (reference PR-AUC), `retrain.py` (fallback reference), `retrain_flow.py` (gate result) |
| `register_champion.py`, `promote.py` | `models/champion.json`, registry aliases | `test.py` (reference model), `retrain.py` (live version), `export_champion.py` (alias) |
| `serving/export_champion.py` | `serving/bundle/{model.joblib, meta.json, reference_profile.json}` | `serving/app.py`, `drift_exporter.py`, `deploy/prepare_bundle.py` |
| `deploy/prepare_bundle.py` | `deploy/bundle/{model.joblib, meta.json}` | `deploy/Dockerfile` (baked into the Render image) |
| `serving/app.py` | `serving/logs/predictions.jsonl` | `drift_report.py`, `drift_exporter.py`, `performance_report.py` |
| `serving/replay.py` | `data/replay/<run>/{plan.json, labels.csv, summary.*}` | `drift_report.py` (plan), `performance_report.py` (labels) |
| `drift_report.py` | `data/drift/<run>/drift_summary.json` | `performance_report.py` (optional), `retrain.py` (unseen-category check) |
| `performance_report.py` | `data/performance/<run>/performance_summary.json` | `retrain.py` |
| `retrain.py` | `data/retrain/<id>/trigger.json` | `retrain_flow.py` |

### 3.3 Subprocess calls (one script runs another)

```
orchestration/retrain_flow.py  --subprocess-->  retrain.py decide, train.py, test.py, register_champion.py --challenger,
                                                promote.py [--dry-run], serving/export_champion.py     (all in the project .venv)
serving/rollback_drill.py      --subprocess-->  serving/export_champion.py, promote.py --rollback
```

The flow and the drill call these as separate processes (not imports) so each runs in the project environment with its own pinned packages, and a
crash in one cannot corrupt the caller.

### 3.4 Order of the chain

```
data_cleaning -> leak_review (optional) -> data_split -> train -> test -> register_champion -> export_champion
   -> (docker compose up | prepare_bundle + git push) -> replay -> drift_report -> performance_report -> retrain decide
   -> retrain_flow -> (train -> test -> register --challenger -> promote) -> export_champion -> reload
```

---

## 4. Data preparation (`code/`)

### 4.1 `code/data_cleaning.py`

**Purpose.** Turn the raw UCI file into a clean, readable dataset (strings, not integer codes) and record exactly what was done.
**Run.** `python code/data_cleaning.py` (no arguments; behaviour is switched in the CONFIG block at the top).
**Imports.** `json`, `pathlib.Path`, `pandas` (everything is a DataFrame operation; no model packages).

| Input | Output |
|---|---|
| `data/diabetic_data.csv` (never modified) | `data/diabetic_data_cleaned.csv`, `data/cleaning_config.json`, `data/cleaning_report.txt` |

| Function | What it does |
|---|---|
| `icd9_group(code)` | Maps a diagnosis code string to a broad group: `missing`, `supplementary` (V), `external` (E), then numeric ranges (`diabetes` 250.x, `circulatory` 390-459 and 785, `respiratory` 460-519 and 786, `digestive` 520-579 and 787, `genitourinary` 580-629 and 788, `neoplasms` 140-239, `musculoskeletal` 710-739, `injury` 800-999, `infectious` 1-139), otherwise `other`. Non-numeric text returns `other`. Leading zeros are missing in the source, so `38` is infectious |
| `main()` | Reads every column as a string with only `?` as missing; checks required columns exist and `encounter_id` is unique; keeps the raw target as `readmitted_raw`; drops expired/hospice discharges and invalid gender; fills missing values (`not_measured`, `missing`); merges unknown ids into `unknown`; maps the target (`<30` to 1, else 0); validates drug columns; adds `n_drugs_taken`, `n_drugs_changed`, `diag_*_group`; drops `weight`, `examide`, `citoglipton` and any constant column; writes outputs |

**Depends on / used by.** Nothing upstream. Read by `leak_review.py`, `data_split.py`; `icd9_group` is unit-tested.
**Failure.** `FileNotFoundError` if the raw file is missing. `ValueError` if expected columns are absent, `encounter_id` is not unique, the target has an unexpected value, or a drug column has a value other than No/Down/Steady/Up. Nothing is written in those cases (the output is written at the end), and the raw file is never touched.

### 4.2 `code/leak_review.py`

**Purpose.** Show that no feature gives away the answer. A hand-written table decides which columns are allowed; statistics only flag suspects.
**Run.** `python code/leak_review.py`.
**Imports.** `numpy`, `pandas`, scikit-learn (`HistGradientBoostingClassifier`, `roc_auc_score`, `GroupKFold`, `GroupShuffleSplit`, `OrdinalEncoder`): a fast, categorical-aware model for the whole-model sanity check; patient-grouped splitters so the checks themselves do not leak.

| Input | Output |
|---|---|
| `data/diabetic_data_cleaned.csv` | `data/leak_review_report.txt`, `data/leak_review_columns.csv` |

| Function | What it does |
|---|---|
| `build_review()` | The verdict table: column to (`EXCLUDE` / `KEEP` / `KEEP-WATCH` / `TARGET`, reason) |
| `two_sided_auc(score, y)` | AUC that ignores direction (`max(a, 1-a)`), so a strongly negative column also flags |
| `oof_target_encoding_auc(x, y, groups)` | Out-of-fold AUC of a category's mean target rate, folds grouped by patient |
| `grouped_model_auc(df, features, y, groups)` | Ordinal-encodes categories (fit on train rows only), trains gradient boosting on a patient-grouped 75/25 split, returns test AUC |
| `main()` | Runs the seven checks (verdicts, single numeric AUC, identifier AUC, categorical AUC, rate tables, split-leak demo, whole-model AUC three ways) and writes both files |

**Failure.** `FileNotFoundError` without the cleaned file. `ValueError` if any column has no verdict (new columns cannot slip through). `assert grouped_overlap == 0` stops the script if the grouped split ever shares a patient.

### 4.3 `code/data_split.py`

**Purpose.** Create train / test / holdout by patient, stratified on the label, reproducibly.
**Run.** `python code/data_split.py`.
**Imports.** `numpy`, `pandas`, `StratifiedGroupKFold` (groups by patient while keeping the positive rate similar in each fold).

| Input | Output |
|---|---|
| `data/diabetic_data_cleaned.csv` | `data/splits/{train,test,holdout}.csv`, `split_assignments.csv` (source of truth), `split_report.txt` |

`main()` is the only function: assigns each row to one of 20 folds (`N_FOLDS`), maps folds 12 / 3 / 5 to train / test / holdout, runs the hard checks, writes files, and appends a report (sizes, positive rates, test/holdout categories never seen in train).
**Failure.** `FileNotFoundError` without the cleaned file. `AssertionError` if a row is unassigned, `encounter_id` repeats, any patient appears in two splits, or the three files do not add up to the cleaned data. A split more than 1.5 points off its target share only adds a warning to the report.

---

## 5. `train.py`

**Purpose.** Train logistic regression (baseline), XGBoost and LightGBM, score each with patient-grouped cross-validation, fit on all of train, log everything to MLflow and save ranked joblib files.
**Run.** `python train.py [--models logistic,xgboost,lightgbm] [--cv-folds 5] [--train-file PATH] [--data-tag LABEL]`.
**Imports.** Standard library (`argparse`, `hashlib`, `json`, `os`, `tempfile`, `traceback`), `joblib`, `matplotlib` (Agg), `mlflow`, `mlflow.sklearn`, `numpy`, `pandas`, `shap`, `lightgbm.LGBMClassifier`, `xgboost.XGBClassifier`, scikit-learn (`ColumnTransformer`, `LogisticRegression`, `StratifiedGroupKFold`, `cross_validate`, `Pipeline`, `FunctionTransformer`, `OneHotEncoder`, `StandardScaler`).

| Input | Output |
|---|---|
| `data/splits/train.csv` (or `--train-file`); `split_report.txt` and `cleaning_config.json` if present; `MLFLOW_TRACKING_URI` (default `sqlite:///<root>/mlflow.db`) | `models/01_<model>_prauc<cv>.joblib` ..., `models/best_model.json`, `models/model_report.md`, `data/training_report.txt`, MLflow runs in experiment `readmission-30d` |

| Function | What it does |
|---|---|
| `find_project_root()` | See conventions |
| `file_sha256(path)` | SHA-256 of the training file; the first 16 characters go into MLflow as `train_file_sha256` |
| `build_preprocessor(categorical)` | `ColumnTransformer`: log counts (`log1p` then scale), other numerics (scale), categoricals (`OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)`) |
| `build_model(name, pos_weight)` | Returns the classifier and its parameter dict. Trees get `scale_pos_weight` = negatives / positives (7.704); logistic uses `class_weight="balanced"` |
| `log_explanations(name, pipe, X, tmp)` | Logistic: sorted coefficients to CSV. Trees: 2,000-row sample, `shap.TreeExplainer`, handles the different SHAP output shapes, mean absolute SHAP to CSV and a bar chart; logs artifacts; returns the ranking |
| `log_pipeline(pipe, name, example)` | `mlflow.sklearn.log_model` with cloudpickle serialisation; falls back to the MLflow 2 argument name on `TypeError` |
| `export_ranked_models(ok)` | Deletes only previous `NN_*_prauc*.joblib` files, ranks by CV PR-AUC, saves new files, writes `best_model.json` |
| `original_column(feature, categorical)` | Maps a one-hot name like `age_[70-80)` back to `age` (longest matching column name) |
| `write_model_report(...)` | The plain-language report: results table, verdict, SHAP and coefficient lists, glossary |
| `main()` | Parses args, validates, loads data, loops over models (one MLflow run each: params, CV metrics per fold, fit, artifacts, explanations, pipeline), then ranks, exports and writes reports |

**Depends on / used by.** Needs `code/data_split.py`. Its joblibs feed `test.py`; its MLflow runs are found later by `register_champion.py`. Called as a subprocess by `retrain_flow.py` with `--data-tag retrain-<id>`.

**Failure.**
- `ValueError` for an unknown model name, or **a refusal to train on `test.csv` / `holdout.csv`** (a hard guard against leakage).
- `FileNotFoundError` if the train file is missing.
- `cross_validate(..., error_score="raise")`: a failing fold stops that model with the real error instead of returning NaN (which MLflow would fail to store and hide the cause).
- Any exception in one model is caught, recorded as FAILED with a traceback in `training_report.txt`, and **the other models still run**. If all fail, no joblibs are exported and the report says so.
- Running with `--models xgboost` alone would delete the other ranked joblibs (it only rebuilds files for the models in that run).

---

## 6. `test.py`

**Purpose.** The evaluation gate. Score every saved model once, unchanged, on the test split (or `--data`), and apply pass/fail rules fixed in advance. It never chooses the champion.
**Run.** `python test.py [--reference MODEL] [--no-mlflow] [--data CSV] [--name LABEL]`.
**Imports.** `argparse`, `json`, `os`, `re` (parse joblib file names), `datetime`, `pathlib`, `joblib`, `numpy`, `pandas`, scikit-learn metrics; `mlflow` is imported only when logging.

**Gate constants** (never change after seeing results): `MIN_LIFT 1.5`, `MAX_CV_DROP 0.02`, `MUST_BEAT_BASELINE True`, `MAX_GROUP_AUC_GAP 0.05`, `MIN_GROUP_ROWS 500`, `MIN_GROUP_POSITIVES 50`, `FLAG_FRACTION 0.20`, `MAX_RECALL_GAP 0.10`, `FAIRNESS_COLUMNS [gender, race, age]`, `N_BOOT 200`.

| Input | Output |
|---|---|
| `data/splits/test.csv` (or `--data`), `models/NN_*_prauc*.joblib`, `models/champion.json` (optional) | `models/evaluation.json`, `models/test_report.md`, `data/test_report.txt`, `eval-<model>` MLflow runs, a line in `models/data_use_log.jsonl`. With `--data X --name n`: `models/evaluation_n.json`, `models/n_report.md`, `data/n_report.txt` instead |

| Function | What it does |
|---|---|
| `load_models()` | Globs `NN_<model>_prauc<cv>.joblib`, parses family and CV score from the name with a regex, loads each pipeline |
| `ece(y, p, bins=10)` | Expected calibration error with equal-count bins |
| `recall_at_top(y, s, frac)` | Recall if the top `frac` by score is flagged (defined for completeness; `main` uses a cutoff-based recall instead) |
| `patient_bootstrap(y, scores, groups, n_boot, seed)` | Resamples whole patients with replacement; computes PR-AUC for **every model on the same resample**, so differences are paired. Skips resamples with no positives |
| `fairness(df, y, s, cutoff, overall_auc, overall_recall)` | One row per level of gender / race / age: size, positives, ROC-AUC, recall at the cutoff, flag rate, whether it is gated, and the gaps to overall |
| `choose_reference(requested, results)` | `--reference`, else the family in `champion.json`, else best test PR-AUC. Only a comparison anchor |
| `relabel(text, label)` | Rewords reports written for "test" when another dataset is evaluated |
| `write_plain_report(...)` | Writes the plain-language Markdown report |
| `main()` | Loads data and models, checks each model's expected columns exist, scores, bootstraps, builds per-model metrics, applies the checks (stored as plain Python bools so JSON works), picks the reference, writes reports and JSON, appends the data-use log, logs MLflow runs |

**Used by.** `promote.py` loads this file by path and reuses its constants and helpers; `register_champion.py` and others read `evaluation.json`.
**Failure.**
- `FileNotFoundError` if there are no joblibs or the data file is missing.
- `ValueError` if the data lacks `readmitted` or `patient_nbr`, or lacks a column a model was trained on (`"test data lacks columns the model was trained on"`). Extra columns are ignored with a printed note.
- `SystemExit` if `--name test` is used for a file that is not the real test split (the name is reserved).
- A repeat use of non-test data prints a warning from `data_use_log.jsonl` but proceeds.
- A model that fails a check simply gets `passed: false`; it does not stop the script.

---

## 7. `register_champion.py`

**Purpose.** Put models in the MLflow registry `readmission` with aliases and evidence tags; verify the registered copy equals the saved joblib.
**Run.** `python register_champion.py --champion MODEL --reason "..."` or `--challenger MODEL --reason "..."` (mutually exclusive, one required); `--no-verify`, `--force`.
**Imports.** `argparse`, `json`, `os`, `datetime`, `pathlib`, `joblib`, `mlflow`, `mlflow.sklearn`, `numpy`, `pandas`, `MlflowException`, `MlflowClient`.

| Input | Output |
|---|---|
| `models/evaluation.json`, MLflow runs of experiment `readmission-30d`, one joblib (verification), `data/splits/test.csv` (first 500 rows) | Registry versions, aliases and tags; `models/champion.json`, `models/champion_report.md`; in challenger mode a `challenger_registered` line in `promotion_log.jsonl` |

| Function | What it does |
|---|---|
| `find_training_run(client, exp_id, family, cv_expected)` | Searches finished runs tagged with the family, skips evaluation runs, returns the latest whose `cv_pr_auc_mean` is within 0.0006 of the value in `evaluation.json` |
| `model_uri_for(client, exp_id, run_id, family)` | MLflow 3 "logged model" URI if one exists for that run, else `runs:/<run>/readmission_<family>` |
| `register_version(client, uri, run_id)` | Reuses an existing version for the same run, else creates the registered model (if needed) and registers a new version |
| `verify_against_joblib(uri, joblib_name, label)` | Predictions of the registry model and the joblib must match to 1e-9 on 500 rows |
| `register_challenger(...)` | One new version with alias `challenger`; evidence tags; the previous challenger is tagged `retired_challenger` (not deleted); optional verification; log line |
| `main()` | Argument handling, gate and safety-guard checks, champion mode registers every model (champion, baseline, other passing models as challengers), verifies, writes files |
| `write_champion_files(...)` | Writes `champion.json` and `champion_report.md` |

**Failure.** Each is a deliberate stop with nothing partially applied:
- `FileNotFoundError` with no `evaluation.json` ("Run test.py first").
- `ValueError` if the model is not in `evaluation.json`.
- `SystemExit "REFUSED: <model> did not pass the evaluation gate"`.
- `SystemExit "REFUSED: promote.py has changed the champion ..."` when champion mode is re-run after a promotion or rollback, unless `--force` (it would reset aliases to the old champion).
- `SystemExit "REFUSED: this exact training run is already the champion"` for a challenger identical to the champion.
- `RuntimeError` if the experiment does not exist, or no training run matches the CV score ("evaluation.json and MLflow are out of step").
- `SystemExit "VERIFY FAILED"` if registry and joblib predictions differ.
The retrain flow treats a message containing `REFUSED` as a normal outcome and any other non-zero exit as a real failure.

---

## 8. `promote.py`

**Purpose.** Decide with a rule whether the challenger replaces the champion, apply it, and roll back.
**Run.** `python promote.py [--dry-run] [--candidate-version N] [--data CSV] [--rollback]`.
**Imports.** `argparse`, `importlib.util` (load `test.py` by path), `json`, `os`, `datetime`, `pathlib`, `mlflow`, `mlflow.sklearn`, `numpy`, `pandas`, `MlflowException`, `MlflowClient`, scikit-learn metrics.

| Input | Output |
|---|---|
| Registry aliases `champion`, `challenger` (or `--candidate-version`), `baseline`; test split (or `--data`); thresholds from `test.py` | `models/promotion_report.md`, a line in `promotion_log.jsonl` (not for dry runs), and on a promotion/rollback: alias changes, version tags, `champion.json`, `champion_report.md` |

| Function | What it does |
|---|---|
| `load_gate()` | Loads `test.py` as module `evaluation_gate`; run at import time, so `g.ROOT`, `g.MIN_LIFT`, `g.fairness`, `g.patient_bootstrap` are available. Thresholds are shared, never copied |
| `get_version(client, ref)` | A digit string means a version number; otherwise an alias |
| `score(version, df)` | Loads `models:/readmission/<version>`, selects the columns the pipeline expects, returns `predict_proba[:, 1]` |
| `family(v)`, `cv_score(client, v)` | Read the model family tag; read `cv_pr_auc_mean` from the training run |
| `append_log(entry)` | Appends a timestamped JSON line to `promotion_log.jsonl` |
| `refresh_champion_files(client, v, extra)` | Rewrites `champion.json` (removing stale fields) and `champion_report.md` from the version's tags |
| `swap_aliases(client, new_v, old_v)` | `previous_champion` to old, `champion` to new, updates `role` tags, deletes `challenger` if it pointed at the new champion |
| `do_rollback(client, dry_run)` | Swaps `previous_champion` back to `champion`, refreshes files, logs, and prints the export-and-reload reminder |
| `main()` | Scores champion and candidate on the same data, computes PR-AUC, lift, fairness gaps and the **paired bootstrap difference**, builds the checks dict, promotes only if **every** check passes including "95% range of the PR-AUC gain above 0", writes the report, applies the swap |

**Failure.** `SystemExit` with a fix-it message when there is no `@champion`, no `@challenger` (or the requested version does not exist), candidate equals champion, or no `@previous_champion` for a rollback. A candidate that fails any check is not an error: the decision is KEEP CHAMPION, a report and a log line are written, and the registry is untouched.
**Note.** The default comparison data is the test split, already used for selection; the script prints that caution.

---

## 9. `retrain.py`

**Purpose.** Apply the written retrain rule to labelled performance. It decides; it retrains nothing.
**Run.** `python retrain.py decide --run-id RUN [--performance FILE] [--champion-version N] [--scope clean|all] [--force-retrain] [--id NAME]`.
**Imports.** `argparse`, `json`, `datetime`, `pathlib` only (pure logic, so it is easy to unit test).

| Input | Output |
|---|---|
| `performance_summary.json` (searched under `data/` by run id, or `--performance`), optional `data/drift/<run>/drift_summary.json`, `models/champion.json`, `models/evaluation.json` (fallback reference) | `data/retrain/<id>/trigger.json` and a printed summary; decision is one of `keep`, `fix_data_first`, `retrain` |

| Function | What it does |
|---|---|
| `find_performance(run_id, explicit)` | Explicit path, else the newest `performance_summary.json` in a folder named after the run |
| `champion_version(explicit)`, `champion_family()` | Live version and family from `champion.json` |
| `unseen_weeks(run_id, weeks)` | Weeks whose drift detail mentions `__unseen__` |
| `reference_for(entry, family)` | The model's test PR-AUC: from the performance summary, else `evaluation.json` for that family |
| `decide(perf, version, scope, run_id, family)` | The rule: no stays for the live version means keep; fewer than `MIN_WEEKS` (4) weeks means keep; no reference means keep with a clear reason; pooled upper bound not below the reference means keep; below with unseen categories means `fix_data_first`; otherwise `retrain` |
| `main()` | Argument parsing, runs `decide`, applies `--force-retrain` (decision becomes `retrain`, original kept as `evidence_decision`, `forced: true`), writes `trigger.json` |

**Failure.** `SystemExit` if the performance file is missing, or `champion.json` is missing with no `--champion-version`. Every "cannot judge" situation resolves to `keep` with a stated reason, never to a silent retrain.
**Tested by** `tests/test_retrain_decide.py` (the rule is pinned by unit tests).

---

## 10. `orchestration/retrain_flow.py`

**Purpose.** A Prefect flow that runs the existing scripts in order for the retrain loop. It makes no decisions itself.
**Run.** From `.venv-prefect`: `python orchestration/retrain_flow.py --run-id RUN [--force-retrain] [--apply] [--allow-promote] [--scope] [--performance] [--champion-version] [--main-python] [--api-url]`.
**Imports.** `argparse`, `json`, `os`, `shutil` (backup), `subprocess` (run scripts), `urllib` (POST `/reload`), `datetime`, `pathlib`, and `prefect` (`flow`, `task`, `get_run_logger`).

| Mode | What happens |
|---|---|
| default | only `retrain.py decide` (dry run) |
| `--apply` | back up `models/`, train, gate, register a challenger, show `promote.py --dry-run`; champion untouched |
| `--apply --allow-promote` | also promote if the rule says PROMOTE, then export the bundle and reload the API |

| Function / task | What it does |
|---|---|
| `project_python(explicit)` | `--main-python`, else `RETRAIN_PYTHON`, else `<root>/.venv/Scripts/python.exe` or `.venv/bin/python`; `FileNotFoundError` if none exists |
| `run_script(py, args, logger, name)` | `subprocess.run` in the project root with UTF-8 output; returns exit code and combined output; logs the tail |
| `log_decision(entry)` | Appends a `retrain_flow` line to `promotion_log.jsonl` |
| `t_decide` | Runs `retrain.py decide`, returns the parsed `trigger.json`; raises on a non-zero exit |
| `t_backup` | Copies joblibs and key `models/*` files to `data/retrain/<id>/models_before/` |
| `t_train` | `train.py --data-tag retrain-<id>` |
| `t_gate` | `test.py`; returns `{family: {passed, test_pr_auc}}` from `evaluation.json` |
| `pick_challenger(gate)` | Among **non-baseline** models that passed, the highest test PR-AUC (rule written in advance); `None` if none |
| `t_register` | `register_champion.py --challenger`; exit code 1 with `REFUSED` is a normal outcome, any other failure raises |
| `t_promote(py, dry_run)` | `promote.py [--dry-run]`; `"DECISION: PROMOTE" in output` is the result |
| `t_export_reload(py, url, token)` | Runs `export_champion.py`, then POSTs `/reload` (with `X-Admin-Token` from `ADMIN_TOKEN`); a failed reload is a warning, not an error |
| `write_report(retrain_id, summary)` | Writes `retrain_report.md` |
| `retrain_flow(...)` | The orchestration, producing an outcome sentence |

**Normal outcomes (flow ends cleanly):** decision keep or fix_data_first; no model passes the gate; registration refused; promote says KEEP CHAMPION; the challenger would be promoted but `--allow-promote` was not given.
**Real failures (Prefect marks the run failed):** a script exits non-zero unexpectedly, a required file is missing, the project Python cannot be found.
**Safety.** `--allow-promote` without `--apply` is refused by the argument check. The backup is taken before anything is overwritten.

---

## 11. `serving/export_champion.py`

**Purpose.** Export the registry `@champion` into a self-contained bundle the API loads, with the high-risk cutoff and the drift reference profile.
**Run.** `python serving/export_champion.py [--name readmission] [--alias champion] [--out DIR] [--reference-data CSV] [--flag-fraction 0.20] [--no-ledger]`.
**Imports.** `argparse`, `hashlib`, `importlib.metadata`, `json`, `os`, `platform`, `sys`, `datetime`, `pathlib`, `joblib`, `numpy`, `pandas`; `mlflow` and `MlflowClient` inside `main`; `drift_lib` via a `sys.path` insert of `monitoring/`.

| Input | Output |
|---|---|
| Registry model by alias, `data/splits/test.csv` (features and scores only, no labels used) | `serving/bundle/model.joblib`, `reference_profile.json`, `meta.json` (written last); a line in `data_use_log.jsonl` |

| Function | What it does |
|---|---|
| `find_project_root()`, `sha256(path)`, `package_versions()` | See conventions; hashes in 1 MB chunks; records the versions of key packages |
| `write_reference_profile(out_dir, reference, cols, scores, meta_base, ref_name)` | Builds the drift reference with `drift_lib.build_reference`, writes it atomically, returns its checksum and size info |
| `write_bundle(pipe, out_dir, meta_base, reference, flag_fraction, reference_name)` | Needs no MLflow: scores the reference, sets `cutoff = quantile(scores, 1 - flag_fraction)`, writes model atomically, then the profile, then `meta.json` with checksum, cutoff, example record, packages, Python version, timestamp |
| `main()` | Resolves the alias, loads `models:/readmission/<version>`, builds registry-evidence metadata, writes the bundle, logs the data use |

**Failure.** `SystemExit` if `--flag-fraction` is not between 0 and 1 or the alias does not exist ("Register one first"). `ValueError` if the reference data lacks a column the model needs. Because `meta.json` is written last and holds the model checksum, an interrupted export leaves a bundle the API refuses to load (the previous model keeps serving).

---

## 12. `serving/app.py`

**Purpose.** The FastAPI service: loads the bundle, validates input, scores, explains, flags, logs predictions, exposes Prometheus metrics, supports hot reload and serves the web console.
**Run.** `uvicorn app:app --app-dir serving --port 8000` (one worker: counters live in the process). Local Docker uses `serving/Dockerfile`; Render uses `deploy/Dockerfile`.
**Imports.** `hashlib`, `importlib.metadata`, `json`, `logging`, `os`, `re`, `threading`, `time`, `uuid`, `warnings`, `contextlib`, `dataclasses`, `datetime`, `pathlib`, `typing`, `joblib`, `numpy`, `pandas`, `fastapi` (`FastAPI`, `Header`, `HTTPException`, `Request`, `Response`), `HTMLResponse`, `JSONResponse`, `StaticFiles`, `prometheus_client`, `pydantic` (`BaseModel`, `Field`), `sklearn.preprocessing.OneHotEncoder`; `shap` and `LogisticRegression` are imported inside `load_state`.

**Environment variables:** `BUNDLE_DIR`, `MAX_RECORDS` (500; Render 100), `ID_FIELD` (`encounter_id`), `LOG_PREDICTIONS` (true; Render false), `PREDICTION_LOG`, `ADMIN_TOKEN`, `LOG_LEVEL`, and `PORT` on Render (read by the `CMD`, not the app).

| Endpoint | Input | Output | Failure |
|---|---|---|---|
| `GET /` | none | `ui/index.html` (no-cache) | 404 if the file is missing |
| `POST /predict` | `{records:[...], explain, top_k}` plus optional `X-Replay-Run`, `X-Replay-Week` | `request_id`, `model`, `flag_rule`, `predictions[{id, score, high_risk, top_drivers}]`, `ignored_columns`, `unseen_categories`, `explain_error` | 503 no model; 422 with a structured `problems` list (missing columns, nulls, not numeric, too many records, bad replay header); pydantic's own 422 for a malformed envelope |
| `GET /example`, `GET /model-info` | none | an example request; live model, evidence, columns, categories, flag rule, packages, limits | 503 no model; 404 no example |
| `GET /health` | none | 200 `{status, model_version, model_family}` | 503 `{status: no_model_loaded}` |
| `GET /metrics` | none | Prometheus text | none |
| `POST /reload` | optional `X-Admin-Token` | `{reloaded, previous_version, model_version, model_family}` | 403 wrong/missing token; 500 with `still_serving_version` if the new bundle is bad |

| Function / class | What it does |
|---|---|
| Metric objects | `HTTP_REQUESTS`, `HTTP_LATENCY`, `PREDICTIONS`, `SCORES`, `BATCH_SIZE`, `VALIDATION_ERRORS`, `UNSEEN`, `MODEL_INFO`, `MODEL_LOADED_AT`, `RELOADS`, `REPLAY_WEEK` (full catalogue in the deployment document) |
| `ModelState` (dataclass) | Everything about the live model: pipeline, meta, required/numeric/categorical columns, known categories, feature names and their source columns, cutoff, explainer kind, load time, package warnings. Swapped as one object |
| `file_sha256(path)` | Streamed SHA-256 |
| `inspect_pipeline(pipe)` | Reads the **input contract from the fitted pipeline**: required columns, numeric columns, each `OneHotEncoder`'s learned categories (and whether string or numeric), and a map from each one-hot feature name back to its source column |
| `load_state(bundle_dir)` | Checks both files exist, verifies `model.joblib` against `meta.json`'s checksum, loads the pipeline, requires steps named `prep` and `model`, builds a SHAP `TreeExplainer` (logistic is marked `linear`; explainer failure only disables explanations), compares package versions with the export and records mismatches |
| `activate(state)` | Sets the module-level `_state`, resets and sets the `MODEL_INFO` gauge, stamps `MODEL_LOADED_AT` |
| `try_reload()` | Under a lock: load then activate. On failure the old state stays, `RELOADS{outcome="failed"}` increments, the exception propagates |
| `lifespan(app)` | Tries to load at startup; if it fails it logs the error and **still starts** (health returns 503) |
| `metrics_middleware` | Times every request and counts it by **route template** (`/ui/*`, known doc paths, or `unmatched`), so label cardinality stays bounded; skips `/metrics` itself |
| `PredictRequest` | `records` (at least 1), `explain` (default true), `top_k` (1 to 20) |
| `require_state()` | 503 when no model |
| `reject(reason, message, problems)` | Counts `VALIDATION_ERRORS{reason}` and raises 422 with up to 50 problems |
| `parse_replay_headers(run, week)` | Validates `X-Replay-Run` (`^[A-Za-z0-9_.-]{1,64}$`) and `X-Replay-Week` (0 to 1000) |
| `bad_rows(mask)` | First 5 offending row indices, for readable errors |
| `build_frame(records, st)` | In order: missing columns, nulls, numeric conversion, categorical normalisation (a float like `1.0` becomes `"1"`), unseen-category detection (counted in `UNSEEN`, returned, **not** rejected). Returns the frame, ignored columns, unseen values, ids |
| `explain(st, df, records, top_k)` | Transforms the frame, computes SHAP values (or coefficient times value for logistic), returns the top-k drivers per record with the original input value, in log-odds |
| `log_predictions(...)` | Under a lock appends one JSON line per record; **any logging failure is caught and logged and never fails the request** |
| `predict`, `example`, `model_info`, `health`, `metrics`, `reload_model`, `console`, `favicon` | Endpoint handlers as in the table |
| `NoCacheStaticFiles` | Serves `/ui/*` with `Cache-Control: no-cache` so replaced files are never shown stale |

**Failure behaviour summary.**
- No or corrupt bundle at startup: the service starts, `/health` is 503, the console shows a banner, `/predict` returns 503.
- Checksum mismatch: `ValueError` in `load_state`; the model is not loaded (at startup) or the old one keeps serving (on reload).
- Package-version mismatch: warnings only, shown in `/model-info` as `package_warnings`.
- SHAP failure: the prediction still returns; the error is in `explain_error`.
- Unseen category: scored in the "infrequent" bucket, counted in `readmission_unseen_category_total`, listed in the response.

---

## 13. `serving/ui/app.js`

**Purpose.** The web console's behaviour. It talks only to its own origin (the API that serves it) and uses no external libraries or fonts.
**Inputs.** The API's JSON and the Prometheus text from `/metrics`; files chosen by the user. **Outputs.** DOM updates, a downloaded `readmission_scores.csv`.

| Group | Functions | What they do |
|---|---|---|
| DOM helpers | `$`, `$$`, `el`, `put` | Build elements with `textContent`/text nodes only, so server data is never injected as HTML |
| Network | `request(path, opts)`, `toast`, `copy` | `fetch` wrapper that parses JSON when possible and throws `{status, body}`; feedback and clipboard |
| Labels | `LABELS`, `GROUPS`, `labelFor` | Friendly names and the grouping of form fields |
| Navigation and status | `showView`, `pollHealth`, `init` | Hash-based tabs (`#score`, `#batch`, `#model`, `#monitor`); status pill from `/health` every 15 s |
| Form | `makeField`, `buildForm`, `fillForm`, `clearForm`, `collect`, `loadExample` | One input per required column (number box, or dropdown with an "Other" box), built from `/model-info` |
| Single score | `score`, `sentence`, `meter`, `drivers`, `jsonDetails`, `renderResult`, `renderError` | POST one record; show score, cutoff meter, driver bars; map 422 problems onto the offending fields |
| Batch | `parseCSV`, `handleFile`, `loadCSV`, `rowToRecord`, `runBatch`, `batchError`, `histogram`, `explainRow`, `renderBatch`, `stat`, `downloadResults` | Browser-side CSV parser (quotes handled), column check, chunked requests of at most the API's `max_records_per_request`, progress and cancel, histogram, top-100 table, search, click-through, CSV download |
| Model tab | `kv`, `renderModel`, `reloadModel` | Registry evidence, rules, columns; reload (asks for the admin token on 403) |
| Monitoring tab | `parseProm`, `total`, `quantileLE`, `loadMonitor`, `startMonitor`, `stopMonitor` | Parse Prometheus text in the browser, compute totals and an approximate p95 from histogram buckets |

**Failure.** `request` converts network errors to `{status: 0}`; the UI shows "Cannot reach the API", a banner when `/model-info` returns 503, and per-field highlights for 422 problems. A malformed CSV or a missing required column disables scoring with a message.

---

## 14. `serving/send_sample.py`

**Purpose.** Smoke test: send rows of a CSV to a running API and print what comes back. **Run.** `python serving/send_sample.py [--csv FILE] [--url URL] [--n 20] [--batch-size 20] [--no-explain]` (works against the Render URL too).
**Imports.** `argparse`, `json`, `urllib.error`, `urllib.request`, `pathlib`, `pandas`.
**Input.** A CSV (default the test split). `readmitted` and `readmitted_raw` are dropped before sending. **Output.** Counts of rows scored and flagged, the cutoff, the first record.

| Function | What it does |
|---|---|
| `post(url, body)` | JSON POST; returns `(status, body)` even for HTTP errors |
| `main()` | Batches the first `n` rows and aggregates results |

**Failure.** `SystemExit` with startup hints if the API is unreachable; on a non-200 it prints the (truncated) error body and exits 1. On a public service with `MAX_RECORDS=100`, keep `--batch-size` at or below 100.

---

## 15. `serving/replay.py`

**Purpose.** Replay the sealed holdout through the **running** API as simulated weeks, injecting one kind of drift at a time, and save the withheld labels.
**Run.** `python serving/replay.py [--preset demo] [--run-id ID] [--weeks 12] [--start-week 1] [--seconds-per-week 60] [--batch-size 100] [--inject NAME[:FIRST[-LAST]][:SEVERITY]]... [--explain] [--seed 42] [--csv] [--url] [--out-dir] [--dry-run] [--show-examples]`.
**Imports.** `argparse`, `copy`, `json`, `re`, `sys`, `time`, `urllib`, `datetime`, `pathlib`, `numpy`, `pandas`.

| Input | Output |
|---|---|
| `data/splits/holdout.csv`; the API's `/health`, `/model-info`, `/metrics`, `/predict` | `data/replay/<run>/{plan.json, labels.csv, summary.json, summary.md}`; a line in `data_use_log.jsonl`; log lines in the API |

| Function | What it does |
|---|---|
| `call(method, url, body, headers, timeout)` | `urllib` request; HTTP errors return `(status, text)`; connection/timeout errors raise `SystemExit` with Docker hints |
| `as_json`, `api_counters(base)` | Safe JSON parse; reads the unseen-value and rejected-request totals from `/metrics` so each week reports its own change |
| `parse_inject(spec, n_weeks)` | Validates scenario name, week range and severity |
| `assign_weeks(df, n_weeks, seed)` | Shuffles patients with a fixed seed, then walks the list cumulatively so weeks have about equal stays and a patient never spans two weeks |
| `covariate_shift`, `population_shift`, `population_shift_young`, `new_categories`, `drop_by_age` | Scenario functions applied to a week's in-memory feature frame (they keep the original index so labels still line up). `new_categories` injects a value certainly not in the model's known categories |
| `corrupt_batch(records, k, info)` | Rotates through null / text-in-number / removed-column so all three rejection reasons appear |
| `info_from_holdout`, `show_examples` | `--show-examples`: real before/after rows per scenario, no sending |
| `post_predict`, `reject_reason` | One timed `/predict` call with replay headers; extract the 422 reason |
| `write_summary(...)` | `summary.json` and `summary.md`, including the clean-week flagged share and weeks outside the alert band |
| `main()` | Validate, build and print the plan, check `/health` and `/model-info`, write `plan.json` and the data-use line, then loop over weeks: scenarios, batches (for `bad_data` a corrupted copy first, then the clean batch), collect results, append labels for stays actually scored, sleep out the rest of the week |

**Never sends** `readmitted`, `readmitted_raw` or `patient_nbr`; only columns the live model asks for plus `encounter_id`. The holdout file is never modified.
**Failure.** `SystemExit` on a bad run id, week arguments, batch size, missing holdout file, an unhealthy API, missing columns, or an existing run folder (run ids are never overwritten; use `--start-week` to resume). A non-200 on a clean batch **stops the run** and records `stopped in week N`. `KeyboardInterrupt` writes what was done so far. Resuming drops the old rows of the resumed week from `labels.csv`.

---

## 16. `serving/rollback_drill.py`

**Purpose.** Rehearse a bad model going live and the real rollback. **Run.** `python serving/rollback_drill.py --start [--to baseline] [--yes]` then `--finish`; also `--url`, `--admin-token`, `--skip-export`.
**Imports.** `argparse`, `json`, `os`, `subprocess`, `sys`, `urllib`, `datetime`, `pathlib`, `mlflow`, and `promote` as `P` (after inserting the project root in `sys.path`).

| Function | What it does |
|---|---|
| `export_and_reload(url, token, skip_export)` | Runs `export_champion.py` as a subprocess, then `POST /reload`; a refused reload is reported and the old model keeps serving |
| `main()` `--start` | Resolves champion and target, asks for confirmation (unless `--yes`), calls `P.swap_aliases`, `P.refresh_champion_files`, `P.append_log("drill_swap")`, then exports and reloads |
| `main()` `--finish` | Requires `@previous_champion`, runs the real `promote.py --rollback`, restores the baseline version's `role` tag (which `swap_aliases` overwrote), exports and reloads |

**Failure.** `SystemExit` if the target is already champion, the confirmation is not "yes", or there is no `@previous_champion`. `subprocess.run(..., check=True)` aborts if the export fails.

---

## 17. `monitoring/drift_lib.py`

**Purpose.** The shared PSI functions, used by the report, the live exporter and the export step. **Imports.** `numpy`, `pandas` only, so the exporter image stays small.
**Constants.** `EPS 1e-4` (floor for empty bins), `RARE_SHARE 0.005`, `MAX_DISCRETE 12`, `N_BINS 10`, `MODERATE 0.10`, `MAJOR 0.25`, `SCORE "__score__"`.

| Function | What it does |
|---|---|
| `_label(v)`, `_labels(series, numeric)` | Make `3`, `3.0` and `"3"` the same category label; text stays text |
| `make_profile(series)` | Numeric with more than 12 distinct values: decile bin edges. Otherwise: the category list, with categories under 0.5% pooled as `__rare__` |
| `shares(profile, series)` | Share of stays per bin or category; values outside the reference go to `__unseen__` |
| `psi(expected, actual)` | `sum((a - e) * ln(a / e))` with empty bins floored |
| `level(v)` | `stable` below 0.10, `moderate` below 0.25, else `major` |
| `describe(profile, base, week, e, a)` | One sentence: a mean shift for numerics, the three biggest share changes for categories |
| `profile_with_shares`, `profile_to_json`, `profile_from_json`, `build_reference` | Add reference shares, convert to and from JSON (sets and arrays handled), build the file `export_champion.py` saves |

**Failure.** Pure functions with no I/O. Empty input gives zero counts; `shares` divides by `max(1, ...)` so it never divides by zero. Tested by `tests/test_drift_lib.py`.

---

## 18. `monitoring/drift_report.py`

**Purpose.** Per-week drift report for one replay run. **Run.** `python monitoring/drift_report.py --run-id RUN [--baseline-weeks 1-4] [--top 5] [--min-rows 300] [--log] [--out-dir] [--seed 42]`.
**Imports.** `argparse`, `json`, `datetime`, `pathlib`, `matplotlib` (Agg; `BoundaryNorm`, `ListedColormap` for the heatmap), `numpy`, `pandas`, and from `drift_lib` the PSI functions.

| Input | Output |
|---|---|
| `serving/logs/predictions.jsonl`, optional `data/replay/<run>/plan.json` | `data/drift/<run>/{drift_report.md, psi_by_week.csv, drift_summary.json, drift_heatmap.png}` |

| Function | What it does |
|---|---|
| `parse_weeks("1-4")` | `1-4` or `1,2,3` to a sorted list |
| `load_log(path, run_id)` | Keeps rows of the run with a replay week, flattens `features` into columns, drops duplicate `(id, week)` (a resumed week is logged twice) |
| `main()` | Builds profiles from baseline weeks only; measures a **noise floor** (a clean sample of typical week size against the rest of the baseline); computes PSI per week and feature plus the score; builds the "did we catch it" table from `plan.json` (`DETECT_FEATURES` maps scenario to the features that should move; `bad_data` is "not visible here"); draws the heatmap; writes files |

**Failure.** `SystemExit` if the log is missing, no stays carry the run id, or the baseline weeks hold fewer than `--min-rows` stays. Weeks below `--min-rows` are marked "too small to judge" rather than judged.

---

## 19. `monitoring/drift_exporter.py`

**Purpose.** Turn the drift report into live Prometheus metrics. Runs as the `drift-exporter` container. **Run.** `python monitoring/drift_exporter.py [--once]` (`--once` evaluates, prints the metrics and exits).
**Imports.** `argparse`, `json`, `os`, `sys`, `time`, `datetime`, `pathlib`, `pandas`, `prometheus_client` (`CollectorRegistry`, `Gauge`, `generate_latest`, `start_http_server`), `drift_lib`.
**Environment.** `LOG_PATH`, `PROFILE_PATH`, `WINDOW_ROWS` (1000), `MIN_ROWS` (500), `MAX_AGE_SECONDS` (600), `INTERVAL` (15), `PORT` (9100), `RUN_ID` (empty = all traffic).

| Function | What it does |
|---|---|
| `load_profile()` | Loads `reference_profile.json` and reloads it when the file's modification time changes (after a new export) |
| `tail_lines(path, n)` | Reads only the end of the log (1 MB chunks) |
| `read_window(path, n, run_id)` | Last `n` lines as a frame with timestamp, version, score and features; optional run filter |
| `clear_verdict()` | Removes the PSI series, sets score PSI to NaN and `ready` to 0, so alerts clear instead of sticking |
| `evaluate()` | One cycle (below); returns a sentence explaining what happened |
| `main()` | Starts the HTTP server and loops forever; one failed cycle is caught, clears the verdict and prints to stderr |

`evaluate()` gives **no verdict** when: no profile, no log, an empty window, fewer than `MIN_ROWS` stays, or the newest stay is older than `MAX_AGE_SECONDS`. Otherwise it publishes PSI for every feature, counts features per level, and publishes the score PSI **only from stays scored by the same model version as the profile** (if fewer than `MIN_ROWS` such stays exist there is no score verdict).
**Failure.** Never exits on a cycle error. A missing profile or log is reported in the log line, not raised.

---

## 20. `monitoring/performance_report.py`

**Purpose.** Real performance from the late labels. **Run.** `python monitoring/performance_report.py --run-id RUN [--versions 2=lightgbm,3=logistic] [--label-delay-weeks 4] [--as-of-week N] [--min-positives 30] [--n-boot 200] [--seed 42] [--log] [--labels] [--evaluation] [--out-dir]`.
**Imports.** `argparse`, `json`, `datetime`, `pathlib`, `matplotlib` (Agg), `numpy`, `pandas`, `average_precision_score`, `roc_auc_score`.

| Input | Output |
|---|---|
| `serving/logs/predictions.jsonl`, `data/replay/<run>/labels.csv`, `models/evaluation.json`, optional `data/drift/<run>/drift_summary.json` | `data/performance/<run>/{performance_report.md, performance_by_week.csv, performance_summary.json, performance_chart.png}`; a line in `data_use_log.jsonl` |

| Function | What it does |
|---|---|
| `parse_versions(text)` | Maps registry versions to families, so each version is judged against **its own** gate result. Default `1=xgboost,2=lightgbm,3=logistic` |
| `load_predictions(path, run_id)` | id, week, score, flag, version for the run; drops duplicate `(id, week)` |
| `load_references(path, versions)` | Per version: family, test PR-AUC, its 95% range and the readmission rate (`PR-AUC / lift`) from `evaluation.json` |
| `metrics(g)` | n, positives, prevalence, flagged share, PR-AUC, ROC-AUC, lift, precision and recall of the flagged stays (NaN where undefined) |
| `bootstrap_pr(g, n_boot, seed)` | Patient-level resampling; 2.5th and 97.5th percentiles; NaN if fewer than 20 valid resamples |
| `verdict(pr, lo, hi, ref, positives, min_pos)` | `too few readmissions`, `no reference`, `below reference` (upper end under the reference), `above reference`, else `consistent` |
| `summarize(g, ref, args, seed_offset)` | `metrics` plus range plus verdict |
| `fmt`, `rng_text` | Formatting helpers |
| `main()` | Joins predictions and labels on `(id, week)` (unmatched stays are counted), applies the label delay and `--as-of-week`, builds weekly, by-version and by-injected-test tables, combines with drift into one of four readings (healthy, harmless drift, harmful drift, silent degradation), writes the outputs |

**Failure.** `SystemExit` if the log or labels file is missing, no logged stays carry the run id, no stay matches a label, or no labels would be known yet as of the chosen week. If `evaluation.json` has no entry for a version, its weeks read `no reference` rather than failing.

---

## 21. `deploy/prepare_bundle.py`

**Purpose.** Copy the champion bundle into `deploy/bundle/`, the committed folder the Render image is built from. **Run.** `python deploy/prepare_bundle.py` after `export_champion.py`.
**Imports.** `hashlib`, `json`, `shutil`, `pathlib`.

| Function | What it does |
|---|---|
| `sha256(path)` | Streamed SHA-256 |
| `main()` | Requires `model.joblib` and `meta.json` in `serving/bundle/`; verifies the model against `meta.json`'s checksum; copies both; prints version, family, export time and size; warns above 50 MB and refuses above 100 MB (GitHub limits) |

**Failure.** `SystemExit` if a file is missing ("Run: python serving/export_champion.py"), if the checksum does not match (an unfinished export is never deployed), or if the model exceeds 100 MB.

---

## 22. `tools/mlflow_runs.py`

**Purpose.** List every MLflow run with its group (original, a rejected `v2-...` attempt, retrain flow runs) and optionally soft-delete the v2 attempt. **Run.** `python tools/mlflow_runs.py [--delete-v2 [--apply]]`.
**Imports.** `argparse`, `os`, `datetime`, `pathlib`, `mlflow`, `MlflowClient`.
`main()` groups runs by the `data_tag` of the most recent training run before them (evaluation runs carry no tag), prints a table, and with `--delete-v2 --apply` calls `client.delete_run` for runs of groups starting `v2-`.
**Safety.** Runs that back a registered version are never deleted; only groups starting `v2-` can be deleted; deletion is soft (nothing in `mlruns/` is removed). `--apply` without `--delete-v2` is refused. Copy `mlflow.db` first.

---

## 23. Tests

Run with `pytest` (config in `pytest.ini`). `tests/conftest.py` creates an empty `data/` folder (the scripts use it to find the root, and git ignores it) and adds the root, `code/`, `serving/` and `monitoring/` to `sys.path`.

| File | Target | What it pins down |
|---|---|---|
| `test_retrain_decide.py` | `retrain.decide`, `retrain.main` | keep when not below the reference; the optimistic end touching the reference is not enough; retrain only when the whole range is below; fewer than 4 weeks keeps; no stays for the live version keeps; scope matters; `fix_data_first` with unseen categories; unseen categories alone never trigger retrain; reference fallbacks (evaluation.json, champion family); missing reference keeps with a reason; `--force-retrain` is recorded as forced and keeps the evidence decision |
| `test_icd9_group.py` | `data_cleaning.icd9_group` | every group, missing leading zeros, V and E codes, junk input |
| `test_drift_lib.py` | `drift_lib` | identical data gives zero PSI; a shift is major; a new category lands in `__unseen__`; level thresholds; JSON round trip of a profile |
| `test_build_frame.py` | `app.build_frame`, `app.inspect_pipeline` | valid conversion (`1.0` becomes `"1"`), unseen categories reported not rejected, missing column / null / text-in-number rejected with the right reason, schema read from a tiny fitted pipeline |

CI (`.github/workflows/ci.yml`) runs `ruff check --select E9,F63,F7,F82 .` (syntax errors and undefined names only) and `pytest`, then builds both Docker images.

---

## 24. Failure handling cheat sheet

| Situation | What happens | State left behind | What to do |
|---|---|---|---|
| One model crashes in `train.py` | Recorded as FAILED; others finish | Other joblibs and runs are fine | Read `training_report.txt`, fix, rerun all three |
| Train file is `test.csv` or `holdout.csv` | `ValueError`, nothing trained | None | Use `train.csv` or a proper training file |
| Test data lacks a trained column | `ValueError` in `test.py` | `evaluation.json` unchanged | Clean the data the same way as training |
| Model fails the gate | `passed: false` | Report written | It cannot be registered |
| Registering a failed model | `REFUSED`, exit 1 | Registry unchanged | Do not register it |
| `evaluation.json` stale versus MLflow | `RuntimeError` (no matching run) | Registry may be partly updated only for earlier families in champion mode | Rerun `train.py`, `test.py` |
| Challenger does not clearly win | `KEEP CHAMPION`, report and log line | Registry untouched | Nothing; it is a normal outcome |
| No `@previous_champion` | `SystemExit` on rollback | None | Nothing to roll back to |
| Export interrupted | Checksum mismatch at load | Old bundle's `meta.json` may be missing or stale | Rerun the export; the API keeps its old model meanwhile |
| API starts with no bundle | Starts; `/health` 503 | None | Export, then Reload |
| `/reload` with a bad bundle | 500 with `still_serving_version` | Old model keeps serving | Re-export |
| Prediction log cannot be written | Logged and ignored | Request served | Fix permissions or disable logging |
| Replay gets a 422 on a clean batch | Run stops and records it | Partial outputs and labels for completed weeks | Fix the data, resume with `--start-week` |
| Drift exporter has too few or stale stays | No verdict, PSI series removed | Alerts clear | Send traffic; normal |
| A cycle of the drift exporter throws | Caught, verdict cleared, error to stderr | Exporter keeps running | Read `docker compose logs drift-exporter` |
| Retrain flow script crashes | Prefect run fails | `models_before/` backup exists if `--apply` | Inspect the log; restore from the backup if needed |
| Retrain flow reload fails | Warning only | Bundle exported | Start the API, `POST /reload` |

---

## 25. Known quirks

Small things worth knowing, none of which breaks behaviour:

- `test.py` defines `recall_at_top` but `main` computes recall from a cutoff instead; the function is currently unused.
- `drift_report.py` imports `N_BINS` from `drift_lib` without using it; `promote.py` defines `EXPERIMENT` without using it; `register_champion.py` defines `NOT_FEATURES` without using it.
- Several scripts' messages still say `split_data.py` and `train_models.py`; the real files are `code/data_split.py` and `train.py`.
- `promote.py` executes the top of `test.py` at import time, so importing `promote` (as `rollback_drill.py` does) also resolves `test.py`'s project root.
- Registry version numbers change when models are re-registered after a rebuild (currently LightGBM v6, XGBoost v5, logistic v7). `performance_report.py --versions` and `retrain.py --champion-version` must match the run being analysed.
- The monitoring and alert files are loaded from disk by Prometheus; the console, by contrast, is baked into the Render image, so UI edits reach the public site only after a push.