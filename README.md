# Clinical readmission MLOps project

Predict which diabetic hospital stays will end in a readmission within 30 days, and manage the models behind that
prediction in a disciplined, repeatable way: clean, split, train, gate, register, promote, serve over an API,
monitor, detect drift, measure real performance from late labels, decide when to retrain, and deploy online.

**Live demo:** https://clinical-risk-prediction.onrender.com
(batch scoring: https://clinical-risk-prediction.onrender.com/#batch, API docs: `/docs`, health: `/health`).
The service runs on Render's free plan, so after a quiet period the first request can take 30 to 60 seconds while it wakes up.

> **Not a clinical tool.** The Project covers full MLOps Lifecycle, Not for clinical use. The data used here is the public UCI "Diabetes 130-US Hospitals (1999-2008)" dataset
> (CC BY 4.0, https://doi.org/10.24432/C5230J).

Last updated: 2026-11-03.

---

## Contents

1. [What this project is](#1-what-this-project-is)
2. [The big picture](#2-the-big-picture)
3. [Run it from a fresh clone](#3-run-it-from-a-fresh-clone)
4. [Folder map](#4-folder-map)
5. [Words you will meet](#5-words-you-will-meet)
6. [Docker and Docker Desktop](#6-docker-and-docker-desktop)
7. [The pipeline, step by step](#7-the-pipeline-step-by-step)
8. [Deployment on Render (plan step 18)](#8-deployment-on-render-plan-step-18)
9. [Monitoring (Prometheus and Grafana)](#9-monitoring-prometheus-and-grafana)
10. [Tests and CI](#10-tests-and-ci)
11. [How everything affects everything else](#11-how-everything-affects-everything-else)
12. [Recipes](#12-recipes)
13. [Current results](#13-current-results)
14. [Limits and rules to respect](#14-limits-and-rules-to-respect)
15. [Status and what is not built yet](#15-status-and-what-is-not-built-yet)
16. [Troubleshooting](#16-troubleshooting)
17. [File-by-file guide](#17-file-by-file-guide)
18. [Publishing to GitHub](#18-publishing-to-github)

---

## 1. What this project is

**The prediction task.** Each row of data is one hospital stay of a diabetic patient. The label is whether the patient
came back within 30 days (about 11.5% did). A model gives every stay a risk score so that the highest-risk stays can be flagged.

**The real point: MLOps.** MLOps is the machinery around a model. Training a model once is easy. The hard part is being
able to answer, months later:

- Which model is in use, and why was it chosen?
- Was it tested on data it had never seen, using rules written down beforehand?
- Can a newer model replace it safely, and can we undo that?
- Can someone else rebuild everything and get the same numbers?
- Is the live service healthy, are its inputs and scores behaving normally, and are its predictions still correct?
- When should it be retrained, and who decides?

This project answers those questions with scripts for each stage, an experiment tracker and model registry (MLflow),
an HTTP API with a web console (FastAPI), a monitoring stack (Prometheus, Grafana), drift and performance reports,
a retrain decision rule with an orchestrated flow (Prefect), automated tests and CI (GitHub Actions), and a public
deployment (Render).

**An honest note on model quality.** Readmission is hard to predict from hospital records. The current best model finds
readmissions about 1.9 times better than random flagging, with a ROC-AUC of about 0.67. That is typical for this kind of
data and is not a sign of a bug.

---

## 2. The big picture

```
 raw data (data/diabetic_data.csv, from UCI)
    |  code/data_cleaning.py  ->  code/leak_review.py (checks)  ->  code/data_split.py   (split BY PATIENT)
    v
 data/splits/   train.csv        test.csv         holdout.csv
                   |                |                |
         train.py  |                |                | (only used by the replay,
                   v                |                |  serving/replay.py)
  models/*.joblib + MLflow runs     |
                   |                |
                   +----> test.py <-+      "the gate": every model is scored on data it has never seen
                             |
                             v
         register_champion.py --champion X --reason "..."     (YOU decide)
                             |
                             v
        MLflow registry "readmission"  ---->  champion.json      @champion  @challenger  @baseline
                  ^          |
                  |          |  serving/export_champion.py: registry @champion -> serving/bundle/
                  |          |  deploy/prepare_bundle.py:   serving/bundle/    -> deploy/bundle/  (committed)
                  |          v                                    |
   promote.py: challenger   Local Docker stack:                   v
   vs champion; swaps       API + drift exporter +          GitHub -> Render builds deploy/Dockerfile
   aliases only if the      Prometheus + Grafana            -> public API + web console (the live demo)
   challenger clearly wins        |                               |
   (and can roll back)            +--- Prometheus also scrapes ---+   (job readmission-render, over HTTPS)

 Feedback loop:  replay -> prediction log -> drift report / exporter (inputs, scores)
                                          -> performance report (late labels = real accuracy)
                                          -> retrain.py decide -> retrain flow -> challenger -> gate -> promotion rule
```

Four ideas hold everything together:

1. **Separate data for separate jobs.** Train is for learning, test is for the final exam, the holdout is sealed for the replay.
2. **Gate, then register, then promote.** A model must pass a pre-written test before it can be registered, and must clearly beat the champion to replace it.
3. **The registry is the source of truth.** "Which model is champion?" is answered by the MLflow registry, not by a file name. The API serves an exported copy (a "bundle").
4. **Drift is a symptom, performance is the verdict.** A model is retrained only when real, labelled performance says it has gone stale.

---

## 3. Run it from a fresh clone

The repository does **not** contain the training data splits, the trained models, or the MLflow database (see
[section 18](#18-publishing-to-github)). You rebuild them in a few commands. Everything is seeded (42), so a rebuild reproduces
the numbers in [section 13](#13-current-results).

### 3.1 What you need

| Need | Why |
|---|---|
| Python 3.9 (built on 3.9.7) | The pinned packages are matched to it |
| About 5 GB free disk | Packages, MLflow artifacts, Docker images |
| Docker (Docker Desktop on Windows/macOS, Docker Engine on Linux) | Only for the local API and monitoring stack |
| Git | To clone |

### 3.2 Clone and create the environment

Windows (PowerShell):

```powershell
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
python -m venv .venv                       # use a Python 3.9 interpreter
.\.venv\Scripts\Activate.ps1               # prompt shows (.venv)
python -m pip install --upgrade pip
python -m pip install --prefer-binary --no-cache-dir -r requirements.txt
```

macOS / Linux:

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --prefer-binary --no-cache-dir -r requirements.txt
```

If PowerShell refuses to run the activation script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
`requirements-lock.txt` is a full Windows snapshot (it includes Windows-only packages such as `pywin32`); on macOS or Linux use `requirements.txt`.

### 3.3 Get the data

The two raw UCI files are tracked in the repository (`.gitignore` keeps them). If they are missing, download them from
https://doi.org/10.24432/C5230J ("Diabetes 130-US Hospitals for years 1999-2008") into `data/`:

```
data/diabetic_data.csv
data/IDS_mapping.csv
```

### 3.4 Prepare the data

```bash
python code/data_cleaning.py     # raw -> data/diabetic_data_cleaned.csv (+ cleaning_config.json, cleaning_report.txt)
python code/leak_review.py       # optional: leak checks -> data/leak_review_report.txt
python code/data_split.py        # -> data/splits/train.csv, test.csv, holdout.csv, split_assignments.csv, split_report.txt
```

Expected sizes: train 59,753 stays / 42,054 patients, test 14,907 stays / 10,483 patients, holdout 24,680 stays / 17,450 patients.

### 3.5 Train, gate, register, compare

```bash
python train.py                                  # 1. train all three models (always all three)
python test.py                                   # 2. the gate: score them on the test split
python register_champion.py --champion lightgbm --reason "Passed the gate; chosen for credible feature reasoning"
python promote.py --dry-run                      # 4. would the challenger (XGBoost) replace the champion?
```

`python test.py --no-mlflow` should show LightGBM 0.222, XGBoost 0.218 and logistic 0.206 test PR-AUC. If it does, your
environment reproduces the project. To browse runs and the registry:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # open http://127.0.0.1:5000 (Experiments and Models tabs)
```

### 3.6 Serve the champion locally (Docker)

```bash
python serving/export_champion.py        # registry @champion -> serving/bundle/ (model.joblib, meta.json, reference_profile.json)
docker compose up -d --build             # builds the API and drift images, starts API + drift exporter + Prometheus + Grafana
python serving/send_sample.py --n 20     # send 20 test rows, print the answers
```

Start Docker Desktop first on Windows or macOS and wait until it says **Engine running** (see [section 6](#6-docker-and-docker-desktop)).

| Thing | Address | Login |
|---|---|---|
| Web console | http://localhost:8000/ | none |
| API docs (Swagger) | http://localhost:8000/docs | none |
| Health | http://localhost:8000/health | none |
| Prometheus | http://localhost:9090 (Status > Targets, Alerts) | none |
| Grafana | http://localhost:3000 | admin / admin (change it) |
| Drift exporter metrics | http://localhost:9100/metrics | none |

Stop everything: `docker compose down` (keeps metric history) or `docker compose down -v` (deletes it).

Without Docker: `pip install -r serving/requirements-serve.txt`, export the bundle, then
`python -m uvicorn app:app --app-dir serving --port 8000`.

### 3.7 Replay, drift report, performance report, retrain decision

With the stack up and a bundle exported:

```bash
python serving/replay.py --preset demo --dry-run                                 # show the 12-week plan, send nothing
python serving/replay.py --preset demo --run-id demo1 --seconds-per-week 120     # about 24 minutes

python monitoring/drift_report.py --run-id demo1 --baseline-weeks 1-4            # step 13: which inputs moved
python monitoring/performance_report.py --run-id demo1                           # step 14: real performance from late labels
python retrain.py decide --run-id demo1                                          # step 15: should we retrain?
```

Outputs land in `data/replay/`, `data/drift/`, `data/performance/` and `data/retrain/` (each in a folder named after the run id).
To rehearse a bad model and the rollback under traffic: start `python serving/replay.py --run-id drill3 --weeks 8 --seconds-per-week 90`,
then in a second window run `python serving/rollback_drill.py --start` as week 3 begins and `--finish` as week 7 begins.

### 3.8 After a promotion or rollback

```bash
python serving/export_champion.py            # refresh the local bundle from the registry
curl -X POST http://localhost:8000/reload    # or press "Reload model" on the console's Model tab
python deploy/prepare_bundle.py              # to also update the public deployment (see section 8)
git add deploy/bundle && git commit -m "Deploy new champion" && git push
```

---

## 4. Folder map

```
<project root>/
|-- code/                       data preparation: data_cleaning.py, leak_review.py, data_split.py (+ .md explanations)
|-- data/                       (mostly not in git) data and reports
|   |-- diabetic_data.csv, IDS_mapping.csv    the raw UCI files (the only data tracked in git)
|   |-- diabetic_data_cleaned.csv, splits/    produced by code/ scripts
|   |-- replay/<run-id>/        written by serving/replay.py: plan.json, labels.csv, summary.json, summary.md
|   |-- drift/<run-id>/         written by monitoring/drift_report.py
|   |-- performance/<run-id>/   written by monitoring/performance_report.py
|   `-- retrain/<retrain-id>/   written by retrain.py and the retrain flow: trigger.json, retrain_report.md, models_before/
|-- models/                     (not in git, except four .md reports) everything the pipeline produces
|   |-- 01_<model>_prauc0.xxx.joblib ...   saved models, ranked by CV score (train.py)
|   |-- best_model.json, model_report.md   training outputs
|   |-- evaluation.json, test_report.md    the gate (test.py)
|   |-- champion.json, champion_report.md  the current champion
|   |-- promotion_log.jsonl, promotion_report.md   audit trail and latest decision
|   `-- data_use_log.jsonl      every time any dataset was scored or used
|-- mlflow.db, mlruns/          (not in git) MLflow history AND registry   (DO NOT DELETE)
|-- serving/                    the API
|   |-- app.py, export_champion.py, send_sample.py, replay.py, rollback_drill.py
|   |-- Dockerfile, requirements-serve.txt, .dockerignore
|   |-- ui/                     the web console: index.html, styles.css, app.js
|   |-- bundle/                 (created, not in git) model.joblib, meta.json, reference_profile.json
|   `-- logs/                   (created, not in git) predictions.jsonl
|-- deploy/                     the public deployment (Render)
|   |-- Dockerfile              image with the model baked in
|   |-- prepare_bundle.py       serving/bundle -> deploy/bundle (checksum-verified)
|   `-- bundle/                 (committed) model.joblib + meta.json that Render deploys
|-- render.yaml                 Render Blueprint: one web service built from deploy/Dockerfile
|-- monitoring/                 drift, performance, Prometheus, Grafana
|   |-- drift_lib.py, drift_report.py, drift_exporter.py, performance_report.py
|   |-- Dockerfile.drift, requirements-drift.txt, .dockerignore   the exporter's image
|   |-- prometheus/             prometheus.yml, alerts.yml, alerts_replay.yml, alerts_drift.yml
|   `-- grafana/                provisioning/datasources/prometheus.yml, starter_dashboard.json, drift_dashboard.json
|-- orchestration/retrain_flow.py   Prefect flow for the retrain loop (runs in its own environment)
|-- tests/                      pytest tests: retrain rule, ICD-9 grouping, PSI, API input validation
|-- tools/mlflow_runs.py        list MLflow runs by group, optionally soft-delete a rejected attempt
|-- .github/workflows/ci.yml    CI: lint, tests, Docker builds
|-- docker-compose.yml          local stack: API + drift exporter + Prometheus + Grafana
|-- train.py, test.py, register_champion.py, promote.py, retrain.py    the pipeline scripts
|-- requirements.txt, requirements-lock.txt, requirements-dev.txt, pytest.ini
|-- .gitignore
|-- .venv/, .venv-prefect/      (not in git) project environment and the separate Prefect environment
`-- __pycache__/                (not in git)
```

Two pairs of files to keep straight:

- `mlflow.db` + `mlruns/` together are the experiment history and the registry. Delete either one and the registry is damaged.
- `.venv/` is only installed packages. It is safe to delete and rebuild.

---

## 5. Words you will meet

| Word | Plain meaning |
|---|---|
| **MLOps** | Managing models like software: tracked, tested, versioned, monitored and replaceable. |
| **Train / test / holdout** | Three separate piles of patients. Train: the model learns from it. Test: the final exam used to judge models. Holdout: sealed, for the replay. |
| **Split by patient** | All stays of one patient go into the same pile, so the model is never tested on a patient it has seen. |
| **Positive rate** | The share of stays that were readmitted (about 11.5%). Because it is small, plain accuracy is misleading. |
| **Cross-validation (CV)** | Training on part of the training data and scoring on the rest, repeated 5 times. |
| **PR-AUC** | The main score: how well truly-readmitted stays are ranked near the top without flagging too many others. No skill scores about 0.115 (the positive rate). |
| **ROC-AUC** | The chance a random readmitted stay outranks a random other one. 0.5 is a coin flip, 1.0 is perfect. |
| **Precision / recall (flagged)** | Of the stays flagged high risk, the share really readmitted (precision); of all readmitted stays, the share flagged (recall). |
| **Lift** | PR-AUC divided by the positive rate. 1.9x means about twice as good as chance. |
| **95% range (bootstrap)** | Whole patients are resampled 200 times to show how much a score could wobble by luck. |
| **Evaluation gate** | Pass/fail rules, written before looking at test results, that a model must meet. |
| **SHAP** | A method that measures how much each input pushed a prediction up or down. |
| **Calibration** | Whether a predicted 20% risk really means 20% of such patients are readmitted. Our scores are not calibrated. |
| **MLflow, experiment, registry** | MLflow records every run. The experiment (`readmission-30d`) is the history; the registry (`readmission`) is the short list of models that matter. |
| **Alias** | A movable label on a registered version: `champion`, `challenger`, `baseline`, `previous_champion`. |
| **Bundle** | The exported champion (`model.joblib`, `meta.json`, `reference_profile.json`). The API never talks to MLflow. |
| **PSI** | Population stability index: how far a feature's (or the score's) distribution moved from a reference. Below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 major. |
| **Late labels** | The true outcomes, which in real life arrive about 30 days after discharge. |
| **Drift** | Live data (or scores) behaving differently from what the model was built on. It is a symptom, not a verdict. |
| **Prefect flow** | A scripted sequence of steps with logging and failure handling; here it runs the retrain loop. |
| **Blueprint (Render)** | `render.yaml`: the file that tells Render which service to build and how to run it. |
| **Image / container / compose** | An image is a packaged OS + Python + packages + code; a container is a running copy; compose starts several together. |

### 5.1 Metrics in one table

| Metric | What it measures | The question it answers |
|---|---|---|
| Request rate, latency (p50, p95), error rate | API traffic and health | Is the service working? |
| Feature PSI | Input distribution change | Has the patient data changed? |
| Score PSI | Score distribution change | Has the model's output changed? |
| PR-AUC | Ranking of rare positives | Can we find readmissions? |
| ROC-AUC | Positive versus negative ranking | Do readmitted stays score higher? |
| Precision, recall | Correctness and coverage of the flags | How many flagged stays were readmitted; how many readmissions did we catch? |
| Lift | Richness of the flagged group | Is the flagged group riskier than average? |
| 95% range | Uncertainty of an estimate | How reliable is this number? |

PR-AUC is the area under the precision-recall curve (scikit-learn `average_precision_score`). It is not accuracy and not F1.
A model saying "nobody is readmitted" is right 88.5% of the time and useless, which is why PR-AUC is the main metric.

### 5.2 Steps 11 to 15 are one feedback system

They answer a single question: **is the model still making useful predictions, and if not, what changed?** The difficulty is
delayed ground truth: a prediction is made at discharge, but whether the patient came back is only known about 30 days later.

| Step | Question | What it does |
|---|---|---|
| 11, Replay | What happens when new data arrives? | Sends the holdout week by week through the API, deliberately damaging some weeks; withholds the outcomes. |
| 12, Monitoring | Is the application working? | Prometheus and Grafana track request rate, latency, errors, flagged share. |
| 13, Drift | Has the incoming data changed? | PSI of every input feature and of the scores against a reference. |
| 14, Late labels | Are the predictions actually correct? | Joins the true outcomes to the logged predictions; real PR-AUC, precision, recall. |
| 15, Retrain | Should we replace the model? | A written rule decides; an orchestrated flow trains, gates and registers a challenger; the promotion rule decides. |

The decision, in order: (1) is the data wrong (broken feed, renamed code)? Fix it, do not retrain. (2) Did pooled real performance
drop? If not, keep the model; drift alone is not harm. (3) If it did, train a challenger and promote it only if it clearly wins.

What this cannot tell you: drift is not causation; labels create a blind period of weeks; a single week is noisy (pool weeks);
PR-AUC says nothing about calibration; and retraining does not fix a broken feed.

---

## 6. Docker and Docker Desktop

**Docker is the engine; Docker Desktop is the easiest way to get that engine on Windows or macOS.** The commands are identical
(`docker compose up`). Containers use the Linux kernel, so Docker Desktop runs a small hidden Linux VM (WSL 2 on Windows).

Docker is used here for **serving and monitoring**, not for training:

1. **Reproducible runtime.** The image fixes the OS, Python 3.9 and exact package versions.
2. **Deployment realism.** A model API ships as a container; Render builds the same kind of image.
3. **One command for several services.** `docker compose up` starts the API, drift exporter, Prometheus and Grafana on a private network.
4. **Isolation.** The API runs as a non-root user and sees only the folders mounted into it.
5. **Health checks and restarts.** Containers are probed at `/health` or `/metrics` and restart after a reboot.
6. **Swappable model without rebuilding.** Locally the model is a read-only mounted folder (`serving/bundle`); a promotion needs a new export and `POST /reload`.
   On Render there are no mounted folders, so the model is baked into the image (`deploy/bundle`).

**Why a bundle instead of reading MLflow inside the container?** MLflow stores absolute Windows paths in `mlflow.db` that do not exist in a
Linux container, and SQLite does not behave on mounted Windows folders. `export_champion.py` runs on the host and writes a self-contained
`model.joblib` plus `meta.json` with a SHA-256 checksum.

Everyday commands:

```bash
docker compose up -d --build     # build images if needed, start all services in the background
docker compose ps                # what is running, and is it healthy?
docker compose logs -f readmission-api
docker compose restart prometheus
docker compose down              # stop and remove containers (named volumes, so metric history, are kept)
docker compose down -v           # ALSO delete the volumes (Prometheus history, Grafana settings)
```

If `docker version` prints only a **Client** section and every command hangs, the engine is not running: see Troubleshooting.

---

## 7. The pipeline, step by step

**Where the plan's 15 steps (plus deployment) live:**

| Plan step | What | Where | Status |
|---|---|---|---|
| 1 to 5 | Look at the data, clean, label, leak review, split | `code/` | Done |
| 6, 7 | Train models, track runs | `train.py`, MLflow | Done |
| 8 | Evaluation gate | `test.py` | Done (calibration reported, not gated) |
| 9 | Register, alias `champion` | `register_champion.py` (and `promote.py`, added beyond the plan) | Done |
| 10 | Serve (FastAPI + Docker) | `serving/`, `docker-compose.yml` | Done |
| 11 | Holdout replay, drift injection, rollback drill | `serving/replay.py`, `serving/rollback_drill.py` | Done |
| 12 | Monitoring | `monitoring/`, metrics in `serving/app.py` | Done |
| 13 | Drift: weekly report and live alerts | `monitoring/drift_*.py`, `alerts_drift.yml` | Done |
| 14 | Late labels, real performance | `monitoring/performance_report.py` | Done |
| 15 | Retrain loop | `retrain.py`, `orchestration/retrain_flow.py` | Done |
| 17 | Tests and CI | `tests/`, `.github/workflows/ci.yml` | Done |
| 18 | Deploy online | `render.yaml`, `deploy/` | Done: live |

### Step 0. Data preparation and split (`code/`)

1. **`data_cleaning.py`:** drops patients who died or went to hospice and 3 invalid-gender rows; groups the "unknown" ID codes; fills missing
   values with explicit categories; builds the label (`<30` = 1, else 0, keeping `readmitted_raw`); adds `n_drugs_taken`, `n_drugs_changed`
   and `diag_*_group`; drops `weight` and constant columns. Result: 99,340 rows, 69,987 patients, 11.39% positive.
2. **`leak_review.py`:** a hand-written table decides which columns are allowed ("known at discharge?"); statistics only flag suspects.
   Result: no leak flagged (whole-model AUC 0.669).
3. **`data_split.py`:** `StratifiedGroupKFold`, 20 folds grouped by `patient_nbr`, folds assigned 12 / 3 / 5 to train 60% / test 15% / holdout 25%.
   Asserts zero patient overlap.

Details: `code/data_cleaning.md` and `code/leak_review.md`.

### Step 1. `train.py`: train and compare models

Trains three models, all inside one scikit-learn `Pipeline` (preprocessing + model), so the same steps run in training, the gate and the API:

| Model | What it is | Handling of rare readmissions |
|---|---|---|
| Logistic regression | Simple weighted-evidence model; the **baseline** | `class_weight="balanced"` |
| XGBoost | Boosted trees | `scale_pos_weight` = 7.704 |
| LightGBM | Boosted trees, built differently | `scale_pos_weight` = 7.704 |

Preprocessing: skewed visit counts `log1p` + standardise, other numbers standardised, categories one-hot encoded with rare and unseen values
in one "infrequent" bucket (`MIN_CATEGORY_ROWS = 50`). Each model is scored with 5-fold cross-validation, stratified and grouped by patient.

```bash
python train.py                                   # all three models (always run all three)
python train.py --cv-folds 3                      # quicker check
python train.py --data-tag my-version             # label recorded in MLflow for this data version
python train.py --train-file path/to/other.csv    # train on another file (never test.csv or holdout.csv: refused)
```

Every MLflow run gets the tags `data_tag` and `train_file` and the parameter `train_file_sha256`, so you can tell which data trained which model.
Never run `--models xgboost` alone: the ranked joblib files are rebuilt from only the models in that run, so the others would be deleted.

Reads `data/splits/train.csv` only. Creates the ranked `models/NN_<model>_prauc<cv>.joblib` files, `best_model.json` (top of the CV ranking only,
not the champion), `model_report.md`, `data/training_report.txt` and MLflow runs.

### Step 2. `test.py`: the evaluation gate

Scores every saved model, unchanged, on the test split against rules written before the first result was seen (constants at the top of `test.py`;
never change them after seeing results):

| Check | Rule |
|---|---|
| Lift | at least 1.5x over chance |
| Overfitting | CV score minus test score at most 0.02 |
| Beats the baseline | non-baseline models must score at least as high as logistic |
| Fairness (ROC-AUC) | no gender, race or age group more than 0.05 below overall |
| Fairness (recall) | flagging the top 20%, no group's recall more than 0.10 below overall |
| Small groups | under 500 stays or 50 readmissions: listed, not judged |

Calibration is reported, not judged. The gate never chooses the champion; a person does. `--reference` picks the comparison anchor (default: the
current champion). `--data FILE [--name NAME]` runs the same gate on other data, writing separate files so `evaluation.json` is never overwritten;
every dataset use is appended to `models/data_use_log.jsonl`.

### Step 3. `register_champion.py`: the registry

```bash
python register_champion.py --champion lightgbm --reason "why"      # first-time setup (a human decision)
python register_champion.py --challenger xgboost --reason "what changed"   # every later model
```

Refuses a model that failed the gate. Champion mode registers one version per model and sets `champion`, `baseline` (logistic, always) and
`challenger` aliases, tags each version with its evidence, and verifies the registered champion predicts identically to the saved joblib.
Challenger mode registers one new version and leaves the champion untouched. After `promote.py` has changed the champion, champion mode refuses
unless `--force`.

### Step 4. `promote.py`: challenger versus champion

```bash
python promote.py --dry-run     # show the decision, change nothing
python promote.py               # decide and, if it wins, apply
python promote.py --rollback    # put the previous champion back
```

A challenger replaces the champion only if it passes every gate check (the thresholds are imported from `test.py`) **and** its PR-AUC gain over the
champion has a paired 95% range entirely above zero. A tie keeps the champion. On a win the aliases swap, `champion.json`/`champion_report.md` are
rewritten and a line is added to `promotion_log.jsonl`. By default it compares on the test split, which has already been used for selection, so prefer
fresh data (`--data`) for a final check.

### Step 5. `serving/`: the API and web console (plan step 10)

Serves the champion over HTTP: a risk score, a high-risk flag and the top SHAP drivers per stay.

| Endpoint | What it does |
|---|---|
| `GET /` | The web console (tabs `#score`, `#batch`, `#model`, `#monitor`) |
| `POST /predict` | Score, `high_risk` flag and drivers for 1 to `MAX_RECORDS` records |
| `GET /example`, `GET /model-info` | A ready request body; live model, evidence, required columns, allowed categories, flag rule |
| `GET /health` | 200 if a model is loaded, otherwise 503 |
| `GET /metrics` | Prometheus metrics: latency, errors, volume, scores, model version, unseen categories |
| `POST /reload` | Re-read the bundle; a bad bundle is refused and the old model keeps serving. Needs `X-Admin-Token` if `ADMIN_TOKEN` is set |

**Console views.** *Score a stay* (form with dropdowns built from training values; meter, verdict and driver bars), *Score a file* (upload a CSV,
chunked requests, histogram, top-100 table, search, click-through to a stay's explanation, CSV download), *Model* (live version, registry evidence,
reload) and *Monitoring* (parses `/metrics` in the browser, refreshes every 5 s).

**Rules the API enforces.** Every feature column present, no nulls, numeric columns numeric, categories matched as the model saw them (`1`, `1.0`
and `"1"` are the same); a never-seen category is still scored (in the "infrequent" bucket) and counted. Extra columns are ignored.

**The score is a ranking score, not a probability.** `high_risk` means score at or above a cutoff: the score the top 20% of reference (test) stays reach.
**Prediction log:** every scored record is appended to `serving/logs/predictions.jsonl` (set `LOG_PREDICTIONS=false` to disable); it holds patient-level
rows, so never commit or share it.

### Step 6. `serving/replay.py`: the holdout replay (plan step 11)

The sealed holdout is cut into simulated weeks (whole patients per week, seed 42) and sent week by week through the running API, with
`X-Replay-Run` and `X-Replay-Week` headers. From chosen weeks onward one kind of damage is injected at a time. The true outcomes are withheld from
the API and saved in `data/replay/<run-id>/labels.csv`. The holdout file itself is never modified.

```bash
python serving/replay.py --preset demo --dry-run                        # plan only
python serving/replay.py --preset demo --show-examples                  # real before/after rows per scenario, sends nothing
python serving/replay.py --preset demo --run-id demo1 --seconds-per-week 120
python serving/replay.py --run-id demo1 --start-week 7                  # resume a stopped run
```

Scenarios (`--inject NAME[:FIRST[-LAST]][:SEVERITY]`): `covariate_shift`, `population_shift`, `population_shift_young`, `new_categories`, `bad_data`.

**Results of `demo1` (12 weeks).** Clean weeks flagged 20.7% on average (design value 20%).

| Test | Weeks | What we saw | Caught by |
|---|---|---|---|
| Covariate shift | 5, 6 | Flagged share about 49%, mean score 0.52 to 0.54 | Replay flagged-share alert |
| New categories | 8, 9 | Unseen-value counter +204 and +198; scores barely moved | Unseen-category alerts |
| Bad data | 11 | 7 of 28 requests rejected (25%, as injected); every stay still scored after the clean resend | Error-share alert |
| Population shift | 12 | Flagged 19.1%, scores normal | **Not caught by score alerts** (caught later by feature PSI) |

**Rollback drill (`drill3`).** The logistic baseline was made champion at week 3 and rolled back at week 7. No request failed, about 12,300 stays were
scored by the weaker model, and the flagged share stayed near 20% because the export recomputes the cutoff for whichever model it exports. Label-free
monitoring cannot see a worse model; only labels can.

### Step 7. Drift detection (plan step 13)

Two tools answer: *did the inputs or scores move away from normal?*

| Tool | Runs | Output |
|---|---|---|
| `monitoring/drift_report.py` | On demand, for one replay run | Per-week PSI of every feature and the score, a "did we catch it" table, heatmap |
| `monitoring/drift_exporter.py` | Continuously, in Docker (port 9100) | Prometheus metrics and alerts (`alerts_drift.yml`) |

`PSI = sum over bins of (a - e) * ln(a / e)` where `a` is the new share and `e` the reference share. Numeric features use the reference deciles; categories
are compared value by value with `__rare__` and `__unseen__` buckets. The shared code is `monitoring/drift_lib.py`.

The exporter compares the most recent 1,000 logged stays with `serving/bundle/reference_profile.json` (written by `export_champion.py` from the test
split). It gives **no verdict** (and removes the PSI series so alerts clear) when there are too few stays, no recent traffic, or no profile. Settings
(`WINDOW_ROWS`, `MIN_ROWS`, `MAX_AGE_SECONDS`, `INTERVAL`, `RUN_ID`) are in the `drift-exporter` block of `docker-compose.yml`.

| Alert | Fires when |
|---|---|
| `FeatureDriftMajor` / `FeatureDriftModerate` | A feature's PSI is above 0.25 / between 0.10 and 0.25 for 1 minute (one alert per feature) |
| `ScoreDriftMajor` | The score PSI is above 0.25 |
| `DriftExporterDown` | The exporter cannot be scraped for 2 minutes |

**Findings from `demo1`.** PSI names the broken column (covariate shift: `number_inpatient` 1.26 to 1.29; new categories: `discharge_disposition_id` and
`medical_specialty` about 0.31 to 0.34; population shift: `age` 0.19, which score alerts missed). Clean weeks stay quiet (maximum 0.032). One change
ripples: the age shift also moved `payer_code`, `medical_specialty` and `discharge_disposition_id`. Read a burst of alerts together; the largest PSI is
usually the cause.

### Step 8. Late labels and real performance (plan step 14)

`monitoring/performance_report.py` joins the logged predictions to the withheld outcomes and computes real PR-AUC, ROC-AUC, lift, precision and recall of
the flagged stays, per week, per model version and per injected test, with patient-level 95% ranges, judged against each model's gate PR-AUC. With a drift
summary present each week gets one of four readings: *harmful drift*, *harmless drift*, *silent degradation* or *healthy*.

```bash
python monitoring/performance_report.py --run-id demo1
python monitoring/performance_report.py --run-id drill3 --versions <version>=lightgbm,<version>=logistic
python monitoring/performance_report.py --run-id demo1 --as-of-week 8     # only labels known by week 8
```

`--versions` maps registry version numbers to model families. Check your registry (Models tab) for the numbers; after a rebuild they differ from the defaults.

**Results of `demo1`:**

| Test | Weeks | PR-AUC | 95% range | Reading against clean weeks |
|---|---|---|---|---|
| none (clean) | 1, 2, 3, 4, 7, 10 | 0.234 | 0.213 to 0.254 | yardstick (lift 2.05x; gate said 0.222) |
| covariate shift | 5, 6 | 0.188 | 0.151 to 0.228 | below: recall rose to 0.60 to 0.67, precision fell to 0.13 to 0.15 |
| new categories | 8, 9 | 0.181 | 0.153 to 0.214 | below, although the score distribution barely moved |
| population shift | 12 | 0.214 | 0.163 to 0.275 | consistent: drift without harm |
| bad data | 11 | 0.249 | 0.182 to 0.335 | consistent: nothing bad reached the model |

Lessons: one week (about 240 readmissions) is a noisy judge, so judge by pooled tables; a feature-level drift alarm can be right when score PSI is silent
(new categories); a modestly worse model (about 0.016 PR-AUC) cannot be caught by labels at this volume, so compare it deliberately as a challenger instead.

### Step 9. The retrain decision and flow (plan step 15)

**`retrain.py decide`** applies a rule written before any result was looked at (do not loosen it afterwards):

1. Only weeks scored by the **live champion's version** count. "Clean" weeks are weeks with no injected test (in production, every week).
2. Fewer than `MIN_WEEKS` (4) labelled weeks pooled: **keep** (not enough evidence).
3. Pooled PR-AUC range **not entirely below** the reference (the model's gate PR-AUC): **keep** (drift without a performance drop needs no retrain).
4. Pooled range entirely below the reference: if any labelled week shows never-seen category values, **fix_data_first** (do not retrain on a data problem);
   otherwise **retrain**.

```bash
python retrain.py decide --run-id demo1                       # judge clean weeks
python retrain.py decide --run-id demo1 --scope all           # judge all weeks
python retrain.py decide --run-id demo1 --force-retrain       # a person overrides; recorded as forced, the evidence decision is kept
```

It reads `performance_summary.json` (found by run id), optionally `drift_summary.json`, and `models/champion.json`; it writes
`data/retrain/<id>/trigger.json` and retrains nothing.

**`orchestration/retrain_flow.py`** is a Prefect flow that runs your existing scripts in order and decides nothing itself. Run it with the separate
Prefect environment (`.venv-prefect`), not the project's `.venv`, so Prefect's dependencies never touch the pinned MLflow, pydantic or scikit-learn. Each step
runs in the project environment (`--main-python`, `RETRAIN_PYTHON`, or `.venv`).

| Command | What happens |
|---|---|
| `python orchestration/retrain_flow.py --run-id demo1` | **Dry run:** only `retrain.py decide` |
| `... --force-retrain` | Dry run of a forced retrain: shows the plan |
| `... --force-retrain --apply` | Backs up `models/`, trains, gates, registers a challenger, shows the promotion decision (`promote.py --dry-run`). The champion is not changed |
| `... --force-retrain --apply --allow-promote` | Same, and if the promotion rule says PROMOTE: promotes, exports the bundle, reloads the API |

The challenger is picked by a rule written in advance: among non-baseline models that passed the gate, the highest test PR-AUC. Normal outcomes that end the
flow cleanly: decision keep or fix_data_first, no model passes the gate, registration refused, promote says KEEP CHAMPION. A script crash raises and Prefect
marks the run failed. The flow writes `data/retrain/<id>/retrain_report.md` and (with `--apply`) a `retrain_flow` line in `promotion_log.jsonl`.
A retrained model **never goes straight live**: it enters as a challenger through the gate and the promotion rule, and every promotion can be undone with
`python promote.py --rollback`.

`tools/mlflow_runs.py` lists every MLflow run by group (original, a rejected `v2-...` attempt, retrain flow runs). `--delete-v2 --apply` soft-deletes the
rejected attempt's runs (never runs that back a registered version; copy `mlflow.db` first).

---

## 8. Deployment on Render (plan step 18)

The live service is https://clinical-risk-prediction.onrender.com. Three files work together:

| File | Role |
|---|---|
| `deploy/prepare_bundle.py` | Copies the exported model (`model.joblib`, `meta.json`) from `serving/bundle/` (ignored by git) to `deploy/bundle/` (committed), after checking the checksum so a half-finished export is never deployed. Warns above 50 MB, refuses above 100 MB (GitHub limits) |
| `deploy/Dockerfile` | Builds the image: pinned packages first (layer cache), then `app.py`, `ui/` and `deploy/bundle`. Runs as a non-root user, listens on Render's `$PORT` (default 10000), one worker |
| `render.yaml` | The Blueprint: one `web` service, `runtime: docker`, `plan: free`, `healthCheckPath: /health` |

Environment set by `render.yaml`: `ADMIN_TOKEN` (generated by Render; protects `POST /reload`, see the service's Environment tab), `LOG_PREDICTIONS=false`
(a public service must not keep patient-level rows, and the disk is temporary anyway), `MAX_RECORDS=100` (limits the work one public request can ask for),
`ID_FIELD=encounter_id`.

**First deploy:** Render dashboard > New > Blueprint > pick the repository > Apply. Pushing to GitHub alone does not create a service. After that, pushes to
the connected branch redeploy automatically (if auto-deploy is on). Make sure the service name in `render.yaml` matches the service shown in the dashboard,
so a Blueprint sync does not create a second one.

**Promoting a new champion to the web:**

```bash
python promote.py                          # (or the retrain flow with --allow-promote)
python serving/export_champion.py
python deploy/prepare_bundle.py
git add deploy/bundle
git commit -m "Deploy new champion"
git push                                   # Render rebuilds and redeploys
```

**Checking it:** open `/health` (expect `{"status":"ok",...}`), `/docs`, and `/metrics` (lines starting with `readmission_`). Smoke test:

```bash
python serving/send_sample.py --url https://clinical-risk-prediction.onrender.com --n 20
```

**Things to know about the free plan.** Instances sleep when idle (first request is slow), disks are temporary, all counters reset on every restart or
redeploy, and memory is limited (if the build or service runs out of memory, use the `starter` plan). `/metrics` is public. A hostname that answers with the
header `x-render-routing: no-server` means no service exists at that address (not created yet, wrong URL, or a failed build; check the dashboard's Events tab).
To avoid a rebuild on pushes that only touch monitoring files, set Render's **Build Filters** to `deploy/`, `serving/` and `render.yaml`.

---

## 9. Monitoring (Prometheus and Grafana)

`docker compose up` starts **Prometheus** (port 9090, 15 s scrapes, 30 days of history in the named volume `prometheus_data`), **Grafana** (port 3000,
provisioned Prometheus data source) and the **drift exporter** (port 9100).

Dashboards: Grafana > Dashboards > New > Import > upload `monitoring/grafana/starter_dashboard.json` (API health, flagged share, latency, errors, score heatmap)
and `monitoring/grafana/drift_dashboard.json` (drift verdict, counts, top drifting features, score PSI, worst-first table).

### 9.1 Watching the live Render service

The local Prometheus can also scrape the public service over HTTPS. In `monitoring/prometheus/prometheus.yml` the job is:

```yaml
  - job_name: readmission-render            # the public API on Render
    metrics_path: /metrics
    scheme: https                           # Render only serves HTTPS
    scrape_interval: 30s                    # gentler on a free instance
    scrape_timeout: 20s                     # a sleeping instance can be slow to answer
    static_configs:
      - targets: ["clinical-risk-prediction.onrender.com"]   # hostname only: no https://, no path
        labels: {env: "render"}
```

Apply with `docker compose restart prometheus` (or `docker compose up -d`), then check http://localhost:9090/targets: `readmission-render` should be **UP**.
Generate traffic with `send_sample.py --url ...` and try in Grafana Explore:

```promql
sum(rate(readmission_http_requests_total{job="readmission-render"}[5m])) by (path)
sum(rate(readmission_predictions_total{job="readmission-render"}[5m]))
```

Caveats: the alert rules and starter dashboard do **not** filter by job, so `sum(...)` queries add local and Render traffic together (add
`{job="readmission-render"}` where you want only Render); drift metrics stay local, because the exporter reads the local prediction log and Render has
prediction logging off; counters reset on every Render restart, so always chart `rate()` or `increase()`. Without Docker, a standalone `prometheus.exe`
with a one-job config works too, and the console's own **Monitoring** tab (`/#monitor`) needs nothing.

### 9.2 Alert rules (shown at http://localhost:9090/alerts; no notifications are sent anywhere yet)

| File | Rules |
|---|---|
| `alerts.yml` (real traffic, long windows) | `ApiDown`, `FlaggedShareOutOfRange` (outside 12% to 28% for 15 min), `PredictLatencyHigh` (p95 above 1 s), `ErrorShareHigh` (above 2% non-2xx), `UnseenCategoriesRising` (more than 25 in an hour) |
| `alerts_replay.yml` (3-minute windows, for replay and drills) | `ReplayFlaggedShareOutOfRange`, `ReplayUnseenCategories`, `ReplayErrorShare`, `ReplayModelChanged` |
| `alerts_drift.yml` | `FeatureDriftMajor`, `FeatureDriftModerate`, `ScoreDriftMajor`, `DriftExporterDown` |

After editing a rule file run `docker compose restart prometheus`. Use 90 to 120 s per simulated week so 3-minute windows fire cleanly. After a replay ends,
drift alerts stay on for up to `MAX_AGE_SECONDS` (10 minutes) until the exporter reports no verdict; `UnseenCategoriesRising` clears about an hour later.
`docker compose down -v` deletes the metric history.

---

## 10. Tests and CI

```bash
pip install -r serving/requirements-serve.txt -r requirements-dev.txt
ruff check --select E9,F63,F7,F82 .       # real errors only: syntax errors and undefined names
pytest                                    # config in pytest.ini (testpaths = tests)
```

| Test file | What it covers |
|---|---|
| `tests/test_retrain_decide.py` | The retrain rule: keep / retrain / fix_data_first, minimum weeks, the optimistic-end boundary, scope, reference fallbacks, `--force-retrain` recorded as forced |
| `tests/test_icd9_group.py` | ICD-9 grouping, including codes whose leading zeros are missing in the source |
| `tests/test_drift_lib.py` | PSI: identical data is 0, shifts are major, unseen categories land in `__unseen__`, profiles survive a JSON round trip |
| `tests/test_build_frame.py` | API input validation (missing columns, nulls, text in numeric columns, unseen categories) and reading the schema from a fitted pipeline |
| `tests/conftest.py` | Creates an empty `data/` folder (git ignores it, but the scripts use it to find the project root) and sets import paths |

**CI** (`.github/workflows/ci.yml`, on push to main/master and on pull requests, Python 3.9): job `lint-and-test` installs the serving pins plus the dev tools,
lints and runs the tests; job `docker-build` checks that `serving/Dockerfile` exists with the right case (Linux is case-sensitive) and builds both the API image
and the drift-exporter image. Nothing in CI needs the data or the model.

---

## 11. How everything affects everything else

### 11.1 The chain

```
code/data_*.py        writes data/splits/*.csv, cleaning_config.json, split_report.txt
train.py              reads train.csv; writes models/*.joblib and MLflow runs
test.py               reads test.csv + joblibs; writes evaluation.json; defines the gate thresholds (imported by promote.py)
register_champion.py  reads evaluation.json + MLflow; writes registry aliases and champion.json
promote.py            reads registry + test.py rules + test data; writes aliases, champion.json, logs, reports
serving/export_champion.py  reads registry @champion + test.csv (scores only); writes serving/bundle/ (+ reference_profile.json)
deploy/prepare_bundle.py    copies serving/bundle/model.joblib + meta.json to deploy/bundle/   -> committed -> Render
serving/app.py       reads the bundle; writes predictions.jsonl (locally) and /metrics
serving/replay.py    reads holdout.csv + /model-info; sends weeks to /predict; writes data/replay/<run>/
monitoring/drift_report.py        reads predictions.jsonl (+ plan.json); writes data/drift/<run>/
monitoring/drift_exporter.py      reads predictions.jsonl + reference_profile.json; serves PSI on :9100 -> Prometheus
monitoring/performance_report.py  reads predictions.jsonl + labels.csv + evaluation.json (+ drift summary); writes data/performance/<run>/
retrain.py decide     reads performance_summary.json (+ drift summary) + champion.json; writes data/retrain/<id>/trigger.json
orchestration/retrain_flow.py     runs retrain.py, train.py, test.py, register_champion.py, promote.py, export_champion.py in order
```

### 11.2 Source of truth

| Question | Authoritative answer | Not authoritative |
|---|---|---|
| Which model is champion? | The registry alias `@champion` (MLflow **Models** tab) | `champion.json` is a copy; `best_model.json` is only the top of the CV ranking |
| Which model is the local API serving? | `GET /model-info` or `serving/bundle/meta.json` | The registry may already have moved on until you export and reload |
| Which model is the public site serving? | `deploy/bundle/meta.json` as last deployed (or `/model-info` on the live URL) | The registry and `serving/bundle/` |
| Did a model pass the gate last time? | `evaluation.json` (the last `test.py` run only) | Older reports are overwritten |
| What decisions were made? | `promotion_log.jsonl` and `data/retrain/*/trigger.json` | Reports are overwritten each run |

### 11.3 If you change X, then rerun Y

| If you change... | Then... |
|---|---|
| The data cleaning or the split | Rerun everything from `train.py`; old scores are not comparable, so new models enter as challengers |
| Model settings or features in `train.py` | `train.py` (all three) -> `test.py` -> `register_champion.py --challenger` -> `promote.py` |
| Gate thresholds in `test.py` | They change the gate and `promote.py`; changing them after seeing results defeats their purpose |
| The retrain rule in `retrain.py` | Same warning; rerun `retrain.py decide`; the unit tests encode the rule |
| The random seed or package versions | Scores shift or saved models may not load; update requirements, retrain, re-verify, rebuild images |
| The champion | `export_champion.py` then `POST /reload`; for the public site also `prepare_bundle.py`, commit, push |
| `serving/app.py` or a Dockerfile | `docker compose up -d --build` (local); push (Render rebuilds) |
| `serving/ui/*` | Refresh the browser locally (mounted folder); push for Render (the console is baked into the image) |
| `monitoring/prometheus/*.yml` | `docker compose restart prometheus` |
| Drift window or limits | Edit `drift-exporter` environment in `docker-compose.yml`, then `docker compose up -d` |
| `mlflow.db` or `mlruns/` (deleting) | Registry and history are gone; joblibs survive but `register_champion.py` cannot find runs |

### 11.4 Hidden couplings worth knowing

1. **`promote.py` imports `test.py` by file path** (a file named `test.py` clashes with Python's built-in `test` package). Keep them in the same folder.
2. **`test.py` reads `champion.json`** to choose its reference model.
3. **`register_champion.py` finds a model's training run by matching CV scores** (tolerance 0.0006). If `evaluation.json` is out of date relative to MLflow it stops with a clear message.
4. **The registry holds its own copy of each model** inside `mlruns/`; retraining replaces the joblibs but not what the registry serves.
5. **`train.py` renumbers and replaces the joblib files every run**, so `evaluation.json` is stale until `test.py` runs again.
6. **Everything is seeded (42).** A different result means something changed.
7. **Test-set reuse** slightly overfits it, which is why promotion demands a clear win, not a tie.
8. **The flag cutoff comes from the test split's scores** and is recomputed for whichever model is exported.
9. **Registry version numbers change on rebuild.** In the current registry LightGBM is v6 (`@champion`), XGBoost v5 (`@challenger`) and logistic v7 (`@baseline`); older drill and replay results in this README were produced when the same models were v2, v1 and v3. Use `--versions` and `--champion-version` accordingly.
10. **Folder-name case matters on Linux** (`serving/ui`, `serving/Dockerfile`), in compose, the Dockerfiles and CI.
11. **Two places keep a copy of the model for serving:** `serving/bundle/` (local, mounted, git-ignored) and `deploy/bundle/` (committed, baked into the Render image). Updating one does not update the other.

---

## 12. Recipes

**A. First run from scratch:** [section 3](#3-run-it-from-a-fresh-clone).

**B. Try an improved model:** change a setting in `train.py`; `python train.py`; `python test.py`;
`python register_champion.py --challenger xgboost --reason "what you changed"`; `python promote.py --dry-run`, then `python promote.py` if you agree; if promoted,
export, reload, and update `deploy/bundle` (section 8). A new model always enters as a challenger, so it cannot skip the promotion rule.

**C. Undo a promotion:** `python promote.py --rollback`, then `python serving/export_champion.py`, `POST /reload`, and (for the public site) `python deploy/prepare_bundle.py` + commit + push.

**D. See who the champion is:** read `models/champion_report.md`; or open MLflow > Models > readmission and look at the Aliases column; or run
`python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); c=mlflow.MlflowClient(); print(c.get_model_version_by_alias('readmission','champion').version)"`.

**E. Check what the API is serving:** `/model-info` on the local or the live URL, or the Model tab of the console.

**F. Run the whole monitoring loop once:** replay (3.7) -> `drift_report.py` -> `performance_report.py` -> `retrain.py decide` -> (optionally) the retrain flow in dry-run mode.

---

## 13. Current results

All numbers are on the test split (14,907 stays, no-skill PR-AUC 0.116).

| Model | CV PR-AUC | Test PR-AUC | 95% range | ROC-AUC | Lift | Gate | Registry (current) |
|---|---|---|---|---|---|---|---|
| LightGBM | 0.224 | 0.222 | 0.200 to 0.247 | 0.670 | 1.93x | PASS | v6, `@champion` |
| XGBoost | 0.226 | 0.218 | 0.197 to 0.240 | 0.670 | 1.89x | PASS | v5, `@challenger` |
| Logistic | 0.211 | 0.206 | 0.188 to 0.229 | 0.660 | 1.78x | FAIL (fairness) | v7, `@baseline` |

**How the champion was chosen.** LightGBM and XGBoost are statistically tied (paired difference -0.005, range -0.010 to +0.001). XGBoost ranked first by CV by 0.002,
but its SHAP ranking leaned on tiny rare categories (for example `medical_specialty_Otolaryngology`), while LightGBM relied on sensible signals (prior inpatient
visits, discharge disposition, lab procedures). LightGBM was chosen by a person on that basis, with more room under the fairness limits (worst AUC gap 0.040 against
0.048, limit 0.05). `promote.py` confirms XGBoost does not clearly beat it.

**Logistic regression fails the gate** on fairness (worst ROC-AUC gap 0.0504 against a limit of 0.05; worst recall gap 0.126 against 0.10) but is still registered as the baseline.
**Fairness:** the champion's weakest group is patients aged 80 to 90 (ROC-AUC 0.630 against 0.670 overall). **Calibration:** mean predicted risk is 0.428 against an actual rate
of 0.116 (ECE 0.313) because of class weights; scores are for ranking only.

**Live behaviour.** On clean replay weeks the champion scored a pooled PR-AUC of 0.234 (lift 2.05x), consistent with the gate. See Steps 6 to 8 for the stress tests.

---

## 14. Limits and rules to respect

- **Not a clinical tool.** Not validated for use with real patients. Do not send real patient data to the public demo.
- **Scores are not probabilities.** Class weights inflate them; use them to rank and flag until a calibration step is added.
- **The test split is spent for selection.** Do not tune on it. The holdout is used by the replay: every use is logged, and it is for decisions made in advance, not for tuning.
- **Never loosen a gate threshold or the retrain rule after seeing results.**
- **No dates in the data**, so behaviour over time is simulated (weeks are made by shuffling whole patients).
- **The high-risk cutoff is derived from test-split scores** and assumes live traffic resembles them.
- **Damaged replay weeks are stress tests:** features were changed but outcomes were not, so they measure what altered inputs do, not a real change in the world.
- **One week is a noisy judge** (about 240 readmissions): pool weeks and use ranges.
- **Rare categories** inflate some feature rankings (especially XGBoost and logistic); raising `MIN_CATEGORY_ROWS` is a candidate experiment.
- **The prediction log holds patient-level rows.** Never commit it or share it.
- **Public deployment:** `/metrics` is public; `/reload` is protected by `ADMIN_TOKEN`; free-plan instances sleep, have temporary disks and reset counters on restart.
- **Default Grafana password is admin / admin.** Change it (`GRAFANA_PASSWORD` in an uncommitted `.env`) before exposing the port beyond your machine.

---

## 15. Status and what is not built yet

**Built:** data preparation, training, gate, registry, promotion and rollback, API and web console, local Docker stack with Prometheus and Grafana, replay and rollback drill,
drift detection (report and live alerts), real performance from late labels, the retrain rule and Prefect flow, tests and CI, and a public deployment on Render that Prometheus can scrape.

**Not built yet (possible next steps):**

1. **Alert notifications.** Prometheus alerts only show on the `/alerts` page; routing them to email or chat (Alertmanager) is not set up. Alert rules and dashboards are not yet split per job (local versus Render).
2. **Calibration and model improvement.** Make scores usable as probabilities; test how the model treats never-seen category values; run the experiments of section 14 as challengers.
3. **Monitoring of the public service end to end.** Drift and late-label performance only cover the local stack, because the public service does not log predictions. Shipping a sanitised prediction log to proper storage would extend them.
4. **Protecting the public API** further (API key or rate limit on `/predict`, `/metrics` behind a token).
5. **A scheduled retrain** (for example a Prefect deployment on a timer) and a real label feed instead of the replay's saved `labels.csv`.
6. **More unit tests** (for example `test.py`'s gate helpers and `promote.py`'s rule).

---

## 16. Troubleshooting

Errors that have actually come up in this project, plus the ones to expect.

| Message or symptom | Meaning and fix |
|---|---|
| `Pipeline should either be a classifier ... Got a regressor with response_method=predict_proba` | scikit-learn is newer than this XGBoost supports. Use scikit-learn 1.5.2 with xgboost 2.0.3 |
| `UNIQUE constraint failed: metrics...` (MLflow) | A model produced NaN scores and MLflow could not store them; read the first error above it |
| `OSError: [Errno 28] No space left on device` | Disk is full. Free about 5 GB; install with `--no-cache-dir` |
| `Failed building wheel for greenlet` / `Microsoft Visual C++ 14.0 or greater is required` | Pip is compiling. Install with `--prefer-binary` |
| `No matching distribution found for pywin32` (macOS/Linux) | You used `requirements-lock.txt` (a Windows snapshot). Use `requirements.txt` |
| `the following arguments are required: --champion, --reason` | Run `register_champion.py` from the terminal and name the model and reason |
| `REFUSED: ... did not pass the evaluation gate` | The model failed the gate in `evaluation.json`. Do not register it |
| `REFUSED: this exact training run is already the champion` | No new model to register. Retrain first |
| `REFUSED: promote.py has changed the champion ...` | Re-running champion mode would reset labels. Use `--force` only if you mean it |
| `No finished training run for '<model>' matches CV PR-AUC ...` | `evaluation.json` and MLflow are out of step. Run `train.py`, then `test.py`, then retry |
| `VERIFY FAILED: ... differs from ...joblib` | Registry copy and saved file do not match. Retrain and re-register |
| `Refusing to train on test.csv` | `--train-file` must never point at test or holdout data |
| `Missing .../diabetic_data_cleaned.csv` / `.../train.csv` | Run `code/data_cleaning.py`, then `code/data_split.py` |
| `docker version` shows only a Client section; `docker compose ps` prints nothing and commands hang | The Docker engine is not running. Open Docker Desktop and wait for **Engine running**; if it is stuck: quit it, run `wsl --shutdown`, reopen. Then `docker compose up -d` (use `up`, not `restart`, for stopped containers) |
| Docker Desktop will not open | End leftover `Docker Desktop.exe`, `com.docker.backend.exe`, `vmmem` in Task Manager, run `wsl --shutdown`, reopen (try as administrator). Check virtualization is enabled (Task Manager > Performance > CPU) and the WSL / Virtual Machine Platform features; as a last resort repair or reinstall (volumes, so metric history, are lost) |
| `port is already allocated` (8000, 9090, 3000, 9100) | Another program uses that port; stop it or change the left side of the port mapping |
| `failed to read dockerfile` on Linux | The file must be `serving/Dockerfile` (capital D); Linux is case-sensitive |
| Console unstyled or "Loading the form..." forever; `/ui/...` is 404 | `serving/ui/` is missing or named `UI`. Must be lowercase with `index.html`, `styles.css`, `app.js`. Hard refresh with Ctrl+F5 |
| `/health` returns 503 or "No model is loaded" | No bundle. Run `python serving/export_champion.py`, then Reload (on Render: the image is missing `deploy/bundle`) |
| `/predict` returns 422 | The message names the column and the problem. Fix the request |
| `/reload` returns 500 with `still_serving_version` | New bundle incomplete or corrupt; the old model keeps serving. Re-run the export |
| `/reload` returns 403 | `ADMIN_TOKEN` is set. Send header `X-Admin-Token` |
| `package_warnings` in `/model-info` | Container packages differ from the export's. Rebuild the image |
| Render: `x-render-routing: no-server` | No service at that hostname: not created (New > Blueprint > Apply), wrong URL, or failed build. Check the dashboard's service list and Events tab |
| Render: first request very slow | A free instance was asleep; wait 30 to 60 seconds |
| Render did not update after a push | Auto-deploy off, or `deploy/bundle` not committed (`git ls-files deploy` should list `deploy/Dockerfile`, `deploy/prepare_bundle.py`, `deploy/bundle/model.joblib`, `deploy/bundle/meta.json`) |
| Render build or service runs out of memory | Switch `plan: free` to `starter` in `render.yaml` |
| Prometheus target `readmission-render` DOWN | Check the error text on `/targets`; confirm `https://<host>/metrics` opens in a browser; use hostname only (no `https://`, no path); raise `scrape_timeout` for a sleeping instance |
| `docker compose restart prometheus` finishes but Prometheus will not start | Check `docker compose logs --tail 30 prometheus` for `parsing YAML file`; fix the indentation (spaces, not tabs) |
| `replay.py`: `Cannot reach the API` / `The API is not healthy` | Start the stack and export a bundle, then reload |
| `replay.py`: `... already exists. Choose another --run-id` | Run ids name output folders and are never overwritten; pick a new one or resume with `--start-week N` |
| `replay.py` stops with `HTTP 422` on a clean week | A real data problem, not an injected one; the message names the column |
| Replay alerts never fire | Weeks too short for the 3-minute windows (use `--seconds-per-week 120`), or Prometheus not restarted after editing the rules |
| `rollback_drill.py --finish`: no `@previous_champion` | The drill was not started or already finished. Check the Models tab |
| `performance_report.py`: `Missing .../labels.csv` | The file is written by `serving/replay.py`; check the run id |
| `performance_report.py`: every week says `no reference` | `evaluation.json` has no entry for that family; check `--versions` (registry version = family) and that `test.py` ran |
| `retrain.py`: `No performance_summary.json for run ...` | Run `python monitoring/performance_report.py --run-id <run>` first |
| `retrain.py` decides keep with "No labelled stays were scored by version N" | The run was scored by another registry version; pass `--champion-version` to match the run |
| Retrain flow: `Project Python not found` | Pass `--main-python` or set `RETRAIN_PYTHON` to the project environment's python |
| Retrain flow stops on a Prefect import error | You are in the project `.venv`; activate `.venv-prefect` |
| Drift alerts stay on after the replay ended | Expected for up to `MAX_AGE_SECONDS` (10 minutes); `UnseenCategoriesRising` uses a one-hour window |
| `readmission_drift_ready` is 0 and no `readmission_feature_psi` series | No verdict; the exporter log says why (too few stays, no recent traffic, no profile). Try `python monitoring/drift_exporter.py --once` |
| Several drift alerts fire together | Usually one cause; the feature with the largest PSI (often `age`) is the root |
| Score PSI high but no feature drifted, or missing after a rollback | The model version changed; a different model gives different scores, and the exporter waits for enough stays of the new version |
| `drift_report.py`: `No logged stays with run_id ...` | Wrong run id, log deleted, or the replay ran against another API |
| Grafana panels say "No data" | Check the data source is Prometheus, send traffic (`send_sample.py --n 200`) and widen the time range |
| CI fails on `serving/Dockerfile is missing or has the wrong case` | Rename with two `git mv` steps so Git records the case change (section 18) |

---

## 17. File-by-file guide

Each script starts with a docstring describing what it reads, writes and why; read that first. A short map:

### Data preparation (`code/`)

| File | What it does |
|---|---|
| `data_cleaning.py` | Raw UCI file to a clean, readable dataset; records every setting in `cleaning_config.json`. Never touches the raw file, does not encode categories |
| `leak_review.py` | Hand-written verdict table (EXCLUDE / KEEP / KEEP-WATCH / TARGET) plus statistical checks; stops if a column has no verdict |
| `data_split.py` | `StratifiedGroupKFold` split by patient into train 60% / test 15% / holdout 25%; asserts no patient overlap |
| `data_cleaning.md`, `leak_review.md` | Human explanations and results of those steps |

### Pipeline scripts (project root)

| File | What it does |
|---|---|
| `train.py` | One `Pipeline` per model, grouped cross-validation, MLflow logging, SHAP or coefficients, ranked joblibs, reports; records the data version |
| `test.py` | The gate: PR-AUC, ROC-AUC, lift, bootstrap ranges, calibration, fairness, pass/fail, `evaluation.json` |
| `register_champion.py` | Registers models with aliases and evidence; verifies against the joblib; challenger mode |
| `promote.py` | The promotion rule, aliases swap, rollback, audit log |
| `retrain.py` | The retrain decision rule (`decide`), writes `trigger.json` |
| `requirements.txt`, `requirements-lock.txt`, `requirements-dev.txt` | Pipeline pins; full Windows snapshot; test and lint tools |
| `docker-compose.yml` | Local stack: API, drift exporter, Prometheus, Grafana, two named volumes |
| `render.yaml` | Render Blueprint for the public API |
| `pytest.ini`, `.gitignore`, `.github/workflows/ci.yml` | Test config, ignore rules, CI |

### Serving, deployment and orchestration

| File | What it does |
|---|---|
| `serving/app.py` | FastAPI app: bundle loading and checksum, validation, scoring, SHAP, flag, prediction log, Prometheus metrics, hot reload, console |
| `serving/export_champion.py` | Registry `@champion` to `serving/bundle/` (model, metadata last as the commit marker, drift reference profile) |
| `serving/send_sample.py` | Sends CSV rows to a running API (local or the Render URL) |
| `serving/replay.py`, `serving/rollback_drill.py` | Holdout replay with scenarios; deliberate bad-model swap and real rollback |
| `serving/Dockerfile`, `requirements-serve.txt`, `.dockerignore` | The local API image (model mounted) and its pinned packages |
| `serving/ui/` | The web console (`index.html`, `styles.css`, `app.js`); light and dark mode, no external libraries |
| `deploy/Dockerfile`, `deploy/prepare_bundle.py`, `deploy/bundle/` | The public image (model baked in) and the committed model copy |
| `orchestration/retrain_flow.py` | Prefect flow: decide, back up, train, gate, register, promote (dry-run by default) |
| `tools/mlflow_runs.py` | List MLflow runs by group; soft-delete a rejected attempt |

### Monitoring (`monitoring/`)

| File | What it does |
|---|---|
| `drift_lib.py` | Shared PSI code: profiles, shares, PSI, levels, JSON profiles (pandas and numpy only) |
| `drift_report.py`, `drift_exporter.py` | Per-week drift report; live PSI metrics for Prometheus |
| `performance_report.py` | Late labels joined to predictions: real performance by week, version and test |
| `Dockerfile.drift`, `requirements-drift.txt`, `.dockerignore` | The drift exporter's image |
| `prometheus/*.yml`, `grafana/...` | Scrape config (local API, Render, drift exporter, itself), alert rules, data source, dashboards |

### Generated files (not in git)

| File | Written by | Read by | Safe to delete? |
|---|---|---|---|
| `data/splits/*.csv`, `cleaning_config.json` | `code/` scripts | train, test, export | Regenerate with the `code/` scripts |
| `models/*.joblib`, `best_model.json`, `model_report.md` | `train.py` | `test.py`, `register_champion.py` | Yes, rerun training |
| `models/evaluation.json`, `test_report.md` | `test.py` | `register_champion.py` | Yes, rerun `test.py` |
| `models/champion.json`, `champion_report.md`, `promotion_report.md` | register / promote | `test.py`, people | Rewritten on next change |
| `models/promotion_log.jsonl`, `models/data_use_log.jsonl` | promote, register, test, export, replay, reports, retrain flow | people | Avoid: they are your audit trail (and contain local paths) |
| `serving/bundle/*` | `export_champion.py` | the local API, drift exporter | Yes, re-export |
| `deploy/bundle/*` | `prepare_bundle.py` | the Render image | Committed: regenerate with `prepare_bundle.py` |
| `serving/logs/predictions.jsonl` | the local API | drift and performance tools | Avoid: it is the live-traffic record; never commit |
| `data/replay/<run>/*` | `replay.py` | people, performance report | Keep `labels.csv` of runs you still want to analyse |
| `data/drift/<run>/*`, `data/performance/<run>/*`, `data/retrain/<id>/*` | the matching report or flow | people, `retrain.py` | Yes, regenerate |
| `mlflow.db`, `mlruns/` | MLflow (all scripts) | all scripts | **No** |
| `.venv/`, `.venv-prefect/` | `python -m venv` | Python | Yes, rebuild |

---

## 18. Publishing to GitHub

**What is committed:** code, configs, documentation, the two raw UCI files, `deploy/bundle/` (the deployed model, so Render can build) and four plain-language
reports (`models/model_report.md`, `test_report.md`, `champion_report.md`, `promotion_report.md`). **What is not:** the data splits and cleaned data, the other
`models/` files, `mlflow.db`, `mlruns/`, the serving bundle and logs, environments and caches. The `.jsonl` logs hold local file paths and `mlflow.db` stores absolute
paths from your machine, so neither would work for anyone else; the prediction log holds patient-level rows. A clone rebuilds everything in minutes (section 3).

The ignore rules live in `.gitignore`; check them against section 4 if you add new output folders.

**Checklist before you push:**

1. **Folder and file name case.** Docker on Linux (and CI) is case-sensitive; the code uses `serving/ui` and `serving/Dockerfile`. On a case-insensitive Windows disk Git can
   keep the wrong case (`serving/UI`, `serving/dockerfile`). Rename in two steps so Git records it:
   ```bash
   git mv serving/UI serving/ui_tmp && git mv serving/ui_tmp serving/ui
   git mv serving/dockerfile serving/Dockerfile_tmp && git mv serving/Dockerfile_tmp serving/Dockerfile
   ```
   (The same applies to `monitoring/.dockerignore` and `serving/.dockerignore` if they were saved without the leading dot.)
2. **Remove stale copies:** `serving/docker-compose.yml` (the real one is at the root), `code/readme.md` (an old copy of this README) and the empty `practice.py`.
3. **Check `deploy/` is tracked:** `git ls-files render.yaml deploy` should list `render.yaml`, `deploy/Dockerfile`, `deploy/prepare_bundle.py`, `deploy/bundle/model.joblib`, `deploy/bundle/meta.json`.
4. **Add a `LICENSE`** for your code and keep the UCI citation (CC BY 4.0).
5. **Replace the placeholder** repository URL in section 3.2.
6. If you already pushed files that should be ignored, stop tracking them:
   ```bash
   git rm -r --cached data models mlruns mlflow.db serving/bundle serving/logs
   git commit -m "Stop tracking data, models and local artifacts"
   ```
   They remain in earlier commits. If any contained private paths or patient-level rows, remove them from history (for example `git filter-repo`) or recreate the repository.
7. **Run the checks locally** (`ruff check --select E9,F63,F7,F82 .` and `pytest`) so CI is green on the first push.
