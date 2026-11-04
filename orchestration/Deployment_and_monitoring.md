# Deployment and monitoring

How the service is packaged, deployed (locally with Docker and publicly on Render), monitored (Prometheus, Grafana, the drift exporter and the console), and
operated when something goes wrong.

Related documents: [README](../README.md) (how to run it), [ARCHITECTURE](ARCHITECTURE.md) (the MLOps life cycle and design decisions),
[CODE_WALKTHROUGH](CODE_WALKTHROUGH.md) (every script and function).

Live service: https://clinical-risk-prediction.onrender.com (console `/`, batch scoring `/#batch`, docs `/docs`, health `/health`, metrics `/metrics`).

> Not a clinical tool. Do not send real patient data to the public demo.

---

## Contents

1. [Quick reference](#1-quick-reference)
2. [The two deployments](#2-the-two-deployments)
3. [Local Docker stack](#3-local-docker-stack)
4. [Public deployment on Render](#4-public-deployment-on-render)
5. [Promoting a model to production](#5-promoting-a-model-to-production)
6. [Environment variables](#6-environment-variables)
7. [Prometheus](#7-prometheus)
8. [Metrics catalogue](#8-metrics-catalogue)
9. [Alert rules](#9-alert-rules)
10. [Grafana](#10-grafana)
11. [PromQL cookbook](#11-promql-cookbook)
12. [The drift exporter](#12-the-drift-exporter)
13. [The console's Monitoring tab](#13-the-consoles-monitoring-tab)
14. [Runbooks](#14-runbooks)
15. [Operating checklists](#15-operating-checklists)
16. [Hardening and limits](#16-hardening-and-limits)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. Quick reference

| Thing | Local | Public |
|---|---|---|
| Console | http://localhost:8000/ | https://clinical-risk-prediction.onrender.com/ |
| API docs | http://localhost:8000/docs | `/docs` on the live URL |
| Health | http://localhost:8000/health | `/health` on the live URL |
| Metrics | http://localhost:8000/metrics | `/metrics` on the live URL (public) |
| Prometheus | http://localhost:9090 (Targets, Alerts, Graph) | not hosted |
| Grafana | http://localhost:3000 (admin / admin: change it) | not hosted |
| Drift exporter | http://localhost:9100/metrics | not deployed |

Everyday commands:

```bash
docker compose up -d --build        # start or update the whole local stack
docker compose ps                   # is everything running and healthy?
docker compose logs -f readmission-api
docker compose restart prometheus   # after editing prometheus.yml or an alert file
docker compose down                 # stop (keeps metric history)
docker compose down -v              # stop and DELETE metric history and Grafana settings
python serving/send_sample.py --url https://clinical-risk-prediction.onrender.com --n 200   # make some public traffic
```

---

## 2. The two deployments

| | Local stack | Public (Render) |
|---|---|---|
| Purpose | Development, replay, drift, monitoring, drills | Let anyone use the model |
| Image | `serving/Dockerfile` | `deploy/Dockerfile` |
| How the model gets in | **Mounted folder** `serving/bundle/` (read-only); change it with an export and `POST /reload` | **Baked into the image** from `deploy/bundle/`; change it by committing and pushing |
| Prediction log | Written to `serving/logs/predictions.jsonl` (needed by drift and late-label tools) | Off (`LOG_PREDICTIONS=false`) |
| Batch limit | `MAX_RECORDS=500` | `MAX_RECORDS=100` |
| Monitoring | Prometheus, Grafana, drift exporter | Scraped by the local Prometheus over HTTPS |
| Restarts | `restart: unless-stopped` | Render manages; free instances sleep when idle |

The two model copies are independent: updating `serving/bundle/` does **not** update the public site, and `POST /reload` on Render re-reads the bundle that is already inside the image, so it
changes nothing. To change what the public site serves you must redeploy (section 5).

---

## 3. Local Docker stack

`docker-compose.yml` defines four services on one private network. Containers reach each other by service name (`readmission-api:8000`, `prometheus:9090`, `drift-exporter:9100`).

| Service | Image | Port | Mounts | Notes |
|---|---|---|---|---|
| `readmission-api` | built from `./serving` (`readmission-api:local`) | 8000 | `./serving/bundle:/app/bundle:ro`, `./serving/logs:/app/logs`, `./serving/ui:/app/ui:ro` | Edits to `serving/ui/*` show after a browser refresh; changes to `app.py` need `--build` |
| `drift-exporter` | built from `./monitoring` with `Dockerfile.drift` (`readmission-drift:local`) | 9100 | `./serving/logs:/data/logs:ro`, `./serving/bundle:/data/bundle:ro` | Starts after the API |
| `prometheus` | `prom/prometheus:v2.55.1` | 9090 | `./monitoring/prometheus:/etc/prometheus:ro`, volume `prometheus_data` | 30-day retention, `--storage.tsdb.retention.time=30d` |
| `grafana` | `grafana/grafana:11.3.0` | 3000 | volume `grafana_data`, `./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro` | Prometheus data source (uid `prometheus`) is provisioned; dashboards are imported by hand |

Named volumes `prometheus_data` and `grafana_data` survive container restarts and rebuilds; `docker compose down -v` deletes them.

### 3.1 The API images

Both Dockerfiles share the same ideas:

- `python:3.9-slim`, `libgomp1` installed (LightGBM needs the OpenMP runtime).
- `requirements-serve.txt` is copied and installed **before** the code, so Docker's layer cache skips the slow install when only code changes.
- Runs as a non-root user (`appuser`, uid 1000); `/app/logs` is created and owned by it.
- **One worker** (`--workers 1`): the Prometheus counters live inside the process, and several workers would each report their own partial counts.
- The local Dockerfile has a `HEALTHCHECK` on `/health` (every 30 s). Docker marks the container "unhealthy" but does not restart it by itself.

`serving/.dockerignore` keeps `bundle/`, `logs/`, caches and the host-side scripts (`export_champion.py`, `send_sample.py`, `replay.py`, `rollback_drill.py`) out of the image.

### 3.2 Starting it from scratch

```bash
python serving/export_champion.py      # needs the registry; writes serving/bundle/ (model, meta, reference profile)
docker compose up -d --build
docker compose ps                      # all four services Up; the API healthy
python serving/send_sample.py --n 20   # see scores come back
```

Then check http://localhost:9090/targets (every job UP), http://localhost:3000 (log in, import the dashboards, section 10).

### 3.3 When Docker itself is the problem

Symptom: `docker version` prints only a **Client** section and every `docker` command hangs. The engine is not running. Restart Docker Desktop and wait for **Engine running**; if it is
stuck, quit it, run `wsl --shutdown`, reopen it, then `docker compose up -d` (use `up`, not `restart`, for containers that are not running). Details are in section 17. The Render service is independent of all of this.

---

## 4. Public deployment on Render

Three files work together.

| File | Role |
|---|---|
| `deploy/prepare_bundle.py` | Copies `model.joblib` and `meta.json` from `serving/bundle/` (ignored by git) to `deploy/bundle/` (committed), after verifying the model against `meta.json`'s checksum. Warns above 50 MB, refuses above 100 MB (GitHub limits) |
| `deploy/Dockerfile` | Builds the image: pinned packages, then `app.py`, `ui/` and `deploy/bundle`. Non-root user; `CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1`; Render supplies `$PORT` |
| `render.yaml` | The Blueprint: one `web` service, `runtime: docker`, `dockerfilePath: ./deploy/Dockerfile`, `dockerContext: .`, `plan: free`, `healthCheckPath: /health`, and the environment below |

### 4.1 Creating the service (once)

1. Push the repository to GitHub with `render.yaml`, `deploy/Dockerfile`, `deploy/prepare_bundle.py`, `deploy/bundle/model.joblib`, `deploy/bundle/meta.json` and `serving/` committed
   (`git ls-files render.yaml deploy` should list the first five).
2. Render dashboard > **New > Blueprint** > pick the repository > **Apply**. Pushing alone does not create a service.
3. Wait for the first build (a few minutes). The service's Events tab should end in **Live**.
4. Confirm: open `/health` (expect `{"status":"ok",...}`), `/docs`, and the console.

The service name in `render.yaml` should match the service shown in the dashboard, otherwise a Blueprint sync can create a second service.

### 4.2 What `render.yaml` sets

| Variable | Value | Why |
|---|---|---|
| `ADMIN_TOKEN` | generated by Render (see the service's Environment tab) | Protects `POST /reload` |
| `LOG_PREDICTIONS` | `"false"` | A public service must not keep patient-level rows; the disk is temporary anyway |
| `MAX_RECORDS` | `"100"` | Limits the work one public request can ask for (batch scoring in the console chunks at the API's limit) |
| `ID_FIELD` | `encounter_id` | Column returned as `id` |

### 4.3 Behaviour of the free plan

- Instances **sleep when idle**; the first request afterwards can take 30 to 60 seconds. A health-check path does not prevent sleeping.
- The disk is temporary and every restart or redeploy **resets all counters** (`readmission_*_total`). Always chart `rate()` or `increase()`.
- If the build or the service runs out of memory, change `plan: free` to `starter`.
- Scraping from a local Prometheus every 30 seconds keeps the instance awake while Prometheus runs, which uses your free instance hours. Check Render's current limits.
- To avoid rebuilding on pushes that only touch monitoring files, set **Settings > Build Filters** to `deploy/`, `serving/` and `render.yaml`.

### 4.4 Reading Render's responses

| You see | Meaning |
|---|---|
| Console loads | The service is live; "Checking..." and "Loading the form..." are shown briefly before the page script runs |
| `x-render-routing: no-server` | Render has no running service at that hostname: never created, wrong URL, or a failed build |
| HTTP 503 on `/health` | The container runs but no model loaded: `deploy/bundle/` missing or the checksum failed |
| Very slow first request | The free instance was asleep |

---

## 5. Promoting a model to production

A promotion happens in the registry; serving changes only after the steps below. Run them in order.

```bash
python promote.py --dry-run          # see the decision
python promote.py                    # apply if it wins (or use the retrain flow with --allow-promote)

# local stack
python serving/export_champion.py    # new bundle, new cutoff, new drift reference profile
curl -X POST http://localhost:8000/reload      # or the Reload button on the console's Model tab
#   (if ADMIN_TOKEN is set locally: add -H "X-Admin-Token: <token>")

# public site
python deploy/prepare_bundle.py      # copy the verified bundle into deploy/bundle/
git add deploy/bundle
git commit -m "Deploy new champion"
git push                             # Render rebuilds and redeploys
```

**After the reload or redeploy, verify:**

1. `/model-info` (local and public) shows the new `version` and `model_family`.
2. Grafana's "Live model version" panel (`readmission_model_info`) changed.
3. The `ReplayModelChanged` alert fired (informational) and `readmission_model_loaded_timestamp_seconds` moved.
4. Flagged share is still near 20%. **Caution:** it will be near 20% even for a worse model, because the export recomputes the cutoff; only labelled performance shows quality.
5. The drift exporter logs `Loaded reference profile: model version N`.

**Rollback:**

```bash
python promote.py --rollback
python serving/export_champion.py && curl -X POST http://localhost:8000/reload
python deploy/prepare_bundle.py && git add deploy/bundle && git commit -m "Roll back" && git push
```

A corrupt or half-written bundle is refused at load (checksum mismatch), and the previous model keeps serving.

---

## 6. Environment variables

### 6.1 The API (`serving/app.py`)

| Variable | Default | Meaning |
|---|---|---|
| `BUNDLE_DIR` | `serving/bundle` next to the app (`/app/bundle` in images) | Folder with `model.joblib` and `meta.json` |
| `MAX_RECORDS` | 500 (100 on Render) | Most records per `/predict` request |
| `ID_FIELD` | `encounter_id` | Column returned as `id` and logged |
| `LOG_PREDICTIONS` | `true` (`false` on Render) | Append every prediction to the JSONL log |
| `PREDICTION_LOG` | `serving/logs/predictions.jsonl` (`/app/logs/...` in images) | Where the log is written |
| `ADMIN_TOKEN` | empty | If set, `POST /reload` needs header `X-Admin-Token` with this value |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `PORT` | 10000 (Render) | Read by the Render image's `CMD`, not by `app.py` |

### 6.2 The drift exporter (`docker-compose.yml`)

| Variable | Default | Meaning |
|---|---|---|
| `LOG_PATH` | `/data/logs/predictions.jsonl` | The API's prediction log (mounted read-only) |
| `PROFILE_PATH` | `/data/bundle/reference_profile.json` | Reference profile (reloaded when the file changes) |
| `WINDOW_ROWS` | 1000 | Most recent stays to compare |
| `MIN_ROWS` | 500 | Fewer stays: no verdict |
| `MAX_AGE_SECONDS` | 600 | Newest stay older than this: no verdict |
| `INTERVAL` | 15 | Seconds between evaluations |
| `PORT` | 9100 | Metrics port |
| `RUN_ID` | empty | Only use stays of one replay run (empty = all traffic) |

### 6.3 Other

| Variable | Used by | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | all MLflow scripts | Default `sqlite:///<root>/mlflow.db` |
| `GRAFANA_PASSWORD` | `docker-compose.yml` | Grafana admin password (default `admin`); put it in an uncommitted `.env` |
| `ADMIN_TOKEN` | `rollback_drill.py`, `retrain_flow.py` | Sent as `X-Admin-Token` when they call `/reload` |
| `RETRAIN_PYTHON` | `retrain_flow.py` | Python of the project environment |

---

## 7. Prometheus

Prometheus visits each target's `/metrics` page on a timer and stores every number with a timestamp. `monitoring/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s        # keep at 15 s: replay weeks must last at least 20 s to show up
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/alerts.yml
  - /etc/prometheus/alerts_replay.yml
  - /etc/prometheus/alerts_drift.yml

scrape_configs:
  - job_name: readmission-api            # the local container
    metrics_path: /metrics
    static_configs:
      - targets: ["readmission-api:8000"]

  - job_name: readmission-render         # the public API on Render
    metrics_path: /metrics
    scheme: https                        # Render only serves HTTPS
    scrape_interval: 30s                 # gentler on a free instance
    scrape_timeout: 20s                  # a sleeping instance can be slow to answer
    static_configs:
      - targets: ["clinical-risk-prediction.onrender.com"]   # hostname only: no https://, no path
        labels: {env: "render"}

  - job_name: drift-exporter
    metrics_path: /metrics
    static_configs:
      - targets: ["drift-exporter:9100"]

  - job_name: prometheus
    static_configs:
      - targets: ["localhost:9090"]
```

| Job | Target | What it gives you |
|---|---|---|
| `readmission-api` | local container | Service metrics for the local stack and replay |
| `readmission-render` | public service | Service metrics for the live site (label `env="render"`) |
| `drift-exporter` | drift exporter container | PSI of every feature and the score |
| `prometheus` | itself | Proof Prometheus is alive |

After editing any file under `monitoring/prometheus/`: `docker compose restart prometheus`, then http://localhost:9090/targets (every job **UP**) and `/alerts`.
If Prometheus fails to start after an edit, `docker compose logs --tail 30 prometheus` shows `parsing YAML file ...`: fix the indentation (spaces, never tabs).

Without Docker, a standalone `prometheus.exe` with a one-job config pointed at the Render hostname works the same way.

---

## 8. Metrics catalogue

### 8.1 From the API (`/metrics`, defined in `serving/app.py`)

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `readmission_http_requests_total` | counter | `method`, `path` (route template, `/ui/*`, `unmatched`), `status` | Requests (excluding `/metrics`) |
| `readmission_http_request_duration_seconds` | histogram | `method`, `path` | Latency; buckets 5 ms to 10 s |
| `readmission_predictions_total` | counter | `outcome` (`high_risk`, `low_risk`) | Records scored |
| `readmission_prediction_score` | histogram | none | Score distribution; 20 buckets of 0.05 |
| `readmission_batch_size` | histogram | none | Records per `/predict` request |
| `readmission_validation_errors_total` | counter | `reason` (`missing_columns`, `nulls`, `not_numeric`, `too_many_records`, `bad_replay_header`) | Rejected requests |
| `readmission_unseen_category_total` | counter | `column` | Values never seen in training |
| `readmission_model_info` | gauge | `name`, `version`, `family`, `alias` | Live model (value always 1) |
| `readmission_model_loaded_timestamp_seconds` | gauge | none | When the live model was loaded |
| `readmission_model_reloads_total` | counter | `outcome` (`ok`, `failed`) | Load attempts |
| `readmission_replay_week` | gauge | none | Simulated week of the latest replay request (0 unless `X-Replay-Week` is sent) |

`prometheus_client` also exposes the standard `python_gc_*`, `python_info` and `process_*` metrics (memory, CPU, open file descriptors).

### 8.2 From the drift exporter (`:9100/metrics`)

| Metric | Labels | Meaning |
|---|---|---|
| `readmission_feature_psi` | `feature` | PSI of one input feature over the window (46 series); removed when there is no verdict |
| `readmission_score_psi` | none | PSI of the scores, only from stays scored by the profile's model version; NaN when unavailable |
| `readmission_drift_window_rows` | none | Stays in the window |
| `readmission_drift_features` | `level` (`stable`, `moderate`, `major`) | Features per level |
| `readmission_drift_ready` | none | 1 if a verdict was computed this cycle |
| `readmission_drift_age_seconds` | none | Age of the newest logged stay |
| `readmission_drift_profile_info` | `model_version`, `reference_rows` | Which profile is loaded |
| `readmission_drift_last_update_timestamp_seconds` | none | When it last evaluated |

### 8.3 Reading counters correctly

Counters only go up until the process restarts, then drop to zero. On Render this happens on every redeploy and after sleep/wake cycles that restart the process. **Never graph raw totals;
use `rate()` (per second) or `increase()` (over a window).**

---

## 9. Alert rules

Alerts appear at http://localhost:9090/alerts. **No notification is sent anywhere**; Prometheus only records that the rule is firing. States: inactive, pending (true but for less than `for:`), firing.

### 9.1 `alerts.yml` (real traffic, long windows)

| Alert | Fires when | For |
|---|---|---|
| `ApiDown` | `up{job="readmission-api"} == 0` | 1 m |
| `FlaggedShareOutOfRange` | flagged share over 15 m is above 0.28 or below 0.12 (quiet when nothing is scored) | 15 m |
| `PredictLatencyHigh` | p95 of `/predict` over 5 m above 1 s | 5 m |
| `ErrorShareHigh` | non-2xx share over 5 m above 2% | 5 m |
| `UnseenCategoriesRising` | `increase(readmission_unseen_category_total[1h])` above 25 | 10 m |

### 9.2 `alerts_replay.yml` (3-minute windows, for the replay and drills)

| Alert | Fires when |
|---|---|
| `ReplayFlaggedShareOutOfRange` | flagged share over 3 m outside 12% to 28%, with more than 200 stays in the window (for 1 m) |
| `ReplayUnseenCategories` | more than 25 unseen values in 3 m (for 30 s) |
| `ReplayErrorShare` | non-2xx share above 5% over 3 m, with more than 20 requests in the window (for 30 s) |
| `ReplayModelChanged` | `changes(readmission_model_loaded_timestamp_seconds[10m]) > 0` (informational) |

Use 90 to 120 seconds per simulated week so 3-minute windows fire cleanly.

### 9.3 `alerts_drift.yml`

| Alert | Fires when | For |
|---|---|---|
| `FeatureDriftMajor` | `readmission_feature_psi > 0.25` (one alert per feature) | 1 m |
| `FeatureDriftModerate` | PSI between 0.10 and 0.25 | 1 m |
| `ScoreDriftMajor` | `readmission_score_psi > 0.25` | 1 m |
| `DriftExporterDown` | `up{job="drift-exporter"} == 0` | 2 m |

### 9.4 What the rules do and do not cover

- Only `ApiDown` filters by job. The other rules use `sum(...)` over all series, so with the Render job added they **add local and Render traffic together**. Add `{job="readmission-api"}` or
  `{job="readmission-render"}` to the metric names to limit a rule to one service.
- Nothing watches whether the **Render** target is up. A rule for it:
  ```yaml
  - alert: RenderApiDown
    expr: up{job="readmission-render"} == 0
    for: 5m
    labels: {severity: warning}
    annotations:
      summary: "The public API cannot be scraped (asleep, crashed or redeploying)"
  ```
  Use a longer `for:` than the local rule: a free instance that is waking up can miss a scrape or two.
- Drift alerts stay on for up to `MAX_AGE_SECONDS` (10 minutes) after traffic stops, because the exporter keeps judging the last window until it goes stale; then it reports no verdict and they clear.
  `UnseenCategoriesRising` uses a one-hour window and clears about an hour after the last unseen values.
- Drift alerts cover the local stack only (the public service keeps no prediction log).

---

## 10. Grafana

Log in at http://localhost:3000 (admin / admin; change it, or set `GRAFANA_PASSWORD` in an uncommitted `.env`). The Prometheus data source (uid `prometheus`, `http://prometheus:9090`) is provisioned automatically.
Dashboards are imported by hand: **Dashboards > New > Import > upload JSON file**.

| Dashboard | File | Panels |
|---|---|---|
| Readmission API health (uid `readmission-api`) | `monitoring/grafana/starter_dashboard.json` | API up, live model version, replay week, share flagged (15 m gauge, green 12% to 28%), time since model load, requests per second by path, stays scored per minute, `/predict` latency p50/p95, share of non-2xx, flagged share over time, score-distribution heatmap, unseen categories per hour, rejected requests by reason |
| Readmission drift (PSI) (uid `readmission-drift`) | `monitoring/grafana/drift_dashboard.json` | Drift verdict, features at major drift, features at moderate drift, stays in the window, newest stay age, top 8 drifting features with 0.10 and 0.25 lines, score PSI, a worst-first table of every feature |

### 10.1 Showing only the Render service

The starter dashboard's queries do not filter by job, so they mix local and Render traffic. Two ways to separate them:

1. **Explore** (left menu): run a query with the job filter, for example the ones in section 11.
2. **A dashboard variable:** Dashboard settings > Variables > New: name `job`, type Query, query `label_values(up, job)`. Then edit each panel's query to add `{job="$job"}` to the
   `readmission_*` metrics. Pick `readmission-render` or `readmission-api` at the top of the dashboard.

Time range matters: widen it (for example last 1 hour) when nothing appears. "No data" usually means the data source is wrong, no traffic has been sent, or the range is too short.

---

## 11. PromQL cookbook

Replace `readmission-render` with `readmission-api` for the local service.

**Traffic and health**

```promql
sum by (path) (rate(readmission_http_requests_total{job="readmission-render"}[5m]))        # requests per second by endpoint
sum(rate(readmission_predictions_total{job="readmission-render"}[5m])) * 60                # stays scored per minute
sum(rate(readmission_http_requests_total{job="readmission-render",status!~"2.."}[5m]))
  / sum(rate(readmission_http_requests_total{job="readmission-render"}[5m]))               # error share
up{job="readmission-render"}                                                               # 1 = scrape works
```

**Latency**

```promql
histogram_quantile(0.95, sum by (le) (rate(readmission_http_request_duration_seconds_bucket{job="readmission-render",path="/predict"}[5m])))
histogram_quantile(0.50, sum by (le) (rate(readmission_http_request_duration_seconds_bucket{job="readmission-render",path="/predict"}[5m])))
```

**Model behaviour**

```promql
sum(rate(readmission_predictions_total{job="readmission-render",outcome="high_risk"}[15m]))
  / sum(rate(readmission_predictions_total{job="readmission-render"}[15m]))                # flagged share (expect about 0.20)
readmission_model_info{job="readmission-render"}                                           # which version and family is live
time() - readmission_model_loaded_timestamp_seconds{job="readmission-render"}              # seconds since the model loaded
sum by (le) (increase(readmission_prediction_score_bucket{job="readmission-render"}[1m]))  # for the score heatmap (format: heatmap)
```

**Data quality**

```promql
sum by (column) (increase(readmission_unseen_category_total{job="readmission-render"}[1h]))   # never-seen values by column
sum by (reason) (increase(readmission_validation_errors_total{job="readmission-render"}[1h])) # rejections by reason
```

**Drift (local only)**

```promql
topk(8, readmission_feature_psi)               # the most drifted features
readmission_score_psi                          # score drift
readmission_drift_features{level="major"}      # how many features are at major drift
readmission_drift_ready                        # 1 = a verdict exists, 0 = no verdict
ALERTS{alertstate="firing"}                    # every alert firing right now (graph it as stacked bars to see alerts over time)
```

---

## 12. The drift exporter

A small container that turns the weekly drift report into live metrics so drift can raise an alert.

```
 serving/export_champion.py --writes--> serving/bundle/reference_profile.json    what "normal" looks like (from the test split)
 API (local) --appends--> serving/logs/predictions.jsonl                          every scored stay with its features
                                  |  (both mounted read-only)
                                  v
        drift-exporter, every 15 s: last 1,000 stays vs the profile -> PSI per feature and for the score
                                  |  /metrics on :9100
                                  v
        Prometheus (job drift-exporter) -> alerts_drift.yml -> Grafana "Readmission drift (PSI)"
```

**PSI** = sum over bins of `(a - e) * ln(a / e)`, with `a` the share in the new data and `e` the reference share. Stable below 0.10, moderate 0.10 to 0.25, major above 0.25 (conventions, not laws).
Numeric features use the reference deciles; categories compare value by value with `__rare__` (under 0.5% in the reference) and `__unseen__` (never seen) buckets.

**No verdict** (`readmission_drift_ready = 0`, PSI series removed so alerts clear) when: no reference profile, no log, an empty window, fewer than `MIN_ROWS` stays, or the newest stay is older than `MAX_AGE_SECONDS`.
After a model swap the profile changes with the export; until enough stays of the new model version have arrived there is no score verdict.

**Reading a burst of alerts.** One real change ripples: an age shift also moves `payer_code`, `medical_specialty` and `discharge_disposition_id`. Read the alerts together; the feature with the largest PSI is usually the cause.
Without Docker: `python monitoring/drift_exporter.py --once` evaluates once and prints the metrics and the reason for any missing verdict.

---

## 13. The console's Monitoring tab

`/#monitor` (local or public) parses `/metrics` in the browser every 5 seconds and shows stays scored, flagged share, average response, 95% response time, rejected or failed requests, model loaded time,
the score distribution with the cutoff line, rejected requests by reason, never-seen values by column, and requests by endpoint. It needs no Prometheus or Docker, so it is the quickest way to see
public traffic. Totals are since the API process last started (they reset on restart). It does not keep history; Prometheus does.

---

## 14. Runbooks

Each runbook: symptom, likely cause, how to check, what to do.

### 14.1 The public site is down or slow

| Check | Meaning | Action |
|---|---|---|
| Open `/health` | 200 `{"status":"ok"}` | Service is fine; the slowness was a wake-up |
| Very slow, then works | Free instance was asleep | Nothing to fix; wait 30 to 60 s |
| `x-render-routing: no-server` | No service at that hostname | Render dashboard: does it exist, is the URL right, did the build fail? Check Events |
| 503 on `/health` | No model loaded | `deploy/bundle/` missing or checksum failed; re-run `prepare_bundle.py`, commit, push |
| Render build fails | Build error or out of memory | Read the build log; try `plan: starter` for memory |
| Prometheus shows `readmission-render` DOWN | Timeout or error | Check the error text on `/targets`; open `/metrics` in a browser; raise `scrape_timeout` |

### 14.2 Flagged share outside 12% to 28%

1. Check the **Drift** dashboard and `readmission_feature_psi`: did an input move (covariate shift)? A flagged share near 49% with `number_inpatient` PSI above 1 is the signature seen in the replay.
2. Check unseen categories and rejected requests: a changed upstream code or feed.
3. Check `readmission_model_info`: did the model version change (after a promotion the share should still be near 20% because the cutoff is recomputed)?
4. If the data is wrong, fix the feed. If the data is right, check labelled performance (section 14.6) before deciding anything.

### 14.3 Unseen categories rising

A value the model never saw is arriving (a new code, a renamed category). The stay is still scored, in the "infrequent" bucket. Check which column (`sum by (column) (...)`), find the source of the change, and fix the mapping.
The retrain rule will refuse to retrain on this (`fix_data_first`) because retraining on a broken feed reproduces it.

### 14.4 Error share high or many rejected requests

`sum by (reason) (increase(readmission_validation_errors_total[1h]))` tells you why: `missing_columns`, `nulls`, `not_numeric`. The API's 422 response names the column and the first bad rows. The caller must fix its payload;
nothing is wrong with the server.

### 14.5 A promoted model looks wrong

Label-free metrics cannot prove a model is worse (flagged share stays near 20%). If there is a reason to doubt it: roll back (section 5) and compare with `performance_report.py` on labelled weeks. Pool enough stays: a modest
difference of about 0.016 PR-AUC needs far more than a few thousand stays to show.

### 14.6 Drift alerts fired: do we retrain?

```bash
python monitoring/drift_report.py --run-id <run> --baseline-weeks 1-4
python monitoring/performance_report.py --run-id <run>
python retrain.py decide --run-id <run>
```

Follow the decision: `keep` (including "drift without a performance drop"), `fix_data_first`, or `retrain`. A retrain runs through the Prefect flow and never goes straight live (gate, then challenger, then the promotion rule).

### 14.7 `/reload` fails

| Response | Meaning | Action |
|---|---|---|
| 403 | `ADMIN_TOKEN` is set | Send `X-Admin-Token` |
| 500 with `still_serving_version` | The new bundle is incomplete or corrupt | The old model keeps serving; re-run `export_champion.py` |
| 200 but the public site is unchanged | `/reload` on Render re-reads the baked bundle | Redeploy by committing `deploy/bundle` (section 5) |

### 14.8 Prediction log cannot be written (Linux)

`Could not write the prediction log` in the API log: the container user (uid 1000) cannot write `serving/logs`. Run `mkdir -p serving/logs && sudo chown 1000 serving/logs`. Requests are still served.

---

## 15. Operating checklists

**After every promotion or rollback**
- [ ] `export_champion.py` then reload (local); `prepare_bundle.py`, commit, push (public)
- [ ] `/model-info` shows the expected version on both
- [ ] Drift exporter loaded the new profile; dashboards show the new `readmission_model_info`
- [ ] `promotion_log.jsonl` has the decision

**Daily or when watching traffic**
- [ ] `docker compose ps` all Up and healthy; `/targets` all UP
- [ ] Flagged share near 20%; no rejected-request spike; latency normal
- [ ] No unexpected firing alerts (and remember the `for:` delays)

**After a replay run**
- [ ] `drift_report.py`, `performance_report.py`, `retrain.py decide`
- [ ] Review `data/retrain/<id>/trigger.json`; keep `labels.csv` for runs you still want to analyse

**Before pushing**
- [ ] `ruff check --select E9,F63,F7,F82 .` and `pytest` pass
- [ ] `git ls-files render.yaml deploy` lists the five required files
- [ ] No data, logs, MLflow files or `.env` staged

---

## 16. Hardening and limits

**Before relying on the public service**
- Protect `/metrics` (a token or proxy rule; Prometheus then needs `authorization:` in the job) or accept that anyone can read it.
- Add an API key or rate limit to `/predict`; `MAX_RECORDS=100` limits batch size but not request rate.
- Keep `LOG_PREDICTIONS=false`; if you need logs, send a sanitised version to proper storage.
- Change the default Grafana password before exposing port 3000.

**Known limits**
- Free-plan instances sleep, have temporary disks and limited memory.
- Counters reset on every restart.
- Alerts are not routed anywhere (no Alertmanager); alert rules and the starter dashboard do not separate local and Render jobs.
- Drift and late-label performance cover the local stack and replay only.
- The console on Render is baked into the image: UI edits need a push.
- A single worker handles all requests (by design, for the counters); heavy load needs a different metrics approach.

---

## 17. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker version` shows only a Client section; `docker compose ps` prints nothing; commands hang | Docker engine not running | Open Docker Desktop and wait for **Engine running**; if stuck, quit it, run `wsl --shutdown`, reopen; then `docker compose up -d` |
| Docker Desktop will not open | Stuck processes, WSL or virtualization issue | End `Docker Desktop.exe`, `com.docker.backend.exe`, `vmmem` in Task Manager; `wsl --shutdown`; reopen (try as administrator); check virtualization is enabled and the WSL and Virtual Machine Platform features are on; last resort repair or reinstall (volumes, so metric history, are lost) |
| `docker compose restart prometheus` hangs | Docker engine not responding (not a YAML problem) | As above |
| Prometheus restarts but will not stay up | YAML error | `docker compose logs --tail 30 prometheus`; look for `parsing YAML file`; fix indentation |
| `readmission-render` target DOWN | Timeout, wrong hostname, service asleep or down | Read the error on `/targets`; hostname only (no `https://`, no path); `scheme: https`; raise `scrape_timeout`; open `/metrics` in a browser |
| Grafana panels say "No data" | Wrong data source, no traffic, range too short | Data source must be Prometheus; send traffic with `send_sample.py`; widen the time range |
| Dashboard numbers mix local and Render | Queries have no job filter | Add `{job="..."}` or a `$job` variable (section 10.1) |
| `port is already allocated` (8000, 9090, 3000, 9100) | Another program uses the port | Stop it or change the left side of the port mapping |
| Container "unhealthy" | No bundle or a bad bundle | `docker compose logs readmission-api`; run the export; Docker does not restart it by itself |
| `/health` 503 locally | No bundle | `python serving/export_champion.py`, then Reload |
| Console unstyled or "Loading the form..." forever; `/ui/...` 404 | `serving/ui/` missing or named `UI` | Must be lowercase `ui` with `index.html`, `styles.css`, `app.js`; hard refresh with Ctrl+F5 |
| `failed to read dockerfile` (Linux, CI) | File must be `serving/Dockerfile` (capital D) | Rename with two `git mv` steps so Git records the case change |
| Render did not redeploy after a push | Auto-deploy off, build filter excludes the change, or `deploy/bundle` not committed | Check Events and Build Filters; `git ls-files deploy` |
| Replay alerts never fire | Weeks too short for the 3-minute windows, or Prometheus not restarted after a rule edit | `--seconds-per-week 120`; `docker compose restart prometheus` |
| Drift alerts stay on after the replay | Staleness window | Wait up to `MAX_AGE_SECONDS` (10 minutes); `UnseenCategoriesRising` clears about an hour later |
| `readmission_drift_ready` 0, no PSI series | No verdict | Read `docker compose logs drift-exporter` or run `drift_exporter.py --once` |
| Score PSI missing after a promotion | Too few stays scored by the new version yet | It returns once `MIN_ROWS` stays have arrived |
| Metrics look cut in half after a while on Render | The process restarted and counters reset | Use `rate()` and `increase()`, not raw totals |