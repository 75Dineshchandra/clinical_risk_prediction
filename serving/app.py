"""
Readmission risk API (plan step 10, extended in step 12).

Serves the model in the bundle folder (made by serving/export_champion.py from the registry's @champion).
The API never talks to MLflow: the bundle is a joblib pipeline plus a meta.json with the evidence.

Endpoints
    GET  /             the web console (serving/ui/): score a stay, batch CSV, model, monitoring
    POST /predict      score + high-risk flag + top SHAP drivers for 1..MAX_RECORDS records
    GET  /example      a ready-to-send example request body
    GET  /model-info   which model is live, its evidence, the required columns, the flag rule
    GET  /health       200 if a model is loaded, else 503 (used by Docker)
    GET  /metrics      Prometheus metrics (latency, errors, volume, score distribution, model version)
    POST /reload       re-read the bundle after a new export (swap is atomic; a bad bundle is refused)
    GET  /docs         interactive documentation

Replay headers (step 12, optional, used by the replay in step 11)
    X-Replay-Run      run id, 1 to 64 characters of letters, digits, _ . -   -> logged as "run_id"
    X-Replay-Week     simulated week number, an integer 0..1000              -> logged as "replay_week"
                                                                                and shown in the gauge
                                                                                readmission_replay_week
    Normal traffic does not send them; the log then holds null for both.

Settings (environment variables)
    BUNDLE_DIR        folder with model.joblib and meta.json   (default: serving/bundle next to this file)
    MAX_RECORDS       most records accepted per request        (default 500)
    ID_FIELD          column returned as "id" and logged       (default encounter_id)
    LOG_PREDICTIONS   true/false, append every prediction to a JSONL file (default true)
    PREDICTION_LOG    path of that file                        (default serving/logs/predictions.jsonl)
    ADMIN_TOKEN       if set, POST /reload needs header X-Admin-Token with this value
    LOG_LEVEL         INFO, DEBUG, ...

Run one worker only: the Prometheus counters live inside the process.
"""

import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import re
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", message="X does not have valid feature names")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("readmission-api")

HERE = Path(__file__).resolve().parent
BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", HERE / "bundle"))
MAX_RECORDS = int(os.environ.get("MAX_RECORDS", "500"))
ID_FIELD = os.environ.get("ID_FIELD", "encounter_id")
LOG_PREDICTIONS = os.environ.get("LOG_PREDICTIONS", "true").lower() in ("1", "true", "yes")
PREDICTION_LOG = os.environ.get("PREDICTION_LOG", str(HERE / "logs" / "predictions.jsonl"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
UI_DIR = HERE / "ui"
DOC_PATHS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_REPLAY_WEEK = 1000
FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#4f46e5"/>'
           '<path d="M4 17h6l3-8 5 15 3-7h7" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')
SCORE_NOTE = ("Ranking score, not a probability: the model was trained with class weights, so scores run "
              "higher than real risk. Use it to rank and flag.")
CHECK_PACKAGES = ["scikit-learn", "numpy", "pandas", "scipy", "lightgbm", "xgboost", "joblib"]

HTTP_REQUESTS = Counter("readmission_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("readmission_http_request_duration_seconds", "HTTP request latency",
                         ["method", "path"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
PREDICTIONS = Counter("readmission_predictions_total", "Records scored", ["outcome"])
SCORES = Histogram("readmission_prediction_score", "Distribution of scores", buckets=[i / 20 for i in range(1, 21)])
BATCH_SIZE = Histogram("readmission_batch_size", "Records per /predict request", buckets=(1, 5, 10, 50, 100, 250, 500, 1000))
VALIDATION_ERRORS = Counter("readmission_validation_errors_total", "Rejected requests", ["reason"])
UNSEEN = Counter("readmission_unseen_category_total", "Category values never seen in training", ["column"])
MODEL_INFO = Gauge("readmission_model_info", "Live model (value is always 1)", ["name", "version", "family", "alias"])
MODEL_LOADED_AT = Gauge("readmission_model_loaded_timestamp_seconds", "When the live model was loaded")
RELOADS = Counter("readmission_model_reloads_total", "Model load attempts", ["outcome"])
REPLAY_WEEK = Gauge("readmission_replay_week",
                    "Simulated week of the most recent replay request (only set when X-Replay-Week is sent)")


@dataclass
class ModelState:
    pipe: Any
    meta: dict
    required: List[str]
    numeric: List[str]
    categorical: Dict[str, str]          # column -> "str" or "num" (how the encoder saw it in training)
    categories: Dict[str, set]
    feature_names: List[str]
    feature_to_column: Dict[str, str]
    cutoff: Optional[float]
    explain_kind: str                    # "tree", "linear" or "none"
    explainer: Any
    loaded_at: str
    package_warnings: List[str]


_state: Optional[ModelState] = None
_reload_lock = threading.Lock()
_log_lock = threading.Lock()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_pipeline(pipe) -> dict:
    """Read the input contract from the fitted pipeline itself (the pipeline is the source of truth)."""
    prep = pipe.named_steps["prep"]
    required = [str(c) for c in prep.feature_names_in_]
    numeric, categorical, categories = [], {}, {}
    for name, trans, cols in prep.transformers_:
        if name == "remainder" or isinstance(trans, str):
            continue
        if isinstance(trans, OneHotEncoder):
            for col, cats in zip(cols, trans.categories_):
                categorical[str(col)] = "str" if cats.dtype == object else "num"
                categories[str(col)] = set(cats.tolist())
        else:
            numeric.extend(str(c) for c in cols)
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    cat_cols = sorted(categorical, key=len, reverse=True)
    numeric_set = set(numeric)
    mapping = {}
    for n in names:
        if n in numeric_set:
            mapping[n] = n
        else:
            mapping[n] = next((c for c in cat_cols if n.startswith(c + "_")), n)
    return {"required": required, "numeric": numeric, "categorical": categorical,
            "categories": categories, "feature_names": names, "feature_to_column": mapping}


def load_state(bundle_dir: Path) -> ModelState:
    meta_path, model_path = bundle_dir / "meta.json", bundle_dir / "model.joblib"
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"No model bundle in {bundle_dir}. Run: python serving/export_champion.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if file_sha256(model_path) != meta.get("model_sha256"):
        raise ValueError("model.joblib does not match meta.json (an export is unfinished, or the files are out of sync)")
    pipe = joblib.load(model_path)
    steps = getattr(pipe, "named_steps", {})
    if "prep" not in steps or "model" not in steps:
        raise ValueError("The bundle must hold a Pipeline with steps named 'prep' and 'model'")
    schema = inspect_pipeline(pipe)

    explain_kind, explainer = "none", None
    try:
        from sklearn.linear_model import LogisticRegression
        if isinstance(steps["model"], LogisticRegression):
            explain_kind = "linear"
        else:
            import shap
            explainer, explain_kind = shap.TreeExplainer(steps["model"]), "tree"
    except Exception as exc:
        log.warning("Explanations are disabled for this model: %s: %s", type(exc).__name__, exc)

    package_warnings = []
    for name, wanted in (meta.get("packages") or {}).items():
        if name not in CHECK_PACKAGES:
            continue
        try:
            have = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            have = None
        if have != wanted:
            package_warnings.append(f"{name}: bundle made with {wanted}, running {have}")
    for w in package_warnings:
        log.warning("Package mismatch: %s", w)

    return ModelState(pipe=pipe, meta=meta, cutoff=meta.get("flag_cutoff"), explain_kind=explain_kind,
                      explainer=explainer, loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      package_warnings=package_warnings, **schema)


def activate(state: ModelState) -> None:
    global _state
    _state = state
    m = state.meta
    MODEL_INFO.clear()
    MODEL_INFO.labels(str(m.get("registered_model", "")), str(m.get("version", "")),
                      str(m.get("model_family", "")), str(m.get("alias", ""))).set(1)
    MODEL_LOADED_AT.set(time.time())


def try_reload() -> ModelState:
    with _reload_lock:
        try:
            state = load_state(BUNDLE_DIR)
        except Exception:
            RELOADS.labels("failed").inc()
            raise
        activate(state)
        RELOADS.labels("ok").inc()
        log.info("Loaded %s version %s (%s)", state.meta.get("registered_model"),
                 state.meta.get("version"), state.meta.get("model_family"))
        return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        try_reload()
    except Exception as exc:
        log.error("No model loaded at startup: %s: %s", type(exc).__name__, exc)
    yield


app = FastAPI(title="Readmission risk API", version="1.1",
              description="Scores hospital stays for 30-day readmission risk. " + SCORE_NOTE, lifespan=lifespan)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start, status = time.perf_counter(), 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        if request.url.path != "/metrics":
            route = request.scope.get("route")
            raw = request.url.path
            path = getattr(route, "path", None) or ("/ui/*" if raw.startswith("/ui/") else raw if raw in DOC_PATHS else "unmatched")
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - start)


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1,
                                          description="One object per stay, with every feature column the model needs.")
    explain: bool = Field(True, description="Include the top SHAP drivers for each record.")
    top_k: int = Field(5, ge=1, le=20, description="How many drivers to return per record.")


def require_state() -> ModelState:
    if _state is None:
        raise HTTPException(503, detail="No model is loaded. Run serving/export_champion.py, then POST /reload.")
    return _state


def reject(reason: str, message: str, problems: list) -> None:
    VALIDATION_ERRORS.labels(reason).inc()
    raise HTTPException(422, detail={"error": reason, "message": message, "problems": problems[:50]})


def parse_replay_headers(run: Optional[str], week: Optional[str]):
    """Validate the optional replay headers. Returns (run_id or None, week number or None)."""
    run_id, week_no = None, None
    if run is not None:
        if not RUN_ID_RE.match(run):
            reject("bad_replay_header", "X-Replay-Run must be 1 to 64 characters: letters, digits, _ . -",
                   [{"header": "X-Replay-Run", "issue": "invalid"}])
        run_id = run
    if week is not None:
        try:
            week_no = int(week)
            if not 0 <= week_no <= MAX_REPLAY_WEEK:
                raise ValueError("out of range")
        except ValueError:
            reject("bad_replay_header", f"X-Replay-Week must be a whole number from 0 to {MAX_REPLAY_WEEK}",
                   [{"header": "X-Replay-Week", "issue": "invalid"}])
    return run_id, week_no


def bad_rows(mask: pd.Series) -> list:
    return [int(i) for i in np.flatnonzero(mask.to_numpy())[:5]]


def build_frame(records: List[Dict[str, Any]], st: ModelState):
    """Turn JSON records into the frame the pipeline was trained on, or explain exactly what is wrong."""
    df = pd.DataFrame.from_records(records)
    missing = [c for c in st.required if c not in df.columns]
    if missing:
        reject("missing_columns", "Required columns are missing.", [{"column": c, "issue": "missing"} for c in missing])
    ignored = [c for c in df.columns if c not in st.required and c != ID_FIELD]
    ids = df[ID_FIELD].tolist() if ID_FIELD in df.columns else [None] * len(df)
    df = df[st.required].copy()

    problems = [{"column": c, "issue": "null", "rows": bad_rows(df[c].isna())} for c in st.required if df[c].isna().any()]
    if problems:
        reject("nulls", "Null values are not allowed. Use the same encoding as the training data (for example 'missing').", problems)

    for c in st.numeric:
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.isna().any():
            problems.append({"column": c, "issue": "not_numeric", "rows": bad_rows(conv.isna())})
        else:
            df[c] = conv
    if problems:
        reject("not_numeric", "Numeric columns must contain numbers.", problems)

    unseen: Dict[str, List[str]] = {}
    for c, kind in st.categorical.items():
        if kind == "str":
            col = df[c]
            if pd.api.types.is_float_dtype(col) and (col % 1 == 0).all():
                col = col.astype("int64")
            df[c] = col.astype(str)
        else:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.isna().any():
                problems.append({"column": c, "issue": "not_numeric", "rows": bad_rows(conv.isna())})
            else:
                df[c] = conv
        if problems:
            continue
        new = set(df[c].unique().tolist()) - st.categories[c]
        if new:
            unseen[c] = sorted(str(v) for v in new)[:10]
            UNSEEN.labels(c).inc(int(df[c].isin(new).sum()))
    if problems:
        reject("not_numeric", "This column was numeric in training and must contain numbers.", problems)
    return df, ignored, unseen, ids


def explain(st: ModelState, df: pd.DataFrame, records: List[Dict[str, Any]], top_k: int):
    """Per-record top drivers, in log-odds units (the model's raw output)."""
    Xt = st.pipe.named_steps["prep"].transform(df)
    X = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    model = st.pipe.named_steps["model"]
    if st.explain_kind == "linear":
        values, base = X * model.coef_[0], float(np.ravel(model.intercept_)[0])
    elif st.explain_kind == "tree":
        values = st.explainer.shap_values(X)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        base = float(np.ravel(st.explainer.expected_value)[-1])
    else:
        raise RuntimeError("explanations are not available for this model type")
    out = []
    for i in range(len(df)):
        top = np.argsort(-np.abs(values[i]))[:top_k]
        row = []
        for j in top:
            feature = st.feature_names[j]
            column = st.feature_to_column.get(feature, feature)
            contribution = float(values[i, j])
            row.append({"feature": feature, "column": column, "input_value": records[i].get(column),
                        "contribution": round(contribution, 4),
                        "effect": "raises_risk" if contribution > 0 else "lowers_risk"})
        out.append(row)
    return out, base


def log_predictions(request_id: str, st: ModelState, records, ids, scores, flags,
                    run_id: Optional[str] = None, replay_week: Optional[int] = None) -> None:
    if not LOG_PREDICTIONS or not PREDICTION_LOG:
        return
    try:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        version = st.meta.get("version")
        lines = [json.dumps({"ts": ts, "request_id": request_id, "model_version": version,
                             "run_id": run_id, "replay_week": replay_week, "id": ids[i],
                             "score": round(float(scores[i]), 6), "high_risk": flags[i],
                             "features": {c: records[i].get(c) for c in st.required}}, default=str)
                 for i in range(len(records))]
        with _log_lock:
            path = Path(PREDICTION_LOG)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
    except Exception:
        log.exception("Could not write the prediction log (the request itself was served)")


@app.get("/", include_in_schema=False)
def console():
    index = UI_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, detail="serving/ui/index.html is missing")
    return HTMLResponse(index.read_text(encoding="utf-8"), headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(FAVICON, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.post("/predict")
def predict(req: PredictRequest,
            x_replay_run: Optional[str] = Header(default=None),
            x_replay_week: Optional[str] = Header(default=None)):
    st = require_state()
    n = len(req.records)
    if n > MAX_RECORDS:
        reject("too_many_records", f"Send at most {MAX_RECORDS} records per request (got {n}).", [])
    run_id, replay_week = parse_replay_headers(x_replay_run, x_replay_week)
    df, ignored, unseen, ids = build_frame(req.records, st)
    scores = st.pipe.predict_proba(df)[:, 1]
    flags = [bool(s >= st.cutoff) for s in scores] if st.cutoff is not None else [None] * n

    drivers, base, explain_error = None, None, None
    if req.explain:
        try:
            drivers, base = explain(st, df, req.records, req.top_k)
        except Exception as exc:
            log.exception("Explanation failed")
            explain_error = f"{type(exc).__name__}: {exc}"

    BATCH_SIZE.observe(n)
    for s, f in zip(scores, flags):
        SCORES.observe(float(s))
        PREDICTIONS.labels("high_risk" if f else "low_risk").inc()
    if replay_week is not None:
        REPLAY_WEEK.set(replay_week)
    request_id = uuid.uuid4().hex[:12]
    log_predictions(request_id, st, req.records, ids, scores, flags, run_id, replay_week)

    m = st.meta
    predictions = []
    for i in range(n):
        item = {"id": ids[i], "score": round(float(scores[i]), 6), "high_risk": flags[i]}
        if drivers is not None:
            item["top_drivers"] = drivers[i]
        predictions.append(item)
    return {"request_id": request_id,
            "model": {"name": m.get("registered_model"), "alias": m.get("alias"),
                      "version": m.get("version"), "family": m.get("model_family")},
            "flag_rule": {"top_fraction": m.get("flag_fraction"), "cutoff": st.cutoff, "rule": "high_risk if score >= cutoff"},
            "score_note": SCORE_NOTE,
            "drivers_unit": "log-odds (the model's raw output); positive raises risk",
            "base_value": base, "predictions": predictions,
            "ignored_columns": ignored, "unseen_categories": unseen, "explain_error": explain_error}


@app.get("/example")
def example():
    st = require_state()
    record = st.meta.get("example_record")
    if not record:
        raise HTTPException(404, detail="This bundle has no example record.")
    return {"records": [record], "explain": True, "top_k": 5}


@app.get("/model-info")
def model_info():
    st = require_state()
    m = st.meta
    return {"model": {k: m.get(k) for k in ("registered_model", "alias", "version", "model_family", "training_run_id")},
            "evidence": m.get("registry_tags", {}),
            "bundle": {"exported_at": m.get("exported_at"), "loaded_at": st.loaded_at,
                       "model_sha256": (m.get("model_sha256") or "")[:12], "python_at_export": m.get("python")},
            "flag_rule": {"top_fraction": m.get("flag_fraction"), "cutoff": st.cutoff,
                          "based_on": m.get("flag_cutoff_based_on")},
            "features": {"required": st.required, "numeric": st.numeric, "categorical": sorted(st.categorical),
                         "categories": {c: sorted(st.categories[c], key=(lambda v: v) if st.categorical[c] == "num" else str)
                                        for c in st.categorical},
                         "optional_id_field": ID_FIELD},
            "explanations": st.explain_kind, "package_warnings": st.package_warnings,
            "limits": {"max_records_per_request": MAX_RECORDS}, "score_note": SCORE_NOTE}


@app.get("/health")
def health():
    if _state is None:
        return JSONResponse(status_code=503, content={"status": "no_model_loaded"})
    return {"status": "ok", "model_version": _state.meta.get("version"),
            "model_family": _state.meta.get("model_family")}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
def reload_model(x_admin_token: Optional[str] = Header(default=None)):
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(403, detail="Missing or wrong X-Admin-Token.")
    before = _state.meta.get("version") if _state else None
    try:
        st = try_reload()
    except Exception as exc:
        raise HTTPException(500, detail={"error": f"{type(exc).__name__}: {exc}", "still_serving_version": before})
    return {"reloaded": True, "previous_version": before, "model_version": st.meta.get("version"),
            "model_family": st.meta.get("model_family")}


class NoCacheStaticFiles(StaticFiles):
    """Console files are re-checked on every load, so a replaced file is never shown stale."""
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


if UI_DIR.is_dir():
    app.mount("/ui", NoCacheStaticFiles(directory=str(UI_DIR)), name="ui")