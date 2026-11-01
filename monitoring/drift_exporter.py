"""
Live drift exporter (plan step 13, Phase B): turns the weekly drift report into Prometheus metrics, so drift can raise
an alert instead of waiting for someone to run drift_report.py.

It runs as a small container next to the API (see docker-compose.yml, service drift-exporter). Every INTERVAL seconds it
    1. reads the most recent WINDOW_ROWS stays from the API's prediction log (serving/logs/predictions.jsonl),
    2. compares their input features and scores with the REFERENCE PROFILE saved next to the model
       (serving/bundle/reference_profile.json, written by serving/export_champion.py from the reference data),
    3. publishes the PSI of every feature, and of the score, on http://localhost:9100/metrics for Prometheus.

Metrics
    readmission_feature_psi{feature="age"}      PSI of that input feature over the window (46 series)
    readmission_score_psi                       PSI of the scores (only stays scored by the live model version)
    readmission_drift_window_rows               stays in the window
    readmission_drift_features{level="major"}   how many features are at that level (stable, moderate, major)
    readmission_drift_ready                     1 if a verdict was computed this cycle, 0 if not (see below)
    readmission_drift_age_seconds               age of the newest logged stay
    readmission_drift_profile_info{model_version=...,reference_rows=...}   which profile is loaded (value always 1)
    readmission_drift_last_update_timestamp_seconds

When it gives NO verdict (ready = 0, and the PSI series are removed, so alerts clear instead of sticking):
    - the profile or the log does not exist yet,
    - fewer than MIN_ROWS stays are in the window (too few to judge),
    - the newest stay is older than MAX_AGE_SECONDS (no recent traffic: a verdict about old data would be misleading).

Score PSI is only computed from stays scored by the same model version as the profile. After a model swap, the export writes
a new profile, and until enough stays of the new model arrive there is no score verdict.

Settings (environment variables)
    LOG_PATH         prediction log                     default /data/logs/predictions.jsonl
    PROFILE_PATH     reference profile                  default /data/bundle/reference_profile.json
    WINDOW_ROWS      most recent stays to compare       default 1000
    MIN_ROWS         fewer stays than this: no verdict  default 500
    MAX_AGE_SECONDS  older newest stay: no verdict      default 600
    INTERVAL         seconds between evaluations        default 15
    PORT             metrics port                       default 9100
    RUN_ID           only use stays of this run id      default (empty: all traffic)

Try it without Docker:   python monitoring/drift_exporter.py --once      (prints the metrics once and exits)

Needs: pandas, numpy, prometheus-client.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, generate_latest, start_http_server

from drift_lib import MAJOR, MODERATE, SCORE, level, profile_from_json, psi, shares

LOG_PATH = Path(os.environ.get("LOG_PATH", "/data/logs/predictions.jsonl"))
PROFILE_PATH = Path(os.environ.get("PROFILE_PATH", "/data/bundle/reference_profile.json"))
WINDOW_ROWS = int(os.environ.get("WINDOW_ROWS", "1000"))
MIN_ROWS = int(os.environ.get("MIN_ROWS", "500"))
MAX_AGE_SECONDS = float(os.environ.get("MAX_AGE_SECONDS", "600"))
INTERVAL = float(os.environ.get("INTERVAL", "15"))
PORT = int(os.environ.get("PORT", "9100"))
RUN_ID = os.environ.get("RUN_ID", "")

REGISTRY = CollectorRegistry()
FEATURE_PSI = Gauge("readmission_feature_psi", "PSI of one input feature over the recent window", ["feature"], registry=REGISTRY)
SCORE_PSI = Gauge("readmission_score_psi", "PSI of the scores over the recent window", registry=REGISTRY)
WINDOW = Gauge("readmission_drift_window_rows", "Stays in the drift window", registry=REGISTRY)
LEVELS = Gauge("readmission_drift_features", "Features per drift level", ["level"], registry=REGISTRY)
READY = Gauge("readmission_drift_ready", "1 if a drift verdict was computed this cycle", registry=REGISTRY)
AGE = Gauge("readmission_drift_age_seconds", "Age of the newest logged stay", registry=REGISTRY)
PROFILE_INFO = Gauge("readmission_drift_profile_info", "Reference profile in use (value is always 1)",
                     ["model_version", "reference_rows"], registry=REGISTRY)
UPDATED = Gauge("readmission_drift_last_update_timestamp_seconds", "When the metrics were last refreshed", registry=REGISTRY)

_profile = {"mtime": None, "data": None}


def load_profile():
    """Load the reference profile, and reload it when a new export replaces the file."""
    if not PROFILE_PATH.exists():
        _profile.update(mtime=None, data=None)
        return None
    mtime = PROFILE_PATH.stat().st_mtime
    if mtime != _profile["mtime"]:
        raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        _profile["data"] = {
            "model_version": raw.get("model_version"), "reference_rows": raw.get("reference_rows"),
            "features": {c: profile_from_json(p) for c, p in raw["features"].items()},
            "score": profile_from_json(raw["score"])}
        _profile["mtime"] = mtime
        print(f"Loaded reference profile: model version {raw.get('model_version')}, "
              f"{raw.get('reference_rows')} reference stays, {len(raw['features'])} features", flush=True)
    return _profile["data"]


def tail_lines(path: Path, n: int, chunk: int = 1 << 20) -> list:
    """The last n lines of a (possibly large) file, reading only the end."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos, data = f.tell(), b""
        while pos > 0 and data.count(b"\n") <= n:
            step = min(chunk, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    return data.splitlines()[-n:]


def read_window(path: Path, n: int, run_id: str) -> pd.DataFrame:
    rows = []
    for raw in tail_lines(path, n):
        try:
            obj = json.loads(raw.decode("utf-8"))
        except ValueError:
            continue
        if run_id and obj.get("run_id") != run_id:
            continue
        row = {"__ts": obj.get("ts"), "__version": obj.get("model_version"), "__score": obj.get("score")}
        row.update(obj.get("features") or {})
        rows.append(row)
    return pd.DataFrame(rows)


def clear_verdict() -> None:
    FEATURE_PSI.clear()
    SCORE_PSI.set(float("nan"))
    LEVELS.clear()
    READY.set(0)


def evaluate() -> str:
    """One cycle. Returns a short text saying what happened (printed by --once)."""
    UPDATED.set(time.time())
    profile = load_profile()
    if profile is None:
        clear_verdict()
        return f"no verdict: no reference profile at {PROFILE_PATH} (run python serving/export_champion.py)"
    PROFILE_INFO.clear()
    PROFILE_INFO.labels(str(profile["model_version"]), str(profile["reference_rows"])).set(1)
    if not LOG_PATH.exists():
        clear_verdict()
        return f"no verdict: no prediction log at {LOG_PATH}"
    df = read_window(LOG_PATH, WINDOW_ROWS, RUN_ID)
    WINDOW.set(len(df))
    if df.empty:
        clear_verdict()
        return "no verdict: the log has no usable rows"
    newest = pd.to_datetime(df["__ts"], utc=True, errors="coerce").max()
    age = (datetime.now(timezone.utc) - newest.to_pydatetime()).total_seconds() if pd.notna(newest) else float("inf")
    AGE.set(age if age != float("inf") else -1)
    if len(df) < MIN_ROWS:
        clear_verdict()
        return f"no verdict: only {len(df)} stays in the window (need {MIN_ROWS})"
    if age > MAX_AGE_SECONDS:
        clear_verdict()
        return f"no verdict: the newest stay is {age:,.0f} s old (limit {MAX_AGE_SECONDS:,.0f} s), so there is no recent traffic"

    FEATURE_PSI.clear()
    values = {}
    for name, prof in profile["features"].items():
        if name not in df.columns:
            continue
        values[name] = psi(prof["shares"], shares(prof, df[name]))
        FEATURE_PSI.labels(name).set(values[name])
    counts = {"stable": 0, "moderate": 0, "major": 0}
    for v in values.values():
        counts[level(v)] += 1
    for k, c in counts.items():
        LEVELS.labels(k).set(c)

    same_model = df[df["__version"] == profile["model_version"]]
    if len(same_model) >= MIN_ROWS:
        SCORE_PSI.set(psi(profile["score"]["shares"], shares(profile["score"], same_model["__score"])))
        score_text = f"score PSI from {len(same_model)} stays of model version {profile['model_version']}"
    else:
        SCORE_PSI.set(float("nan"))
        score_text = (f"no score verdict: only {len(same_model)} stays scored by model version "
                      f"{profile['model_version']} in the window")
    READY.set(1)
    worst = sorted(values.items(), key=lambda kv: -kv[1])[:3]
    return (f"verdict from {len(df)} stays: {counts['major']} major, {counts['moderate']} moderate, {counts['stable']} stable; "
            f"worst {', '.join(f'{k} {v:.2f}' for k, v in worst)}; {score_text}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Expose feature and score PSI as Prometheus metrics.")
    ap.add_argument("--once", action="store_true", help="evaluate once, print the metrics and exit (for testing)")
    args = ap.parse_args()
    if args.once:
        print(evaluate())
        print(generate_latest(REGISTRY).decode("utf-8"))
        return
    start_http_server(PORT, registry=REGISTRY)
    print(f"Drift exporter on port {PORT}: window {WINDOW_ROWS} stays (minimum {MIN_ROWS}), "
          f"every {INTERVAL:g} s, no verdict if the newest stay is older than {MAX_AGE_SECONDS:g} s. "
          f"Thresholds: moderate {MODERATE}, major {MAJOR}.", flush=True)
    while True:
        try:
            print(time.strftime("%H:%M:%S"), evaluate(), flush=True)
        except Exception as exc:                          # never let one bad cycle stop the exporter
            clear_verdict()
            print(f"cycle failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
