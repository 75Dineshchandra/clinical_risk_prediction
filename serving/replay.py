"""
Holdout replay (plan step 11): feed the sealed holdout through the RUNNING API as if it were live traffic.

The holdout (25% of patients, never used for training, selection or the gate) is cut into simulated weeks,
whole patients per week. Each week is sent to POST /predict with the replay headers, so the API logs the
run id and week and the dashboards show them. From a chosen week onward you can inject drift, one
scenario at a time, and watch whether monitoring notices.

Run from the project root, with the stack up (docker compose up -d) and a bundle exported:
    python serving/replay.py --preset demo --run-id demo1 --seconds-per-week 120
    python serving/replay.py --run-id r2 --weeks 8 --inject covariate_shift:4-6:0.5 --inject bad_data:8
    python serving/replay.py --preset demo --dry-run          # show the plan, send nothing
    python serving/replay.py --preset demo --show-examples    # print real before/after rows for each scenario, send nothing

Scenarios   (--inject NAME[:FIRST[-LAST]][:SEVERITY], repeat the flag for several; weeks count from 1)
    covariate_shift   a share SEVERITY of stays get 3 more prior inpatient stays and 3 more days in hospital
    population_shift  a share SEVERITY of stays of patients under 60 are dropped, so the week is older and smaller
    population_shift_young  the opposite: a share SEVERITY of stays of patients aged 60 and over are dropped, so the week
                      is younger and smaller (a drastic change of mix; default severity 0.8)
    new_categories    a share SEVERITY of stays get a never-seen value in discharge_disposition_id and medical_specialty
    bad_data          a share SEVERITY of requests contain one broken record (null, text in a number, missing column);
                      the API rejects them with 422 and the replay then resends the clean batch (an upstream bug fixed by a retry)

What it never does
    - It never sends the outcome columns (readmitted, readmitted_raw) or patient_nbr. It sends only the columns the
      live model asks for (read from /model-info) plus encounter_id.
    - It never changes the model, the registry or the saved holdout file.

What it writes   (data/replay/<run-id>/)
    plan.json       the weeks, their sizes and the scenarios, so the run can be repeated exactly
    labels.csv      encounter_id, patient_nbr, week, readmitted, scenario for every stay actually scored.
                    These are the LATE LABELS for step 14: they are withheld from the API on purpose.
    summary.json    per-week results
    summary.md      the same as a table, plus what to look at in Grafana
It also appends one line to models/data_use_log.jsonl (the holdout is single-use: look, then decide).

Pacing: Prometheus scrapes every 15 s, so each simulated week should last at least 20 s to show up on a chart.
The replay alert rules (monitoring/prometheus/alerts_replay.yml) look at 3-minute windows, so use 90 to 120 s per week
if you want alerts to fire cleanly.

Uses only the standard library plus pandas and numpy.
"""

import argparse
import copy
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PATIENT, ENCOUNTER, LABEL = "patient_nbr", "encounter_id", "readmitted"
OUTCOME_COLUMNS = ["readmitted", "readmitted_raw"]
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")          # same rule as the API's X-Replay-Run header
FLAG_LOW, FLAG_HIGH = 0.12, 0.28                            # same band as the FlaggedShare alerts
MIN_WEEK_SECONDS = 20                                       # Prometheus scrapes every 15 s
NEW_CATEGORY_COLUMNS = ["discharge_disposition_id", "medical_specialty"]
DEFAULT_SEVERITY = {"covariate_shift": 0.5, "population_shift": 0.6, "population_shift_young": 0.8,
                    "new_categories": 0.05, "bad_data": 0.25}
PRESETS = {  # 12 weeks: 1-4 clean baseline, then one scenario at a time with clean weeks between
    "demo": ["covariate_shift:5-6", "new_categories:8-9", "bad_data:11", "population_shift:12"],
}
EXPECT = {
    "covariate_shift": "Flagged share rises above its clean-week level; the score heatmap shifts up.",
    "population_shift": "Weeks get smaller and older; flagged share and the score distribution move.",
    "population_shift_young": "Weeks get much younger and smaller; the age mix changes drastically. Scores may or may not move.",
    "new_categories": "Unseen-category panel rises; ReplayUnseenCategories alert.",
    "bad_data": "Rejected-requests panel (nulls, not_numeric, missing_columns); error share; ReplayErrorShare alert.",
}


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find a data/ folder next to or above {here}")


# ---------------------------------------------------------------- HTTP helpers
def call(method: str, url: str, body=None, headers=None, timeout: int = 120):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    hdrs = {"Content-Type": "application/json"} if body is not None else {}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as err:      # refused, reset, or timed out (socket.timeout is an OSError)
        reason = getattr(err, "reason", err)
        raise SystemExit(f"Cannot get an answer from the API at {url} ({reason}).\n"
                         "If nothing is listening, start it:  docker compose up -d --build   (export a bundle first:\n"
                         "python serving/export_champion.py). If something IS listening but never answers (a timeout),\n"
                         "Docker's port forwarding is up while the container is not, or an old uvicorn is hung on the port:\n"
                         "check  docker compose ps  and  netstat -ano | findstr :8000 .")


def as_json(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return {"raw": text[:300]}


def api_counters(base: str) -> dict:
    """Totals of two API counters, so each week can report its own change. Totals reset when the API restarts."""
    status, text = call("GET", base + "/metrics")
    out = {"unseen": 0.0, "rejected": 0.0}
    if status != 200:
        return out
    for line in text.splitlines():
        if line.startswith("readmission_unseen_category_total{"):
            out["unseen"] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("readmission_validation_errors_total{"):
            out["rejected"] += float(line.rsplit(" ", 1)[1])
    return out


# ---------------------------------------------------------------- weeks and scenarios
def parse_inject(spec: str, n_weeks: int) -> dict:
    parts = spec.split(":")
    name = parts[0]
    if name not in DEFAULT_SEVERITY:
        raise SystemExit(f"Unknown scenario '{name}'. Choose from {sorted(DEFAULT_SEVERITY)}.")
    first, last = 1, n_weeks
    if len(parts) > 1 and parts[1]:
        bounds = parts[1].split("-")
        try:
            first, last = int(bounds[0]), int(bounds[-1])
        except ValueError:
            raise SystemExit(f"Bad week range in '{spec}'. Use for example {name}:5-8 or {name}:5.")
    severity = float(parts[2]) if len(parts) > 2 and parts[2] else DEFAULT_SEVERITY[name]
    if not (1 <= first <= last <= n_weeks):
        raise SystemExit(f"'{spec}': weeks must satisfy 1 <= first <= last <= {n_weeks}.")
    if not (0 < severity <= 1):
        raise SystemExit(f"'{spec}': severity must be above 0 and at most 1.")
    return {"name": name, "first": first, "last": last, "severity": severity}


def assign_weeks(df: pd.DataFrame, n_weeks: int, seed: int) -> pd.Series:
    """Whole patients go to one week; weeks have about equal numbers of stays. Fixed seed = repeatable."""
    rng = np.random.default_rng(seed)
    patients = np.array(df[PATIENT].unique())
    rng.shuffle(patients)
    rows = df[PATIENT].value_counts().reindex(patients).to_numpy()
    before = np.cumsum(rows) - rows
    week = np.minimum((before / rows.sum() * n_weeks).astype(int), n_weeks - 1) + 1
    return df[PATIENT].map(dict(zip(patients, week))).astype(int)


def covariate_shift(df, severity, rng, info):
    mask = rng.random(len(df)) < severity
    if "number_inpatient" in df:
        df.loc[mask, "number_inpatient"] = df.loc[mask, "number_inpatient"] + 3
    if "time_in_hospital" in df:
        df.loc[mask, "time_in_hospital"] = (df.loc[mask, "time_in_hospital"] + 3).clip(upper=14)
    return df, {"rows_changed": int(mask.sum())}


def drop_by_age(df, severity, rng, older: bool):
    """Drop a share `severity` of the stays of patients aged 60+ (older=True) or under 60 (older=False)."""
    if "age" not in df:
        return df, {"skipped": "no age column"}
    lower = df["age"].astype(str).str.extract(r"\[(\d+)")[0].astype(float)
    group = (lower >= 60).fillna(False).to_numpy() if older else (lower < 60).fillna(False).to_numpy()
    drop = group & (rng.random(len(df)) < severity)
    return df.loc[~drop].copy(), {"rows_dropped": int(drop.sum())}


def population_shift(df, severity, rng, info):          # the week becomes older and smaller
    return drop_by_age(df, severity, rng, older=False)


def population_shift_young(df, severity, rng, info):    # the week becomes younger and smaller
    return drop_by_age(df, severity, rng, older=True)


def new_categories(df, severity, rng, info):
    cats = info["categories"]
    cols = [c for c in NEW_CATEGORY_COLUMNS if c in cats] or sorted(cats)[:1]
    changed = {}
    for c in cols:
        known = set(cats[c])
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in known)
        new_value = 9999 if numeric else "REPLAY_NEW_VALUE"
        while new_value in known:                      # extremely unlikely, but never inject a value that is known
            new_value = new_value + 1 if numeric else str(new_value) + "_X"
        mask = rng.random(len(df)) < severity
        df[c] = df[c].astype(object)
        df.loc[mask, c] = new_value
        changed[c] = int(mask.sum())
    return df, {"rows_changed": changed}


DATA_SCENARIOS = {"covariate_shift": covariate_shift, "population_shift": population_shift,
                  "population_shift_young": population_shift_young, "new_categories": new_categories}


def corrupt_batch(records: list, k: int, info: dict) -> str:
    """Break one batch in place. The kinds rotate so all three rejection reasons show up."""
    numeric, required = info["numeric"], info["required"]
    kind = ["null", "not_numeric", "missing_column"][k % 3]
    if not numeric and kind != "missing_column":
        kind = "missing_column"
    if kind == "null":
        records[0][numeric[0]] = None
    elif kind == "not_numeric":
        records[0][numeric[min(1, len(numeric) - 1)]] = "abc"
    else:
        for r in records:
            r.pop(required[-1], None)
    return kind


# ---------------------------------------------------------------- examples
def info_from_holdout(df: pd.DataFrame) -> dict:
    """Stand-in for /model-info when the API is not running (only used by --show-examples)."""
    skip = {ENCOUNTER, PATIENT, "_week", "diag_1", "diag_2", "diag_3", *OUTCOME_COLUMNS}
    required = [c for c in df.columns if c not in skip]
    numeric = [c for c in required if pd.api.types.is_numeric_dtype(df[c])]
    categories = {c: sorted(df[c].astype(str).unique()) for c in required if c not in numeric}
    return {"required": required, "numeric": numeric, "categories": categories}


def show_examples(df: pd.DataFrame, injects: list, info: dict, seed: int, n: int = 3) -> None:
    """Print real before/after rows for each scenario, using the same functions as the real run."""
    reasons = {"null": "nulls", "not_numeric": "not_numeric", "missing_column": "missing_columns"}
    done = set()
    for s in injects:
        if s["name"] in done:
            continue
        done.add(s["name"])
        week = s["first"]
        base = df[df["_week"] == week][[ENCOUNTER] + info["required"]].copy()
        rng = np.random.default_rng([seed, week])
        print(f"\n=== {s['name']} (severity {s['severity']:.2f}); first active in week {week}, "
              f"which has {len(base):,} stays ===")
        if s["name"] == "bad_data":
            print(f"{s['severity']:.0%} of the requests (batches of stays) get one broken record. Three kinds rotate:")
            recs = json.loads(base.head(2).to_json(orient="records"))
            for k in range(3):
                broken = copy.deepcopy(recs)
                kind = corrupt_batch(broken, k, info)
                for key in recs[0]:
                    if key not in broken[0]:
                        print(f"  {kind:<15} column '{key}' removed from every record in the batch  ->  API answers 422 '{reasons[kind]}'")
                    elif broken[0][key] != recs[0][key]:
                        print(f"  {kind:<15} record 1, '{key}': {recs[0][key]!r} -> {broken[0][key]!r}  ->  API answers 422 '{reasons[kind]}'")
            continue
        after, notes = DATA_SCENARIOS[s["name"]](base.copy(), s["severity"], rng, info)
        print("Counts:", notes)
        if s["name"].startswith("population_shift"):
            table = pd.DataFrame({"before": base["age"].value_counts(), "after": after["age"].value_counts()}).fillna(0).astype(int)
            table["share_before"] = (table["before"] / table["before"].sum()).map("{:.1%}".format)
            table["share_after"] = (table["after"] / table["after"].sum()).map("{:.1%}".format)
            print(table.sort_index().to_string())
            continue
        common = base.index.intersection(after.index)
        differs = (base.loc[common].astype(str) != after.loc[common].astype(str))
        shown = 0
        for idx in common[differs.any(axis=1).to_numpy()][:n]:
            cols = [c for c in differs.columns if differs.loc[idx, c]]
            changes = ", ".join(f"{c}: {base.loc[idx, c]} -> {after.loc[idx, c]}" for c in cols)
            print(f"  encounter {base.loc[idx, ENCOUNTER]}: {changes}")
            shown += 1
        if not shown:
            print("  (no row changed; check that the scenario's columns exist in the data)")
    print("\nThe holdout file itself is never modified: these changes happen on an in-memory copy of each week.")


# ---------------------------------------------------------------- one request
def post_predict(base, records, run_id, week, explain):
    started = time.perf_counter()
    status, text = call("POST", base + "/predict", {"records": records, "explain": explain, "top_k": 3},
                        {"X-Replay-Run": run_id, "X-Replay-Week": str(week)})
    return status, as_json(text), time.perf_counter() - started


def reject_reason(body: dict) -> str:
    detail = body.get("detail", body)
    return str(detail.get("error", "unknown")) if isinstance(detail, dict) else "unknown"


# ---------------------------------------------------------------- reports
def write_summary(out_dir: Path, args, weeks_done: list, model_versions: set, stopped: str) -> None:
    clean = [w for w in weeks_done if not w["scenarios"] and w["scored"]]
    clean_share = (sum(w["flagged"] for w in clean) / sum(w["scored"] for w in clean)) if clean else None
    summary = {"run_id": args.run_id, "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "stopped": stopped,
               "model_versions_seen": sorted(model_versions), "seed": args.seed,
               "seconds_per_week": args.seconds_per_week, "clean_week_flagged_share": clean_share,
               "weeks": weeks_done}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    pct = lambda x: "-" if x is None else f"{x:.1%}"
    lines = [f"# Replay summary: {args.run_id}", "",
             f"Created {summary['created']}. Status: {stopped}. Model version(s) seen: "
             f"{', '.join(map(str, sorted(model_versions))) or 'none'}. Seed {args.seed}, "
             f"{args.seconds_per_week} s per simulated week.", "",
             f"Flagged share in clean weeks: {pct(clean_share)} (the cutoff is set so that about 20% of "
             f"reference stays are flagged; the alert band is {FLAG_LOW:.0%} to {FLAG_HIGH:.0%}).", "",
             "| Week | Injected | Stays scored | Requests | Rejected (422) | Flagged | Mean score | Unseen values (API counter change) |",
             "|---|---|---|---|---|---|---|---|"]
    for w in weeks_done:
        inj = ", ".join(f"{s['name']} {s['severity']:.2f}" for s in w["scenarios"]) or "none"
        flagged = pct(w["flagged"] / w["scored"]) if w["scored"] else "-"
        if w["scored"] and not FLAG_LOW <= w["flagged"] / w["scored"] <= FLAG_HIGH:
            flagged += " (outside band)"
        mean = f"{w['mean_score']:.3f}" if w["mean_score"] is not None else "-"
        lines.append(f"| {w['week']} | {inj} | {w['scored']:,} | {w['requests']} | {w['rejected_requests']} | "
                     f"{flagged} | {mean} | {w['api_unseen_change']:.0f} |")
    injected = sorted({s["name"] for w in weeks_done for s in w["scenarios"]})
    if injected:
        lines += ["", "## What to look at", "",
                  "Grafana (http://localhost:3000, dashboard Readmission API health), Prometheus alerts "
                  "(http://localhost:9090/alerts) and the console's Monitoring tab.", ""]
        lines += [f"- **{n}:** {EXPECT[n]}" for n in injected]
        lines += ["", "Record in your notes: which scenario, the first week it was visible, which alert fired and how "
                  "long after the first affected week. A scenario nothing reacted to is a monitoring gap, "
                  "which is a finding, not a failure of the replay."]
    lines += ["", "## Limits", "",
              "- Everything above is label-free: it can show that inputs or scores moved, not that predictions got worse. "
              "Real performance needs the late labels in `labels.csv` (step 14).",
              "- Weeks are simulated by shuffling whole patients; the data has no dates.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- main
def main() -> None:
    root = find_project_root()
    ap = argparse.ArgumentParser(description="Replay the holdout through the running API, week by week.")
    ap.add_argument("--csv", default=str(root / "data" / "splits" / "holdout.csv"))
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--run-id", default=f"replay-{datetime.now():%Y%m%d-%H%M}",
                    help="1 to 64 characters: letters, digits, _ . -  (sent as X-Replay-Run, names the output folder)")
    ap.add_argument("--weeks", type=int, default=12)
    ap.add_argument("--start-week", type=int, default=1, help="resume a stopped run from this week")
    ap.add_argument("--seconds-per-week", type=float, default=60, help="pause so each week lasts this long (0 = no pause)")
    ap.add_argument("--batch-size", type=int, default=100, help="stays per request (the API accepts up to 500)")
    ap.add_argument("--inject", action="append", default=[], metavar="NAME[:FIRST[-LAST]][:SEVERITY]")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="a ready-made timeline (adds its --inject items)")
    ap.add_argument("--explain", action="store_true", help="also compute SHAP drivers (much slower, not needed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(root / "data" / "replay"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan and send nothing")
    ap.add_argument("--show-examples", action="store_true",
                    help="print real before/after rows for each scenario, send nothing (uses /model-info if the API is up)")
    args = ap.parse_args()

    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit("--run-id must be 1 to 64 characters: letters, digits, _ . -")
    if args.weeks < 1 or not 1 <= args.start_week <= args.weeks:
        raise SystemExit("--weeks must be at least 1 and --start-week must be between 1 and --weeks.")
    if not 1 <= args.batch_size <= 500:
        raise SystemExit("--batch-size must be between 1 and 500.")
    if 0 < args.seconds_per_week < MIN_WEEK_SECONDS:
        print(f"WARNING: weeks shorter than {MIN_WEEK_SECONDS} s may fall between Prometheus scrapes (every 15 s) "
              "and not show on charts.")
    specs = (PRESETS[args.preset] if args.preset else []) + args.inject
    injects = [parse_inject(s, args.weeks) for s in specs]

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run code/data_split.py first.")
    df = pd.read_csv(csv_path, dtype={ENCOUNTER: str, PATIENT: str})
    for needed in (ENCOUNTER, PATIENT, LABEL):
        if needed not in df.columns:
            raise SystemExit(f"{csv_path.name} has no '{needed}' column.")
    df["_week"] = assign_weeks(df, args.weeks, args.seed)

    plan = []
    for w in range(1, args.weeks + 1):
        sub = df[df["_week"] == w]
        plan.append({"week": w, "stays": int(len(sub)), "patients": int(sub[PATIENT].nunique()),
                     "positive_rate": round(float(sub[LABEL].mean()), 4),
                     "scenarios": [s for s in injects if s["first"] <= w <= s["last"]]})
    print(f"Holdout: {len(df):,} stays, {df[PATIENT].nunique():,} patients, {args.weeks} weeks, run '{args.run_id}'")
    for p in plan:
        inj = ", ".join(f"{s['name']} {s['severity']:.2f}" for s in p["scenarios"]) or "clean"
        print(f"  week {p['week']:>2}: {p['stays']:>6,} stays, {p['patients']:>6,} patients   {inj}")
    if args.show_examples:
        if not injects:
            raise SystemExit("Nothing to show: add --preset demo or --inject NAME:FIRST-LAST.")
        info, source = None, "the live API"
        try:
            st, txt = call("GET", args.url.rstrip("/") + "/model-info", timeout=5)
            if st == 200:
                m = as_json(txt)["features"]
                info = {"required": m["required"], "numeric": m["numeric"], "categories": m["categories"]}
        except SystemExit:
            pass
        if info is None:
            info, source = info_from_holdout(df), "the holdout file (API not reachable)"
        print(f"\nColumn information taken from {source}.")
        show_examples(df, injects, info, args.seed)
        return
    if args.dry_run:
        print("\nDry run: nothing was sent and nothing was written.")
        return

    base = args.url.rstrip("/")
    status, text = call("GET", base + "/health", timeout=10)
    if status != 200:
        raise SystemExit(f"The API is not healthy (HTTP {status}). Export a bundle and reload: "
                         "python serving/export_champion.py, then POST /reload.")
    status, text = call("GET", base + "/model-info", timeout=30)
    if status != 200:
        raise SystemExit(f"/model-info failed (HTTP {status}).")
    meta = as_json(text)
    info = {"required": meta["features"]["required"], "numeric": meta["features"]["numeric"],
            "categories": meta["features"]["categories"]}
    expected_share = (meta.get("flag_rule") or {}).get("top_fraction") or 0.2
    missing = [c for c in info["required"] if c not in df.columns]
    if missing:
        raise SystemExit(f"The holdout lacks columns the live model needs: {missing}")
    print(f"\nLive model: {meta['model'].get('registered_model')} version {meta['model'].get('version')} "
          f"({meta['model'].get('model_family')}); expected flagged share about {expected_share:.0%}.")

    out_dir = Path(args.out_dir) / args.run_id
    if out_dir.exists() and args.start_week == 1:
        raise SystemExit(f"{out_dir} already exists. Choose another --run-id, or use --start-week to resume.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps({"args": vars(args), "weeks": plan}, indent=2, default=str), encoding="utf-8")
    ledger = root / "models" / "data_use_log.jsonl"
    ledger.parent.mkdir(exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "data_name": "holdout",
                            "data_file": str(csv_path), "rows": int(len(df)),
                            "purpose": f"replay run {args.run_id}: {args.weeks} weeks, scenarios "
                                       f"{[s['name'] for s in injects] or 'none'}"}) + "\n")

    labels_path = out_dir / "labels.csv"
    send_cols = [ENCOUNTER] + info["required"]
    weeks_done, versions, stopped = [], set(), "finished"
    if args.start_week > 1 and (out_dir / "summary.json").exists():
        weeks_done = json.loads((out_dir / "summary.json").read_text(encoding="utf-8")).get("weeks", [])
        weeks_done = [w for w in weeks_done if w["week"] < args.start_week]
    if args.start_week > 1 and labels_path.exists():     # a resumed week is redone, so drop its old labels
        old = pd.read_csv(labels_path, dtype={ENCOUNTER: str, PATIENT: str})
        old[old["week"] < args.start_week].to_csv(labels_path, index=False)

    try:
        for week in range(args.start_week, args.weeks + 1):
            began = time.monotonic()
            part = df[df["_week"] == week]
            active = [s for s in injects if s["first"] <= week <= s["last"]]
            rng = np.random.default_rng([args.seed, week])
            feats = part[send_cols].copy()
            for s in active:
                if s["name"] in DATA_SCENARIOS:
                    feats, _ = DATA_SCENARIOS[s["name"]](feats, s["severity"], rng, info)
            bad = next((s for s in active if s["name"] == "bad_data"), None)
            records = json.loads(feats.to_json(orient="records"))

            before = api_counters(base)
            scores, flags, scored_ids, latencies = [], [], [], []
            requests = rejected = 0
            reasons, unseen_requests, failed = {}, 0, None
            for k, start in enumerate(range(0, len(records), args.batch_size)):
                chunk = records[start:start + args.batch_size]
                if bad and rng.random() < bad["severity"]:
                    broken = copy.deepcopy(chunk)
                    kind = corrupt_batch(broken, k, info)
                    st, body, secs = post_predict(base, broken, args.run_id, week, args.explain)
                    requests += 1
                    if st == 200:
                        reasons[f"accepted_{kind}"] = reasons.get(f"accepted_{kind}", 0) + 1
                    else:
                        rejected += 1
                        reasons[reject_reason(body)] = reasons.get(reject_reason(body), 0) + 1
                st, body, secs = post_predict(base, chunk, args.run_id, week, args.explain)
                requests += 1
                if st != 200:
                    failed = f"HTTP {st}: {json.dumps(body)[:400]}"
                    break
                latencies.append(secs)
                versions.add(body["model"]["version"])
                for p in body["predictions"]:
                    scores.append(p["score"])
                    flags.append(bool(p["high_risk"]))
                    scored_ids.append(p["id"])
                if body.get("unseen_categories"):
                    unseen_requests += 1

            after = api_counters(base)
            if scored_ids:
                lab = part.loc[part[ENCOUNTER].isin(set(scored_ids)), [ENCOUNTER, PATIENT, LABEL]].copy()
                lab["week"] = week
                lab["scenario"] = "+".join(s["name"] for s in active) or "none"
                lab.to_csv(labels_path, mode="a", header=not labels_path.exists(), index=False)
            result = {"week": week, "scenarios": active, "stays_planned": int(len(feats)), "scored": len(scores),
                      "flagged": int(sum(flags)), "mean_score": float(np.mean(scores)) if scores else None,
                      "requests": requests, "rejected_requests": rejected, "reject_reasons": reasons,
                      "requests_with_unseen_values": unseen_requests,
                      "api_unseen_change": after["unseen"] - before["unseen"],
                      "api_rejected_change": after["rejected"] - before["rejected"],
                      "mean_request_seconds": float(np.mean(latencies)) if latencies else None,
                      "error": failed}
            weeks_done.append(result)
            share = result["flagged"] / result["scored"] if result["scored"] else None
            tag = ", ".join(f"{s['name']} {s['severity']:.2f}" for s in active) or "clean"
            line = (f"week {week:>2} [{tag}] scored {result['scored']:,}/{result['stays_planned']:,} in {requests} requests; "
                    f"flagged {'-' if share is None else f'{share:.1%}'} (expected ~{expected_share:.0%}); "
                    f"rejected {rejected}; unseen change {result['api_unseen_change']:.0f}")
            if share is not None and not FLAG_LOW <= share <= FLAG_HIGH:
                line += "   <-- outside the alert band"
            print(line)
            if failed:
                stopped = f"stopped in week {week}: {failed}"
                print("STOPPED: " + failed)
                break
            wait = args.seconds_per_week - (time.monotonic() - began)
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        stopped = "interrupted"
        print("\nInterrupted. Writing what was done so far.")

    write_summary(out_dir, args, weeks_done, versions, stopped)
    if len(versions) > 1:
        print(f"NOTE: the live model changed during the run (versions seen: {sorted(versions)}).")
    print(f"\nWrote {out_dir}/ : plan.json, labels.csv, summary.json, summary.md")
    print("Late labels for step 14 are in labels.csv. Resume a stopped run with --start-week N and the same --run-id.")


if __name__ == "__main__":
    main()