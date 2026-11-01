"""
Retrain decision (plan step 15): do the labelled results say the live model has gone stale?

Run from the project root, after a replay, drift_report.py and performance_report.py:
    python retrain.py decide --run-id demo1
    python retrain.py decide --run-id demo1 --scope all            # judge on all weeks, not only the clean ones
    python retrain.py decide --run-id demo1 --force-retrain        # a person overrides the evidence (recorded as such)

Reads (read-only)
    data/.../performance_summary.json   written by monitoring/performance_report.py (found by run id, or --performance)
    data/drift/<run-id>/drift_summary.json   optional; used only to tell a data problem from a model problem
    models/champion.json                which model version is live (or --champion-version)
Writes
    data/retrain/<retrain-id>/trigger.json   the decision, the reason and the numbers behind it

The rule (written before any result is looked at; do not loosen it afterwards)
    Only weeks scored by the LIVE champion's version count (a model swap mid-run would blur the answer).
    "Clean" weeks are weeks with no injected test. In real production every week is clean.
    1. Fewer than MIN_WEEKS labelled weeks pooled              -> keep        (not enough evidence)
    2. Pooled PR-AUC range NOT entirely below the reference    -> keep        (drift without a performance drop needs no retrain)
    3. Pooled range entirely below the reference:
         a. a labelled week shows never-seen category values   -> fix_data_first (do not retrain on a data problem)
         b. otherwise (with or without input drift)            -> retrain
    "Below the reference" means the OPTIMISTIC end (95% upper bound) of the pooled range is under the model's
    test-split PR-AUC from the evaluation gate.

--force-retrain turns any decision into "retrain" but keeps the original decision in the file and marks
the trigger as forced by a person, so the audit trail never pretends the evidence demanded it.
This script retrains nothing. The Prefect flow (orchestration/retrain_flow.py) does the work.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

MIN_WEEKS = 4                    # labelled weeks that must be pooled before a verdict is possible
UNSEEN_MARK = "__unseen__"       # drift_report.py writes this bucket name when a never-seen value appears


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find a data/ folder next to or above {here}")


ROOT = find_project_root()
DATA = ROOT / "data"
MODELS = ROOT / "models"


def find_performance(run_id: str, explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"Missing {p}.")
        return p
    hits = [p for p in DATA.rglob("performance_summary.json") if p.parent.name == run_id]
    if not hits:
        raise SystemExit(f"No performance_summary.json for run '{run_id}' under {DATA}. "
                         f"Run: python monitoring/performance_report.py --run-id {run_id}   (or pass --performance <file>)")
    return sorted(hits, key=lambda p: p.stat().st_mtime)[-1]


def champion_version(explicit) -> int:
    if explicit is not None:
        return int(explicit)
    cj = MODELS / "champion.json"
    if not cj.exists():
        raise SystemExit("models/champion.json not found. Pass --champion-version N.")
    return int(json.loads(cj.read_text(encoding="utf-8"))["version"])


def champion_family():
    """Model family of the live champion, from models/champion.json (None if the file or field is missing)."""
    cj = MODELS / "champion.json"
    return json.loads(cj.read_text(encoding="utf-8")).get("model_family") if cj.exists() else None


def unseen_weeks(run_id: str, weeks: list) -> list:
    """Weeks (of the given list) whose drift detail mentions a never-seen category value. [] if no drift summary."""
    path = DATA / "drift" / run_id / "drift_summary.json"
    if not path.exists():
        return []
    drift = json.loads(path.read_text(encoding="utf-8"))
    hit = []
    for w in drift.get("weeks", []):
        if w["week"] in weeks and any(UNSEEN_MARK in str(t.get("what", "")) for t in w.get("top", [])):
            hit.append(w["week"])
    return hit


def reference_for(entry: dict, family=None) -> tuple:
    """The model's test-split PR-AUC from the gate: the performance summary's value, else models/evaluation.json."""
    ref = entry.get("reference_pr_auc")
    if ref is not None:
        return float(ref), "performance_summary.json"
    ev = MODELS / "evaluation.json"
    if ev.exists():
        res = json.loads(ev.read_text(encoding="utf-8")).get("results", {}).get(family or entry.get("family"))
        if res and res.get("test_pr_auc") is not None:
            return float(res["test_pr_auc"]), "models/evaluation.json (the gate's test PR-AUC for this model family)"
    return None, None


def decide(perf: dict, version: int, scope: str, run_id: str, family=None) -> dict:
    out = {"champion_version": version, "scope": scope, "min_weeks": MIN_WEEKS}
    entry = next((b for b in perf.get("by_version", []) if b["version"] == version), None)
    if entry is None:
        out.update(decision="keep", reason=f"No labelled stays were scored by version {version} in this run, so there is nothing to judge.")
        return out
    pool = entry.get("clean" if scope == "clean" else "all")
    weeks = [w for w in perf["weeks"] if w["versions"] == [version] and (scope == "all" or not w["scenarios"])]
    wk = [w["week"] for w in weeks]
    fam = entry.get("family")
    if fam in (None, "", "?"):          # the performance report could not name the model: use the live champion's family
        fam = family
    out.update(model_family=fam, weeks_used=wk, n_weeks=len(wk))
    if not pool or len(wk) < MIN_WEEKS:
        out.update(decision="keep", reason=f"Only {len(wk)} labelled {scope} week(s) for version {version}; the rule needs at least {MIN_WEEKS}.")
        return out

    ref, ref_source = reference_for(entry, fam)
    if ref is None:
        out.update(decision="keep", reason=(
            f"No reference PR-AUC is available for version {version} ({fam}): it is missing from the performance "
            "summary and from models/evaluation.json, so the pooled result cannot be judged. Check monitoring/performance_report.py."))
        return out
    below = pool["pr_hi"] < ref
    out["evidence"] = {"stays": pool["n"], "pooled_pr_auc": round(pool["pr_auc"], 4),
                       "pooled_range": [round(pool["pr_lo"], 4), round(pool["pr_hi"], 4)],
                       "reference_pr_auc": round(ref, 4), "reference_source": ref_source,
                       "below_reference": bool(below)}
    readings = [w["reading"] for w in weeks]
    out["week_readings"] = {r: readings.count(r) for r in sorted(set(readings))}
    drift_weeks = [w["week"] for w in weeks if w["drift"]["major"] or w["drift"]["moderate"]]
    out["weeks_with_input_drift"] = drift_weeks
    bad_codes = unseen_weeks(run_id, wk)
    out["weeks_with_unseen_categories"] = bad_codes

    if not below:
        extra = (f" Weeks {bad_codes} show never-seen category values: worth fixing the data feed, but not a reason to retrain."
                 if bad_codes else "")
        out.update(decision="keep", reason=(
            f"Pooled PR-AUC {pool['pr_auc']:.3f} (range {pool['pr_lo']:.3f} to {pool['pr_hi']:.3f}) is not entirely below the "
            f"reference {ref:.3f}. Drift without a performance drop needs no retrain." + extra))
    elif bad_codes:
        out.update(decision="fix_data_first", reason=(
            f"Pooled PR-AUC {pool['pr_auc']:.3f} (upper bound {pool['pr_hi']:.3f}) is below the reference {ref:.3f}, but weeks {bad_codes} "
            "contain never-seen category values. Check the data feed or the code mapping first; do not retrain on a data problem."))
    else:
        out.update(decision="retrain", reason=(
            f"Pooled PR-AUC {pool['pr_auc']:.3f} (upper bound {pool['pr_hi']:.3f}) is entirely below the reference {ref:.3f} "
            f"over {len(wk)} labelled weeks, and no data problem was found."))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 15: decide whether to retrain.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide", help="apply the retrain rule to a performance summary")
    d.add_argument("--run-id", required=True, help="the replay run whose labelled results are judged")
    d.add_argument("--performance", default=None, help="path to performance_summary.json (default: found by run id)")
    d.add_argument("--champion-version", default=None, help="registry version that is live (default: from models/champion.json)")
    d.add_argument("--scope", choices=["clean", "all"], default="clean", help="which weeks to pool (default: clean weeks)")
    d.add_argument("--force-retrain", action="store_true", help="override the evidence; recorded as forced by a person")
    d.add_argument("--id", default=f"retrain-{datetime.now():%Y%m%d-%H%M}", help="name of the output folder")
    args = ap.parse_args()

    perf_path = find_performance(args.run_id, args.performance)
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    result = decide(perf, champion_version(args.champion_version), args.scope, args.run_id, champion_family())
    result.update(retrain_id=args.id, run_id=args.run_id, performance_file=str(perf_path),
                  created=f"{datetime.now():%Y-%m-%d %H:%M:%S}", forced=False)
    if args.force_retrain:
        result.update(forced=True, evidence_decision=result["decision"], decision="retrain",
                      reason="FORCED by a person (--force-retrain). What the evidence said: " + result["reason"])
    out_dir = DATA / "retrain" / args.id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trigger.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Run '{args.run_id}', live champion version {result['champion_version']}"
          f" ({result.get('model_family', 'unknown')}), scope: {args.scope}")
    if "evidence" in result:
        e = result["evidence"]
        print(f"  pooled over {result['n_weeks']} weeks / {e['stays']:,} stays: PR-AUC {e['pooled_pr_auc']} "
              f"(range {e['pooled_range'][0]} to {e['pooled_range'][1]}), reference {e['reference_pr_auc']}")
        print(f"  week readings: {result['week_readings']}")
        print(f"  weeks with input drift: {result['weeks_with_input_drift'] or 'none'}; "
              f"with never-seen categories: {result['weeks_with_unseen_categories'] or 'none'}")
    print(f"  {result['reason']}")
    print(f"\nDECISION: {result['decision'].upper()}" + ("  (forced)" if result["forced"] else ""))
    print(f"Saved: {out_dir / 'trigger.json'}")


if __name__ == "__main__":
    main()