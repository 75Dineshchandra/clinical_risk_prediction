"""
Drift report (plan step 13): did the INPUTS and SCORES of a replay week move away from normal?

The replay (step 11) showed that score-based alerts miss some changes, for example a different patient mix. This
script looks at every input feature, week by week, against a baseline of clean weeks, and says which features moved
and by how much. It reads what the API already logged; it needs no new data and does not touch the API, the registry
or any model.

Run from the project root, after a replay:
    python monitoring/drift_report.py --run-id demo1                      # baseline = weeks 1-4
    python monitoring/drift_report.py --run-id demo1 --baseline-weeks 1-4 --top 8
    python monitoring/drift_report.py --run-id drill3 --baseline-weeks 1-2

Reads
    serving/logs/predictions.jsonl        every scored stay: week, model version, score, flag and the input features
    data/replay/<run-id>/plan.json        (optional) which test was injected in which week, for the "caught it?" table

Writes   data/drift/<run-id>/
    drift_report.md       the report: weekly summary, "did we catch it" table, details of the weeks that drifted
    psi_by_week.csv       PSI of every feature (and the score) for every week
    drift_summary.json    the same in machine-readable form
    drift_heatmap.png     weeks across, features down, coloured by drift level

The measure: PSI (population stability index)
    Split a feature into bins using the BASELINE data. For each bin compare the share of stays in the week (a) with the
    share in the baseline (e). PSI = sum over bins of (a - e) * ln(a / e). 0 means identical. Common rule of thumb:
        below 0.10  stable        0.10 to 0.25  moderate        above 0.25  major
    Numeric features with many values use the baseline's deciles as bins. Features with few values and categories are
    compared value by value. For categories there are two extra buckets: "__rare__" (seen in the baseline, but under 0.5%)
    and "__unseen__" (never seen in the baseline), so a brand-new code shows up on its own.

Read the thresholds together with the NOISE FLOOR printed in the report: the PSI you get from a clean sample of the same
size as a week. A week is only interesting when it is clearly above that.

What it cannot do
    - It shows that inputs or scores MOVED, not that predictions got WORSE. That needs the late labels (step 14).
    - Rejected requests (bad data) are never logged, so this report cannot see them; the error-share alert covers that.
    - When the model version changes mid-run, the score PSI jumps because a different model produces different scores.
      That is expected and is not input drift; the report prints the model version of every week.

The PSI functions live in monitoring/drift_lib.py, shared with the live exporter (drift_exporter.py).

Needs: pandas, numpy, matplotlib.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from drift_lib import MAJOR, MODERATE, N_BINS, SCORE, describe, level, make_profile, psi, shares   # shared PSI code

META = ["replay_week", "model_version", "id", "score", "high_risk"]
DETECT_FEATURES = {     # which features should react to which injected test
    "covariate_shift": ["number_inpatient", "time_in_hospital"],
    "new_categories": ["discharge_disposition_id", "medical_specialty"],
    "population_shift": ["age"],
    "population_shift_young": ["age"],
}


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find a data/ folder next to or above {here}")


# ---------------------------------------------------------------- loading
def parse_weeks(text: str) -> list:
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def load_log(path: Path, run_id: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("run_id") != run_id or obj.get("replay_week") is None:
                continue
            row = {k: obj.get(k) for k in META}
            row.update(obj.get("features") or {})
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["id", "replay_week"], keep="last")      # a resumed week is logged twice
    df["replay_week"] = df["replay_week"].astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- main
def main() -> None:
    root = find_project_root()
    ap = argparse.ArgumentParser(description="Per-week drift report (PSI) for a replay run.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline-weeks", default="1-4", help="clean weeks that define normal, e.g. 1-4 or 1,2,3")
    ap.add_argument("--log", default=str(root / "serving" / "logs" / "predictions.jsonl"))
    ap.add_argument("--out-dir", default=str(root / "data" / "drift"))
    ap.add_argument("--top", type=int, default=5, help="how many drifting features to list per week")
    ap.add_argument("--min-rows", type=int, default=300, help="weeks with fewer stays are marked as too small to judge")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Missing {log_path}. Run a replay first (python serving/replay.py ...).")
    df = load_log(log_path, args.run_id)
    if df.empty:
        raise SystemExit(f"No logged stays with run_id '{args.run_id}' in {log_path}.")
    baseline_weeks = parse_weeks(args.baseline_weeks)
    weeks = sorted(df["replay_week"].unique())
    base = df[df["replay_week"].isin(baseline_weeks)]
    if len(base) < args.min_rows:
        raise SystemExit(f"The baseline weeks {baseline_weeks} hold only {len(base)} stays in the log (need at least "
                         f"{args.min_rows}). Check --baseline-weeks.")
    features = [c for c in df.columns if c not in META]
    print(f"Run '{args.run_id}': {len(df):,} logged stays in weeks {weeks[0]}-{weeks[-1]}; baseline weeks {baseline_weeks} "
          f"= {len(base):,} stays; {len(features)} features.")

    # ---- baseline profiles and shares
    profiles = {f: make_profile(base[f]) for f in features}
    profiles[SCORE] = make_profile(base["score"])
    base_shares = {f: shares(profiles[f], base[f]) for f in features}
    base_shares[SCORE] = shares(profiles[SCORE], base["score"])
    source = {f: f for f in features}
    source[SCORE] = "score"

    # ---- noise floor: a clean sample of typical week size against the rest of the baseline
    rng = np.random.default_rng(args.seed)
    typical = int(df.groupby("replay_week").size().median())
    n = int(min(typical, len(base) // 2))
    order = rng.permutation(len(base))
    sample, rest = base.iloc[order[:n]], base.iloc[order[n:]]
    noise = []
    for f in features + [SCORE]:
        col = source[f]
        prof = make_profile(rest[col])
        noise.append(psi(shares(prof, rest[col]), shares(prof, sample[col])))
    noise_median, noise_max = float(np.median(noise)), float(np.max(noise))
    print(f"Noise floor (clean sample of {n:,} stays vs the rest of the baseline): median PSI {noise_median:.3f}, "
          f"highest {noise_max:.3f}.")

    # ---- PSI per week
    table, details, weekly = {}, {}, []
    for w in weeks:
        part = df[df["replay_week"] == w]
        row, det = {}, {}
        for f in features + [SCORE]:
            col = source[f]
            a = shares(profiles[f], part[col])
            row[f] = psi(base_shares[f], a)
            det[f] = describe(profiles[f], base[col], part[col], base_shares[f], a)
        table[w] = row
        details[w] = det
        versions = sorted(part["model_version"].dropna().astype(int).unique().tolist())
        flagged = float(part["high_risk"].astype(bool).mean())
        ranked = sorted(((v, f) for f, v in row.items() if f != SCORE), reverse=True)
        weekly.append({"week": int(w), "stays": int(len(part)), "baseline": w in baseline_weeks,
                       "model_versions": versions, "flagged_share": round(flagged, 4),
                       "score_psi": round(row[SCORE], 4),
                       "n_moderate": sum(MODERATE <= v < MAJOR for v, _ in ranked),
                       "n_major": sum(v >= MAJOR for v, _ in ranked),
                       "top": [{"feature": f, "psi": round(v, 4), "what": details[w][f]} for v, f in ranked[:args.top]
                               if v >= MODERATE]})
    psi_df = pd.DataFrame(table).T
    psi_df.index.name = "week"

    # ---- which injected test was in which week (optional)
    plan_path = root / "data" / "replay" / args.run_id / "plan.json"
    injected = {}
    if plan_path.exists():
        for wk in json.loads(plan_path.read_text(encoding="utf-8")).get("weeks", []):
            injected[int(wk["week"])] = [s["name"] for s in wk.get("scenarios", [])]
    for rec in weekly:
        rec["injected"] = injected.get(rec["week"], [])

    checks = []
    if injected:
        for rec in weekly:
            w = rec["week"]
            if rec["baseline"]:
                continue
            small = rec["stays"] < args.min_rows
            if not rec["injected"]:
                verdict = "too small to judge" if small else (
                    f"FALSE ALARM: {rec['n_major']} feature(s) above {MAJOR}" if rec["n_major"] else "ok (nothing major)")
                checks.append({"week": w, "injected": "none (clean)", "verdict": verdict})
                continue
            for name in rec["injected"]:
                if name == "bad_data":
                    checks.append({"week": w, "injected": name,
                                   "verdict": "not visible here: rejected requests are never logged (the error-share alert covers it)"})
                    continue
                feats = [f for f in DETECT_FEATURES.get(name, []) if f in psi_df.columns]
                best = max(((psi_df.loc[w, f], f) for f in feats), default=(0.0, "-"))
                hit = best[0] >= MODERATE and not small
                checks.append({"week": w, "injected": name,
                               "verdict": f"{'CAUGHT' if hit else 'MISSED'}: {best[1]} PSI {best[0]:.2f} ({level(best[0])})"})

    # ---- outputs
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    psi_df.round(4).to_csv(out_dir / "psi_by_week.csv")
    (out_dir / "drift_summary.json").write_text(json.dumps({
        "run_id": args.run_id, "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "baseline_weeks": baseline_weeks,
        "baseline_stays": int(len(base)), "thresholds": {"moderate": MODERATE, "major": MAJOR},
        "noise_floor": {"sample_size": n, "median_psi": round(noise_median, 4), "max_psi": round(noise_max, 4)},
        "weeks": weekly, "checks": checks}, indent=2, default=str), encoding="utf-8")

    after = psi_df.loc[[w for w in weeks if w not in baseline_weeks]]
    focus = (after if len(after) else psi_df).drop(columns=[SCORE]).max().sort_values(ascending=False).index[:25].tolist()
    grid = psi_df[focus + [SCORE]].T.rename(index={SCORE: "SCORE"})
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(weeks), 0.35 * len(grid) + 1.6))
    cmap = ListedColormap(["#2a9d6f", "#e9b949", "#d64545"])
    ax.imshow(grid.to_numpy(), aspect="auto", cmap=cmap, norm=BoundaryNorm([0, MODERATE, MAJOR, 1e9], 3))
    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels([f"{w}*" if w in baseline_weeks else str(w) for w in weeks])
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels(grid.index, fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            if v >= MODERATE:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xlabel("simulated week (* = baseline)")
    ax.set_title(f"Drift by week, run {args.run_id}: green stable, amber moderate (>=0.10), red major (>=0.25)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "drift_heatmap.png", dpi=130)
    plt.close(fig)

    lines = [f"# Drift report: {args.run_id}", "",
             f"Created {datetime.now():%Y-%m-%d %H:%M}. Generated by `monitoring/drift_report.py` from `{log_path.name}`.", "",
             f"- Baseline (normal): weeks {', '.join(map(str, baseline_weeks))}, {len(base):,} stays.",
             f"- Measure: PSI per feature against the baseline. Stable below {MODERATE}, moderate {MODERATE} to {MAJOR}, major above {MAJOR}.",
             f"- **Noise floor:** a clean sample of {n:,} stays (one typical week) gives a median PSI of {noise_median:.3f} and a "
             f"highest PSI of {noise_max:.3f} across {len(features) + 1} features. Only values clearly above that mean something.",
             "- PSI shows that inputs or scores moved, not that predictions got worse (that needs the late labels, step 14).", "",
             "## Weekly summary", "",
             "| Week | Role | Injected | Stays | Model | Flagged | Score PSI | Moderate | Major | Most drifted features (PSI) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for rec in weekly:
        top = ", ".join(f"{t['feature']} {t['psi']:.2f}" for t in rec["top"]) or "-"
        role = "baseline" if rec["baseline"] else ("too small" if rec["stays"] < args.min_rows else "test")
        lines.append(f"| {rec['week']} | {role} | {', '.join(rec['injected']) or '-'} | {rec['stays']:,} | "
                     f"{'/'.join(map(str, rec['model_versions'])) or '-'} | {rec['flagged_share']:.1%} | "
                     f"{rec['score_psi']:.2f} | {rec['n_moderate']} | {rec['n_major']} | {top} |")
    if len({v for r in weekly for v in r["model_versions"]}) > 1:
        lines += ["", "The model version changed during this run. The score PSI of those weeks jumps because a different "
                  "model produces different scores; that is not input drift."]
    if checks:
        lines += ["", "## Did the drift report catch each injected test?", "",
                  "| Week | Injected | Result |", "|---|---|---|"]
        lines += [f"| {c['week']} | {c['injected']} | {c['verdict']} |" for c in checks]
    lines += ["", "## Details of the weeks that drifted", ""]
    drifted = [r for r in weekly if not r["baseline"] and r["top"]]
    if not drifted:
        lines.append("No week had a feature at or above the moderate level.")
    for rec in drifted:
        lines += [f"### Week {rec['week']} ({', '.join(rec['injected']) or 'clean'})", ""]
        lines += [f"- `{t['feature']}`: PSI {t['psi']:.2f} ({level(t['psi'])}); {t['what']}" for t in rec["top"]]
        lines.append("")
    lines += ["## Files", "",
              "- `psi_by_week.csv`: PSI of every feature and the score, every week.",
              "- `drift_summary.json`: the same in machine-readable form.",
              "- `drift_heatmap.png`: the 25 most drifted features by week.", "",
              "## How to act on this", "",
              "Drift is a symptom, not a verdict. First check whether the data is wrong (a broken feed or a renamed code): fix it, "
              "do not retrain. If the data is right, check real performance on the late labels (step 14). Only if performance "
              "dropped, retrain, and let the new model enter as a challenger through the gate and promotion rule.", ""]
    (out_dir / "drift_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{'week':>4} {'role':<9} {'injected':<26} {'stays':>6} {'score PSI':>9} {'mod':>4} {'major':>5}  top feature")
    for rec in weekly:
        top = f"{rec['top'][0]['feature']} {rec['top'][0]['psi']:.2f}" if rec["top"] else "-"
        print(f"{rec['week']:>4} {'baseline' if rec['baseline'] else 'test':<9} {', '.join(rec['injected']) or '-':<26} "
              f"{rec['stays']:>6,} {rec['score_psi']:>9.2f} {rec['n_moderate']:>4} {rec['n_major']:>5}  {top}")
    for c in checks:
        print(f"  week {c['week']:>2} {c['injected']:<24} {c['verdict']}")
    print(f"\nWrote {out_dir}/ : drift_report.md, psi_by_week.csv, drift_summary.json, drift_heatmap.png")


if __name__ == "__main__":
    main()
