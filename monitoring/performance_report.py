"""
Performance report (plan step 14): real performance from the LATE LABELS.

Drift alerts (step 13) show that inputs or scores moved. They cannot show that predictions got worse, because the true
outcome (readmitted within 30 days) only arrives about a month after discharge. The replay (step 11) withheld the outcomes
on purpose and saved them in data/replay/<run-id>/labels.csv. This script brings them back:

    predictions.jsonl  (id, score, flag, model version, week)   +   labels.csv  (the true outcome per stay)
        -> PR-AUC, ROC-AUC, lift, precision and recall of the flagged stays, per week, per model version and per test,
           each with a 95% range from resampling whole patients (as the evaluation gate does),
        -> compared with the model's reference performance from the gate (models/evaluation.json),
        -> put next to the drift result of the same week (data/drift/<run-id>/drift_summary.json, if present).

Run from the project root, after a replay (and, for the drift columns, after drift_report.py):
    python monitoring/performance_report.py --run-id demo1
    python monitoring/performance_report.py --run-id drill3 --versions 2=lightgbm,3=logistic
    python monitoring/performance_report.py --run-id demo1 --as-of-week 8       # only labels that would be known by week 8

Reads
    serving/logs/predictions.jsonl            scored stays of the run
    data/replay/<run-id>/labels.csv           the withheld outcomes (written by serving/replay.py)
    models/evaluation.json                    reference PR-AUC and 95% range per model (written by test.py)
    data/drift/<run-id>/drift_summary.json    (optional) drift level per week

Writes   data/performance/<run-id>/
    performance_report.md    weekly table, by model version, by test, drift versus performance, how to act
    performance_by_week.csv  the weekly numbers
    performance_summary.json the same, machine-readable
    performance_chart.png    PR-AUC by week with its 95% range and the reference line(s)
It also appends one line to models/data_use_log.jsonl (the labels come from the holdout).

How a week is judged
    "below reference"        even the optimistic end of its 95% range is under the reference PR-AUC
    "above reference"        even the pessimistic end is over it
    "consistent"             otherwise (a week of about 2,000 stays has a range of roughly plus or minus 0.04 to 0.06)
    "too few readmissions"   fewer than --min-positives readmissions, so no judgement
Together with drift this gives four readings:
    drift + below   harmful drift: a retrain candidate
    drift + fine    harmless drift: keep watching
    no drift + below silent degradation: only labels can see it
    no drift + fine  healthy

Caveats
    - In weeks where the replay damaged the inputs (covariate shift, new categories, population shift) the features were
      changed but the outcomes were NOT, so those weeks are stress tests, not a real change in the world.
    - A comparison of two models on different stays is not paired, so a small difference needs a lot of stays to show.
    - Real labels arrive about 30 days after discharge: --label-delay-weeks says when each week would really be known.

Needs: pandas, numpy, scikit-learn, matplotlib.
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
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_VERSIONS = "1=xgboost,2=lightgbm,3=logistic"     # registry versions from the README (override with --versions)


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find a data/ folder next to or above {here}")


# ---------------------------------------------------------------- loading
def parse_versions(text: str) -> dict:
    out = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[int(k.strip())] = v.strip()
    return out


def load_predictions(path: Path, run_id: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("run_id") != run_id or o.get("replay_week") is None:
                continue
            rows.append({"id": str(o.get("id")), "week": int(o["replay_week"]), "score": float(o["score"]),
                         "flag": bool(o.get("high_risk")), "version": o.get("model_version")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["id", "week"], keep="last")      # a resumed week is logged twice
    return df


def load_references(path: Path, versions: dict) -> dict:
    """version -> {family, pr_auc, lo, hi, prevalence} from the gate's evaluation.json."""
    if not path.exists():
        return {}
    res = json.loads(path.read_text(encoding="utf-8")).get("results", {})
    out = {}
    for v, fam in versions.items():
        r = res.get(fam)
        if r:
            out[v] = {"family": fam, "pr_auc": float(r["test_pr_auc"]), "lo": float(r["pr_auc_ci95"][0]),
                      "hi": float(r["pr_auc_ci95"][1]), "prevalence": float(r["test_pr_auc"]) / float(r["lift"])}
    return out


# ---------------------------------------------------------------- metrics
def metrics(g: pd.DataFrame) -> dict:
    y, s, f = g["y"].to_numpy(), g["score"].to_numpy(), g["flag"].to_numpy()
    n, pos = len(y), int(y.sum())
    out = {"n": n, "positives": pos, "prevalence": pos / n if n else float("nan"),
           "flagged_share": float(f.mean()) if n else float("nan"),
           "pr_auc": float("nan"), "roc_auc": float("nan"), "lift": float("nan"),
           "precision_flagged": float("nan"), "recall_flagged": float("nan")}
    if 0 < pos < n:
        out["pr_auc"] = float(average_precision_score(y, s))
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["lift"] = out["pr_auc"] / out["prevalence"]
    if f.sum() > 0:
        out["precision_flagged"] = float(y[f].mean())
    if pos > 0:
        out["recall_flagged"] = float(y[f].sum() / pos)
    return out


def bootstrap_pr(g: pd.DataFrame, n_boot: int, seed: int):
    """95% range of PR-AUC from resampling whole patients."""
    y, s = g["y"].to_numpy(), g["score"].to_numpy()
    codes, _ = pd.factorize(g["patient"])
    order = np.argsort(codes, kind="stable")
    by_patient = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by_patient[i] for i in rng.integers(0, len(by_patient), len(by_patient))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(average_precision_score(y[idx], s[idx]))
    if len(vals) < 20:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def verdict(pr: float, lo: float, hi: float, ref, positives: int, min_pos: int) -> str:
    if positives < min_pos or np.isnan(pr) or np.isnan(lo):
        return "too few readmissions"
    if ref is None:
        return "no reference"
    if hi < ref:
        return "below reference"
    if lo > ref:
        return "above reference"
    return "consistent"


def summarize(g: pd.DataFrame, ref, args, seed_offset: int = 0) -> dict:
    m = metrics(g)
    m["pr_lo"], m["pr_hi"] = bootstrap_pr(g, args.n_boot, args.seed + seed_offset)
    m["verdict"] = verdict(m["pr_auc"], m["pr_lo"], m["pr_hi"], ref, m["positives"], args.min_positives)
    return m


def fmt(x, d=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def rng_text(m):
    return "-" if np.isnan(m["pr_lo"]) else f"{m['pr_lo']:.3f} to {m['pr_hi']:.3f}"


# ---------------------------------------------------------------- main
def main() -> None:
    root = find_project_root()
    ap = argparse.ArgumentParser(description="Real performance of a replay run from the late labels.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--log", default=str(root / "serving" / "logs" / "predictions.jsonl"))
    ap.add_argument("--labels", default=None, help="default: data/replay/<run-id>/labels.csv")
    ap.add_argument("--evaluation", default=str(root / "models" / "evaluation.json"))
    ap.add_argument("--versions", default=DEFAULT_VERSIONS, help="registry version = model family, e.g. 2=lightgbm,3=logistic")
    ap.add_argument("--label-delay-weeks", type=int, default=4, help="weeks from a stay to its label (30 days is about 4)")
    ap.add_argument("--as-of-week", type=int, default=None, help="only use weeks whose labels would be known by this week")
    ap.add_argument("--min-positives", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(root / "data" / "performance"))
    args = ap.parse_args()

    log_path = Path(args.log)
    labels_path = Path(args.labels) if args.labels else root / "data" / "replay" / args.run_id / "labels.csv"
    for p, hint in ((log_path, "Run a replay first."), (labels_path, "It is written by serving/replay.py.")):
        if not p.exists():
            raise SystemExit(f"Missing {p}. {hint}")
    preds = load_predictions(log_path, args.run_id)
    if preds.empty:
        raise SystemExit(f"No logged stays with run_id '{args.run_id}' in {log_path}.")
    labels = pd.read_csv(labels_path, dtype={"encounter_id": str, "patient_nbr": str})
    labels = labels.rename(columns={"encounter_id": "id", "readmitted": "y", "patient_nbr": "patient"})
    df = preds.merge(labels[["id", "week", "y", "patient", "scenario"]], on=["id", "week"], how="left")
    unmatched = int(df["y"].isna().sum())
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)
    if df.empty:
        raise SystemExit("No logged stay matches a label (check the run id and that labels.csv belongs to this run).")
    last_week = int(df["week"].max())
    as_of = args.as_of_week if args.as_of_week is not None else last_week + args.label_delay_weeks
    known = df["week"] + args.label_delay_weeks <= as_of
    pending_weeks = sorted(df.loc[~known, "week"].unique().tolist())
    df = df[known]
    if df.empty:
        raise SystemExit(f"As of week {as_of} no labels would be known yet (delay {args.label_delay_weeks} weeks).")
    refs = load_references(Path(args.evaluation), parse_versions(args.versions))
    ref_pr = lambda v: refs.get(int(v), {}).get("pr_auc") if pd.notna(v) else None

    drift = {}
    dpath = root / "data" / "drift" / args.run_id / "drift_summary.json"
    if dpath.exists():
        for w in json.loads(dpath.read_text(encoding="utf-8")).get("weeks", []):
            top = w["top"][0] if w.get("top") else None
            drift[int(w["week"])] = {"major": w["n_major"], "moderate": w["n_moderate"], "score_psi": w["score_psi"],
                                     "top": f"{top['feature']} {top['psi']:.2f}" if top else "-"}

    print(f"Run '{args.run_id}': {len(df):,} stays with labels in weeks {int(df['week'].min())}-{int(df['week'].max())}"
          f"{f' ({unmatched:,} logged stays had no label)' if unmatched else ''}; labels known as of week {as_of}.")

    # ---- per week
    weekly = []
    for w, g in df.groupby("week"):
        versions = sorted(g["version"].dropna().astype(int).unique().tolist())
        ref = ref_pr(versions[0]) if len(versions) == 1 else None
        m = summarize(g, ref, args, seed_offset=int(w))
        scen = sorted(s for s in g["scenario"].dropna().unique() if s != "none")
        d = drift.get(int(w))
        has_drift = bool(d and (d["major"] or d["moderate"]))
        below = m["verdict"] == "below reference"
        if d is None:
            reading = "-"
        elif has_drift and below:
            reading = "harmful drift"
        elif has_drift:
            reading = "harmless drift" if m["verdict"] != "too few readmissions" else "drift, too few labels"
        elif below:
            reading = "silent degradation"
        else:
            reading = "healthy"
        weekly.append({"week": int(w), "versions": versions, "scenarios": scen, "reference_pr_auc": ref, **m,
                       "drift": d, "reading": reading, "labels_known_from_week": int(w) + args.label_delay_weeks})
    wk = pd.DataFrame(weekly)

    # ---- by model version (clean weeks, and all weeks)
    by_version = []
    for v, g in df.groupby("version"):
        ref = ref_pr(v)
        clean = g[g["scenario"] == "none"]
        row = {"version": int(v), "family": refs.get(int(v), {}).get("family", "?"), "reference_pr_auc": ref,
               "reference_range": (refs[int(v)]["lo"], refs[int(v)]["hi"]) if int(v) in refs else None,
               "weeks": sorted(g["week"].unique().tolist())}
        row["all"] = summarize(g, ref, args, seed_offset=100 + int(v))
        row["clean"] = summarize(clean, ref, args, seed_offset=200 + int(v)) if len(clean) else None
        by_version.append(row)

    # ---- by scenario (inputs altered, outcomes original)
    clean_all = df[df["scenario"] == "none"]
    clean_m = summarize(clean_all, None, args, seed_offset=300) if len(clean_all) else None
    clean_pr = clean_m["pr_auc"] if clean_m else None
    by_scenario = []
    for sc, g in df.groupby("scenario"):
        m = summarize(g, None if sc == "none" else clean_pr, args, seed_offset=400 + len(by_scenario))
        by_scenario.append({"scenario": sc, "weeks": sorted(g["week"].unique().tolist()), **m})

    # ---- outputs
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = wk.drop(columns=["drift"]).copy()
    flat["versions"] = flat["versions"].map(lambda x: "/".join(map(str, x)))
    flat["scenarios"] = flat["scenarios"].map(lambda x: "+".join(x) or "none")
    flat["drift_major"] = [d["major"] if d else None for d in wk["drift"]]
    flat["drift_moderate"] = [d["moderate"] if d else None for d in wk["drift"]]
    flat.round(4).to_csv(out_dir / "performance_by_week.csv", index=False)
    (out_dir / "performance_summary.json").write_text(json.dumps({
        "run_id": args.run_id, "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "label_delay_weeks": args.label_delay_weeks,
        "as_of_week": as_of, "pending_weeks": pending_weeks, "unmatched_logged_stays": unmatched,
        "references": refs, "weeks": weekly, "by_version": by_version, "by_scenario": by_scenario},
        indent=2, default=str), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(0.9 * max(6, len(wk)) + 1.5, 4.2))
    palette = {"none": "#5F5E5A", "covariate_shift": "#D85A30", "new_categories": "#7F77DD", "bad_data": "#BA7517",
               "population_shift": "#1D9E75", "population_shift_young": "#0F6E56"}
    markers = {2: "o", 3: "s", 1: "^"}
    for _, r in wk.iterrows():
        key = r["scenarios"][0] if r["scenarios"] else "none"
        c = palette.get(key, "#378ADD")
        if not np.isnan(r["pr_auc"]):
            ax.errorbar(r["week"], r["pr_auc"], yerr=[[r["pr_auc"] - r["pr_lo"]], [r["pr_hi"] - r["pr_auc"]]] if not np.isnan(r["pr_lo"]) else None,
                        fmt=markers.get(r["versions"][0] if r["versions"] else 2, "o"), color=c, capsize=3, markersize=6)
    for v, rf in refs.items():
        if v in df["version"].dropna().astype(int).unique():
            ax.axhline(rf["pr_auc"], linestyle="--", linewidth=1, color="#888780")
            ax.text(wk["week"].max(), rf["pr_auc"], f"{rf['family']} reference {rf['pr_auc']:.3f}", ha="right", va="bottom", fontsize=8, color="#5F5E5A")
    ax.set_xlabel("simulated week")
    ax.set_ylabel("PR-AUC (95% range)")
    ax.set_title(f"Real performance by week, run {args.run_id} (colour = injected test, marker = model version)", fontsize=9)
    ax.set_xticks(wk["week"])
    ax.grid(alpha=0.25)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=k) for k, c in palette.items()
               if k == "none" or any(k in s for s in wk["scenarios"])]
    ax.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "performance_chart.png", dpi=130)
    plt.close(fig)

    L = [f"# Performance report: {args.run_id}", "",
         f"Created {datetime.now():%Y-%m-%d %H:%M}. Generated by `monitoring/performance_report.py` from `{log_path.name}` and `{labels_path.name}`.", "",
         f"- {len(df):,} scored stays with their true outcome, weeks {int(df['week'].min())} to {int(df['week'].max())}."
         + (f" {unmatched:,} logged stays had no label." if unmatched else ""),
         f"- Label delay: {args.label_delay_weeks} weeks (30 days is about 4), so the labels of week W are known from week W + {args.label_delay_weeks}. "
         f"Report as of week {as_of}." + (f" Weeks not yet known: {', '.join(map(str, pending_weeks))}." if pending_weeks else ""),
         "- Reference: each model's PR-AUC on the test split from the gate (`models/evaluation.json`): "
         + (", ".join(f"version {v} {r['family']} {r['pr_auc']:.3f} (range {r['lo']:.3f} to {r['hi']:.3f})" for v, r in refs.items()) or "not found") + ".",
         "- Ranges are 95% from resampling whole patients (200 times). A week is 'below reference' only when even the optimistic end is under the reference.", "",
         "## Weekly performance", "",
         "| Week | Model | Injected | Stays | Readmitted | PR-AUC | 95% range | ROC-AUC | Lift | Precision flagged | Recall flagged | vs reference |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in weekly:
        L.append(f"| {r['week']} | {'/'.join(map(str, r['versions'])) or '-'} | {', '.join(r['scenarios']) or '-'} | {r['n']:,} | {r['positives']:,} | "
                 f"{fmt(r['pr_auc'])} | {rng_text(r)} | {fmt(r['roc_auc'])} | {fmt(r['lift'], 2)}x | {fmt(r['precision_flagged'])} | "
                 f"{fmt(r['recall_flagged'])} | {r['verdict']} |")
    L += ["", "## By model version", "",
          "| Version | Model | Weeks | Scope | Stays | PR-AUC | 95% range | Reference | vs reference |", "|---|---|---|---|---|---|---|---|---|"]
    for r in by_version:
        for scope, m in (("clean weeks", r["clean"]), ("all weeks", r["all"])):
            if m is None:
                continue
            L.append(f"| {r['version']} | {r['family']} | {', '.join(map(str, r['weeks']))} | {scope} | {m['n']:,} | {fmt(m['pr_auc'])} | "
                     f"{rng_text(m)} | {fmt(r['reference_pr_auc'])} | {m['verdict']} |")
    if len(by_version) > 1:
        L += ["", "Two models scored different stays, so this comparison is not paired: if their ranges overlap, the data cannot tell them apart. "
                  "The models differ by about 0.016 PR-AUC on the test split, so telling them apart needs a lot of stays."]
    L += ["", "## By injected test (inputs altered, outcomes original)", "",
          "| Test | Weeks | Stays | PR-AUC | 95% range | Lift | Reading against clean weeks |", "|---|---|---|---|---|---|---|"]
    for r in by_scenario:
        label = "none (clean weeks)" if r["scenario"] == "none" else r["scenario"]
        reading = "reference for this table" if r["scenario"] == "none" else r["verdict"].replace("reference", "clean weeks")
        L.append(f"| {label} | {', '.join(map(str, r['weeks']))} | {r['n']:,} | {fmt(r['pr_auc'])} | {rng_text(r)} | {fmt(r['lift'], 2)}x | {reading} |")
    L += ["", "In the injected weeks the replay changed the features but kept the real outcomes, so these rows measure what happens when "
              "altered inputs reach the model. They are stress tests, not a real change in the world. The comparison is one-sided: it asks whether even the optimistic "
              "end of the test's range is under the clean weeks' point value and ignores the clean weeks' own uncertainty, so a result close to the line is a hint, not proof.", ""]
    if drift:
        L += ["## Drift versus real performance", "",
              "| Week | Drift (step 13) | Strongest drift | Score PSI | Performance | Reading |", "|---|---|---|---|---|---|"]
        for r in weekly:
            d = r["drift"]
            if d is None:
                continue
            L.append(f"| {r['week']} | {d['major']} major, {d['moderate']} moderate | {d['top']} | {d['score_psi']:.2f} | {r['verdict']} | {r['reading']} |")
        L += ["", "- **harmful drift** (drift and below reference): a retrain candidate, step 15.",
              "- **harmless drift** (drift, performance fine): keep watching; the model copes with this change.",
              "- **silent degradation** (no drift, below reference): only labels can see this; investigate.",
              "- **healthy**: no drift and performance consistent with the reference.", ""]
    else:
        L += ["No drift summary found for this run (run `python monitoring/drift_report.py --run-id " + args.run_id + "` first to add the drift columns).", ""]
    L += ["## How to act on this", "",
          "Retrain only when drift is real and performance dropped (harmful drift or silent degradation that persists), and let the new model enter as a "
          "challenger through the gate and the promotion rule. A drifted week with healthy performance needs no action.", "",
          "## Limits", "",
          "- A week of about 2,000 stays has about 240 readmissions, so its PR-AUC range is wide (roughly plus or minus 0.04 to 0.06 on real data; clean weeks of one model ranged from 0.18 to 0.28): "
          "single weeks flip between verdicts, so judge by the pooled tables.",
          "- Weeks with altered inputs are stress tests. For a real change in the world the outcomes would change too.",
          "- The reference is the model's test-split performance; the holdout has its own prevalence and noise.",
          "- The labels come from the holdout. Every use of this report is logged in `models/data_use_log.jsonl`.", "",
          "## Files", "",
          "- `performance_by_week.csv`, `performance_summary.json`: the numbers.",
          "- `performance_chart.png`: PR-AUC by week with its range and the reference line.", ""]
    (out_dir / "performance_report.md").write_text("\n".join(L), encoding="utf-8")

    ledger = root / "models" / "data_use_log.jsonl"
    ledger.parent.mkdir(exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "data_name": "holdout", "data_file": str(labels_path),
                            "rows": int(len(df)), "purpose": f"performance report on replay run {args.run_id} (late labels)"}) + "\n")

    print(f"\n{'week':>4} {'model':<6} {'injected':<24} {'stays':>6} {'PR-AUC':>7} {'95% range':>15} {'vs reference':<22} reading")
    for r in weekly:
        print(f"{r['week']:>4} {'/'.join(map(str, r['versions'])) or '-':<6} {', '.join(r['scenarios']) or '-':<24} {r['n']:>6,} "
              f"{fmt(r['pr_auc']):>7} {rng_text(r):>15} {r['verdict']:<22} {r['reading']}")
    print(f"\nWrote {out_dir}/ : performance_report.md, performance_by_week.csv, performance_summary.json, performance_chart.png")


if __name__ == "__main__":
    main()