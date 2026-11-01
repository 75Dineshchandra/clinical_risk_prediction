"""
Prefect flow for the retrain loop (plan step 15). It runs your EXISTING scripts, in order, and decides nothing itself:
every decision is made by retrain.py (should we retrain?), test.py (gate) and promote.py (promotion rule).

Run from the project root, with the PREFECT environment active (.venv-prefect), not the project's .venv:
    python orchestration/retrain_flow.py --run-id demo1                      # DRY RUN: decide only, change nothing
    python orchestration/retrain_flow.py --run-id demo1 --force-retrain      # dry run of a forced retrain: shows the plan
    python orchestration/retrain_flow.py --run-id demo1 --force-retrain --apply
        # trains, gates, registers a challenger, and SHOWS the promotion decision (promote.py --dry-run). The champion is not changed.
    python orchestration/retrain_flow.py --run-id demo1 --force-retrain --apply --allow-promote
        # same, and if the promotion rule says PROMOTE: promote, export the bundle, reload the API.

Safety levels
    default            only runs retrain.py decide (writes data/retrain/<id>/trigger.json)
    --apply            also backs up models/, then train.py, test.py, register_champion.py --challenger, promote.py --dry-run
                       (this REPLACES models/*.joblib and evaluation.json, and adds a registry version; the champion alias is untouched)
    --allow-promote    also lets promote.py change the champion, then export_champion.py and POST /reload

What counts as a normal outcome (the flow ends cleanly, nothing is "failed")
    - the decision is keep or fix_data_first
    - no model passes the gate
    - register_champion.py refuses
    - promote.py says KEEP CHAMPION
A real error (a script crashing, a missing file) raises, and Prefect marks the run as failed.

Each step runs in the PROJECT environment: pass --main-python, or set RETRAIN_PYTHON, or the flow uses
<project>/.venv/Scripts/python.exe (Windows) or <project>/.venv/bin/python. Prefect itself only needs to be installed
in the separate environment, so its dependencies never touch your pinned MLflow / pydantic / scikit-learn.
"""

import argparse
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
BASELINE = "logistic"
PROMOTION_LOG = MODELS / "promotion_log.jsonl"
BACKUP_FILES = ["evaluation.json", "test_report.md", "model_report.md", "best_model.json",
                "champion.json", "champion_report.md", "promotion_report.md"]


def project_python(explicit) -> str:
    if explicit:
        return explicit
    if os.environ.get("RETRAIN_PYTHON"):
        return os.environ["RETRAIN_PYTHON"]
    exe = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not exe.exists():
        raise FileNotFoundError(f"Project Python not found at {exe}. Pass --main-python or set RETRAIN_PYTHON.")
    return str(exe)


def run_script(py: str, args: list, logger, name: str) -> tuple:
    """Run one project script in the project environment. Returns (exit code, combined output)."""
    cmd = [py, *args]
    logger.info("[%s] %s", name, " ".join(args))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-25:])
    logger.info("[%s] exit code %s. Last lines:\n%s", name, proc.returncode, tail)
    return proc.returncode, out


def log_decision(entry: dict) -> None:
    entry = {"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", **entry}
    with open(PROMOTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


@task(name="decide: should we retrain?")
def t_decide(py, run_id, retrain_id, force, scope, performance, champion_version):
    logger = get_run_logger()
    args = ["retrain.py", "decide", "--run-id", run_id, "--id", retrain_id, "--scope", scope]
    if force:
        args.append("--force-retrain")
    if performance:
        args += ["--performance", performance]
    if champion_version:
        args += ["--champion-version", str(champion_version)]
    code, out = run_script(py, args, logger, "decide")
    if code != 0:
        raise RuntimeError("retrain.py decide failed:\n" + out[-1500:])
    return json.loads((DATA / "retrain" / retrain_id / "trigger.json").read_text(encoding="utf-8"))


@task(name="backup models/ before retraining")
def t_backup(retrain_id):
    logger = get_run_logger()
    dest = DATA / "retrain" / retrain_id / "models_before"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in list(MODELS.glob("[0-9][0-9]_*_prauc*.joblib")) + [MODELS / f for f in BACKUP_FILES]:
        if p.exists():
            shutil.copy2(p, dest / p.name)
            n += 1
    logger.info("Backed up %d files to %s", n, dest)
    return str(dest)


@task(name="train all three models")
def t_train(py, tag):
    code, out = run_script(py, ["train.py", "--data-tag", tag], get_run_logger(), "train")
    if code != 0:
        raise RuntimeError("train.py failed:\n" + out[-1500:])


@task(name="gate: test.py")
def t_gate(py):
    code, out = run_script(py, ["test.py"], get_run_logger(), "gate")
    if code != 0:
        raise RuntimeError("test.py failed:\n" + out[-1500:])
    ev = json.loads((MODELS / "evaluation.json").read_text(encoding="utf-8"))["results"]
    return {fam: {"passed": bool(r["passed"]), "test_pr_auc": r["test_pr_auc"]} for fam, r in ev.items()}


def pick_challenger(gate: dict):
    """Rule written in advance: among NON-baseline models that passed the gate, the highest test PR-AUC."""
    ok = {f: r for f, r in gate.items() if r["passed"] and f != BASELINE}
    return max(ok, key=lambda f: ok[f]["test_pr_auc"]) if ok else None


@task(name="register challenger")
def t_register(py, family, reason):
    code, out = run_script(py, ["register_champion.py", "--challenger", family, "--reason", reason],
                           get_run_logger(), "register")
    # a refusal (exit code 1 with REFUSED) is a normal outcome, not a crash
    refused = code != 0 and "REFUSED" in out
    if code != 0 and not refused:
        raise RuntimeError("register_champion.py failed:\n" + out[-1500:])
    return {"registered": code == 0, "refused": refused, "detail": out.strip().splitlines()[-1] if out.strip() else ""}


@task(name="promotion rule: promote.py")
def t_promote(py, dry_run):
    args = ["promote.py"] + (["--dry-run"] if dry_run else [])
    code, out = run_script(py, args, get_run_logger(), "promote-dry" if dry_run else "promote")
    if code != 0:
        raise RuntimeError("promote.py failed:\n" + out[-1500:])
    return {"promote": "DECISION: PROMOTE" in out, "output_tail": "\n".join(out.strip().splitlines()[-12:])}


@task(name="export bundle and reload the API")
def t_export_reload(py, url, token):
    logger = get_run_logger()
    code, out = run_script(py, ["serving/export_champion.py"], logger, "export")
    if code != 0:
        raise RuntimeError("export_champion.py failed:\n" + out[-1500:])
    req = urllib.request.Request(url.rstrip("/") + "/reload", data=b"", method="POST",
                                 headers={"X-Admin-Token": token} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        return {"exported": True, "reloaded": True, "detail": body}
    except (urllib.error.URLError, OSError) as err:
        logger.warning("Bundle exported, but the API did not reload (%s). Start the API and POST /reload.", err)
        return {"exported": True, "reloaded": False, "detail": str(err)}


def write_report(retrain_id, summary: dict) -> Path:
    path = DATA / "retrain" / retrain_id / "retrain_report.md"
    t = summary["trigger"]
    lines = [f"# Retrain report: {retrain_id}", "",
             f"Created {datetime.now():%Y-%m-%d %H:%M}. Generated by `orchestration/retrain_flow.py`.", "",
             f"- Mode: {summary['mode']}.",
             f"- Evidence from run `{t['run_id']}`: {t['reason']}",
             f"- Decision: **{t['decision']}**" + (" (forced by a person)" if t.get("forced") else "") + ".", ""]
    for step in summary["steps"]:
        lines.append(f"- {step}")
    lines += ["", f"**Outcome: {summary['outcome']}**", "",
              "The champion changes only through `promote.py`'s rule. Every promotion can be undone with `python promote.py --rollback`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@flow(name="readmission-retrain", log_prints=True)
def retrain_flow(run_id: str, force_retrain: bool = False, apply: bool = False, allow_promote: bool = False,
                 scope: str = "clean", performance: str = "", champion_version: int = 0,
                 main_python: str = "", api_url: str = "http://localhost:8000") -> dict:
    logger = get_run_logger()
    py = project_python(main_python or None)
    retrain_id = f"retrain-{datetime.now():%Y%m%d-%H%M%S}"
    mode = ("full run, promotion allowed" if allow_promote else "apply, promotion shown but not applied") if apply else "dry run (decide only)"
    steps = []

    trigger = t_decide(py, run_id, retrain_id, force_retrain, scope, performance, champion_version)
    steps.append(f"Decide: {trigger['decision']}.")
    summary = {"retrain_id": retrain_id, "mode": mode, "trigger": trigger, "steps": steps}
    if trigger["decision"] != "retrain":
        summary["outcome"] = {"keep": "no retrain: the champion stays.",
                              "fix_data_first": "no retrain: fix the data first."}[trigger["decision"]]
    elif not apply:
        steps.append("Dry run: would back up models/, run train.py, test.py, register_champion.py --challenger and promote.py --dry-run.")
        summary["outcome"] = "dry run: nothing was changed."
    else:
        steps.append(f"Backed up models/ to {t_backup(retrain_id)}.")
        t_train(py, f"retrain-{retrain_id}")
        steps.append("Trained logistic, XGBoost and LightGBM (train.py).")
        gate = t_gate(py)
        steps.append("Gate: " + ", ".join(f"{f} {'PASS' if r['passed'] else 'FAIL'} ({r['test_pr_auc']:.3f})" for f, r in gate.items()) + ".")
        family = pick_challenger(gate)
        if family is None:
            summary["outcome"] = "no model passed the gate: nothing registered, the champion stays."
        else:
            reg = t_register(py, family, f"retrain flow {retrain_id}: {trigger['decision']} (run {run_id})")
            if not reg["registered"]:
                steps.append(f"Registration refused: {reg['detail']}")
                summary["outcome"] = "challenger refused by the registry script: the champion stays."
            else:
                steps.append(f"Registered {family} as @challenger.")
                promo = t_promote(py, dry_run=True)
                steps.append("Promotion rule (dry run): " + ("PROMOTE" if promo["promote"] else "KEEP CHAMPION") + ".")
                if not promo["promote"]:
                    summary["outcome"] = "challenger registered but did not clearly beat the champion: the champion stays."
                elif not allow_promote:
                    summary["outcome"] = "the challenger WOULD be promoted. Re-run with --allow-promote to apply."
                else:
                    t_promote(py, dry_run=False)
                    steps.append("Promoted: the challenger is now @champion, the old one is @previous_champion.")
                    ex = t_export_reload(py, api_url, os.environ.get("ADMIN_TOKEN", ""))
                    steps.append("Exported the bundle" + (" and reloaded the API." if ex["reloaded"] else "; the API was NOT reloaded (start it, then POST /reload)."))
                    summary["outcome"] = "challenger promoted. Watch the dashboards; undo with promote.py --rollback."
    report = write_report(retrain_id, summary)
    if apply:
        log_decision({"decision": "retrain_flow", "retrain_id": retrain_id, "trigger": trigger["decision"],
                      "forced": bool(trigger.get("forced")), "outcome": summary["outcome"]})
    logger.info("Outcome: %s", summary["outcome"])
    print(f"Outcome: {summary['outcome']}\nReport: {report}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prefect flow for the retrain loop (step 15).")
    ap.add_argument("--run-id", required=True, help="replay run whose labelled results are judged, e.g. demo1")
    ap.add_argument("--force-retrain", action="store_true", help="override the evidence (recorded as forced)")
    ap.add_argument("--apply", action="store_true", help="actually train, gate and register a challenger (default: decide only)")
    ap.add_argument("--allow-promote", action="store_true", help="with --apply: let promote.py change the champion, then export and reload")
    ap.add_argument("--scope", choices=["clean", "all"], default="clean")
    ap.add_argument("--performance", default="", help="path to performance_summary.json (default: found by run id)")
    ap.add_argument("--champion-version", type=int, default=0, help="live registry version (default: from models/champion.json)")
    ap.add_argument("--main-python", default="", help="python of the project environment (default: .venv or RETRAIN_PYTHON)")
    ap.add_argument("--api-url", default="http://localhost:8000")
    a = ap.parse_args()
    if a.allow_promote and not a.apply:
        raise SystemExit("--allow-promote only makes sense together with --apply.")
    retrain_flow(run_id=a.run_id, force_retrain=a.force_retrain, apply=a.apply, allow_promote=a.allow_promote,
                 scope=a.scope, performance=a.performance, champion_version=a.champion_version,
                 main_python=a.main_python, api_url=a.api_url)
