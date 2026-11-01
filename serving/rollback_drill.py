"""
Rollback drill (plan step 11): rehearse a bad model going live, then the rollback.

Why a drill? The gate and promote.py exist so that a bad model can never be promoted by the normal route. So to
practise the recovery you have to put one in deliberately. This script does that on purpose, writes it to the audit
log as "drill_swap", and the way back is the REAL rollback, promote.py --rollback.

Run from the project root with the API up. Do it between replay weeks (use --seconds-per-week 90 or more):

    python serving/rollback_drill.py --start       # champion -> the baseline (logistic), export the bundle, reload the API
    ... watch the "Live model version" panel in Grafana and the console's Model tab ...
    python serving/rollback_drill.py --finish      # promote.py --rollback, export the bundle, reload the API

--start
    1. Makes the target version (default @baseline) the champion and the old champion @previous_champion, using the
       same functions as promote.py, so champion.json, champion_report.md and the log stay truthful.
    2. Runs serving/export_champion.py and POST /reload, so the API now serves the target model.
--finish
    1. Runs promote.py --rollback (the same command you would use in a real incident).
    2. Puts the drill target's role tag back to what it was ("baseline" when it came from @baseline).
    3. Runs serving/export_champion.py and POST /reload.

What to expect, and the lesson
    The model-version gauge changes, so the ReplayModelChanged alert fires and the dashboard shows the new version.
    The flagged share stays near 20%: export_champion.py recomputes the cutoff for whatever model it exports, so the
    label-free metrics look healthy even though the live model is worse. Only real labels (step 14) show that.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find a data/ folder next to or above {here}")


ROOT = find_project_root()


def export_and_reload(url: str, token: str, skip_export: bool) -> None:
    if not skip_export:
        print("\nExporting the bundle from the registry ...")
        subprocess.run([sys.executable, str(ROOT / "serving" / "export_champion.py")], cwd=ROOT, check=True)
    headers = {"X-Admin-Token": token} if token else {}
    req = urllib.request.Request(url.rstrip("/") + "/reload", data=b"", headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read())
        print(f"API reloaded: version {out.get('previous_version')} -> {out.get('model_version')} ({out.get('model_family')})")
    except urllib.error.HTTPError as err:
        print(f"Reload failed (HTTP {err.code}): {err.read().decode('utf-8', 'replace')[:300]}")
        print("The old model keeps serving. Fix the cause and POST /reload again.")
    except urllib.error.URLError as err:
        print(f"Cannot reach the API at {url} ({err.reason}). Start it, then POST /reload.")


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--start", action="store_true", help="put a deliberately worse model live")
    mode.add_argument("--finish", action="store_true", help="roll back with promote.py --rollback")
    ap.add_argument("--to", default="baseline", help="alias or version number to make champion (default: baseline)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--admin-token", default=os.environ.get("ADMIN_TOKEN", ""))
    ap.add_argument("--skip-export", action="store_true", help="only change the registry, do not export or reload")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    import mlflow
    import promote as P                      # reuse promote.py's functions so the files and the log stay truthful
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"))
    client = P.MlflowClient()

    if args.start:
        champ = P.get_version(client, "champion")
        target = P.get_version(client, args.to)
        if target.version == champ.version:
            raise SystemExit(f"Version {target.version} is already the champion. Choose another --to.")
        print(f"Drill: champion v{champ.version} ({P.family(champ)}) -> v{target.version} ({P.family(target)}), "
              f"gate result of the target: {target.tags.get('gate', 'unknown')}.")
        print("This bypasses promote.py's rule on purpose and is recorded in the audit log as 'drill_swap'.")
        if not args.yes and input("Type yes to continue: ").strip().lower() != "yes":
            raise SystemExit("Cancelled. Nothing changed.")
        P.swap_aliases(client, target, champ)
        target = client.get_model_version(P.REGISTERED_NAME, target.version)
        P.refresh_champion_files(client, target, {
            "promoted_at": f"{datetime.now():%Y-%m-%d %H:%M}", "previous_version": int(champ.version),
            "reason": "DRILL: deliberate swap to rehearse a rollback. Undo with promote.py --rollback."})
        P.append_log({"decision": "drill_swap", "from_version": int(champ.version), "to_version": int(target.version),
                      "note": "rollback drill; undo with promote.py --rollback"})
        print(f"Registry: @champion is now v{target.version}; @previous_champion is v{champ.version}.")
        export_and_reload(args.url, args.admin_token, args.skip_export)
        print("\nNow watch the dashboard, then run:  python serving/rollback_drill.py --finish")
        return

    # --finish
    current = P.get_version(client, "champion")
    try:
        previous = P.get_version(client, "previous_champion")
    except Exception:
        raise SystemExit("There is no @previous_champion, so there is nothing to roll back to.")
    print(f"Rolling back: champion v{current.version} -> v{previous.version}, using promote.py --rollback.")
    subprocess.run([sys.executable, str(ROOT / "promote.py"), "--rollback"], cwd=ROOT, check=True)
    try:                                     # swap_aliases re-tags the demoted version; restore the baseline's own tag
        baseline = client.get_model_version_by_alias(P.REGISTERED_NAME, "baseline")
        client.set_model_version_tag(P.REGISTERED_NAME, baseline.version, "role", "baseline")
    except Exception:
        pass
    export_and_reload(args.url, args.admin_token, args.skip_export)
    print("\nDrill finished. The champion is back; check the Model tab at http://localhost:8000/#model.")


if __name__ == "__main__":
    main()
