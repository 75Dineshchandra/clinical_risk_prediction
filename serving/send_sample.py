"""
Send rows from a CSV to the running API and print what comes back (plan step 10 check; seed of the step 11 replay).

    python serving/send_sample.py                                  # 20 rows of the test split
    python serving/send_sample.py --csv data/splits/test.csv --n 100 --batch-size 25
    python serving/send_sample.py --no-explain --url http://localhost:8000

Uses only the standard library plus pandas. The outcome columns (readmitted, readmitted_raw) are never sent.
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DROP = ["readmitted", "readmitted_raw"]


def post(url: str, body: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")
    except urllib.error.URLError as err:
        raise SystemExit(f"Cannot reach the API at {url} ({err.reason}).\n"
                         "Is it running? Start it with:   docker compose up -d --build\n"
                         "or, without Docker:           python -m uvicorn app:app --app-dir serving --port 8000\n"
                         "Then check http://localhost:8000/health in a browser (it needs an exported bundle: "
                         "python serving/export_champion.py).")


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent if (here.parent / "data").is_dir() else here
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(root / "data" / "splits" / "test.csv"))
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--no-explain", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype={"encounter_id": str, "patient_nbr": str}).drop(columns=DROP, errors="ignore")
    df = df.head(args.n)
    scored = flagged = 0
    first = None
    for start in range(0, len(df), args.batch_size):
        chunk = df.iloc[start:start + args.batch_size]
        records = json.loads(chunk.to_json(orient="records"))
        status, out = post(args.url.rstrip("/") + "/predict", {"records": records, "explain": not args.no_explain})
        if status != 200:
            print(f"HTTP {status}: {json.dumps(out, indent=2)[:1500]}")
            raise SystemExit(1)
        preds = out["predictions"]
        scored += len(preds)
        flagged += sum(1 for p in preds if p["high_risk"])
        first = first or (out["model"], out["flag_rule"], preds[0])
    model, rule, p0 = first
    print(f"Scored {scored} rows with {model['name']} version {model['version']} ({model['family']}); "
          f"{flagged} flagged high risk (cutoff {rule['cutoff']:.4f}).")
    print("First record:", json.dumps(p0, indent=2))


if __name__ == "__main__":
    main()