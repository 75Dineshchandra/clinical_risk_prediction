"""
Export the registry champion to a self-contained bundle for the API (plan step 10).

Run from the project root, inside the project's virtual environment:
    python serving/export_champion.py
    python serving/export_champion.py --alias challenger --out serving/bundle_challenger

What it does
- Loads models:/readmission@champion from the MLflow registry (this is the "load by alias" step).
- Computes the high-risk cutoff: the score that the top 20% of stays reach on the reference data
  (the test split by default; this only reads scores, nothing is evaluated or tuned).
- Writes serving/bundle/model.joblib and serving/bundle/meta.json. meta.json is written last and holds a
  checksum of the model file, so the API refuses a half-written export.
- Step 13: also writes serving/bundle/reference_profile.json, the REFERENCE the drift exporter compares live
  traffic with: for every input feature (and for the scores) the bins or categories and their shares on the same
  reference data. It is written before meta.json, so it is always in place when the bundle is complete.

Why a bundle? MLflow stores absolute paths (such as C:\\Users\\...\\mlruns) in mlflow.db, which do not exist inside a
Linux container, and SQLite does not behave well on mounted Windows folders. The bundle needs neither.

Run this again after every promotion or rollback, then tell the API:  POST /reload
(the drift exporter notices the new reference_profile.json by itself).
"""

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TARGET, PATIENT = "readmitted", "patient_nbr"
ID_FIELDS = ["encounter_id", PATIENT]
PACKAGES = ["scikit-learn", "numpy", "pandas", "scipy", "lightgbm", "xgboost", "shap", "joblib", "mlflow"]
KEEP_TAGS = ["gate", "cv_pr_auc", "test_pr_auc", "test_roc_auc", "test_lift", "selection_reason", "role",
             "promoted_at", "registration_reason"]


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find a data/ folder next to or above {here}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def write_reference_profile(out_dir: Path, reference: pd.DataFrame, cols: list, scores, meta_base: dict, ref_name: str) -> dict:
    """Step 13: save the drift reference (bins/categories and shares per feature, and for the scores)."""
    sys.path.insert(0, str(find_project_root() / "monitoring"))
    import drift_lib                                       # pandas + numpy only
    profile = drift_lib.build_reference(
        reference[cols], scores,
        {"model_version": meta_base.get("version"), "model_family": meta_base.get("model_family"),
         "reference_file": ref_name, "created": datetime.now().astimezone().isoformat(timespec="seconds")})
    tmp = out_dir / "reference_profile.json.tmp"
    tmp.write_text(json.dumps(profile), encoding="utf-8")
    os.replace(tmp, out_dir / "reference_profile.json")
    return {"file": "reference_profile.json", "sha256": sha256(out_dir / "reference_profile.json"),
            "reference_rows": profile["reference_rows"], "features": len(profile["features"])}


def write_bundle(pipe, out_dir: Path, meta_base: dict, reference: pd.DataFrame, flag_fraction: float,
                 reference_name: str = "reference") -> dict:
    """Write the bundle. Needs no MLflow, so it can be tested on its own."""
    cols = list(pipe.named_steps["prep"].feature_names_in_)
    missing = [c for c in cols if c not in reference.columns]
    if missing:
        raise ValueError(f"The reference data lacks columns the model needs: {missing}")
    scores = pipe.predict_proba(reference[cols])[:, 1]
    cutoff = float(np.quantile(scores, 1 - flag_fraction))
    id_cols = [c for c in ID_FIELDS if c in reference.columns]
    example = json.loads(reference[id_cols + cols].head(1).to_json(orient="records"))[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_model = out_dir / "model.joblib.tmp"
    joblib.dump(pipe, tmp_model)
    os.replace(tmp_model, out_dir / "model.joblib")
    profile_info = write_reference_profile(out_dir, reference, cols, scores, meta_base, reference_name)   # step 13
    meta = {**meta_base, "model_sha256": sha256(out_dir / "model.joblib"),
            "flag_fraction": flag_fraction, "flag_cutoff": cutoff,
            "flag_cutoff_based_on": {"rows": int(len(reference)),
                                     "share_flagged": round(float((scores >= cutoff).mean()), 4)},
            "reference_profile": profile_info,
            "example_record": example, "packages": package_versions(), "python": platform.python_version(),
            "exported_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    tmp_meta = out_dir / "meta.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_meta, out_dir / "meta.json")          # written last: the commit marker
    return meta


def main() -> None:
    root = find_project_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="readmission", help="registered model name")
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--out", default=str(root / "serving" / "bundle"))
    ap.add_argument("--reference-data", default=str(root / "data" / "splits" / "test.csv"))
    ap.add_argument("--flag-fraction", type=float, default=0.20, help="share of stays to flag as high risk")
    ap.add_argument("--no-ledger", action="store_true", help="do not note this use of the data in data_use_log.jsonl")
    args = ap.parse_args()
    if not 0 < args.flag_fraction < 1:
        raise SystemExit("--flag-fraction must be between 0 and 1")

    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(root / 'mlflow.db').as_posix()}"))
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(args.name, args.alias)
    except Exception:
        raise SystemExit(f"No '{args.alias}' alias on registered model '{args.name}'. "
                         f"Register one first: python register_champion.py --champion <model> --reason \"...\"")
    pipe = mlflow.sklearn.load_model(f"models:/{args.name}/{mv.version}")
    ref_path = Path(args.reference_data)
    reference = pd.read_csv(ref_path, dtype={"encounter_id": str, PATIENT: str})

    meta_base = {"registered_model": args.name, "alias": args.alias, "version": int(mv.version),
                 "model_family": mv.tags.get("model_family"), "training_run_id": mv.tags.get("training_run_id", mv.run_id),
                 "registry_tags": {k: v for k, v in mv.tags.items() if k in KEEP_TAGS}}
    out_dir = Path(args.out)
    meta = write_bundle(pipe, out_dir, meta_base, reference, args.flag_fraction, ref_path.name)

    if not args.no_ledger:
        ledger = root / "models" / "data_use_log.jsonl"
        ledger.parent.mkdir(exist_ok=True)
        with open(ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "data_name": ref_path.stem,
                                "data_file": str(ref_path), "rows": int(len(reference)),
                                "purpose": "flag cutoff and drift reference profile for the serving bundle (scores and features only, nothing evaluated)"}) + "\n")

    size_mb = (out_dir / "model.joblib").stat().st_size / 1e6
    print(f"Exported {args.name}@{args.alias}: version {meta['version']} ({meta['model_family']}) -> {out_dir}")
    print(f"  model.joblib {size_mb:.1f} MB, sha256 {meta['model_sha256'][:12]}")
    print(f"  high-risk flag: top {args.flag_fraction:.0%} of {meta['flag_cutoff_based_on']['rows']:,} reference stays, "
          f"cutoff score {meta['flag_cutoff']:.4f}")
    rp = meta["reference_profile"]
    print(f"  drift reference: {rp['features']} features and the scores, from {rp['reference_rows']:,} reference stays "
          f"({rp['file']}, sha256 {rp['sha256'][:12]})")
    print(f"  packages at export: " + ", ".join(f"{k} {v}" for k, v in meta['packages'].items() if k != 'mlflow'))
    print("Next: start or reload the API (docker compose up -d, or POST http://localhost:8000/reload).")


if __name__ == "__main__":
    main()
