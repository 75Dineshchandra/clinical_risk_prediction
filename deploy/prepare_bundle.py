"""
Copy the champion bundle into deploy/bundle, the folder the Render image is built from.

Run from the project root, after `python serving/export_champion.py`:
    python deploy/prepare_bundle.py

Why a second copy? serving/bundle/ is ignored by Git (it is rebuilt on every export), but a hosted service has no mounted
folders, so the model has to be INSIDE the image, and the image is built from the repository. deploy/bundle/ is the committed
copy of the model that is deployed. Run this again after every promotion, then commit and push: Render redeploys.

It copies only what the API loads (model.joblib and meta.json), and checks first that model.joblib matches the checksum
in meta.json, so a half-finished export is never deployed.
"""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "serving" / "bundle"
DST = ROOT / "deploy" / "bundle"
FILES = ["model.joblib", "meta.json"]
WARN_MB, LIMIT_MB = 50, 100          # GitHub warns above 50 MB and refuses files above 100 MB


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    for name in FILES:
        if not (SRC / name).exists():
            raise SystemExit(f"Missing {SRC / name}. Run: python serving/export_champion.py")
    meta = json.loads((SRC / "meta.json").read_text(encoding="utf-8"))
    if sha256(SRC / "model.joblib") != meta.get("model_sha256"):
        raise SystemExit("model.joblib does not match meta.json (an export is unfinished). Run the export again.")
    DST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(SRC / name, DST / name)
    size = (DST / "model.joblib").stat().st_size / 1e6
    print(f"Copied the bundle to {DST}")
    print(f"  model: {meta.get('registered_model')} version {meta.get('version')} ({meta.get('model_family')}), "
          f"exported {meta.get('exported_at')}")
    print(f"  model.joblib {size:.1f} MB, checksum verified")
    if size > LIMIT_MB:
        raise SystemExit(f"model.joblib is {size:.0f} MB: GitHub refuses files above {LIMIT_MB} MB (use Git LFS).")
    if size > WARN_MB:
        print(f"  WARNING: GitHub warns for files above {WARN_MB} MB.")
    print("Next: git add deploy/bundle && git commit && git push")


if __name__ == "__main__":
    main()
