"""Test setup. The scripts look for a data/ folder to find the project root, but data/ is not committed
(it is in .gitignore), so a fresh clone or a CI runner does not have one. Create an empty one here."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "data").mkdir(exist_ok=True)
for sub in ("", "code", "serving", "monitoring"):
    sys.path.insert(0, str(ROOT / sub) if sub else str(ROOT))
