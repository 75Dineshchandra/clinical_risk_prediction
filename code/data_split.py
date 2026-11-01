"""
Split the cleaned data into train / test / holdout (step 5).

Run from the project root, after data_cleaning.py:
    python data/split_data.py

Input:   data/diabetic_data_cleaned.csv
Outputs: data/splits/train.csv               all cleaned columns, train rows only
         data/splits/test.csv                all cleaned columns, test rows only
         data/splits/holdout.csv             all cleaned columns, holdout rows only
         data/splits/split_assignments.csv   encounter_id, patient_nbr, split (source of truth)
         data/splits/split_report.txt        sizes, positive rates, checks

Method
- Split BY PATIENT (patient_nbr): all encounters of a patient go to one split.
- Stratified on the label, so each split keeps about the same positive rate.
- 60% train / 15% test / 25% holdout. The holdout is held back for the
  replay simulation (step 11). Do not use it for model selection or tuning.
- There is no separate validation set. Tune with cross-validation inside train,
  grouped by patient_nbr.
- Fixed random seed, so the split is reproducible.

Needs: pandas, numpy, scikit-learn (1.0 or newer).
This script does not train a model or change the cleaned data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_FILE = DATA_DIR / "diabetic_data_cleaned.csv"
SPLIT_DIR = DATA_DIR / "splits"
ASSIGNMENTS_FILE = SPLIT_DIR / "split_assignments.csv"
SPLIT_FILES = {name: SPLIT_DIR / f"{name}.csv" for name in ("train", "test", "holdout")}
REPORT_FILE = SPLIT_DIR / "split_report.txt"

TARGET = "readmitted"
RAW_TARGET = "readmitted_raw"
PATIENT = "patient_nbr"
ENCOUNTER = "encounter_id"

SEED = 42
N_FOLDS = 20                                   # each fold is about 5% of patients
FOLDS_PER_SPLIT = {"train": 12, "test": 3, "holdout": 5}   # 60% / 15% / 25%
SIZE_TOLERANCE = 0.015                         # warn if a split is >1.5 points off target


def main() -> None:
    if not CLEANED_FILE.exists():
        raise FileNotFoundError(f"Run data_cleaning.py first. Missing: {CLEANED_FILE}")
    assert sum(FOLDS_PER_SPLIT.values()) == N_FOLDS

    df = pd.read_csv(CLEANED_FILE, dtype={ENCOUNTER: str, PATIENT: str})
    y = df[TARGET].astype(int)
    groups = df[PATIENT]

    # ---- Assign every encounter to a fold, grouped by patient, stratified ----
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(df), -1)
    for i, (_, idx) in enumerate(sgkf.split(df, y, groups)):
        fold[idx] = i
    assert (fold >= 0).all(), "Some rows were not assigned to a fold."

    # ---- Map folds to splits ---------------------------------------------------
    fold_to_split, nxt = {}, 0
    for name, n in FOLDS_PER_SPLIT.items():
        for f in range(nxt, nxt + n):
            fold_to_split[f] = name
        nxt += n
    df["split"] = pd.Series(fold).map(fold_to_split).to_numpy()

    # ---- Hard checks -----------------------------------------------------------
    assert df["split"].notna().all()
    assert df[ENCOUNTER].is_unique
    patient_sets = {s: set(df.loc[df.split == s, PATIENT]) for s in FOLDS_PER_SPLIT}
    names = list(FOLDS_PER_SPLIT)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a} & {b}"] = len(patient_sets[a] & patient_sets[b])
    assert all(v == 0 for v in overlaps.values()), f"Patient overlap between splits: {overlaps}"

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df[[ENCOUNTER, PATIENT, "split"]].to_csv(ASSIGNMENTS_FILE, index=False)
    data_columns = [c for c in df.columns if c != "split"]
    rows_written = {}
    for name, path in SPLIT_FILES.items():
        part = df.loc[df["split"] == name, data_columns]
        part.to_csv(path, index=False)
        rows_written[name] = len(part)
    assert sum(rows_written.values()) == len(df), "Split files do not add up to the cleaned data."

    # ---- Report ----------------------------------------------------------------
    target_share = {k: v / N_FOLDS for k, v in FOLDS_PER_SPLIT.items()}
    lines = ["SPLIT REPORT", "=" * 60,
             f"Method: StratifiedGroupKFold({N_FOLDS} folds, shuffle, seed={SEED}), grouped by {PATIENT}",
             f"Rows: {len(df):,}   Patients: {groups.nunique():,}   Overall positive rate: {y.mean():.2%}", "",
             "Split sizes",
             f"{'split':<9}{'target':>8}{'rows':>10}{'row share':>11}{'patients':>10}{'positives':>11}{'pos rate':>10}"]
    warnings = []
    for s in names:
        sub = df[df.split == s]
        share = len(sub) / len(df)
        lines.append(f"{s:<9}{target_share[s]:>8.0%}{len(sub):>10,}{share:>11.1%}"
                     f"{sub[PATIENT].nunique():>10,}{int(sub[TARGET].sum()):>11,}{sub[TARGET].mean():>10.2%}")
        if abs(share - target_share[s]) > SIZE_TOLERANCE:
            warnings.append(f"{s}: row share {share:.1%} is more than {SIZE_TOLERANCE:.1%} from target {target_share[s]:.0%}")
    lines += ["", "Files written (rows)",
              *[f"- {path.name}: {rows_written[name]:,}" for name, path in SPLIT_FILES.items()],
              "", "Patient overlap between splits (must be 0)",
              *[f"- {k}: {v}" for k, v in overlaps.items()],
              "", "Every encounter assigned exactly once: yes"]
    if warnings:
        lines += ["", "WARNINGS", *[f"- {w}" for w in warnings]]

    # ---- Unseen categories: test/holdout values not present in train ------------
    cat_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                and c not in (ENCOUNTER, PATIENT, RAW_TARGET, "split")]
    train = df[df.split == "train"]
    lines += ["", "Rows in test/holdout whose category never appears in train (columns with at least one)"]
    found = False
    for s in ("test", "holdout"):
        sub = df[df.split == s]
        for c in cat_cols:
            unseen = ~sub[c].isin(set(train[c]))
            if unseen.any():
                found = True
                vals = sorted(sub.loc[unseen, c].unique().tolist())
                lines.append(f"- {s}.{c}: {int(unseen.sum())} rows, values {vals[:8]}{' ...' if len(vals) > 8 else ''}")
    if not found:
        lines.append("- none")

    lines += ["", "Notes",
              "- Do not use the holdout for tuning or model selection; it is for the step 11 replay.",
              "- Tune with cross-validation inside train, grouped by patient_nbr.",
              "- Fit all preprocessing on train only.",
              f"- {ASSIGNMENTS_FILE.name} is the source of truth. If data_cleaning.py is re-run, re-run this script too.",
              "- Read the ID columns as strings: dtype={'encounter_id': str, 'patient_nbr': str}."]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()