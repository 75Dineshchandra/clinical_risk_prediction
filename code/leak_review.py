"""
Leak review for the cleaned Diabetes 130-US Hospitals data (step 4).

Run from the project root, after data_cleaning.py:
    python data/leak_review.py

Input:   data/diabetic_data_cleaned.csv
Outputs: data/leak_review_report.txt     results of every check
         data/leak_review_columns.csv    one row per column: verdict + reason

Needs: pandas, numpy, scikit-learn.
This script does not save a model or a split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_FILE = DATA_DIR / "diabetic_data_cleaned.csv"
REPORT_FILE = DATA_DIR / "leak_review_report.txt"
COLUMNS_FILE = DATA_DIR / "leak_review_columns.csv"

TARGET = "readmitted"
RAW_TARGET = "readmitted_raw"
PATIENT = "patient_nbr"
ENCOUNTER = "encounter_id"

SUSPECT_SINGLE_AUC = 0.70   # one column alone above this deserves a look
SUSPECT_MODEL_AUC = 0.75    # whole-model AUC above this is a warning sign
MIN_GROUP_ROWS = 50         # ignore tiny categories in rate tables
RANDOM_STATE = 42

DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def build_review() -> dict:
    """Column -> (verdict, reason). Verdicts: EXCLUDE, KEEP, KEEP-WATCH."""
    r = {
        ENCOUNTER: ("EXCLUDE", "Identifier. May act as a time proxy; not guaranteed chronological."),
        PATIENT: ("EXCLUDE", "Identifier. Lets a model memorize patients. Use only to group the split."),
        RAW_TARGET: ("EXCLUDE", "The target in its original three-class form."),
        TARGET: ("TARGET", "Label: 1 = readmitted in <30 days."),
        "race": ("KEEP-WATCH", "Known at admission. Sensitive: keep at least for fairness slices; decide whether the model sees it."),
        "gender": ("KEEP-WATCH", "Known at admission. Sensitive: same as race."),
        "age": ("KEEP-WATCH", "Known at admission. Sensitive: same as race."),
        "admission_type_id": ("KEEP", "Known at admission."),
        "admission_source_id": ("KEEP", "Known at admission."),
        "discharge_disposition_id": ("KEEP-WATCH", "Decided at discharge, so usable if scoring happens at discharge. Strong signal: check first if AUC looks too high."),
        "time_in_hospital": ("KEEP", "Length of stay, known at discharge."),
        "payer_code": ("KEEP", "Known during the stay. Mostly missing."),
        "medical_specialty": ("KEEP", "Admitting physician specialty. Mostly missing."),
        "num_lab_procedures": ("KEEP", "Count during the stay."),
        "num_procedures": ("KEEP", "Count during the stay."),
        "num_medications": ("KEEP", "Count during the stay."),
        "number_outpatient": ("KEEP", "Visits in the year BEFORE the encounter."),
        "number_emergency": ("KEEP", "Visits in the year BEFORE the encounter."),
        "number_inpatient": ("KEEP", "Visits in the year BEFORE the encounter."),
        "number_diagnoses": ("KEEP-WATCH", "Diagnosis entries may be finalized after discharge in real systems."),
        "max_glu_serum": ("KEEP", "Test done during the stay (or not_measured)."),
        "A1Cresult": ("KEEP", "Test done during the stay (or not_measured)."),
        "change": ("KEEP", "Medication change during the stay."),
        "diabetesMed": ("KEEP", "Diabetic medication prescribed during the stay."),
        "n_drugs_taken": ("KEEP", "Derived from the drug columns."),
        "n_drugs_changed": ("KEEP", "Derived from the drug columns."),
    }
    for c in ("diag_1", "diag_2", "diag_3", "diag_1_group", "diag_2_group", "diag_3_group"):
        r[c] = ("KEEP-WATCH", "Diagnosis codes may be finalized after discharge in real systems.")
    for c in DRUG_COLUMNS:
        r[c] = ("KEEP", "Medication given during the stay.")
    return r


def two_sided_auc(score, y) -> float:
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def oof_target_encoding_auc(x: pd.Series, y: pd.Series, groups, n_splits=5) -> float:
    """AUC of a category's out-of-fold mean target rate, folds grouped by patient."""
    n_splits = min(n_splits, groups.nunique())
    oof = np.zeros(len(x))
    for tr, va in GroupKFold(n_splits=n_splits).split(x, y, groups):
        means = y.iloc[tr].groupby(x.iloc[tr]).mean()
        oof[va] = x.iloc[va].map(means).fillna(y.iloc[tr].mean()).to_numpy()
    return roc_auc_score(y, oof)


def grouped_model_auc(df, features, y, groups):
    """Grouped 75/25 split, gradient boosting, returns test AUC."""
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE).split(df, y, groups))
    cat = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in features if c not in cat]
    parts_tr, parts_te = [], []
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, dtype=float)
        parts_tr.append(pd.DataFrame(enc.fit_transform(df.iloc[tr][cat]), columns=cat))
        parts_te.append(pd.DataFrame(enc.transform(df.iloc[te][cat]), columns=cat))
    if num:
        parts_tr.append(df.iloc[tr][num].reset_index(drop=True))
        parts_te.append(df.iloc[te][num].reset_index(drop=True))
    Xtr, Xte = pd.concat(parts_tr, axis=1), pd.concat(parts_te, axis=1)
    mask = [True] * len(cat) + [False] * len(num)
    model = HistGradientBoostingClassifier(
        categorical_features=mask, max_iter=150, early_stopping=False, random_state=RANDOM_STATE
    )
    model.fit(Xtr, y.iloc[tr])
    return roc_auc_score(y.iloc[te], model.predict_proba(Xte)[:, 1])


def main() -> None:
    if not CLEANED_FILE.exists():
        raise FileNotFoundError(f"Run data_cleaning.py first. Missing: {CLEANED_FILE}")

    df = pd.read_csv(CLEANED_FILE, dtype={ENCOUNTER: str, PATIENT: str})
    y = df[TARGET].astype(int)
    groups = df[PATIENT]
    base_rate = y.mean()
    lines = ["LEAK REVIEW REPORT", "=" * 60,
             f"Rows: {len(df):,}   Patients: {groups.nunique():,}   Positive rate: {base_rate:.2%}", ""]

    # ---- 1. Column-by-column table -------------------------------------------
    review = build_review()
    undecided = [c for c in df.columns if c not in review]
    if undecided:
        raise ValueError(f"No leak-review verdict for new columns: {undecided}. Add them to build_review().")
    table = pd.DataFrame(
        [(c, *review[c]) for c in df.columns], columns=["column", "verdict", "reason"]
    )
    table.to_csv(COLUMNS_FILE, index=False)
    lines += ["1. COLUMN VERDICTS (full table in leak_review_columns.csv)",
              *[f"- {v}: {', '.join(table.loc[table.verdict == v, 'column'])}"
                for v in ("EXCLUDE", "KEEP-WATCH")],
              f"- KEEP: {(table.verdict == 'KEEP').sum()} columns", ""]

    # ---- 2. Single numeric columns -------------------------------------------
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
               and c not in (TARGET,) and review[c][0] != "EXCLUDE"]
    aucs = pd.Series({c: two_sided_auc(df[c], y) for c in numeric}).sort_values(ascending=False)
    lines += [f"2. SINGLE-COLUMN AUC, numeric columns (flag above {SUSPECT_SINGLE_AUC})",
              *[f"- {c}: {a:.3f}{'  <-- SUSPECT' if a > SUSPECT_SINGLE_AUC else ''}" for c, a in aucs.items()], ""]

    # ---- 3. Identifiers as predictors ----------------------------------------
    enc_auc = two_sided_auc(pd.to_numeric(df[ENCOUNTER]), y)
    pat_auc = two_sided_auc(pd.to_numeric(df[PATIENT]), y)
    lines += ["3. IDENTIFIERS AS PREDICTORS (should be close to 0.5)",
              f"- {ENCOUNTER}: {enc_auc:.3f}{'  <-- SUSPECT: time proxy?' if enc_auc > 0.55 else ''}",
              f"- {PATIENT}: {pat_auc:.3f}", ""]

    # ---- 4. Categorical columns, out-of-fold target encoding -----------------
    cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
            and review[c][0] not in ("EXCLUDE", "TARGET")]
    cat_aucs = pd.Series({c: oof_target_encoding_auc(df[c], y, groups) for c in cats}).sort_values(ascending=False)
    lines += [f"4. CATEGORICAL COLUMNS, out-of-fold AUC grouped by patient (flag above {SUSPECT_SINGLE_AUC})",
              *[f"- {c}: {a:.3f}{'  <-- SUSPECT' if a > SUSPECT_SINGLE_AUC else ''}" for c, a in cat_aucs.items()], ""]

    # ---- 5. Readmission rate by key categories -------------------------------
    lines.append(f"5. READMISSION RATE BY CATEGORY (categories with at least {MIN_GROUP_ROWS} rows)")
    for col in ("discharge_disposition_id", "admission_source_id", "admission_type_id", "A1Cresult", "diag_1_group"):
        if col not in df.columns:
            continue
        g = df.groupby(col)[TARGET].agg(rate="mean", n="size")
        g = g[g.n >= MIN_GROUP_ROWS].sort_values("rate")
        lines.append(f"- {col}")
        lines += [f"    {idx}: rate {row.rate:.1%}, n={int(row.n):,}" for idx, row in g.iterrows()]
    lines.append("")
    if "diag_1_group" in df.columns:
        share = df["diag_1_group"].value_counts(normalize=True)
        lines += ["   diag_1_group share of rows:",
                  *[f"    {k}: {v:.1%}" for k, v in share.items()], ""]

    # ---- 6. Split leakage demonstration --------------------------------------
    perm = np.random.RandomState(RANDOM_STATE).permutation(len(df))
    cut = int(0.8 * len(df))
    seen = set(groups.iloc[perm[:cut]])
    random_overlap = groups.iloc[perm[cut:]].isin(seen).mean()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE).split(df, y, groups))
    grouped_overlap = len(set(groups.iloc[tr]) & set(groups.iloc[te]))
    lines += ["6. SPLIT LEAKAGE",
              f"- Random row split: {random_overlap:.1%} of test rows belong to a patient also in train",
              f"- Patient-grouped split: {grouped_overlap} shared patients (must be 0)", ""]
    assert grouped_overlap == 0

    # ---- 7. Whole-model sanity check -----------------------------------------
    # Raw diag columns have too many categories for the boosting model; use the group columns.
    base_features = [c for c in df.columns if review[c][0] in ("KEEP", "KEEP-WATCH")
                     and c not in ("diag_1", "diag_2", "diag_3")]
    df["_encounter_numeric"] = pd.to_numeric(df[ENCOUNTER])
    runs = {
        "allowed features": base_features,
        "allowed features without discharge_disposition_id": [c for c in base_features if c != "discharge_disposition_id"],
        "allowed features + encounter_id (leak demo)": base_features + ["_encounter_numeric"],
    }
    lines += [f"7. GROUPED-SPLIT GRADIENT BOOSTING AUC (warning above {SUSPECT_MODEL_AUC})"]
    for name, feats in runs.items():
        a = grouped_model_auc(df, feats, y, groups)
        lines.append(f"- {name}: {a:.3f}{'  <-- SUSPECT' if a > SUSPECT_MODEL_AUC else ''}")
    lines += ["",
              "How to read this:",
              "- Published models on this dataset land around 0.6-0.7 AUC. Much higher means look for a leak.",
              "- If 'without discharge_disposition_id' drops a lot, the model leans on that column.",
              "- If '+ encounter_id' beats 'allowed features', encounter_id carries time information.",
              "- Write your own decision next to every KEEP-WATCH column before training."]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()