"""
Clean the UCI Diabetes 130-US Hospitals dataset (v2).

Run from the project root:
    python data/data_cleaning.py

Input:   data/diabetic_data.csv   (never modified)
Outputs: data/diabetic_data_cleaned.csv   readable values, NOT integer-encoded
         data/cleaning_config.json        settings + exclusions used
         data/cleaning_report.txt         counts for every step

Design choices
- The cleaned file keeps readable string categories. Encoding (one-hot /
  target / ordinal) belongs in the training pipeline, fit on train data only,
  so unseen categories are handled and the same code runs in the API.
- Phase 1 = corrections (always on). Phase 2 = feature engineering and
  experiments (switch on/off in the CONFIG block).
- This script does not train a model.
"""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "diabetic_data.csv"
CLEANED_FILE = DATA_DIR / "diabetic_data_cleaned.csv"
CONFIG_FILE = DATA_DIR / "cleaning_config.json"
REPORT_FILE = DATA_DIR / "cleaning_report.txt"


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ID_COLUMNS = ["encounter_id", "patient_nbr"]  # tracking/splitting only, never features
TARGET_COLUMN = "readmitted"
RAW_TARGET_COLUMN = "readmitted_raw"

# weight is ~97% missing; examide and citoglipton are "No" in every row.
DROP_COLUMNS = ["weight", "examide", "citoglipton"]

NUMERIC_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]
LAB_COLUMNS = ["max_glu_serum", "A1Cresult"]
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
OTHER_CATEGORICAL_COLUMNS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty", "change", "diabetesMed",
]
DRUG_VALUES = {"No", "Down", "Steady", "Up"}

# Phase 1 exclusions. VERIFY these raw IDs against IDS_mapping.csv.
EXCLUDE_DISCHARGE_IDS = {"11", "13", "14", "19", "20", "21"}  # expired / hospice
INVALID_GENDER = "Unknown/Invalid"

# Phase 1, needs verification before trusting: merge "not available / null /
# not mapped" style IDs into one "unknown" category per column.
GROUP_UNKNOWN_IDS = True
UNKNOWN_ID_GROUPS = {
    "admission_type_id": {"5", "6", "8"},
    "discharge_disposition_id": {"18", "25", "26"},
    "admission_source_id": {"9", "15", "17", "20", "21"},
}

# Phase 2 (feature engineering / experiments)
ADD_DIAGNOSIS_GROUPS = True      # adds diag_1_group, diag_2_group, diag_3_group
ADD_DRUG_SUMMARY = True          # adds n_drugs_taken, n_drugs_changed
DROP_RARE_DRUGS = False          # experiment: compare with and without
RARE_DRUG_MIN_ROWS = 100         # non-"No" rows below this count as rare
ONE_ENCOUNTER_PER_PATIENT = False  # experiment: keeps the lowest encounter_id per patient
# Note: rare specialty/payer collapsing is NOT done here. Do it in the training
# pipeline (e.g. OneHotEncoder(min_frequency=...)) so it is fit on train only.


def icd9_group(code: str) -> str:
    """Broad ICD-9 group for the codes as they appear in this dataset."""
    if code in ("missing", ""):
        return "missing"
    if code.startswith("V"):
        return "supplementary"
    if code.startswith("E"):
        return "external"
    try:
        n = float(code)
    except ValueError:
        return "other"
    if 250 <= n < 251:
        return "diabetes"
    if 390 <= n <= 459 or n == 785:
        return "circulatory"
    if 460 <= n <= 519 or n == 786:
        return "respiratory"
    if 520 <= n <= 579 or n == 787:
        return "digestive"
    if 580 <= n <= 629 or n == 788:
        return "genitourinary"
    if 140 <= n <= 239:
        return "neoplasms"
    if 710 <= n <= 739:
        return "musculoskeletal"
    if 800 <= n <= 999:
        return "injury"
    if 1 <= n <= 139:
        return "infectious"
    return "other"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_FILE}")

    # Everything as string. Only "?" is missing, so the literal "None" in the
    # lab columns survives (it means "test not performed").
    df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False, na_values=["?"])
    df = df.apply(lambda s: s.str.strip())
    original_rows, original_columns = df.shape

    required = set(
        ID_COLUMNS + DROP_COLUMNS + NUMERIC_COLUMNS + DIAGNOSIS_COLUMNS + LAB_COLUMNS
        + DRUG_COLUMNS + OTHER_CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    )
    absent = sorted(required - set(df.columns))
    if absent:
        raise ValueError("Input CSV is missing expected columns: " + ", ".join(absent))
    if df["encounter_id"].duplicated().any():
        raise ValueError("encounter_id is not unique.")

    raw_missing = df.isna().sum()
    raw_missing = raw_missing[raw_missing > 0].sort_values(ascending=False)
    steps = [("Original rows", len(df))]

    # Keep the original target before anything else.
    df[RAW_TARGET_COLUMN] = df[TARGET_COLUMN]

    # ---- Phase 1: row exclusions, using ORIGINAL ids -------------------------
    mask = df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_IDS)
    excluded_by_id = df.loc[mask, "discharge_disposition_id"].value_counts().to_dict()
    df = df[~mask]
    steps.append(("After dropping expired/hospice discharges", len(df)))

    mask = df["gender"] == INVALID_GENDER
    df = df[~mask]
    steps.append(("After dropping invalid gender", len(df)))
    df = df.copy()

    # ---- Phase 1: missing values and special categories ----------------------
    for col in LAB_COLUMNS:
        df[col] = df[col].replace({"None": "not_measured"}).fillna("missing")
    for col in OTHER_CATEGORICAL_COLUMNS + DIAGNOSIS_COLUMNS:
        df[col] = df[col].fillna("missing")

    unknown_group_counts = {}
    if GROUP_UNKNOWN_IDS:
        for col, codes in UNKNOWN_ID_GROUPS.items():
            hit = df[col].isin(codes)
            unknown_group_counts[col] = int(hit.sum())
            df.loc[hit, col] = "unknown"

    # ---- Phase 1: target ------------------------------------------------------
    target_map = {"<30": 1, ">30": 0, "NO": 0}
    raw_target = df[RAW_TARGET_COLUMN].str.upper()
    bad = sorted(set(raw_target.dropna().unique()) - set(target_map))
    if bad or raw_target.isna().any():
        raise ValueError(f"Unexpected/missing values in {TARGET_COLUMN}: {bad}")
    df[TARGET_COLUMN] = raw_target.map(target_map).astype("int8")

    # ---- Phase 1: numeric columns and drug values ----------------------------
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_missing = {c: int(df[c].isna().sum()) for c in NUMERIC_COLUMNS if df[c].isna().any()}

    for col in DRUG_COLUMNS:
        unexpected = set(df[col].dropna().unique()) - DRUG_VALUES
        if unexpected or df[col].isna().any():
            raise ValueError(f"Unexpected values in {col}: {sorted(unexpected)}")

    # ---- Phase 2: engineered features (before any drug column is dropped) ----
    if ADD_DRUG_SUMMARY:
        df["n_drugs_taken"] = (df[DRUG_COLUMNS] != "No").sum(axis=1).astype("int8")
        df["n_drugs_changed"] = df[DRUG_COLUMNS].isin(["Up", "Down"]).sum(axis=1).astype("int8")
    if ADD_DIAGNOSIS_GROUPS:
        for col in DIAGNOSIS_COLUMNS:
            df[f"{col}_group"] = df[col].map(icd9_group)

    drug_nondefault = {c: int((df[c] != "No").sum()) for c in DRUG_COLUMNS}
    rare_drugs = sorted(c for c, n in drug_nondefault.items() if n < RARE_DRUG_MIN_ROWS)
    dropped_rare = []
    if DROP_RARE_DRUGS:
        df = df.drop(columns=rare_drugs)
        dropped_rare = rare_drugs

    if ONE_ENCOUNTER_PER_PATIENT:
        order = df["encounter_id"].astype("int64").argsort()
        df = df.iloc[order].drop_duplicates("patient_nbr", keep="first")
        steps.append(("After keeping one encounter per patient", len(df)))

    # ---- Drop agreed columns, then truly constant columns --------------------
    df = df.drop(columns=DROP_COLUMNS)
    check_cols = [c for c in df.columns if c not in ID_COLUMNS]
    constant = [c for c in check_cols if df[c].nunique(dropna=False) == 1]
    df = df.drop(columns=constant)

    # ---- Write outputs --------------------------------------------------------
    df.to_csv(CLEANED_FILE, index=False)

    config = {
        "exclude_discharge_ids": sorted(EXCLUDE_DISCHARGE_IDS),
        "invalid_gender": INVALID_GENDER,
        "group_unknown_ids": GROUP_UNKNOWN_IDS,
        "unknown_id_groups": {k: sorted(v) for k, v in UNKNOWN_ID_GROUPS.items()},
        "drop_columns": DROP_COLUMNS,
        "constant_columns_dropped": constant,
        "add_diagnosis_groups": ADD_DIAGNOSIS_GROUPS,
        "add_drug_summary": ADD_DRUG_SUMMARY,
        "drop_rare_drugs": DROP_RARE_DRUGS,
        "rare_drug_min_rows": RARE_DRUG_MIN_ROWS,
        "rare_drugs_dropped": dropped_rare,
        "one_encounter_per_patient": ONE_ENCOUNTER_PER_PATIENT,
        "target_mapping": target_map,
        "drug_values_in_cleaned_file": sorted(DRUG_VALUES),
        "steps": steps,
    }
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")

    pos_rate = df[TARGET_COLUMN].mean()
    per_patient = df["patient_nbr"].value_counts()
    lines = [
        "DATA CLEANING REPORT", "=" * 60,
        f"Input: {INPUT_FILE.name}  ({original_rows:,} rows x {original_columns} columns)",
        f"Output rows x columns: {df.shape[0]:,} x {df.shape[1]}",
        "", "Row counts by step:",
        *[f"- {name}: {n:,}" for name, n in steps],
        "", "Excluded discharge IDs (raw id: rows):",
        *([f"- {k}: {v}" for k, v in sorted(excluded_by_id.items())] or ["- none found"]),
        "", "Rows regrouped to 'unknown' per column:",
        *([f"- {k}: {v}" for k, v in unknown_group_counts.items()] or ["- grouping disabled"]),
        "", "Missing values in the RAW file (before filling):",
        *[f"- {k}: {v} ({v / original_rows:.1%})" for k, v in raw_missing.items()],
        "", "Numeric columns still missing after parsing:",
        *([f"- {k}: {v}" for k, v in numeric_missing.items()] or ["- none"]),
        "", f"Dropped columns: {DROP_COLUMNS}",
        f"Constant columns dropped automatically: {constant or 'none'}",
        f"Rare drug columns (<{RARE_DRUG_MIN_ROWS} non-'No' rows): {rare_drugs or 'none'}",
        f"Rare drug columns dropped: {dropped_rare or 'none (DROP_RARE_DRUGS is off)'}",
        "", "Target (1 = readmitted <30 days):",
        *[f"- {k}: {v:,}" for k, v in df[TARGET_COLUMN].value_counts().sort_index().items()],
        f"- positive rate: {pos_rate:.2%}",
        "", f"Unique patients: {per_patient.shape[0]:,}",
        f"Max encounters for one patient: {int(per_patient.max())}",
        "", "Notes:",
        "- IDs are kept for tracking/splitting; exclude them from model features.",
        f"- {RAW_TARGET_COLUMN} keeps the original three-class label. Do not use it as a feature.",
        "- 'NO' readmission means none observed, not proof of complete follow-up.",
        "- Categories are readable strings. Encode in the training pipeline, fit on train only.",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nFirst 5 cleaned rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()