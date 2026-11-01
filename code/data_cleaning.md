# Data Cleaning: Diabetes 130-US Hospitals (1999-2008)

This document describes what `data/data_cleaning.py` does and what it produces. The script cleans the raw UCI dataset. It does not train a model, split the data, or encode categories.

Source: [UCI Diabetes 130-US Hospitals](https://doi.org/10.24432/C5230J) (CC BY 4.0).

## Run

```bash
# from the project root
python data_cleaning.py
```

| | File | Notes |
|---|---|---|
| Input | `data/diabetic_data.csv` | Raw data. Never modified. |
| Reference | `data/IDS_mapping.csv` | Used to verify the exclusion and "unknown" ID lists. |
| Output | `data/diabetic_data_cleaned.csv` | Cleaned dataset with readable string values. |
| Output | `data/cleaning_config.json` | Settings, exclusions and row counts per step. |
| Output | `data/cleaning_report.txt` | Summary of the run. |

## Results of the Run

| Metric | Value |
|---|---|
| Raw rows x columns | 101,766 x 50 |
| Cleaned rows x columns | 99,340 x 53 |
| Unique patients | 69,987 |
| Max encounters for one patient | 40 |
| `readmitted` = 1 | 11,314 (11.39%) |
| `readmitted` = 0 | 88,026 |

## Steps

### 1. Load
- Reads every column as a string, so IDs and diagnosis codes keep their exact values.
- Only `?` is treated as missing. The literal `None` is kept, because in the lab columns it means "test not performed".
- Trims whitespace, checks that all expected columns exist, and checks that `encounter_id` is unique.
- Saves the original target as `readmitted_raw`.

### 2. Drop rows
Filtering uses the original ID values, before any other change.

| Exclusion | Rows removed |
|---|---|
| `discharge_disposition_id` 11 (Expired) | 1,642 |
| 13 (Hospice / home) | 399 |
| 14 (Hospice / medical facility) | 372 |
| 19 (Expired at home, Medicaid hospice) | 8 |
| 20 (Expired in a medical facility, Medicaid hospice) | 2 |
| 21 (Expired, place unknown) | 0 found |
| `gender` = `Unknown/Invalid` | 3 |

Row counts: 101,766 -> 99,343 (after discharge exclusions) -> 99,340 (after gender).

### 3. Missing and special values

| Column(s) | Treatment |
|---|---|
| `max_glu_serum`, `A1Cresult` | `None` becomes `not_measured` |
| Other categorical and diagnosis columns | Missing (`?`) becomes `missing` |
| `admission_type_id` | IDs 5, 6, 8 become `unknown` (10,144 rows) |
| `discharge_disposition_id` | IDs 18, 25, 26 become `unknown` (4,680 rows) |
| `admission_source_id` | IDs 9, 15, 17, 20, 21 become `unknown` (6,854 rows) |

The unknown-ID lists were checked against `IDS_mapping.csv`.

Missing values in the raw file: `weight` 96.9%, `medical_specialty` 49.1%, `payer_code` 39.6%, `race` 2.2%, `diag_3` 1.4%, `diag_2` 0.4%, `diag_1` 0.0%.

### 4. Target
- `readmitted`: `<30` -> 1, `>30` -> 0, `NO` -> 0. Any other value stops the script.
- `readmitted_raw` keeps the original three-class label.

### 5. Validation
- The eight count columns are parsed as numbers. None have missing or invalid values.
- Every drug column must contain only `No`, `Down`, `Steady` or `Up`. Anything else stops the script.

### 6. Added columns

| New column | Definition |
|---|---|
| `n_drugs_taken` | Number of drug columns with a value other than `No` |
| `n_drugs_changed` | Number of drug columns with `Up` or `Down` |
| `diag_1_group`, `diag_2_group`, `diag_3_group` | Broad ICD-9 category: diabetes, circulatory, respiratory, digestive, genitourinary, neoplasms, musculoskeletal, injury, infectious, supplementary (V-codes), external (E-codes), other, missing |

Leading zeros are missing in the source (for example `38`, `8`), so codes 1 to 139 are grouped as `infectious`.

### 7. Dropped columns

| Column | Reason |
|---|---|
| `weight` | 96.9% missing |
| `examide` | Constant (`No` in every row) |
| `citoglipton` | Constant (`No` in every row) |

Any other column with a single unique value is also dropped automatically. None were found.

50 raw columns - 3 dropped + 6 added = **53 columns**.

## Switches (CONFIG block at the top of the script)

| Setting | Default | Purpose |
|---|---|---|
| `GROUP_UNKNOWN_IDS` | `True` | Merge "not available / null / not mapped" IDs into `unknown` |
| `ADD_DIAGNOSIS_GROUPS` | `True` | Add the `diag_*_group` columns |
| `ADD_DRUG_SUMMARY` | `True` | Add `n_drugs_taken` and `n_drugs_changed` |
| `DROP_RARE_DRUGS` | `False` | Drop drug columns with fewer than `RARE_DRUG_MIN_ROWS` non-`No` rows |
| `RARE_DRUG_MIN_ROWS` | `100` | Threshold for the rare-drug check |
| `ONE_ENCOUNTER_PER_PATIENT` | `False` | Keep only the lowest `encounter_id` per patient |

Rare drug columns flagged in the report but not dropped: `acetohexamide`, `chlorpropamide`, `glimepiride-pioglitazone`, `glipizide-metformin`, `metformin-pioglitazone`, `metformin-rosiglitazone`, `miglitol`, `tolazamide`, `tolbutamide`, `troglitazone`.

## Output Format

- Categories are readable strings (for example `Caucasian`, `Up`, `not_measured`), not integer codes.
- Read the ID columns as strings:

```python
import pandas as pd

df = pd.read_csv(
    "data/diabetic_data_cleaned.csv",
    dtype={"encounter_id": str, "patient_nbr": str},
)
```

Example row (first encounter):

| Column | Value |
|---|---|
| `race`, `gender`, `age` | Caucasian, Female, [0-10) |
| `admission_type_id`, `discharge_disposition_id` | unknown, unknown |
| `medical_specialty` | Pediatrics-Endocrinology |
| `diag_1`, `diag_1_group` | 250.83, diabetes |
| `max_glu_serum`, `A1Cresult` | not_measured, not_measured |
| `readmitted`, `readmitted_raw` | 0, NO |

## Notes from the Script's Report

- `encounter_id` and `patient_nbr` are kept for tracking and splitting, not as model features.
- `readmitted_raw` is the target in its original form, not a feature.
- `NO` readmission means none was observed, not proof of complete follow-up.

## Reproducibility

The raw CSV is never overwritten. `cleaning_config.json` records every setting, exclusion list and row count, so a run can be audited and repeated.