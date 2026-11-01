# Leak Review: Diabetes 130-US Hospitals (Step 4)

This document describes what `data/leak_review.py` does, what it found on this dataset, and answers the questions that came up while building it.

## In One Line

Leak review checks that every feature would actually be known at the moment the prediction is made, so test results are not inflated by information the model should not have.

## Run

```bash
# from the project root, after data_cleaning.py
python leak_review.py
```

Requires pandas, numpy and scikit-learn.

| | File | Notes |
|---|---|---|
| Input | `data/diabetic_data_cleaned.csv` | Output of `data_cleaning.py` |
| Output | `data/leak_review_report.txt` | Results of every check |
| Output | `data/leak_review_columns.csv` | One row per column: verdict and reason |

The script does not save a model or a data split.

## Why Do a Leak Review

The model's job is to predict, at discharge, whether a patient will return within 30 days. A leak is any feature or setup that gives the model information it would not have at that moment. A leaky model scores well in testing and fails on real patients.

Types of leakage checked here:

1. **Features that reveal the answer.** A column filled in after the outcome, or the target in another form (`readmitted_raw`).
2. **Identifiers.** `patient_nbr` lets a model memorize patients instead of learning general patterns.
3. **Time proxies.** `encounter_id` may increase with time, so a model could learn the calendar.
4. **Information from the future.** Columns that real systems fill in after discharge, such as final diagnosis codes.
5. **Split leakage.** The same patient in both train and test. About 30% of the rows in this data are repeat visits.

## What the Script Does

### How columns are decided
The decision is made by a **hand-written table**, not by the statistics. `build_review()` holds a verdict and a reason for every column:

| Verdict | Meaning |
|---|---|
| `EXCLUDE` | Must not be a model feature (`encounter_id`, `patient_nbr`, `readmitted_raw`) |
| `KEEP-WATCH` | Allowed, but depends on a decision about when the model is used or on sensitivity |
| `KEEP` | Known by discharge |
| `TARGET` | The label (`readmitted`) |

The script stops with an error if any column has no verdict, so new columns cannot slip through undecided. The verdicts are proposals that you confirm or change.

### The checks
The statistical checks do not remove anything. They flag columns that deserve a second look.

| # | Check | What it measures | Flag |
|---|---|---|---|
| 1 | Column verdict table | "Known at discharge?" with a reason per column | Missing verdict stops the script |
| 2 | Single-column AUC (numeric columns) | How well one column alone separates readmitted from not readmitted | Above 0.70 |
| 3 | Identifiers as predictors | AUC of `encounter_id` and `patient_nbr` | Should be near 0.5 |
| 4 | Categorical columns | Out-of-fold AUC of each category's readmission rate, with folds grouped by patient | Above 0.70 |
| 5 | Rate tables | Readmission rate by category for discharge disposition, admission source, admission type, `A1Cresult`, `diag_1_group`, plus the share of rows per diagnosis group | Visual review |
| 6 | Split leakage | Share of test rows from a patient also in train under a random split, against the patient-grouped split | Grouped split must share 0 patients |
| 7 | Whole-model sanity check | Gradient boosting AUC on a patient-grouped 75/25 split, run three ways (see below) | Above 0.75 |

Section 7 runs three variants:
1. Allowed features.
2. Allowed features without `discharge_disposition_id` (shows how much the model leans on it).
3. Allowed features plus `encounter_id` (a deliberate leak demo: a gain means `encounter_id` carries time information).

## Results on This Dataset

| Metric | Value |
|---|---|
| Rows | 99,340 |
| Patients | 69,987 |
| Positive rate | 11.39% |

| Check | Result |
|---|---|
| Highest single numeric AUC | `number_inpatient` 0.607. Nothing near 0.70. |
| `encounter_id` / `patient_nbr` AUC | 0.508 / 0.510 |
| Highest categorical AUC | 0.573 (`diag_1` and `discharge_disposition_id`) |
| Random row split | 41.5% of test rows share a patient with train |
| Patient-grouped split | 0 shared patients |
| Gradient boosting, allowed features | **0.669** |
| Without `discharge_disposition_id` | 0.638 |
| With `encounter_id` added | 0.672 (gain of 0.003, noise) |

**Conclusion: no leak was flagged.** No check exceeded its threshold, and the whole-model AUC is below the 0.75 warning line.

### Readmission rate by discharge disposition
The cleaned file keeps the original IDs (decode with `IDS_mapping.csv`). The rates make clinical sense:

| ID | Meaning | Rate | Rows |
|---|---|---|---|
| 23 | Long-term care hospital | 7.3% | 412 |
| 1 | Home | 9.3% | 60,232 |
| 5 | Another inpatient institution | 20.9% | 1,184 |
| 22 | Rehab facility | 27.7% | 1,992 |
| 28 | Psychiatric hospital | 36.7% | 139 |
| 15 | Swing bed (within institution) | 44.4% | 63 |

The extremes are small groups, so do not over-read them.

### Diagnosis groups
The `other` group is 13.7% of rows and `missing` and `external` are about 0%, so the grouping works as it is. The `supplementary` (V-code) group has the highest rate (16.2%).

## Decisions Still Open

| Column(s) | Decision needed |
|---|---|
| `discharge_disposition_id`, `diag_*`, `diag_*_group`, `number_diagnoses` | Fine if the model is scored at discharge **and** the diagnoses are already coded at that point. Write the prediction time down as "at discharge". If diagnoses might not exist yet, the model has to run without them. |
| `race`, `gender`, `age` | Keep at least for the fairness slices (step 8). Whether the model itself sees them is a choice. One option is to train with and without them and compare. |

Final exclude list: `encounter_id`, `patient_nbr`, `readmitted_raw`.

## Questions and Answers

### How does the review decide which columns to drop?
It does not drop anything automatically. The hand-written table decides, and the statistical checks only point at suspects. A high AUC is a flag, not proof: a column can be highly predictive for a legitimate reason (for example `number_inpatient`). A column that scores high **and** has no good reason to be known at discharge is a leak.

### What model does it use, and why?
Only section 7 trains a model: scikit-learn's `HistGradientBoostingClassifier` (150 iterations, patient-grouped 75/25 split). It handles categorical and missing values natively, captures nonlinear patterns that would reveal a leak, and trains quickly on 99k rows. Sections 2 to 4 use no model.

### Does section 7 use all the features, and are they numeric?
It uses 46 features. The file has 53 columns. Removing the target and the three EXCLUDE columns leaves 49, and the raw `diag_1`, `diag_2` and `diag_3` are dropped because each has far more than 255 distinct values, the limit for a categorical column in that model. The three `diag_*_group` columns stand in for them.

Numeric columns pass through unchanged. Text categories are turned into integer codes with `OrdinalEncoder`, fit on the training rows only, and the model is told which columns are categorical so it treats the codes as labels and ignores their order.

This encoding exists only for the sanity check. It is not the final preprocessing, and the AUC from section 7 is for leak detection only, not a performance figure to report.

### What happens to categories in the test rows that were not in the training rows?
In this script they become missing values (NaN). The affected rows are few (a handful of rare categories), so it does not distort the review. It would matter later, for three reasons:

- It hides drift. New categories appearing in a batch are a kind of drift, and silently turning them into NaN means nothing signals that the input changed.
- It mixes two meanings. NaN would stand for both "unseen category" and "missing".
- The serving code must preprocess the same way as training.

In the real pipeline:
1. Use `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=...)` so rare and unseen categories go into one explicit bucket.
2. Define fixed category lists where the vocabulary is known (ID columns from `IDS_mapping.csv`, four values for each drug column).
3. Monitor the share of rows with an unseen category as a drift metric (step 13).
4. Validate allowed values and plausible ranges at the API.
5. Save the fitted encoder with the model in MLflow.

### How is this different from VIF and a correlation matrix?
Leak review compares each feature to the target, to catch columns that give away the answer. Correlation and VIF compare features to each other, to catch columns that just repeat one another. They answer different questions, and a column can pass one and fail the other.

- `n_drugs_taken` is not a leak (it is known at discharge) but overlaps with the drug columns it was built from.
- A leaky column could be unique, with no overlap with any other feature, so correlation and VIF would never flag it.

Multicollinearity matters mostly for the logistic regression baseline, where redundant features make coefficients unstable. Tree models such as XGBoost and LightGBM largely ignore it.

Correlation and VIF are **not** part of `leak_review.py`. A reasonable plan:
- Numeric columns: Spearman correlation (the counts are skewed) and VIF with `statsmodels` (above 5 is a concern, above 10 is serious).
- Categorical columns: Cramér's V, which is biased upward for columns with many categories.
- Run them on the training split only, once the split exists.
- Do not remove a column just because it correlates with another. Check first whether it is a leak.

## Limits of This Review

- It catches strong leaks. A weak or subtle one can slip through, and the 0.669 comes from one split with an untuned model.
- The data has no dates, so time effects cannot be ruled out entirely. The `encounter_id` test only shows that no time signal is visible through it.
- The "published models land around 0.6 to 0.7 AUC" benchmark is from memory of the original literature and has not been verified. Use it as a rough guide, and treat a large jump above your other runs as the real warning sign.

## Next Step

Step 5: split the data 60% train, 15% test and 25% held back, grouped by `patient_nbr`, saving the patient IDs for each split and asserting zero overlap.