# Readmission model report

Created 2026-11-01 10:47. Generated automatically by `train_models.py` and replaced on every run.

## What this report shows

We trained models that give each diabetic hospital patient a risk score for being readmitted within 30 days of discharge. This report compares the models and explains the numbers. Every result here comes from the training data only (cross-validation). The test data has not been used yet, so these are **not final results**.

## The data in brief

- 59,753 hospital stays from 42,054 patients (the training file, data version `rebuild-v1`).
- About 11.5% of stays ended in a readmission within 30 days. This is the minority outcome, so the data is imbalanced and plain accuracy would be misleading.
- 46 pieces of information per stay: age, length of stay, prior visits, lab tests, medications, diagnosis groups, and so on.
- Stays from the same patient are always kept together when checking the model, so each model is judged on patients it has not seen.

## Models compared

| Model | What it is |
|---|---|
| Logistic regression | A simple model that adds up weighted evidence from each feature. Easy to explain. It is the baseline: any more complex model has to beat it to be worth using. |
| XGBoost | Many small decision trees built one after another, each fixing the mistakes of the previous ones. It can pick up interactions between features. |
| LightGBM | Another tree-boosting model, built differently from XGBoost. Included as a second opinion. |

## Results

Best first. Scores are averages over 5 validation rounds; the number after the plus-minus sign shows how much the score moved between rounds.

| Rank | Model | PR-AUC | ROC-AUC | Lift over chance |
|---|---|---|---|---|
| 1 | XGBoost | 0.226 +/- 0.007 | 0.671 +/- 0.010 | 1.97x |
| 2 | LightGBM | 0.224 +/- 0.006 | 0.669 +/- 0.007 | 1.95x |
| 3 | Logistic regression | 0.211 +/- 0.008 | 0.663 +/- 0.011 | 1.84x |

### How to read these numbers

- **PR-AUC** (precision-recall area under the curve) measures how well the model puts the patients who really were readmitted near the top of its risk ranking without flagging too many who were not. A model with no skill scores about 0.115 (the share of readmitted patients). 1.0 is perfect.
- **Lift over chance** is the PR-AUC divided by that no-skill value. A lift of 1.0x means no better than guessing; 1.5x means the model finds readmitted patients about one and a half times as well as chance.
- **ROC-AUC** is the chance that a randomly chosen readmitted patient gets a higher risk score than a randomly chosen patient who was not readmitted. 0.5 is a coin flip and 1.0 is perfect.
- **The plus-minus value** shows stability. If two models differ by less than this, treat them as tied.

## Verdict

- Best model by PR-AUC: **XGBoost**, with 0.226 (lift 1.97x over chance).
- Rule of thumb used here: lift under 1.1x is barely better than guessing, 1.1x to 1.5x is a modest signal, above 1.5x is useful. This model shows a useful signal. Readmission is hard to predict from hospital records alone, so modest scores are not a sign of a bug.
- XGBoost is somewhat better than the simple baseline (gain +0.015), but the gap is only a little larger than the round-to-round variation.
- The ranking is a shortlist. The test data decides the final choice in the evaluation gate (step 8).

## What drives the predictions

Names look like `column_value`, for example `age_[70-80)` means the age group 70 to 80, and `_infrequent_sklearn` means rare values of that column grouped together. Influence is not the same as cause: these are patterns in the data, not proof of why patients return.

### XGBoost

Average influence on predictions (mean absolute SHAP value). Bigger means more influence; it does not show direction.

- `medical_specialty_Otolaryngology` (0.667)
- `discharge_disposition_id_28` (0.621)
- `medical_specialty_Pediatrics-Endocrinology` (0.546)
- `medical_specialty_ObstetricsandGynecology` (0.431)
- `number_inpatient` (0.271)
- `discharge_disposition_id_2` (0.244)
- `discharge_disposition_id_infrequent_sklearn` (0.239)
- `rosiglitazone_infrequent_sklearn` (0.239)
- `discharge_disposition_id_23` (0.219)
- `diag_1_group_supplementary` (0.214)

### LightGBM

Average influence on predictions (mean absolute SHAP value). Bigger means more influence; it does not show direction.

- `number_inpatient` (0.299)
- `discharge_disposition_id_1` (0.193)
- `num_lab_procedures` (0.068)
- `number_diagnoses` (0.067)
- `age_[50-60)` (0.060)
- `time_in_hospital` (0.060)
- `num_medications` (0.053)
- `number_emergency` (0.048)
- `diag_1_group_circulatory` (0.047)
- `payer_code_missing` (0.046)

### Logistic regression

Largest coefficients. Positive raises the predicted risk, negative lowers it (features are standardized, so sizes are comparable).

- `discharge_disposition_id_28`: raises risk (+1.29)
- `medical_specialty_Pediatrics-Endocrinology`: lowers risk (-1.11)
- `discharge_disposition_id_22`: raises risk (+0.88)
- `discharge_disposition_id_23`: lowers risk (-0.84)
- `medical_specialty_Otolaryngology`: lowers risk (-0.80)
- `medical_specialty_Hematology/Oncology`: raises risk (+0.80)
- `medical_specialty_ObstetricsandGynecology`: lowers risk (-0.68)
- `discharge_disposition_id_1`: lowers risk (-0.58)
- `medical_specialty_Oncology`: raises risk (+0.51)
- `medical_specialty_Neurology`: lowers risk (-0.48)

Columns in the top 10 of every model: `discharge_disposition_id`. When several different models lean on the same columns, those columns are the more reliable signals.

## What these results do not tell you

- They are not accuracy. With 11% positives, a model that always says no would be right 89% of the time and still useless.
- They come from cross-validation on training data. Performance on the untouched test data may differ.
- The data has no dates, so we could not check how the model holds up over time.
- Fairness across age, race and gender, and whether the scores are well calibrated as probabilities, are checked later (step 8).
- This is not a clinical tool. It has not been validated for use with real patients.

## Where to find things

| What | Where |
|---|---|
| Saved models, ranked, score in the file name | `models/01_...joblib`, `models/02_...joblib` |
| Which model ranked first | `models/best_model.json` |
| Technical summary of this run | `data/training_report.txt` |
| All runs, metrics, SHAP charts | MLflow: run `mlflow ui --backend-store-uri sqlite:///mlflow.db` in the project folder and open http://127.0.0.1:5000 |

In MLflow, open the `readmission-30d` experiment, click **Columns** and tick `cv_pr_auc_mean` and `cv_roc_auc_mean` to see the scores. Click a run, then **Artifacts**, then `explanations` for the SHAP chart or coefficients. The folders with random names are MLflow's internal storage; use the names above instead.

## Glossary

- **Cross-validation**: splitting the training data into parts, training on some and scoring on the held-out part, and repeating, so every score comes from data the model did not train on.
- **Baseline**: a simple model used as the bar every other model must clear.
- **Class weights**: telling the model that a missed readmission costs more than a false alarm, because readmissions are rare.
- **SHAP**: a method that measures how much each feature pushed a prediction up or down.
- **Calibration**: whether a predicted risk of 20% really corresponds to about 20% of such patients being readmitted.
