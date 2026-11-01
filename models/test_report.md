# Test evaluation report

Created 2026-11-01 10:47. Generated automatically by `test.py` and replaced on every run.

## What this report shows

The models were trained earlier on the training data only. This report scores the saved models, unchanged, on the test split they have never seen. Nothing was retrained or tuned here. This is the evaluation gate (step 8): a model is promoted only if it passes rules that were written down before the test results were seen.

## The test data in brief

- 14,907 hospital stays from 10,483 patients.
- About 11.6% ended in a readmission within 30 days, so plain accuracy would be misleading.

## Results

Best first. The 95% range comes from resampling whole patients 200 times; if two models' ranges overlap heavily, treat them as tied.

| Model | CV PR-AUC | Test PR-AUC | 95% range | ROC-AUC | Lift over chance | Gate |
|---|---|---|---|---|---|---|
| LightGBM | 0.224 | 0.222 | 0.200 to 0.247 | 0.670 | 1.93x | PASS |
| XGBoost | 0.226 | 0.218 | 0.197 to 0.240 | 0.670 | 1.89x | PASS |
| Logistic regression | 0.211 | 0.206 | 0.188 to 0.229 | 0.660 | 1.78x | FAIL |

No-skill PR-AUC is about 0.116 (the positive rate). Lift = PR-AUC / positive rate. CV PR-AUC is the training-time estimate; a big drop on test would suggest overfitting.

## Gate rules and outcome

Rules were fixed in `test.py` before the test data was scored.

**XGBoost**: passes all checks

- Pass: lift >= 1.5
- Pass: CV-to-test drop <= 0.02
- Pass: ROC-AUC gap by group <= 0.05
- Pass: recall gap by group <= 0.1
- Pass: beats logistic baseline

**LightGBM**: passes all checks

- Pass: lift >= 1.5
- Pass: CV-to-test drop <= 0.02
- Pass: ROC-AUC gap by group <= 0.05
- Pass: recall gap by group <= 0.1
- Pass: beats logistic baseline

**Logistic regression**: FAILS

- Pass: lift >= 1.5
- Pass: CV-to-test drop <= 0.02
- FAIL: ROC-AUC gap by group <= 0.05
- FAIL: recall gap by group <= 0.1

## LightGBM against the other models

(Reference model: highest test PR-AUC (no champion registered yet).)

Difference in PR-AUC, resampled by patient. If the range includes zero, the data cannot tell the models apart.

- vs XGBoost: +0.005 (range -0.001 to +0.010): no clear difference.
- vs Logistic regression: +0.016 (range +0.005 to +0.027): reference clearly better.

## Decision

Passed the gate (eligible to be registered): XGBoost, LightGBM.
Failed the gate: Logistic regression.

The gate does not choose a champion. Choosing one is a human decision, recorded with a reason by `register_champion.py`.

## Fairness

Flagging the top 20% of stays by risk catches 38.0% of readmissions overall. Worst subgroup shortfall for the reference model (LightGBM): ROC-AUC 0.040 below overall, recall 0.054 below overall (groups under 500 stays or 50 readmissions are listed in `data/test_report.txt` but not judged).

## Calibration

Mean predicted risk for the reference model (LightGBM) is 0.428 against an actual rate of 0.116 (Brier 0.204, ECE 0.313). The models were trained with class weights, so raw scores run higher than real probabilities. They rank patients well but should not be read as 'a 30% chance' until a calibration step is added. This is reported, not gated.

## Parameters and versions used

Models were loaded from their saved files, not rebuilt, so these are the training-time settings.

- **XGBoost** (`01_xgboost_prauc0.226.joblib`): `{"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 5, "n_estimators": 300, "random_state": 42, "scale_pos_weight": 7.704, "subsample": 0.8}`
- **LightGBM** (`02_lightgbm_prauc0.224.joblib`): `{"class_weight": null, "colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": -1, "min_child_samples": 40, "min_child_weight": 0.001, "n_estimators": 300, "num_leaves": 31, "random_state": 42, "subsample": 0.8, "subsample_freq": 1, "scale_pos_weight": 7.704}`
- **Logistic regression** (`03_logistic_prauc0.211.joblib`): `{"C": 1.0, "class_weight": "balanced", "max_iter": 2000, "random_state": 42}`

Packages at test time: scikit-learn 1.5.2, xgboost 2.0.3, lightgbm 4.6.0. Compare with the versions used for training.

## What these results do not tell you

- The test split has now been used to choose between models; do not tune on it. The holdout (step 11) is the next untouched check.
- The data has no dates, so behaviour over time is untested.
- This is not a clinical tool and has not been validated for use with real patients.
