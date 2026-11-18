# Diabetes Prediction with Logistic Regression

Predicts whether a patient has diabetes from eight diagnostic measurements, using the Pima Indians Diabetes dataset. The project covers data cleaning, exploratory analysis, a leakage-safe scikit-learn pipeline, cross-validation, hyperparameter tuning, decision-threshold analysis and coefficient interpretation.

> **Disclaimer:** This is a learning project built on 768 rows from a single population. It is not a medical tool and must not be used for diagnosis.

---

## Results at a glance

| Evaluation | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline, held-out test set (154 rows) | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |
| Tuned, held-out test set (154 rows) | 0.721 | 0.585 | 0.704 | 0.639 | 0.811 |
| Baseline, 5-fold stratified CV (all 768 rows) | 0.772 ± 0.017 | n/a | 0.571 ± 0.047 | n/a | 0.837 ± 0.020 |
| Tuned, best CV ROC-AUC (training split only) | n/a | n/a | n/a | n/a | 0.845 |

Precision, recall and F1 are for the **diabetic** class (`Outcome = 1`).

**Main takeaway:** tuning did not produce a better ranking model (test ROC-AUC 0.813 → 0.811). What it changed was the operating point: `class_weight="balanced"` raised recall on diabetic patients from **50% to 70%** and cost some precision. For screening, where missing a true case is worse than a false alarm, that trade is the useful part.

### Confusion matrices (test set, threshold = 0.5)

Rows are actual, columns are predicted.

| Baseline | Pred. No diabetes | Pred. Diabetes |
|---|---|---|
| **Actual No diabetes** | 82 | 18 |
| **Actual Diabetes** | 27 | 27 |

| Tuned | Pred. No diabetes | Pred. Diabetes |
|---|---|---|
| **Actual No diabetes** | 73 | 27 |
| **Actual Diabetes** | 16 | 38 |

The baseline misses 27 of 54 diabetic patients. The tuned model misses 16 of 54 but raises 27 false alarms instead of 18.

### Best hyperparameters (GridSearchCV, 5-fold, scored on ROC-AUC)

```
C = 0.1, penalty = l2, solver = liblinear, class_weight = balanced
```

### Decision-threshold analysis (tuned model, test set)

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.30 | 0.695 | 0.539 | 0.889 | 0.671 |
| 0.35 | 0.701 | 0.545 | 0.889 | 0.676 |
| 0.40 | 0.721 | 0.570 | 0.833 | 0.677 |
| 0.45 | 0.727 | 0.579 | 0.815 | 0.677 |
| 0.50 | 0.721 | 0.585 | 0.704 | 0.639 |
| 0.55 | 0.734 | 0.614 | 0.648 | 0.631 |
| 0.60 | 0.714 | 0.593 | 0.593 | 0.593 |

Lowering the threshold trades precision for recall. These numbers were computed on the test set, so they illustrate the trade-off but should not be used to pick a "best" threshold. A proper choice needs a separate validation set.

### Feature importance (standardized coefficients, tuned model)

| Feature | Coefficient | Odds ratio (per +1 SD) |
|---|---|---|
| Glucose | 1.022 | 2.778 |
| BMI | 0.597 | 1.817 |
| Pregnancies | 0.320 | 1.377 |
| DiabetesPedigreeFunction | 0.255 | 1.291 |
| Age | 0.188 | 1.206 |
| SkinThickness | 0.057 | 1.059 |
| BloodPressure | 0.017 | 1.018 |
| Insulin | 0.016 | 1.017 |

Glucose dominates: one standard deviation higher glucose is associated with roughly 2.8× the odds of diabetes, holding the other features fixed. SkinThickness and Insulin have near-zero coefficients, but 30% and 49% of their values were missing and imputed, so their weights should not be over-interpreted.

### Example prediction

A hand-written demo patient (not a real record):

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age |
|---|---|---|---|---|---|---|---|
| 2 | 150 | 85 | 30 | 120 | 33.5 | 0.55 | 45 |

```
Probability of diabetes: 71.6%
Prediction: Diabetic
```

---

## Dataset

- **Source:** Pima Indians Diabetes Database (originally from the National Institute of Diabetes and Digestive and Kidney Diseases; widely mirrored on Kaggle and the UCI repository).
- **Size:** 768 rows, 8 features, 1 binary target.
- **Population:** women of Pima Indian heritage, aged 21 and over.
- **Class balance:** 500 non-diabetic (65.1%) vs 268 diabetic (34.9%).
- **Reference:** Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* Proceedings of the Symposium on Computer Applications and Medical Care, 261–265.

| Column | Meaning |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Score for diabetes family history |
| Age | Age in years |
| Outcome | 1 = diabetes, 0 = no diabetes |

---

## Method

1. **Load and inspect:** 768 × 9, no NaNs, no duplicate rows.
2. **Clean:** zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are physiologically impossible, so they are treated as missing values.

   | Column | Zeros converted to NaN | Share missing |
   |---|---|---|
   | Glucose | 5 | 0.7% |
   | BloodPressure | 35 | 4.6% |
   | SkinThickness | 227 | 29.6% |
   | Insulin | 374 | 48.7% |
   | BMI | 11 | 1.4% |

3. **Explore:** class balance, per-class feature distributions and a correlation matrix. Correlation with `Outcome`: Glucose 0.495, BMI 0.314, Insulin 0.303, SkinThickness 0.259, Age 0.238, Pregnancies 0.222, DiabetesPedigreeFunction 0.174, BloodPressure 0.171.
4. **Split:** stratified 80/20 (614 train / 154 test, `random_state=42`) so both sets keep the ~35% diabetic ratio.
5. **Model:** `Pipeline(SimpleImputer(median) → StandardScaler → LogisticRegression)`. Imputation and scaling are fit on training data only, which prevents test-set leakage.
6. **Validate:** 5-fold stratified cross-validation, because a single split with 54 positive test cases is noisy.
7. **Tune:** `GridSearchCV` over `C`, `penalty` and `class_weight`, optimizing ROC-AUC.
8. **Threshold analysis, coefficients, single-patient prediction.**
9. **Export:** the notebook saves the tuned pipeline with `joblib` as `diabetes_logreg_model.pkl`.

---

## Limitations

- **Small, single-population data.** 768 women from one group, so the model may not generalize to other populations, ages or sexes.
- **Heavy imputation.** Nearly half of `Insulin` and a third of `SkinThickness` values are imputed with the median, which weakens those features.
- **Noisy test estimate.** The test set has only 154 rows (54 positive). Prefer the cross-validation numbers when comparing models.
- **Threshold chosen on test data.** The threshold table is illustrative only (see above).
- **Tuning did not improve ROC-AUC.** The gain is in recall at the 0.5 threshold, not in discriminative power.
- **Linear model.** Logistic regression cannot capture feature interactions. Random Forest or gradient boosting would be a fair next comparison.

## Possible improvements

- Add missing-value indicator columns instead of only median imputation.
- Compare against Random Forest, XGBoost or LightGBM using the same CV folds.
- Choose the threshold on a validation fold or with nested cross-validation.
- Calibrate predicted probabilities (e.g. `CalibratedClassifierCV`).
- Add feature engineering such as BMI categories and age bins.

---

## Project structure

```
.
├── README.md
├── requirements.txt
├── diabetes.csv
└── diabetes_logistic_regression.ipynb
```

## How to run

```bash
git clone https://github.com/<iannzee>/<Diabetes-Predicition-by-Linear-Regression>.git
cd <Diabetes-Predicition-by-Linear-Regression>
pip install -r requirements.txt
jupyter notebook diabetes_logistic_regression.ipynb
```

`requirements.txt`:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

Keep `diabetes.csv` in the same folder as the notebook, or change the `FILE` variable in section 1.

To reuse the saved model:

```python
import joblib, pandas as pd
model = joblib.load("diabetes_logreg_model.pkl")
patient = pd.DataFrame([{"Pregnancies": 2, "Glucose": 150, "BloodPressure": 85,
                         "SkinThickness": 30, "Insulin": 120, "BMI": 33.5,
                         "DiabetesPedigreeFunction": 0.55, "Age": 45}])
print(model.predict_proba(patient)[0, 1])
```

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter.

## Author

**Ziyan Patel** · [GitHub](https://github.com/iannzee) · [LinkedIn](https://www.linkedin.com/in/iannzee7841/)
