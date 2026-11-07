# FinWise Lending — Baseline Report
## Checkpoint 2 (Week 3): Data Pipeline + Baseline Model

**Status:** Complete
**Model:** Logistic Regression (class-balanced)
**Purpose:** Establish the data pipeline and the performance floor that any
later model (e.g. the XGBoost champion model) must clearly exceed.

---

## 1. Data Pipeline Summary

- **Source file:** `finwise_loan_applications.csv`
- **Rows ingested:** 1,000
- **Rows after validation:** 1,000
  (0.00% dropped)
- **Duplicate application IDs removed:** 0
- **Rows dropped for missing required fields:** 0
- **Rows dropped for out-of-range values:** 0

**Pipeline stages:** ingest -> validate schema -> clean rows -> engineer
features -> stratified train/val/test split -> fit preprocessor on train
only -> transform all splits.

**Split sizes:**

| Split | Rows | Default Rate |
|---|---|---|
| Train | 700 | 23.71% |
| Validation | 100 | 24.00% |
| Test | 200 | 24.00% |

**Features used (23):** age, income_annual_inr, employment_years, existing_loans, existing_emi_inr, credit_score, loan_amount_inr, ltv_ratio, dti_ratio, emi_to_income_ratio, loan_to_income_ratio, total_debt, debt_to_income, employment_stability, high_dti, high_ltv, low_credit, multiple_loans, risk_score, employment_type_encoded, loan_purpose_encoded, credit_score_band_encoded, age_group_encoded

---

## 2. Baseline Model

- **Algorithm:** Logistic Regression, `class_weight="balanced"`, `max_iter=1000`
- **Why this baseline:** fast, interpretable via coefficients, and the
  standard industry starting point for credit scoring before reaching for
  gradient boosting. It gives a defensible floor: any added model
  complexity later must be justified by a clear lift over these numbers.

---

## 3. Evaluation Benchmark

This is the fixed benchmark all future models are compared against.

| Split | AUC | Gini | KS Statistic | Brier Score |
|---|---|---|---|---|
| Dummy (majority class), Val | 0.4583 | -0.0833 | 0.2083 | 0.3900 |
| Baseline, Train | 0.9309 | 0.8619 | 0.7114 | 0.1079 |
| Baseline, Validation | 0.9194 | 0.8388 | 0.7368 | 0.1158 |
| Baseline, Test | 0.9246 | 0.8492 | 0.6996 | 0.1074 |

**Project target (champion model):** Gini >= 0.45 on test.
**Baseline Gini on test:** 0.8492
**Gap remaining for the champion model to close:** 0.0000

The baseline is not expected to meet the champion-model target — its role
is to prove the pipeline works end-to-end and to give a numeric floor
("beat this or the added complexity isn't worth it").

---

## 4. Top Risk Directions (Baseline Coefficients)

Top absolute-value coefficients from the baseline model, on standardized
features (positive = increases default risk):

| feature              |   coefficient |
|:---------------------|--------------:|
| credit_score         |     -0.920841 |
| dti_ratio            |      0.581454 |
| existing_emi_inr     |      0.519037 |
| income_annual_inr    |     -0.516566 |
| emi_to_income_ratio  |      0.476662 |
| debt_to_income       |      0.349603 |
| loan_to_income_ratio |      0.313957 |
| employment_stability |     -0.300265 |
| high_dti             |      0.288241 |
| high_ltv             |     -0.276995 |

---

## 5. Known Limitations of the Baseline

1. Logistic Regression assumes linear/additive relationships — it will
   miss interaction effects (e.g. high DTI *combined with* low tenure)
   that a tree-based model can capture.
2. No hyperparameter tuning was performed — this is intentionally a
   floor, not an optimized model.
3. Bias/fairness audit is deferred to the champion-model checkpoint.
4. Calibration was not explicitly tuned; `class_weight="balanced"` trades
   calibration for better separation on the minority (default) class.

## 6. Next Steps (Checkpoint 3)

- Train the champion model (XGBoost) using the **same** `data_pipeline.py`
  and `preprocessor.pkl` produced here, so the comparison is apples-to-apples.
- Compare champion vs. baseline on this exact benchmark table.
- Run the full SHAP explainability and bias-audit workstreams on the
  champion model.
