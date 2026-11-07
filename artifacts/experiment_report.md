# FinWise Lending — Core Model & Experimentation Report
## Checkpoint 3: Core Model + Experimentation

**Status:** Complete
**Primary model:** XGBoost Classifier
**Configurations tested:** 3
**Experiment tracking:** MLflow (`sqlite:///mlflow.db`, experiment `finwise_champion_model`)

---

## 1. Experiment Setup

- **Data pipeline:** reused unchanged from Checkpoint 2 (`data_pipeline.py`),
  including the same fitted preprocessor — no leakage between baseline and
  champion comparisons.
- **Train / Validation / Test sizes:** 700 / 100 / 200
- **Early stopping:** 30 rounds on the validation AUC,
  so each configuration's `n_estimators` is a ceiling, not a target.
- **Model selection rule:** the winning configuration is chosen by the
  **highest validation Gini**. The test set is only used afterward, to
  report a final, unbiased performance estimate — it never influences
  which configuration is picked.

---

## 2. Hyperparameter Configurations Tested

**config_A_shallow_fast**: n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, min_child_weight=3, reg_alpha=0.0, reg_lambda=1.0
**config_B_balanced_default**: n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0
**config_C_deep_regularized**: n_estimators=800, max_depth=8, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, min_child_weight=8, gamma=0.2, reg_alpha=0.3, reg_lambda=2.0

---

## 3. Results — All Configurations (Evidence Table)

| Config                    |   Best Iteration |   Val AUC |   Val Gini |   Val KS |   Test AUC |   Test Gini |   Test KS |   Test Brier |   Train-Val Gini Gap |
|:--------------------------|-----------------:|----------:|-----------:|---------:|-----------:|------------:|----------:|-------------:|---------------------:|
| config_B_balanced_default |               15 |    0.9375 |     0.875  |   0.7719 |     0.9084 |      0.8169 |    0.682  |       0.1376 |               0.0681 |
| config_C_deep_regularized |               25 |    0.9364 |     0.8728 |   0.7741 |     0.9116 |      0.8232 |    0.6601 |       0.1452 |               0.0299 |
| config_A_shallow_fast     |               15 |    0.9359 |     0.8717 |   0.7303 |     0.9137 |      0.8274 |    0.6404 |       0.1195 |               0.0341 |

*(Train-Val Gini Gap is an overfitting signal — a large positive gap means
the model memorized the training set more than it generalized.)*

---

## 4. Selected Configuration

- **Winner:** `config_B_balanced_default`
- **MLflow run ID:** `70f079637d3a42c692fd99f3a940c712`
- **Selection basis:** highest validation Gini (0.8750)
- **Best boosting iteration:** 15 (via early stopping)

**Final test-set performance (unbiased, held out from selection):**

| Metric | Value |
|---|---|
| AUC | 0.9084 |
| Gini | 0.8169 |
| KS Statistic | 0.6820 |
| Brier Score | 0.1376 |

---

## 5. Champion vs. Baseline

| Model | Test Gini | Test AUC |
|---|---|---|
| Baseline (Logistic Regression, Checkpoint 2) | 0.8492 | 0.9246 |
| Champion (config_B_balanced_default) | 0.8169 | 0.9084 |
| **Lift** | **-0.0323** | **-0.0162** |

**Project acceptance criterion:** Gini >= 0.45 on test set: **PASSED**
(0.8169 vs. 0.45 target)

---

## 6. Evidence Artifacts Generated

- `experiment_roc_comparison.png` — ROC curves for all 3 configs, winner highlighted
- `experiment_metric_comparison.png` — Gini/AUC/KS bar comparison across configs
- `champion_vs_baseline.png` — champion vs. baseline Gini, with target line
- `champion_feature_importance.png` — top 15 features driving the champion model
- `champion_confusion_matrix.png` — test-set confusion matrix at threshold 0.5
- `artifacts/experiment_comparison.csv` — full metrics table, all configs
- `artifacts/champion_model.pkl` — the selected, trained model
- `artifacts/champion_model_metadata.json` — full config + metrics for the winner
- MLflow run history — `mlflow ui --backend-store-uri sqlite:///mlflow.db`

---

## 7. Next Steps (Checkpoint 4)

- Run SHAP explainability on the champion model (waterfall + summary plots).
- Run the bias/fairness audit (age, employment type, loan purpose) on the
  champion model's predictions.
- Wire the champion model into the FastAPI serving layer.
