# Model Card: FinWise Credit Risk Scorer

| | |
|---|---|
| **Author** | Veda Shivayogi Ramagaondanahalli, AI Intern, Cynaris Solutions |
| **Client** | FinWise (FinTech, Bangalore, growth stage) |
| **Version** | MLflow run `70f079637d3a42c692fd99f3a940c712`, config `config_B_balanced_default`, card dated 2026-11-07 |
| **Model type** | XGBoost binary classifier (gradient-boosted trees), 16 trees used (early stopping at iteration 15) |
| **Task** | Estimate the risk that a loan applicant defaults, and explain which inputs drove the result |
| **Framework** | Python 3.10, xgboost 3.2.0, scikit-learn 1.3.0, numpy 1.26.4, pandas 2.0-2.2 |
| **Status** | Prototype / internship project. **Not approved for real lending decisions** (see sections 2 and 7) |

## 1. Summary
The model scores an application from 11 input fields (income, employment, credit history, loan amount and so on), returns a
score on a 300-900 scale, a **low / medium / high** risk band, the **top 3 factors raising risk**, and a **SHAP waterfall** showing how
each factor moved the result. It is served through a FastAPI REST API with a React front-end.
On the 200-row held-out test set it reaches **Gini 0.817** (project target: 0.45 or higher).

## 2. Intended use and out-of-scope use
**Intended:** a decision-support tool for credit analysts at FinWise to triage applications and see *why* an application looks risky;
a reference implementation of an explainable scoring service.

**Not intended / do not use for:**
- Fully automated approve/decline decisions without human review.
- Real lending before validation on real FinWise portfolio data (the training data appears synthetic, see section 3).
- Applicants or products outside the data's ranges (section 3), for example incomes above Rs 24 lakh, loans above Rs 50 lakh, age outside 22-58.
- Reading the raw output as a true default probability (it is **not calibrated**, section 4).

## 3. Training and evaluation data
| Item | Detail |
|---|---|
| Source | `finwise_loan_applications.csv`, provided with the project brief |
| Size | 1,000 applications; 0 rows dropped by validation (no nulls, no out-of-range values, no duplicate IDs) |
| Target | `default_flag` (1 = defaulted); **23.8% positive** |
| Split | Stratified, seed 42: **700 train / 100 validation / 200 test**. Test set untouched during model selection |
| Inputs | age, annual income (INR), employment type, years employed, number of existing loans, existing monthly EMI, credit score, loan amount, loan purpose, LTV ratio, DTI ratio |
| Engineered | EMI-to-income, loan-to-income, total debt, debt-to-income, employment stability, flags (high DTI > 0.5, high LTV > 0.8, low credit < 600, 2+ loans), risk-flag count, credit-score band, age group (23 model features in total) |
| Ranges seen | age 22-58; income Rs 3-24 lakh; loan Rs 1-50 lakh; credit score 453-900; DTI 0-3.06; 0-3 existing loans |
| Categories | employment: salaried, self_employed, government, contract; purpose: home, personal, vehicle, education, medical, business |

**Data caveats**
- The data looks **synthetic**: clean, no missing values, and `dti_ratio` is an exact formula of other columns
  (`EMI*12/income + 0.096*loan/income`). Real portfolios are noisier, so real-world performance will likely be lower.
- Only 1,000 rows and 238 defaults: metrics have wide uncertainty (the test set has only about 48 defaulters).
- `finwise_credit_risk_features.csv` was **not used for training**: its `risk_category` is just a CIBIL score band and `approved` is almost determined by it (label leakage).
- `finwise_fraud_transactions.csv` is outside this task and cannot be joined to the loan data.
- **No gender field** exists in the loan data. **Age is a model input** (see section 6).

## 4. Metrics
Gini = 2*AUC - 1. KS = maximum separation between defaulters and non-defaulters. Brier = squared error of the probability (lower is better).

| Split | n | Gini | AUC | KS | Brier |
|---|---|---|---|---|---|
| Train | 700 | 0.943 | 0.972 | 0.823 | 0.118 |
| Validation | 100 | 0.875 | 0.938 | 0.772 | 0.130 |
| **Test** | **200** | **0.817** | **0.908** | **0.682** | **0.138** |
| Logistic-regression baseline (test) | 200 | 0.849 | 0.925 | n/a | n/a |
| Majority-class dummy (validation) | 100 | -0.083 | 0.458 | 0.208 | 0.390 |

**Model selection (MLflow, 3 configurations):** chosen on highest *validation* Gini.

| Config | Val Gini | Test Gini | Test Brier | Train-Val gap |
|---|---|---|---|---|
| **B balanced default (selected)** | 0.875 | 0.817 | 0.138 | 0.068 |
| C deep regularized | 0.873 | 0.823 | 0.145 | 0.030 |
| A shallow fast | 0.872 | 0.827 | 0.119 | 0.034 |

**Honest reading:** all three configurations are within about 0.01 Gini of each other on a 100-row validation set, which is inside the noise,
and the simple logistic baseline scores slightly higher (0.849) than the selected XGBoost model. Both clear the 0.45 target. XGBoost was kept
as the required model type and for tree-based SHAP explanations; the train-to-test gap (0.943 to 0.817) shows some overfitting.

**Risk bands (test set, n = 200).** Cut-offs were chosen on validation data and are configurable (`FINWISE_LOW_T`, `FINWISE_HIGH_T`):

| Band | Model output | Applicants | Actual default rate |
|---|---|---|---|
| Low | below 0.25 | 56 | 0.0% |
| Medium | 0.25 to 0.50 | 88 | 12.5% |
| High | 0.50 or above | 56 | 66.1% |

Treating "High" as a flag: precision 66%, recall 77% of true defaulters.

**Calibration:** the model is trained with `scale_pos_weight` = 3.22 to handle the 24% positive rate, so raw outputs are inflated
(minimum around 0.22) and **are not probabilities**. Use the band, not the raw number. Calibration (isotonic/Platt) is a next step.

**What drives predictions (mean absolute SHAP, test set):** debt-to-income ratio 0.57, high-DTI flag 0.15, credit score 0.13,
EMI-to-income 0.05, existing EMIs 0.04, number of existing loans 0.04. Debt-to-income dominates; the credit score matters much less than might be expected.

## 5. Explainability
Per application, the API returns SHAP values from XGBoost's native TreeSHAP computed on the **same 16 trees** used for the score (the saved model
stores 46 trees; using all 46 would change scores by up to 0.20, so this is pinned and tested). Output: top 3 risk-raising factors, up to 3
protective factors, and a waterfall (log-odds) from the average applicant to this applicant. SHAP describes what the **model** did, not a causal
reason a person defaulted. Adverse-action wording for customers needs legal review.

## 6. Bias and fairness audit
**Method:** held-out test split (n = 200), "flagged" = High band (score 0.50 or above) as the adverse outcome. Impact ratio = a group's
approval-side rate divided by the best group's rate (four-fifths rule: should be 0.80 or higher). Reproduce: `python scripts/bias_audit.py`.
CI fails if the lowest age impact ratio drops below 0.80.

| Age group | n | Actual default rate | Flagged high risk | Approval-side rate | Impact ratio | Defaulters caught |
|---|---|---|---|---|---|---|
| 30 and under | 50 | 26.0% | 24.0% | 76.0% | 1.00 | 69% |
| 31-40 | 60 | 26.7% | 31.7% | 68.3% | 0.90 | 75% |
| 41-50 | 50 | 20.0% | 30.0% | 70.0% | 0.92 | 90% |
| 51 and over | 40 | 22.5% | 25.0% | 75.0% | 0.99 | 78% |

**Result:** the age audit passes the four-fifths rule (lowest 0.90). **Caveats:** groups have only 40-60 people, so differences are statistically
weak; defaulter catch rate ranges 69% to 90%, which is a larger spread than the approval rates and should be rechecked on more data.

**Gender: not audited.** The loan data has no gender field and no other file joins to it. No claim of gender fairness is made. Recommended: collect
gender for audit purposes only (kept out of the model), then rerun the script.

**Age is used as a model input** (as `age` and `age_group`). Its SHAP influence is small (not among the top 8 drivers), but using age in credit decisions
can be restricted by regulation or policy. **Recommendation: retrain without age and compare**, and have compliance review before any real use.
Other proxy risks (for example loan purpose or employment type standing in for protected traits) were not tested.

## 7. Limitations and risks
1. **Synthetic-looking, small data:** do not expect Gini 0.82 on production data; validate out-of-time on real FinWise applications first.
2. **Not calibrated:** raw scores are not default probabilities.
3. **Narrow input ranges:** behaviour outside the ranges in section 3 is untested; the API accepts wider values (age 18-100, credit score 300-900) and will still return a score.
4. **Overfitting and noise in selection:** train Gini 0.94 vs test 0.82; config choice rests on 100 validation rows.
5. **Dominated by one feature (DTI)**, which is itself a formula of other fields: the model may be less informative than the Gini suggests.
6. **Possibly spurious patterns:** in the data some categories (for example contract employment: 17.4% default vs 23.8% overall) show lower default rates than expected; check against credit policy.
7. **Fairness coverage is partial:** age only, small groups, gender and intersectional groups not tested.
8. **No monitoring, authentication or rate limiting** in the current service (section 8).
9. **No human-in-the-loop workflow** is built in; the tool must not be wired to automatic decisions.

## 8. Deployment guide
### Run locally (Windows or Linux/macOS, Python 3.10)
```bash
cd FinWise-Lending-Credit-Risk-Project
python -m pip install -r requirements-dev.txt     # pinned to training versions
python -m pytest -q                               # expect 36 passed
python -m uvicorn serving.app:app --port 8000     # UI + API on http://localhost:8000 (docs at /docs)
```
Windows shortcut: double-click `run.bat`.

### API
| Endpoint | Purpose |
|---|---|
| `POST /predict` | One application: score, band, top risk factors, protective factors, waterfall |
| `POST /predict/batch` | 1-500 applications |
| `GET /model-info` | Version, hyperparameters, metrics, band rules, bias audit |
| `GET /health` | Returns 200 only after the model is loaded and warmed up |
Invalid input (out-of-range, unknown category, unknown field) returns HTTP 422 and is never scored.

### Configuration (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `FINWISE_SCORING_PATH` | `fast` | `fast` (NumPy) or `pipeline` (original pandas pipeline); proven identical by tests |
| `FINWISE_LOW_T` / `FINWISE_HIGH_T` | 0.25 / 0.50 | Band cut-offs |
| `FINWISE_ARTIFACTS` | `./artifacts` | Folder with `champion_model.pkl`, `preprocessor.pkl`, metadata JSON |
| `WEB_CONCURRENCY` / `--workers` | 1 | Worker processes; use one per CPU core |

### Docker (provided, not built or tested in this project)
```bash
docker build -t finwise-api . && docker run -p 8000:8000 -e WEB_CONCURRENCY=2 finwise-api
```

### Measured performance (single shared CPU core, load generator on the same machine; repeat runs varied about 20%)
| Path | Throughput at 50 concurrent users | p50 / p95 / p99 latency | Server CPU per request |
|---|---|---|---|
| Fast (default) | about 366 req/s | 126 / 228 / 253 ms | 2.6 ms |
| Original pandas pipeline | about 46 req/s | p95 about 1,290 ms | about 22 ms |
A single idle request takes about 2-5 ms. 0 errors in 26,985 requests. Estimated capacity is about 1000 / 2.6 = 390 req/s per core (upper bound).
Details: `docs/WEEK4_SERVING_AND_LOAD_TEST_REPORT.md`; reproduce with `python loadtest/loadtest.py --client lean`.

### Before production (not done)
- Add authentication, HTTPS (reverse proxy), rate limiting and request logging without personal data.
- Monitor: input drift (for example population stability index per feature), share of applicants per band, default rate per band once outcomes arrive, latency and error rate.
- Retrain/review triggers: band default rates drift from the table in section 4, input ranges shift, or any fairness gate fails.
- Rollback: every response carries `model_version` (MLflow run id); keep previous `artifacts/` folders and redeploy the earlier one.
- Compliance review (adverse-action wording, use of age, data protection law) before any customer-facing use.

## 9. Testing and CI
36 integration tests (`pytest`), including: HTTP scoring of the test split reproduces the training Gini (0.8169); fast path equals the original pipeline to 1e-9 on all
1,000 rows; SHAP additivity and tree-count guard; 422 on 14 kinds of bad input; waterfall ends at the prediction; 50 concurrent requests against a real
2-process server return correct results; loading of artifacts saved from a script's `__main__`. Tests were checked by injecting three deliberate bugs, all were caught.
GitHub Actions (`.github/workflows/ci.yml`) runs all tests, the age fairness gate (impact ratio of at least 0.80) and the front-end build on every push.

## 10. Repository map
`data_pipeline.py` (validation, features, preprocessor) · `core_model_experimentation.py` (MLflow experiments) · `artifacts/` (model, preprocessor, metrics) ·
`serving/` (API) · `frontend/` (React) · `tests/` · `loadtest/` · `scripts/bias_audit.py` · `reports/bias_audit.json` · `docs/` (load-test report, demo script).
