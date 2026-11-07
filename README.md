[![CI](https://github.com/VedaShivayogi/Cynarisolution/actions/workflows/ci.yml/badge.svg)](https://github.com/VedaShivayogi/Cynarisolution/actions/workflows/ci.yml)

# FinWise Lending — Credit Risk Project
## Checkpoint 2 (Week 3): Data Pipeline + Baseline Model

This folder contains everything needed to run the data ingestion pipeline,
train the baseline model, and reproduce the baseline report.

---

## 1. Files in this delivery

| File | Purpose |
|---|---|
| `data_pipeline.py` | Reusable ingestion + preprocessing pipeline. Import this in any notebook/script — do not duplicate its logic. |
| `Checkpoint2_Baseline_Model.ipynb` | Runs the pipeline, trains the Logistic Regression baseline, computes the evaluation benchmark, and auto-writes the report. |
| `baseline_report.md` | The generated write-up (data summary, benchmark table, limitations, next steps). Regenerated every time the notebook runs. |
| `README.md` | This file. |

Running the notebook also creates an `artifacts/` folder (not included here,
since it's generated fresh from your data) containing:

- `preprocessor.pkl` — the fitted encoder/scaler, reused unchanged by later models
- `baseline_model.pkl` — the trained Logistic Regression model
- `evaluation_benchmark.json` — machine-readable version of the benchmark table
- `ingestion_report.json` — row counts, drops, and reasons
- `baseline_coefficients.csv` — every feature's coefficient, sorted by impact
- `baseline_report.md` — same as above, freshly written

---

## 2. Requirements

- Python 3.9+
- Packages:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn jupyter jupytext tabulate
  ```

---

## 3. Setup

1. Create a project folder and place these three files in it.
2. Add your real data file, named exactly:
   ```
   finwise_loan_applications.csv
   ```
   in the **same folder** as `data_pipeline.py`.

3. Your CSV must contain these columns (case-sensitive):

   ```
   application_id, age, income_annual_inr, employment_type, employment_years,
   existing_loans, existing_emi_inr, credit_score, loan_amount_inr,
   loan_purpose, ltv_ratio, dti_ratio, default_flag
   ```

   If your real schema uses different column names or value ranges, edit
   `REQUIRED_COLUMNS` and `VALID_RANGES` near the top of `data_pipeline.py`
   to match before running.

---

## 4. How to run

### Option A — Jupyter (recommended, matches the deliverable)
```bash
cd your-project-folder
jupyter notebook
```
Open `Checkpoint2_Baseline_Model.ipynb`, then **Kernel > Restart & Run All**.

### Option B — Run just the pipeline from the command line
```bash
python data_pipeline.py
```
This ingests the data, engineers features, splits it, fits the
preprocessor, and prints the resulting shapes. Useful for a quick sanity
check before opening the notebook.

---

## 5. What "done" looks like

After a successful run you should have:

- No errors in any notebook cell
- An `artifacts/` folder with all 6 files listed in section 1
- `baseline_report.md` showing your **real** default rate, split sizes,
  and benchmark numbers (AUC, Gini, KS statistic, Brier score)
- Console output ending with `CHECKPOINT 2 COMPLETE`

---

## 6. Notes for Checkpoint 3 (champion model)

- Reuse `data_pipeline.py` and the exact `artifacts/preprocessor.pkl`
  produced here — do not refit a new preprocessor on different data, or
  the baseline vs. champion comparison will not be apples-to-apples.
- Compare the champion model against `artifacts/evaluation_benchmark.json`
  using the same metrics (AUC, Gini, KS, Brier) and the same test split.
- The project's target for the champion model is **Gini >= 0.45 on test**.
  The baseline is not expected to meet this — its job is to set the floor.

---

## 7. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `FileNotFoundError: Could not find data file` | `finwise_loan_applications.csv` isn't in the same folder as `data_pipeline.py`, or the filename doesn't match exactly. |
| `ValueError: Missing required columns` | Your CSV's column names don't match `REQUIRED_COLUMNS` in `data_pipeline.py`. Rename the columns or edit that list. |
| Notebook cell `from data_pipeline import run_pipeline` fails | The notebook and `data_pipeline.py` must be in the same folder, and you must launch Jupyter from that folder. |
| Gini/AUC look very low | Check `artifacts/ingestion_report.json` — if most rows were dropped by validation, your data may not match the expected schema or ranges. |

---

# Checkpoint 4 (Week 4): Serving layer, integration tests, load test

The champion XGBoost model (`artifacts/champion_model.pkl`) and the Week-3 preprocessor are served by a FastAPI app.
Nothing in `data_pipeline.py` was changed: the API loads the same pickled `FinWisePreprocessor`, so there is no
train/serve skew.

## Run
```bash
pip install -r requirements-dev.txt
uvicorn serving.app:app --port 8000 --workers 2        # docs at http://localhost:8000/docs
pytest -v                                              # 36 integration tests (~20 s); add -m "not live" to skip socket tests
python loadtest/loadtest.py                            # full benchmark (~5 min) -> docs/loadtest_results.json
python loadtest/render_report.py                       # -> docs/WEEK4_SERVING_AND_LOAD_TEST_REPORT.md
```

## API
| Endpoint | What it does |
|---|---|
| `POST /predict` | One application -> default probability, 300-900 score, `low/medium/high` band, top-3 risk factors (SHAP), protective factors |
| `POST /predict/batch` | 1-500 applications in one call |
| `GET /model-info` | Model version (MLflow run id), hyperparameters, train/val/test metrics, band rules |
| `GET /health` | 200 only after the model is loaded and warmed up |

Example:
```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "age":35,"income_annual_inr":300000,"employment_type":"contract","employment_years":5,"existing_loans":3,
  "existing_emi_inr":40000,"credit_score":560,"loan_amount_inr":2500000,"loan_purpose":"personal","ltv_ratio":0.9}'
```
`dti_ratio` is optional (derived as `EMI*12/income + 0.096*loan/income`, which reproduces the dataset's column).
Unknown fields, out-of-range values and unseen categories return HTTP 422 instead of being scored.

## Things a reviewer may ask
* **Why `iteration_range`?** The saved model stores 46 trees but early stopping chose 15 (16 trees). `predict_proba`
  uses the first 16; scoring or explaining with all 46 changes probabilities by up to 0.20. The service pins both
  probability and SHAP to those 16 trees, and a test fails if that regresses.
* **Is the probability calibrated?** No. The model was trained with `scale_pos_weight` (~3.2), so raw outputs are
  inflated (floor ~0.22). Use the **band**, not the raw number, as the decision signal. Bands (`low < 0.25 <= medium < 0.50 <= high`) were
  chosen on validation: default rate 0% / 10% / 64-100%; on test: 0% / 12.5% / 61-100%. Override with `FINWISE_LOW_T`, `FINWISE_HIGH_T`.
* **Two scoring paths.** `FINWISE_SCORING_PATH=pipeline` runs the reference pandas pipeline; `fast` (default) is a
  NumPy re-implementation that is ~9x faster per request (2.3 vs 21.8 ms CPU). `test_fast_path_parity_with_reference_pipeline` proves they
  agree to 1e-9 on all 1,000 rows.
* **Champion vs baseline.** In your Week-3 metadata the champion's test Gini (0.817) is *below* the logistic-regression
  baseline (0.849); both clear the 0.45 target. The API serves the champion as specified; `/model-info` reports both numbers.

---

# React UI + bias audit (full project)

`frontend/` is a Vite + React app: applicant form, score ruler with the low/medium/high band, the **top 3 risk factors**,
the **SHAP waterfall chart** for that applicant, and a "Model & fairness" tab (Gini, bands, age audit).
A pre-built copy is in `frontend/dist/`, so the whole app runs from one command:

```bash
pip install -r requirements-serving.txt
uvicorn serving.app:app --port 8000          # UI + API on http://localhost:8000
# to change the UI:  cd frontend && npm install && npm run dev   (proxy to :8000)   then   npm run build
python scripts/bias_audit.py                 # regenerates reports/bias_audit.json
```

**Bias audit (age, 200-row held-out test split):** approval-side rate 68-76%, lowest impact ratio **0.90** (passes the 0.80
four-fifths rule). Each group has only 40-60 applicants, so treat this as indicative. **Gender cannot be audited**: it is not in
`finwise_loan_applications.csv` and no other file joins to it. Add it to the application data (kept out of the model) and re-run the script.

---

## Demo

A walkthrough demo is available here: [Demo video](https://drive.google.com/file/d/1abJWtPxT3-3zkL7aLRfkXpR0BlictvwP/view?usp=drive_link)
