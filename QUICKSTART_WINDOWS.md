# Quick start (Windows, Command Prompt)

## 0. Be in the right folder
The zip unpacks to a folder **inside** a folder. You must `cd` into the inner one, the one that contains
`serving`, `tests`, `frontend`, `data_pipeline.py` and `requirements-serving.txt`:

```bat
cd C:\Users\Admin\Downloads\finwise-full-project\FinWise-Lending-Credit-Risk-Project
dir
```
`dir` must list `serving`, `tests`, `data_pipeline.py`. If it doesn't, you are one level too high or too low.

## 1. Install (once)
```bat
python -m pip install -r requirements-dev.txt
```
If this fails on an `==` version pin (for example xgboost 3.2.0 or scikit-learn 1.3.0 not available for your Python),
install without the pins:
```bat
python -m pip install fastapi uvicorn pydantic xgboost scikit-learn numpy pandas pytest httpx psutil
```
(The model was trained with xgboost 3.2.0 / scikit-learn 1.3.0 on Python 3.10; newer versions work and the tests will tell you if anything differs.)

## 2. Run the app (UI + API)
```bat
python -m uvicorn serving.app:app --port 8000
```
Use `python -m uvicorn`, not plain `uvicorn`: pip's Scripts folder is usually not on PATH on Windows.
Open http://localhost:8000 (UI) or http://localhost:8000/docs (API docs). Stop with Ctrl+C.

## 3. Run the tests (optional)
```bat
python -m pytest -v
```
Expect `36 passed`.

## 4. Load test (optional, ~5 minutes per client)
Close the app from step 2 first (the load test starts its own server on port 8765).
```bat
python loadtest\loadtest.py --client lean
python loadtest\loadtest.py --client httpx
python loadtest\render_report.py
```
Results: `docs\loadtest_results_*.json` and `docs\WEEK4_SERVING_AND_LOAD_TEST_REPORT.md`.
For a 30-second smoke version add `--quick` to the first two commands.
If a server fails to start, the real error is printed and saved in `docs\server.log`.

## Common errors
| Message | Fix |
|---|---|
| `can't open file ... loadtest\render_report.py` | You are in the wrong folder (step 0) |
| `'uvicorn' is not recognized` | Use `python -m uvicorn ...` and run step 1 |
| `ModuleNotFoundError: fastapi / xgboost / psutil` | Run step 1 |
| `server failed to start` | Read the printed log tail; usually missing packages or port in use |
| `Address already in use` | Another server is on that port: close it or add `--port 8001` |
