import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data_pipeline as dp                      # noqa: E402
from serving.app import app                     # noqa: E402
from serving.service import ModelService        # noqa: E402

FEATURE_COLS = ["age", "income_annual_inr", "employment_type", "employment_years", "existing_loans",
                "existing_emi_inr", "credit_score", "loan_amount_inr", "loan_purpose", "ltv_ratio", "dti_ratio"]


def to_payload(row) -> dict:
    d = {k: row[k] for k in FEATURE_COLS}
    for k in ("age", "existing_loans", "credit_score"):
        d[k] = int(d[k])
    for k in ("income_annual_inr", "employment_years", "existing_emi_inr", "loan_amount_inr", "ltv_ratio", "dti_ratio"):
        d[k] = float(d[k])
    return d


@pytest.fixture(scope="session")
def raw_df():
    return pd.read_csv(ROOT / "finwise_loan_applications.csv")


@pytest.fixture(scope="session")
def split(tmp_path_factory):
    """Re-runs the Week-3 pipeline into a temp dir (never touches the repo's artifacts/)."""
    return dp.run_pipeline(str(ROOT / "finwise_loan_applications.csv"), artifacts_dir=str(tmp_path_factory.mktemp("art")))


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:                  # `with` runs the lifespan => model is loaded
        yield c


@pytest.fixture(scope="session")
def svc_fast():
    return ModelService("fast")


@pytest.fixture(scope="session")
def svc_pipeline():
    return ModelService("pipeline")


@pytest.fixture(scope="session")
def good_payload():
    return {"application_id": "APP-T1", "age": 35, "income_annual_inr": 900000, "employment_type": "salaried",
            "employment_years": 8, "existing_loans": 1, "existing_emi_inr": 8000, "credit_score": 720,
            "loan_amount_inr": 500000, "loan_purpose": "vehicle", "ltv_ratio": 0.6}


@pytest.fixture(scope="session")
def live_server():
    """A REAL uvicorn with 2 worker processes on a free port: exercises process start-up, the lifespan hook,
    real sockets and real concurrency, which TestClient cannot."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-W", "ignore", "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1",
                             "--port", str(port), "--workers", "2", "--log-level", "warning"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not become healthy in 60 s")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
