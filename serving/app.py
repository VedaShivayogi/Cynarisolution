"""
FinWise credit-risk scoring API.

    uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 2

Design notes
  * The model + preprocessor are loaded ONCE at startup (lifespan) and shared read-only.
  * Endpoints are plain `def`, so FastAPI runs them in a threadpool and the event loop never blocks.
  * Scale out with --workers (one process per core); XGBoost is pinned to 1 thread per request.
"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pathlib import Path

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .schemas import Application, BatchRequest, BatchResponse, Prediction
from .service import HIGH_T, LOW_T, ModelService

log = logging.getLogger("finwise.api")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = ModelService()
    svc.predict([{"age": 35, "income_annual_inr": 900000, "employment_type": "salaried", "employment_years": 5,
                  "existing_loans": 1, "existing_emi_inr": 5000, "credit_score": 700, "loan_amount_inr": 500000,
                  "loan_purpose": "home", "ltv_ratio": 0.6, "dti_ratio": 0.2}])   # warm-up: first call is slower
    state["svc"] = svc
    log.info("model loaded: %s via %s path", svc.meta["mlflow_run_id"], svc.scoring_path)
    yield
    state.clear()


app = FastAPI(title="FinWise Credit Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1e3:.2f}"
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal scoring error. See server logs."})


def _public(res: dict, app_id, version: str) -> Prediction:
    res = {k: v for k, v in res.items() if not k.startswith("_")}
    return Prediction(application_id=app_id, model_version=version, **res)


@app.get("/health")
def health():
    """Liveness + readiness: 200 only once the model is loaded."""
    svc = state.get("svc")
    if svc is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "scoring_path": svc.scoring_path}


def _bias_audit():
    f = Path(__file__).resolve().parent.parent / "reports/bias_audit.json"
    return json.loads(f.read_text()) if f.exists() else None


@app.get("/model-info")
def model_info():
    svc = state["svc"]
    m = svc.meta
    return {
        "model_version": m["mlflow_run_id"], "selected_config": m["selected_config"],
        "selection_criterion": m["selection_criterion"], "hyperparameters": m["hyperparameters"],
        "trees_used": svc.n_trees, "trees_stored": svc.booster.num_boosted_rounds(),
        "metrics": {k: {"gini": round(v["gini"], 4), "auc": round(v["auc"], 4), "n": v["n"]} for k, v in m["metrics"].items()},
        "baseline_test_gini": m["baseline_test_gini"], "target_gini": m["target_gini"],
        "risk_bands": {"low": f"p < {LOW_T}", "medium": f"{LOW_T} <= p < {HIGH_T}", "high": f"p >= {HIGH_T}"},
        "bias_audit": _bias_audit(), "probability_calibrated": False, "categories": svc.categories, "scoring_path": svc.scoring_path,
    }


@app.post("/predict", response_model=Prediction)
def predict(application: Application):
    svc = state["svc"]
    row = application.model_dump(exclude={"application_id"})
    return _public(svc.predict([row])[0], application.application_id, svc.meta["mlflow_run_id"])


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    svc = state["svc"]
    rows = [a.model_dump(exclude={"application_id"}) for a in req.applications]
    out = svc.predict(rows)
    preds = [_public(r, a.application_id, svc.meta["mlflow_run_id"]) for r, a in zip(out, req.applications)]
    return BatchResponse(count=len(preds), predictions=preds)


# React UI (built with `cd frontend && npm install && npm run build`) is served from the same origin.
_dist = Path(__file__).resolve().parent.parent / "frontend/dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
