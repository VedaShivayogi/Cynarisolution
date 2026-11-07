"""
Integration tests for the FinWise serving layer.   Run:  pytest -v     (or: pytest -m "not live" for no sockets)

What is proven
  1. Contract       - response shape, ranges, band/score rules, validation errors (422), 404/405.
  2. Faithfulness   - scoring the held-out TEST rows over HTTP reproduces the Gini recorded at training time.
  3. No train/serve skew - fast serving path == Week-3 pipeline path == offline model.predict_proba.
  4. Explanations   - SHAP uses the same 16 trees as the probability and is additive.
  5. Concurrency    - 50 simultaneous requests against a real multi-process server: no errors, results identical
                      to sequential scoring (thread/process safe).
"""
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pytest
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from conftest import ROOT, to_payload
from serving.service import HIGH_T, LOW_T

gini = lambda y, p: 2 * roc_auc_score(y, p) - 1


# ------------------------------------------------------------------ 1. contract
def test_health_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    meta = json.load(open(ROOT / "artifacts/champion_model_metadata.json"))
    assert info["model_version"] == meta["mlflow_run_id"]
    assert info["trees_used"] == meta["best_iteration"] + 1          # 16 of 46 stored trees
    assert info["trees_stored"] > info["trees_used"]
    assert info["metrics"]["test"]["gini"] == pytest.approx(meta["metrics"]["test"]["gini"], abs=1e-4)
    assert info["probability_calibrated"] is False


def test_predict_contract(client, good_payload):
    r = client.post("/predict", json=good_payload)
    assert r.status_code == 200
    d = r.json()
    assert d["application_id"] == "APP-T1"
    assert 0.0 <= d["default_probability"] <= 1.0
    assert d["score"] == round(300 + 600 * (1 - d["default_probability"]))
    expected = "low" if d["default_probability"] < LOW_T else "medium" if d["default_probability"] < HIGH_T else "high"
    assert d["risk_band"] == expected
    assert len(d["top_risk_factors"]) <= 3
    assert all(f["shap_logodds"] > 0 for f in d["top_risk_factors"])
    assert all(f["shap_logodds"] < 0 for f in d["top_protective_factors"])
    assert float(r.headers["x-process-time-ms"]) > 0 and r.headers["x-request-id"]


def test_risky_applicant_scores_worse_than_safe(client, good_payload):
    risky = dict(good_payload, income_annual_inr=300000, existing_emi_inr=40000, existing_loans=3, credit_score=520,
                 loan_amount_inr=2500000, ltv_ratio=0.95, loan_purpose="personal", employment_type="contract")
    safe, bad = client.post("/predict", json=good_payload).json(), client.post("/predict", json=risky).json()
    assert bad["default_probability"] > safe["default_probability"] + 0.3
    assert bad["risk_band"] == "high" and safe["risk_band"] == "low"
    assert bad["score"] < safe["score"]
    assert {"Debt-to-income ratio", "Credit score"} & {f["label"] for f in bad["top_risk_factors"]}


@pytest.mark.parametrize("patch", [
    {"credit_score": 299}, {"credit_score": 901}, {"age": 17}, {"age": 101}, {"income_annual_inr": 0},
    {"income_annual_inr": -5}, {"loan_amount_inr": 0}, {"ltv_ratio": 2.5}, {"existing_loans": -1},
    {"employment_type": "astronaut"}, {"loan_purpose": "yacht"}, {"age": "thirty"}, {"credit_score": None},
    {"unexpected_field": 1},
], ids=lambda p: next(iter(p.items())).__repr__())
def test_validation_rejects_bad_input(client, good_payload, patch):
    r = client.post("/predict", json={**good_payload, **patch})
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("missing", ["age", "income_annual_inr", "credit_score", "employment_type", "loan_amount_inr"])
def test_missing_required_field(client, good_payload, missing):
    body = {k: v for k, v in good_payload.items() if k != missing}
    r = client.post("/predict", json=body)
    assert r.status_code == 422 and missing in r.text


def test_malformed_json_and_routes(client):
    assert client.post("/predict", content="{not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.get("/nope").status_code == 404
    # 405 normally; 404 when the built UI is mounted at "/" (the static handler answers unmatched methods)
    assert client.get("/predict").status_code in (404, 405)


def test_dti_is_derived_when_omitted_and_matches_dataset(client, raw_df):
    for _, row in raw_df.sample(60, random_state=1).iterrows():
        body = to_payload(row)
        explicit = client.post("/predict", json=body).json()["default_probability"]
        body.pop("dti_ratio")
        derived = client.post("/predict", json=body).json()["default_probability"]
        assert derived == pytest.approx(explicit, abs=0.02)         # dti is rounded to 2 dp in the source data


def test_batch_matches_single_and_limits(client, raw_df):
    rows = [to_payload(r) for _, r in raw_df.head(25).iterrows()]
    batch = client.post("/predict/batch", json={"applications": rows}).json()
    assert batch["count"] == 25
    for row, b in zip(rows, batch["predictions"]):
        assert client.post("/predict", json=row).json()["default_probability"] == pytest.approx(b["default_probability"], abs=1e-9)
    assert client.post("/predict/batch", json={"applications": []}).status_code == 422
    assert client.post("/predict/batch", json={"applications": rows * 21}).status_code == 422   # 525 > 500


# ------------------------------------------------------------------ 2. faithfulness
def test_http_scoring_of_test_split_reproduces_training_gini(client, split):
    """THE end-to-end check: raw test rows -> HTTP -> Gini must equal the number in champion_model_metadata.json."""
    meta = json.load(open(ROOT / "artifacts/champion_model_metadata.json"))
    raw_test, y = split["raw_test_df"], split["y_test"].to_numpy()
    payloads = [to_payload(r) for _, r in raw_test.iterrows()]
    probs = []
    for i in range(0, len(payloads), 100):
        out = client.post("/predict/batch", json={"applications": payloads[i:i + 100]}).json()["predictions"]
        probs += [p["default_probability"] for p in out]
    g = gini(y, probs)
    assert len(probs) == meta["metrics"]["test"]["n"] == 200
    assert g == pytest.approx(meta["metrics"]["test"]["gini"], abs=1e-4)
    assert g >= meta["target_gini"]                                  # project acceptance target 0.45


def test_risk_bands_are_monotone_on_test_split(client, split):
    raw_test, y = split["raw_test_df"], split["y_test"].to_numpy()
    out = client.post("/predict/batch", json={"applications": [to_payload(r) for _, r in raw_test.iterrows()]}).json()["predictions"]
    bands = np.array([o["risk_band"] for o in out])
    rate = {b: y[bands == b].mean() for b in ("low", "medium", "high") if (bands == b).any()}
    assert rate["low"] < rate["medium"] < rate["high"], rate


# ------------------------------------------------------------------ 3. no train/serve skew
def test_fast_path_parity_with_reference_pipeline(svc_fast, svc_pipeline, raw_df):
    rows = raw_df.drop(columns=["application_id", "default_flag"]).to_dict("records")     # all 1000 rows
    a, b = svc_fast.predict(rows, explain=True), svc_pipeline.predict(rows, explain=True)
    assert max(abs(x["default_probability"] - y["default_probability"]) for x, y in zip(a, b)) < 1e-9
    for x, y in zip(a, b):
        assert [f["feature"] for f in x["top_risk_factors"]] == [f["feature"] for f in y["top_risk_factors"]]
        assert [f["value"] for f in x["top_risk_factors"]] == [f["value"] for f in y["top_risk_factors"]]


def test_service_equals_offline_model(svc_fast, split):
    X, raw = split["X_test"], split["raw_test_df"].drop(columns=["application_id", "default_flag"]).to_dict("records")
    offline = svc_fast.model.predict_proba(X)[:, 1]
    served = np.array([r["default_probability"] for r in svc_fast.predict(raw, explain=False)])
    assert np.abs(offline - served).max() < 1e-5


# ------------------------------------------------------------------ 4. explanations
def test_shap_uses_the_same_trees_and_is_additive(svc_fast, split):
    X = xgb.DMatrix(split["X_test"], feature_names=svc_fast.features)
    contrib = svc_fast.booster.predict(X, pred_contribs=True, iteration_range=svc_fast.irange)
    margin = svc_fast.booster.predict(X, output_margin=True, iteration_range=svc_fast.irange)
    assert np.abs(contrib.sum(1) - margin).max() < 1e-4                                   # SHAP additivity
    proba = 1 / (1 + np.exp(-margin))
    assert np.abs(proba - svc_fast.model.predict_proba(split["X_test"])[:, 1]).max() < 1e-5
    # pitfall guard: the stored model has extra patience trees; using them would change predictions materially
    all_trees = svc_fast.booster.predict(X)
    assert np.abs(all_trees - proba).max() > 0.05


def test_top_factors_are_sorted_by_impact(client, good_payload):
    risky = dict(good_payload, income_annual_inr=300000, existing_emi_inr=40000, credit_score=520, loan_amount_inr=2500000)
    f = client.post("/predict", json=risky).json()["top_risk_factors"]
    vals = [x["shap_logodds"] for x in f]
    assert vals == sorted(vals, reverse=True) and len(f) == 3


# ------------------------------------------------------------------ 5. concurrency (real server)
@pytest.mark.live
def test_50_concurrent_requests_are_correct(live_server, raw_df):
    payloads = [to_payload(r) for _, r in raw_df.head(50).iterrows()]
    with httpx.Client(base_url=live_server, timeout=30) as c:
        sequential = [c.post("/predict", json=p).json()["default_probability"] for p in payloads]

    def hit(p):
        with httpx.Client(base_url=live_server, timeout=30) as c:       # own connection => genuinely simultaneous
            r = c.post("/predict", json=p)
            return r.status_code, r.json().get("default_probability")

    with ThreadPoolExecutor(max_workers=50) as ex:
        results = list(ex.map(hit, payloads * 2))                         # 100 requests, 50 in flight at once
    assert all(code == 200 for code, _ in results)
    assert [p for _, p in results[:50]] == pytest.approx(sequential, abs=1e-9)
    assert [p for _, p in results[50:]] == pytest.approx(sequential, abs=1e-9)


@pytest.mark.live
def test_bad_requests_under_load_do_not_poison_good_ones(live_server, good_payload):
    def hit(i):
        body = good_payload if i % 2 == 0 else dict(good_payload, credit_score=1)
        with httpx.Client(base_url=live_server, timeout=30) as c:
            return i % 2, c.post("/predict", json=body).status_code
    with ThreadPoolExecutor(max_workers=50) as ex:
        res = list(ex.map(hit, range(100)))
    assert all(code == (200 if parity == 0 else 422) for parity, code in res)


# ------------------------------------------------------------------ 6. waterfall + UI
def test_waterfall_is_consistent_with_prediction(client, good_payload):
    risky = dict(good_payload, income_annual_inr=300000, existing_emi_inr=40000, credit_score=520, loan_amount_inr=2500000)
    for body in (good_payload, risky):
        d = client.post("/predict", json=body).json()
        w = d["waterfall"]
        assert 5 <= len(w["steps"]) <= 9
        cur = w["base_value"]
        for s in w["steps"]:                                         # steps chain end-to-start and match their SHAP value
            assert s["start"] == pytest.approx(cur, abs=1e-9) and s["end"] - s["start"] == pytest.approx(s["shap_logodds"], abs=1e-3)
            cur = s["end"]
        assert cur == pytest.approx(w["final_value"], abs=1e-3)
        assert 1 / (1 + np.exp(-w["final_value"])) == pytest.approx(d["default_probability"], abs=1e-3)   # ends AT the prediction


def test_ui_is_served_when_built(client):
    if not (ROOT / "frontend/dist/index.html").exists():
        pytest.skip("frontend not built")
    r = client.get("/")
    assert r.status_code == 200 and "<div id=\"root\">" in r.text
    assert client.post("/predict", json={}).status_code == 422       # API routes still win over the static mount


# ------------------------------------------------------------------ 7. artifact compatibility
def test_loads_preprocessor_pickled_from_main(tmp_path):
    """Regression: a preprocessor.pkl saved by a directly-run script references __main__.FinWisePreprocessor.
    The service must still load it (this crashed the server on Windows)."""
    import subprocess, sys
    helper = '''
import sys, shutil, pickle, os
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
import data_pipeline
from data_pipeline import FinWisePreprocessor
shutil.copytree(os.path.join(root, "artifacts"), out, dirs_exist_ok=True)
pre = pickle.load(open(os.path.join(root, "artifacts/preprocessor.pkl"), "rb"))
FinWisePreprocessor.__module__ = "__main__"
pickle.dump(pre, open(os.path.join(out, "preprocessor.pkl"), "wb"))
'''
    out = tmp_path / "art"
    subprocess.run([sys.executable, "-W", "ignore", "-c", helper, str(ROOT), str(out)], check=True, capture_output=True)
    code = ("from serving.service import ModelService\n"
            "s = ModelService('fast')\n"
            "r = s.predict([{'age':35,'income_annual_inr':900000,'employment_type':'salaried','employment_years':5,"
            "'existing_loans':1,'existing_emi_inr':5000,'credit_score':700,'loan_amount_inr':500000,'loan_purpose':'home',"
            "'ltv_ratio':0.6,'dti_ratio':0.2}])\nprint('OK', r[0]['risk_band'])")
    res = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env={**__import__("os").environ, "FINWISE_ARTIFACTS": str(out)})
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-600:]
