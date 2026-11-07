"""
serving/service.py
==================
Model-serving core for FinWise (Week 4). No web code here, so it can be unit-tested and
benchmarked without HTTP.

Two scoring paths exist and MUST agree (tests/test_integration.py::test_fast_path_parity):

  * "pipeline"  - the reference path. Calls the Week-3 `data_pipeline.engineer_features`
                  and the pickled `FinWisePreprocessor.transform`, exactly as in training.
  * "fast"      - a plain-Python/NumPy re-implementation of the same maths for one or a few
                  rows (no pandas), used by default because pandas costs ~12 ms per request.

Explanations use XGBoost's native TreeSHAP (`pred_contribs`) restricted to the SAME trees the
probability uses (`iteration_range=(0, best_iteration + 1)`). The saved model holds 46 trees
because early stopping keeps the 30 patience rounds, but only the first 16 are "the model";
scoring with all 46 would change probabilities by up to 0.20.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # the pickles reference module `data_pipeline`
    sys.path.insert(0, str(ROOT))
import data_pipeline as dp             # noqa: E402  (Week-3 module, reused unchanged)

ART = Path(os.environ.get("FINWISE_ARTIFACTS", ROOT / "artifacts"))


class _TolerantUnpickler(pickle.Unpickler):
    """Loads pickles whose classes were recorded as `__main__.X`, which happens when the artifact was produced by
    running a script directly (e.g. `python data_pipeline.py`). Maps them back to the real `data_pipeline` module."""
    def find_class(self, module, name):
        if module == "__main__" and hasattr(dp, name):
            return getattr(dp, name)
        return super().find_class(module, name)


def _load_pickle(path):
    with open(path, "rb") as f:
        return _TolerantUnpickler(f).load()

# Risk bands on the model's raw output. Chosen on the validation split, reported on test
# (docs/WEEK4_...md). Override with env vars if credit policy differs.
LOW_T = float(os.environ.get("FINWISE_LOW_T", 0.25))
HIGH_T = float(os.environ.get("FINWISE_HIGH_T", 0.50))

LABELS = {
    "age": "Age", "income_annual_inr": "Annual income", "employment_years": "Years employed",
    "existing_loans": "Number of existing loans", "existing_emi_inr": "Existing monthly EMIs",
    "credit_score": "Credit score", "loan_amount_inr": "Requested loan amount",
    "ltv_ratio": "Loan-to-value ratio", "dti_ratio": "Debt-to-income ratio",
    "emi_to_income_ratio": "EMI as share of monthly income", "loan_to_income_ratio": "Loan vs annual income",
    "total_debt": "Total debt incl. new loan", "debt_to_income": "Total debt vs annual income",
    "employment_stability": "Employment stability (years employed vs working life)",
    "high_dti": "High DTI flag (>0.5)", "high_ltv": "High LTV flag (>0.8)", "low_credit": "Low credit flag (<600)",
    "multiple_loans": "Multiple loans flag (2+)", "risk_score": "Combined risk-flag count",
    "employment_type_encoded": "Employment type", "loan_purpose_encoded": "Loan purpose",
    "credit_score_band_encoded": "Credit score band", "age_group_encoded": "Age group",
}
_CAT_RAW = {"employment_type_encoded": "employment_type", "loan_purpose_encoded": "loan_purpose",
            "credit_score_band_encoded": "credit_score_band", "age_group_encoded": "age_group"}


def band_for(p: float) -> str:
    return "low" if p < LOW_T else "medium" if p < HIGH_T else "high"


def _fmt(name: str, v) -> str:
    if name in ("income_annual_inr", "loan_amount_inr", "existing_emi_inr", "total_debt"):
        return f"₹{v:,.0f}"
    if isinstance(v, str):
        return v
    if name in ("high_dti", "high_ltv", "low_credit", "multiple_loans"):
        return "yes" if v else "no"
    if name in ("ltv_ratio", "dti_ratio", "emi_to_income_ratio", "loan_to_income_ratio", "debt_to_income",
                "employment_stability"):
        return f"{v:.2f}"
    return f"{v:g}"


class ModelService:
    def __init__(self, scoring_path: str | None = None):
        self.scoring_path = (scoring_path or os.environ.get("FINWISE_SCORING_PATH", "fast")).lower()
        if self.scoring_path not in ("fast", "pipeline"):
            raise ValueError("scoring_path must be 'fast' or 'pipeline'")
        self.pre: dp.FinWisePreprocessor = _load_pickle(ART / "preprocessor.pkl")
        self.model: xgb.XGBClassifier = _load_pickle(ART / "champion_model.pkl")
        self.meta = json.load(open(ART / "champion_model_metadata.json"))
        self.model.set_params(n_jobs=1)            # one thread per request; concurrency comes from workers
        self.booster = self.model.get_booster()
        self.n_trees = int(self.model.best_iteration) + 1
        self.irange = (0, self.n_trees)
        self.features: List[str] = list(self.pre.feature_names_)
        assert self.features == list(self.booster.feature_names), "preprocessor/model feature mismatch"
        self.categories: Dict[str, List[str]] = {c: list(le.classes_) for c, le in self.pre.label_encoders.items()}
        self._enc = {c: {v: i for i, v in enumerate(cls)} for c, cls in self.categories.items()}
        self._mean = self.pre.scaler.mean_
        self._scale = self.pre.scaler.scale_

    # ---- feature construction -------------------------------------------------------
    def _fast_row(self, r: dict) -> tuple[np.ndarray, dict]:
        """Re-implementation of data_pipeline.engineer_features + FinWisePreprocessor.transform for ONE row."""
        inc = float(r["income_annual_inr"]); emi = float(r["existing_emi_inr"]); loan = float(r["loan_amount_inr"])
        cs = float(r["credit_score"]); age = float(r["age"]); yrs = float(r["employment_years"])
        dti, ltv, nloans = float(r["dti_ratio"]), float(r["ltv_ratio"]), float(r["existing_loans"])
        total_debt = emi * 12 + loan
        band = ("Poor" if cs <= 580 else "Fair" if cs <= 670 else "Good" if cs <= 740
                else "Very Good" if cs <= 800 else "Excellent")
        agrp = ("Young" if age <= 30 else "Early Career" if age <= 40 else "Mid Career" if age <= 50
                else "Senior" if age <= 60 else "Elder")
        high_dti, high_ltv, low_credit, multi = int(dti > 0.5), int(ltv > 0.8), int(cs < 600), int(nloans >= 2)
        raw = {
            "age": age, "income_annual_inr": inc, "employment_years": yrs, "existing_loans": nloans,
            "existing_emi_inr": emi, "credit_score": cs, "loan_amount_inr": loan, "ltv_ratio": ltv, "dti_ratio": dti,
            "emi_to_income_ratio": min(max(emi / (inc / 12 + 1), 0.0), 10.0),
            "loan_to_income_ratio": loan / (inc + 1), "total_debt": total_debt, "debt_to_income": total_debt / (inc + 1),
            "employment_stability": yrs / (age - 17 + 1),
            "high_dti": high_dti, "high_ltv": high_ltv, "low_credit": low_credit, "multiple_loans": multi,
            "risk_score": high_dti + high_ltv + low_credit + multi,
        }
        cats = {"employment_type": r["employment_type"], "loan_purpose": r["loan_purpose"],
                "credit_score_band": band, "age_group": agrp}
        for c, v in cats.items():
            raw[c + "_encoded"] = self._enc[c].get(v, -1)
        x = np.array([raw[f] for f in self.features], dtype=np.float64)
        display = {**raw, **{c + "_encoded": v for c, v in cats.items()}}   # human-readable value for categoricals
        return (x - self._mean) / self._scale, display

    def _pipeline_matrix(self, rows: List[dict]) -> tuple[pd.DataFrame, List[dict]]:
        df = pd.DataFrame(rows)
        df["default_flag"] = 0                       # placeholder; dropped by the preprocessor
        df["application_id"] = df.get("application_id", pd.Series(["serving"] * len(df))).fillna("serving")
        fe = dp.engineer_features(df)
        X = self.pre.transform(fe)
        disp = []
        for i in range(len(fe)):
            d = {c: fe.iloc[i][c] for c in self.features if c in fe.columns}
            d.update({c + "_encoded": str(fe.iloc[i][c]) for c in self.categories})
            disp.append(d)
        return X, disp

    def _matrix(self, rows: List[dict]) -> tuple[np.ndarray, List[dict]]:
        if self.scoring_path == "pipeline":
            X, disp = self._pipeline_matrix(rows)
            return X[self.features].to_numpy(dtype=np.float64), disp
        out = [self._fast_row(r) for r in rows]
        return np.vstack([o[0] for o in out]), [o[1] for o in out]

    def _waterfall(self, s: np.ndarray, base: float, disp: dict, top_n: int = 8) -> dict:
        """SHAP waterfall in log-odds space (same convention as shap.plots.waterfall): starts at the average
        applicant, one step per feature (top_n by |impact|, rest lumped), ends at this applicant's margin."""
        order = np.argsort(-np.abs(s))
        steps, cur = [], base
        for j in order[:top_n]:
            f = self.features[j]
            steps.append({"label": f"{LABELS.get(f, f)} = {_fmt(f, disp[f])}", "shap_logodds": round(float(s[j]), 4),
                          "start": cur, "end": cur + float(s[j])})
            cur += float(s[j])
        rest = order[top_n:]
        if len(rest):
            r = float(s[rest].sum())
            steps.append({"label": f"{len(rest)} other features", "shap_logodds": round(r, 4), "start": cur, "end": cur + r})
            cur += r
        return {"base_value": base, "final_value": cur, "steps": steps}

    # ---- scoring --------------------------------------------------------------------
    def predict(self, rows: List[dict], explain: bool = True) -> List[dict]:
        X, disp = self._matrix(rows)
        dm = xgb.DMatrix(X, feature_names=self.features)
        margin = self.booster.predict(dm, output_margin=True, iteration_range=self.irange)
        probs = 1.0 / (1.0 + np.exp(-margin))
        contribs = self.booster.predict(dm, pred_contribs=True, iteration_range=self.irange) if explain else None
        results = []
        for i, p in enumerate(probs):
            p = float(p)
            res = {"default_probability": round(p, 6), "score": int(round(300 + 600 * (1 - p))), "risk_band": band_for(p)}
            if explain:
                s = contribs[i, :-1]
                order = np.argsort(-s)
                def factor(j):
                    f = self.features[j]
                    return {"feature": f, "label": LABELS.get(f, f), "value": _fmt(f, disp[i][f]),
                            "shap_logodds": round(float(s[j]), 4)}
                res["top_risk_factors"] = [factor(j) for j in order[:3] if s[j] > 0]
                res["top_protective_factors"] = [factor(j) for j in order[::-1][:3] if s[j] < -0.05]
                res["waterfall"] = self._waterfall(s, float(contribs[i, -1]), disp[i])
                res["_base_logodds"] = float(contribs[i, -1])
                res["_margin"] = float(margin[i])
            results.append(res)
        return results
