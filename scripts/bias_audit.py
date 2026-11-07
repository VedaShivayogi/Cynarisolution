"""Age bias audit of the champion model on the held-out TEST split (200 rows) -> reports/bias_audit.json.
    python scripts/bias_audit.py
Gender: NOT auditable. finwise_loan_applications.csv has no gender column and no other file joins to it."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sklearn.metrics import roc_auc_score
from serving.service import ModelService, HIGH_T, LOW_T
import data_pipeline as dp

svc = ModelService("pipeline")
res = dp.run_pipeline(str(ROOT / "finwise_loan_applications.csv"), artifacts_dir=tempfile.mkdtemp())
raw, y = res["raw_test_df"], res["y_test"].to_numpy()
rows = raw.drop(columns=["application_id", "default_flag"]).to_dict("records")
p = np.array([r["default_probability"] for r in svc.predict(rows, explain=False)])
bands = ["<=30", "31-40", "41-50", "51+"]
grp = np.array([bands[0] if a <= 30 else bands[1] if a <= 40 else bands[2] if a <= 50 else bands[3] for a in raw.age])
out = []
for g in bands:
    m = grp == g
    fl = p[m] >= HIGH_T
    out.append({"group": g, "n": int(m.sum()), "base_default_rate": float(y[m].mean()), "pct_flagged_high_risk": float(fl.mean()),
                "pct_approved": float(1 - fl.mean()), "tpr": float(fl[y[m] == 1].mean()) if (y[m] == 1).any() else None,
                "auc": float(roc_auc_score(y[m], p[m])) if len(set(y[m])) > 1 else None})
best = max(r["pct_approved"] for r in out)
for r in out: r["impact_ratio"] = r["pct_approved"] / best
audit = {"attribute": "age_band", "data": "held-out test split (n=200)", "threshold": HIGH_T,
         "groups": out, "gender": {"audited": False, "reason": "no gender field in loan_applications.csv; files share no join key"},
         "caveat": "Groups of ~30-60 applicants: impact ratios are noisy. Re-run on more data before drawing conclusions."}
(ROOT / "reports").mkdir(exist_ok=True)
json.dump(audit, open(ROOT / "reports/bias_audit.json", "w"), indent=2)
for r in out: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
