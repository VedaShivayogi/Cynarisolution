import { useState } from "react";
import ScoreRuler from "./ScoreRuler.jsx";
import Waterfall from "./Waterfall.jsx";
import Fairness from "./Fairness.jsx";
import { BAND, inr, pct } from "./format.js";

const PRESETS = {
  "Steady salaried": { age: 38, income_annual_inr: 1800000, employment_type: "salaried", employment_years: 12, existing_loans: 1, existing_emi_inr: 9000, credit_score: 790, loan_amount_inr: 500000, loan_purpose: "vehicle", ltv_ratio: 0.6 },
  "Stretched borrower": { age: 33, income_annual_inr: 600000, employment_type: "self_employed", employment_years: 4, existing_loans: 2, existing_emi_inr: 22000, credit_score: 640, loan_amount_inr: 1000000, loan_purpose: "business", ltv_ratio: 0.8 },
  "Overleveraged": { age: 35, income_annual_inr: 300000, employment_type: "contract", employment_years: 5, existing_loans: 3, existing_emi_inr: 40000, credit_score: 560, loan_amount_inr: 2500000, loan_purpose: "personal", ltv_ratio: 0.9 },
};
const EMP = ["salaried", "self_employed", "government", "contract"];
const PUR = ["home", "personal", "vehicle", "education", "medical", "business"];
const nice = (s) => s.replace("_", " ").replace(/^./, (c) => c.toUpperCase());

function Num({ label, k, form, set, step = 1, hint }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="number" inputMode="decimal" step={step} value={form[k]} required
             onChange={(e) => set(k, e.target.value === "" ? "" : Number(e.target.value))} />
      {hint && <small>{hint}</small>}
    </label>
  );
}

export default function App() {
  const [tab, setTab] = useState("assess");
  const [form, setForm] = useState(PRESETS["Stretched borrower"]);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  async function submit(e) {
    e.preventDefault();
    setBusy(true); setErr(null);
    try {
      const r = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(form) });
      const body = await r.json();
      if (!r.ok) {
        const msg = Array.isArray(body.detail)
          ? body.detail.map((d) => `${d.loc.slice(-1)[0].replaceAll("_", " ")}: ${d.msg}`).join("; ")
          : "The scoring service returned an error.";
        throw new Error(msg);
      }
      setRes(body);
    } catch (x) {
      setErr(x.message === "Failed to fetch" ? "Can't reach the scoring service. Check that the API is running on port 8000." : x.message);
    } finally { setBusy(false); }
  }

  const band = res && BAND[res.risk_band];
  return (
    <>
      <header className="top">
        <h1>FinWise <span>Credit risk</span></h1>
        <nav>
          <button aria-current={tab === "assess"} onClick={() => setTab("assess")}>Assess an applicant</button>
          <button aria-current={tab === "model"} onClick={() => setTab("model")}>Model &amp; fairness</button>
        </nav>
      </header>

      {tab === "model" ? <Fairness /> : (
        <main className="grid">
          <form className="entry" onSubmit={submit}>
            <div className="presets">
              <p>Start from an example</p>
              {Object.keys(PRESETS).map((n) => (
                <button type="button" key={n} onClick={() => { setForm(PRESETS[n]); setRes(null); }}>{n}</button>
              ))}
            </div>
            <fieldset>
              <legend>Applicant</legend>
              <Num label="Age" k="age" form={form} set={set} />
              <Num label="Annual income (₹)" k="income_annual_inr" form={form} set={set} step={10000} hint={inr(form.income_annual_inr || 0)} />
              <label className="field"><span>Employment type</span>
                <select value={form.employment_type} onChange={(e) => set("employment_type", e.target.value)}>
                  {EMP.map((o) => <option key={o} value={o}>{nice(o)}</option>)}
                </select></label>
              <Num label="Years employed" k="employment_years" form={form} set={set} />
              <Num label="Credit (CIBIL) score" k="credit_score" form={form} set={set} hint="300 to 900" />
            </fieldset>
            <fieldset>
              <legend>Existing debt</legend>
              <Num label="Open loans" k="existing_loans" form={form} set={set} />
              <Num label="Monthly EMIs (₹)" k="existing_emi_inr" form={form} set={set} step={500} hint={inr(form.existing_emi_inr || 0)} />
            </fieldset>
            <fieldset>
              <legend>Requested loan</legend>
              <Num label="Loan amount (₹)" k="loan_amount_inr" form={form} set={set} step={50000} hint={inr(form.loan_amount_inr || 0)} />
              <label className="field"><span>Purpose</span>
                <select value={form.loan_purpose} onChange={(e) => set("loan_purpose", e.target.value)}>
                  {PUR.map((o) => <option key={o} value={o}>{nice(o)}</option>)}
                </select></label>
              <Num label="Loan-to-value ratio" k="ltv_ratio" form={form} set={set} step={0.01} hint="0.00 to 1.00" />
            </fieldset>
            <button className="primary" disabled={busy}>{busy ? "Scoring…" : "Score applicant"}</button>
            {err && <p className="error" role="alert">{err}</p>}
          </form>

          <section className="result" aria-live="polite">
            {!res ? (
              <div className="empty">
                <h2>No score yet</h2>
                <p>Enter the applicant's details and choose <b>Score applicant</b>. You'll get a score, a risk band and the three factors pushing risk up the most.</p>
              </div>
            ) : (
              <>
                <div className="verdict" style={{ "--c": band.color }}>
                  <div>
                    <p className="band">{band.label}</p>
                    <p className="note">{band.note}</p>
                  </div>
                  <div className="pd"><b>{pct(res.default_probability)}</b><span>raw model score (uncalibrated)</span></div>
                </div>
                <ScoreRuler score={res.score} band={res.risk_band} />

                <h2>Top 3 risk factors</h2>
                {res.top_risk_factors.length === 0 ? (
                  <p className="muted">No factor is pushing this applicant's risk above the average applicant.</p>
                ) : (
                  <ol className="factors">
                    {res.top_risk_factors.map((f) => (
                      <li key={f.feature} className={f.shap_logodds < 0.05 ? "minor" : ""}>
                        <div><b>{f.label}</b><span>{f.value}</span></div>
                        <em>{f.shap_logodds < 0.05 ? "minor" : "+" + f.shap_logodds.toFixed(2)}</em>
                      </li>
                    ))}
                  </ol>
                )}
                {res.top_protective_factors.length > 0 && (
                  <p className="protect">Working in the applicant's favour: {res.top_protective_factors.map((f) => `${f.label.toLowerCase()} (${f.value})`).join(", ")}.</p>
                )}

                <h2>How the score was built</h2>
                <p className="muted">Each bar is one factor's push on the log-odds of default, starting from the average applicant. Red bars raise risk; green bars lower it.</p>
                <div className="wf-wrap"><Waterfall data={res.waterfall} /></div>
              </>
            )}
          </section>
        </main>
      )}
    </>
  );
}
