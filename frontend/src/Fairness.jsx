import { useEffect, useState } from "react";
import { BAND, pct } from "./format.js";

export default function Fairness() {
  const [d, setD] = useState(null), [err, setErr] = useState(null);
  useEffect(() => { fetch("/model-info").then((r) => r.json()).then(setD).catch(() => setErr("Can't load model details from the API.")); }, []);
  if (err) return <main className="doc"><p className="error">{err}</p></main>;
  if (!d) return <main className="doc"><p className="muted">Loading…</p></main>;
  const t = d.metrics.test, ok = t.gini >= d.target_gini, a = d.bias_audit;
  return (
    <main className="doc">
      <h2>Accuracy</h2>
      <p className="lead">Gini on held-out test set: <b className={ok ? "good" : "bad"}>{t.gini.toFixed(3)}</b> <span className="muted">(target {d.target_gini} or higher)</span></p>
      <p className="muted">Test n={t.n}. Validation Gini {d.metrics.val.gini.toFixed(3)}, train {d.metrics.train.gini.toFixed(3)}. Baseline logistic regression on the same test set: {d.baseline_test_gini.toFixed(3)}.
        Model: {d.selected_config}, {d.trees_used} trees. Probabilities are not calibrated (class-weighted training), so decisions use the risk band.</p>

      <h2>Risk bands</h2>
      <div className="scroll"><table>
        <thead><tr><th>Band</th><th>Rule on model output</th></tr></thead>
        <tbody>{["low", "medium", "high"].map((k) => (
          <tr key={k}><td><i className="dot" style={{ background: BAND[k].color }} />{BAND[k].label}</td><td>{d.risk_bands[k]}</td></tr>))}</tbody>
      </table></div>

      <h2>Fairness by age</h2>
      {a ? (<>
        <p className="muted">{a.data}. A high-risk flag is the adverse outcome. Impact ratio is each group's approval-side rate divided by the best group's; below 0.80 would fail the four-fifths rule. {a.caveat}</p>
        <div className="scroll"><table>
          <thead><tr><th>Age</th><th>Applicants</th><th>Actual default rate</th><th>Flagged high risk</th><th>Approval-side rate</th><th>Impact ratio</th><th>Catches defaulters</th></tr></thead>
          <tbody>{a.groups.map((g) => (
            <tr key={g.group}><td>{g.group}</td><td>{g.n}</td><td>{pct(g.base_default_rate)}</td><td>{pct(g.pct_flagged_high_risk)}</td>
              <td>{pct(g.pct_approved)}</td><td className={g.impact_ratio < 0.8 ? "bad" : ""}>{g.impact_ratio.toFixed(2)}</td><td>{g.tpr != null ? pct(g.tpr, 0) : "n/a"}</td></tr>))}</tbody>
        </table></div>
        <h2>Fairness by gender</h2>
        <p className="warn">Not audited: {a.gender.reason}. Add gender to the application data (held out of the model, used only for auditing) and re-run <code>scripts/bias_audit.py</code>.</p>
      </>) : <p className="muted">Run <code>python scripts/bias_audit.py</code> to generate the audit.</p>}
    </main>
  );
}
