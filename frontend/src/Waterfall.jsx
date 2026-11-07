// SHAP waterfall (log-odds space, same convention as shap.plots.waterfall):
// starts at the average applicant E[f(x)], each bar is one feature's push, ends at this applicant's f(x).
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export default function Waterfall({ data }) {
  const { steps, base_value, final_value } = data;
  const rowH = 34, left = 250, right = 70, top = 34, W = 760;
  const pts = [base_value, final_value, ...steps.flatMap((s) => [s.start, s.end])];
  const lo = Math.min(...pts), hi = Math.max(...pts), pad = (hi - lo) * 0.06 || 0.2;
  const sx = (v) => left + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - left - right);
  const H = top + steps.length * rowH + 40;
  const fmt = (v) => (v > 0 ? "+" : "−") + Math.abs(v).toFixed(2);
  return (
    <svg className="waterfall" viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label="SHAP waterfall chart showing how each factor moves the default risk">
      <line x1={sx(base_value)} x2={sx(base_value)} y1={top - 8} y2={H - 30} className="wf-ref" />
      <text x={sx(base_value)} y={top - 14} textAnchor="middle" className="wf-cap">
        average applicant · model score {(sigmoid(base_value) * 100).toFixed(0)}%
      </text>
      {steps.map((s, i) => {
        const y = top + i * rowH, a = sx(Math.min(s.start, s.end)), b = sx(Math.max(s.start, s.end));
        const up = s.shap_logodds > 0, c = up ? "var(--high)" : "var(--low)";
        return (
          <g key={i}>
            <text x={left - 12} y={y + rowH / 2 + 4} textAnchor="end" className="wf-label">{s.label}</text>
            {i > 0 && <line x1={sx(s.start)} x2={sx(s.start)} y1={y - 8} y2={y + 6} className="wf-link" />}
            <rect x={a} y={y + 5} width={Math.max(b - a, 2)} height={rowH - 12} rx={2} fill={c} />
            <text x={up ? b + 6 : a - 6} y={y + rowH / 2 + 4} textAnchor={up ? "start" : "end"} className="wf-val" fill={c}>
              {fmt(s.shap_logodds)}
            </text>
          </g>
        );
      })}
      <line x1={sx(final_value)} x2={sx(final_value)} y1={top} y2={H - 30} className="wf-final" />
      <text x={sx(final_value)} y={H - 12} textAnchor="middle" className="wf-cap strong">
        this applicant · model score {(sigmoid(final_value) * 100).toFixed(0)}%
      </text>
    </svg>
  );
}
