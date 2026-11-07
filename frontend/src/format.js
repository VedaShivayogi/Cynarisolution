export const inr = (v) => "₹" + Number(v).toLocaleString("en-IN", { maximumFractionDigits: 0 });
export const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
export const BAND = {
  low: { label: "Low risk", note: "Within appetite for standard approval.", color: "var(--low)" },
  medium: { label: "Medium risk", note: "Refer to an underwriter for review.", color: "var(--med)" },
  high: { label: "High risk", note: "Outside appetite; likely decline or restructure.", color: "var(--high)" },
};
