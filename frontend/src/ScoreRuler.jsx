import { BAND } from "./format.js";

// 300–900 score ruler. Band edges come from PD cut-offs: score = 300 + 600*(1-PD).
export default function ScoreRuler({ score, band, lowT = 0.25, highT = 0.5 }) {
  const x = (s) => ((s - 300) / 600) * 100;
  const highEdge = 300 + 600 * (1 - highT), lowEdge = 300 + 600 * (1 - lowT);
  return (
    <figure className="ruler" aria-label={`Score ${score} out of 900, ${BAND[band].label}`}>
      <div className="ruler-bar">
        <span style={{ width: x(highEdge) + "%", background: "var(--high)" }} />
        <span style={{ width: x(lowEdge) - x(highEdge) + "%", background: "var(--med)" }} />
        <span style={{ width: 100 - x(lowEdge) + "%", background: "var(--low)" }} />
        <i className="ruler-mark" style={{ left: x(score) + "%", borderColor: BAND[band].color }}>
          <b style={{ background: BAND[band].color }}>{score}</b>
        </i>
      </div>
      <div className="ruler-scale">
        <span>300</span>
        <span style={{ left: x(highEdge) + "%" }}>{Math.round(highEdge)}</span>
        <span style={{ left: x(lowEdge) + "%" }}>{Math.round(lowEdge)}</span>
        <span style={{ left: "100%" }}>900</span>
      </div>
    </figure>
  );
}
