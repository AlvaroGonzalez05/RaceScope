import { useState } from "react";
import StrategyTimeline from "./StrategyTimeline.jsx";
import DegradationChart from "./DegradationChart.jsx";

export default function StrategyCard({ title, payload }) {
  const strategies = payload?.strategies || [];
  const [selected, setSelected] = useState(0);
  const chosen = strategies[selected] || strategies[0];

  return (
    <section className="strategy-card">
      <div className="strategy-card-head">
        <h3>{title}</h3>
        <div className="metrics">
          <span>Tiempo esperado: {chosen ? chosen.expected_time.toFixed(1) : "-"}s</span>
          <span>Varianza: {chosen ? chosen.variance.toFixed(2) : "-"}</span>
        </div>
      </div>

      <div className="strategy-list">
        {strategies.map((strategy, idx) => (
          <button
            key={`${strategy.type}-${idx}`}
            className={idx === selected ? "active" : ""}
            onClick={() => setSelected(idx)}
          >
            <span className="tag">{strategy.type}</span>
            <span>{strategy.compounds.join(" → ")}</span>
            <span>Stops: {strategy.stop_laps.join(", ")}</span>
          </button>
        ))}
      </div>

      {chosen && (
        <div className="strategy-detail">
          <StrategyTimeline strategy={chosen} totalLaps={payload.context?.total_laps || 60} />
          <DegradationChart degradation={payload.degradation || {}} />
        </div>
      )}
    </section>
  );
}
