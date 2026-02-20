import { memo } from "react";
import { formatRaceDuration } from "../utils/time.js";

function StrategyStrip({ strategies, selectedStrategyId, onSelect }) {
  const ordered = [...(strategies || [])].sort((a, b) => a.expected_time - b.expected_time);

  if (!ordered.length) {
    return <div className="strip-empty">Sin estrategias.</div>;
  }

  return (
    <div className="strategy-strip" role="listbox" aria-label="Estrategias">
      {ordered.map((s, idx) => {
        const strategyId = s.strategy_id || `${s.type}-${s.expected_time}-${idx}`;
        return (
        <button
          key={strategyId}
          className={selectedStrategyId === strategyId ? "active" : ""}
          onClick={(e) => {
            e.stopPropagation();
            onSelect(strategyId);
          }}
        >
          <span className="kind">{s.type}</span>
          <span className="summary">{s.stop_laps?.length || 0} pit</span>
          <span className="time">{formatRaceDuration(s.expected_time)}</span>
        </button>
        );
      })}
    </div>
  );
}

export default memo(StrategyStrip);
