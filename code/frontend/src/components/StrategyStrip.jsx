import { memo } from "react";
import { formatRaceDuration, formatDelta } from "../utils/time.js";
import { COMPOUND_COLORS } from "../constants/compounds.js";
import { useLang } from "../i18n/LangContext.jsx";

function CompoundDots({ compounds }) {
  if (!compounds?.length) return null;
  return (
    <span className="strip-dots" aria-hidden="true">
      {compounds.map((c, i) => (
        <span key={i} className="strip-dot" style={{ background: COMPOUND_COLORS[c] || "#aab2bc" }} />
      ))}
    </span>
  );
}

function StrategyStrip({ strategies, selectedStrategyId, onSelect }) {
  const { t } = useLang();
  const ordered = [...(strategies || [])].sort((a, b) => a.expected_time - b.expected_time);
  const bestTime = ordered[0]?.expected_time ?? 0;

  if (!ordered.length) {
    return <div className="strip-empty">{t("strip.empty")}</div>;
  }

  return (
    <div className="strategy-strip-wrapper">
      <div className="strategy-strip" role="list" aria-label="Estrategias">
        {ordered.map((s, idx) => {
          const strategyId = s.strategy_id || `${s.type}-${s.expected_time}-${idx}`;
          const isActive = selectedStrategyId === strategyId;
          return (
            <div key={strategyId} role="listitem">
              <button
                className={isActive ? "active" : ""}
                aria-pressed={isActive}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect(strategyId);
                }}
              >
                <span className="strip-top-row">
                  <span className="kind">{s.type}</span>
                  <CompoundDots compounds={s.compounds} />
                </span>
                <span className="time">
                  {idx === 0
                    ? formatRaceDuration(s.expected_time)
                    : formatDelta(s.expected_time - bestTime)}
                </span>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default memo(StrategyStrip);
