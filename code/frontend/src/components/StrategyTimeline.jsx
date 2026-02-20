const compoundColors = {
  SOFT: "var(--soft)",
  MEDIUM: "var(--medium)",
  HARD: "var(--hard)",
};

export default function StrategyTimeline({ strategy, totalLaps }) {
  const total = totalLaps || strategy.stints.reduce((a, b) => a + b, 0);
  let currentLap = 0;

  return (
    <div className="timeline">
      <div className="timeline-bar">
        {strategy.stints.map((len, idx) => {
          const compound = strategy.compounds[idx] || "MEDIUM";
          const width = (len / total) * 100;
          const start = currentLap;
          currentLap += len;
          return (
            <div
              key={`${compound}-${idx}`}
              className="stint"
              style={{ width: `${width}%`, background: compoundColors[compound] || "#888" }}
              title={`${compound} · Laps ${start + 1}-${currentLap}`}
            >
              <span>{compound}</span>
            </div>
          );
        })}
      </div>
      <div className="pit-windows">
        {strategy.pit_windows.map((win, idx) => {
          const left = (win.lap_min / total) * 100;
          const right = (win.lap_max / total) * 100;
          return (
            <div
              key={`window-${idx}`}
              className="pit-window"
              style={{ left: `${left}%`, width: `${right - left}%` }}
            >
              <span>{win.lap_min}-{win.lap_max}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
