import { useCallback, useMemo } from "react";
import StrategyStrip from "./StrategyStrip.jsx";
import StrategyCurveChart from "./StrategyCurveChart.jsx";
import { formatRaceDuration } from "../utils/time.js";

const TEAM_COLORS = {
  "Red Bull Racing": "#1f6cff",
  Ferrari: "#ff3b30",
  Mercedes: "#00d6c7",
  McLaren: "#ff8a00",
  Aston: "#2db56f",
  Alpine: "#ff4d94",
  Williams: "#4fa2ff",
  RB: "#3355ff",
  Sauber: "#6dcf38",
  Haas: "#bfc7d1",
};

function teamTint(teamName) {
  const key = Object.keys(TEAM_COLORS).find((k) => teamName?.includes(k));
  return key ? TEAM_COLORS[key] : "#6f7a86";
}

export default function DriverRow({
  row,
  teams,
  drivers,
  onRowChange,
  rowHeight,
}) {
  const rowDrivers = row.team ? drivers.filter((d) => d.team_name === row.team) : drivers;
  const strategies = row.data?.strategies || [];
  const orderedStrategies = [...strategies].sort((a, b) => a.expected_time - b.expected_time);
  const selectedStrategyId = row.selectedStrategyId || orderedStrategies[0]?.strategy_id || null;
  const selectedIndex = Math.max(0, orderedStrategies.findIndex((s) => s.strategy_id === selectedStrategyId));
  const selected = orderedStrategies[selectedIndex] || orderedStrategies[0];
  const tint = teamTint(row.team);
  const yDomain = useMemo(() => {
    const values = orderedStrategies.flatMap((strategy) =>
      (strategy.stint_curves || []).flatMap((stint) =>
        (stint.lap_time_data || []).map(Number).filter(Number.isFinite),
      ),
    );
    if (!values.length) return null;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(0.5, max - min);
    const margin = span * 0.05;
    return { min: min - margin, max: max + margin };
  }, [orderedStrategies]);
  const onSelectStrategy = useCallback(
    (strategyId) => onRowChange(row.id, { selectedStrategyId: strategyId }),
    [onRowChange, row.id],
  );
  const onTeamChange = useCallback(
    (value) => onRowChange(row.id, { team: value, driverId: "", data: null, status: "idle", selectedStrategyId: null }),
    [onRowChange, row.id],
  );
  const onDriverChange = useCallback(
    (value) => onRowChange(row.id, { driverId: Number(value), data: null, status: "idle", selectedStrategyId: null }),
    [onRowChange, row.id],
  );

  return (
    <section className="driver-row" style={{ "--team-tint": tint, "--row-height": rowHeight }}>
      <aside className="driver-rail">
        <div className="row-head">
          <h3>Piloto</h3>
        </div>

        <label>
          Equipo
          <select value={row.team || ""} onChange={(e) => onTeamChange(e.target.value)}>
            <option value="">Seleccionar</option>
            {teams.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>

        <label>
          Piloto
          <select
            value={row.driverId || ""}
            onChange={(e) => onDriverChange(e.target.value)}
            disabled={!rowDrivers.length}
          >
            <option value="">Seleccionar</option>
            {rowDrivers.map((d) => (
              <option key={d.driver_id} value={d.driver_id}>
                {d.driver_code || d.driver_id}
              </option>
            ))}
          </select>
        </label>

        <div className="row-meta">
          <span>{row.status === "loading" ? "Calculando" : row.status === "error" ? "Error" : "Listo"}</span>
          {selected ? <span>{formatRaceDuration(selected.expected_time)}</span> : <span>Sin datos</span>}
        </div>
      </aside>

      <div className="row-main">
        <StrategyStrip
          strategies={orderedStrategies}
          selectedStrategyId={selectedStrategyId}
          onSelect={onSelectStrategy}
        />

        {orderedStrategies.length ? (
          <div className="strategy-stack">
            {orderedStrategies.map((strategy, idx) => {
              const strategyKey = strategy.strategy_id || `${strategy.type}-${strategy.expected_time}-${idx}`;
              return (
                <StrategyCurveChart
                  key={strategyKey}
                  strategy={strategy}
                  totalLaps={row.data?.context?.total_laps || 60}
                  selected={strategyKey === selectedStrategyId}
                  onSelect={() => onSelectStrategy(strategyKey)}
                  yDomain={yDomain}
                />
              );
            })}
          </div>
        ) : (
          <div className="row-empty">
            {row.status === "error"
              ? "Error al calcular estrategias para este piloto."
              : row.status === "idle"
                ? "Selecciona piloto y pulsa Calcular."
                : "Sin estrategia para este piloto."}
          </div>
        )}
      </div>
    </section>
  );
}
