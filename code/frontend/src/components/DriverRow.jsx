import { useCallback, useMemo } from "react";
import StrategyStrip from "./StrategyStrip.jsx";
import StrategyCurveChart from "./StrategyCurveChart.jsx";
import { formatRaceDuration } from "../utils/time.js";
import { teamTint } from "../constants/teams.js";
import { useLang } from "../i18n/LangContext.jsx";

export default function DriverRow({
  row,
  teams,
  drivers,
  onRowChange,
  onRetry,
}) {
  const { t } = useLang();
  const rowDrivers = row.team ? drivers.filter((d) => d.team_name === row.team) : drivers;
  const strategies = row.data?.strategies || [];
  const orderedStrategies = [...strategies].sort((a, b) => a.expected_time - b.expected_time);
  const bestTime = orderedStrategies[0]?.expected_time ?? 0;
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
    <section className="driver-row" style={{ "--team-tint": tint }}>
      <aside className="driver-rail">
        <div className="row-head">
          <span className="row-slot-label">{t("row.slot", { n: row.id })}</span>
          {row.driverId && (() => {
            const d = drivers.find(dr => String(dr.driver_id) === String(row.driverId));
            return d ? (
              <span className="row-driver-hero" style={{ color: tint }}>
                {d.driver_code || d.driver_id}
              </span>
            ) : null;
          })()}
        </div>

        <label>
          {t("row.team")}
          <select value={row.team || ""} onChange={(e) => onTeamChange(e.target.value)}>
            <option value="">{t("row.select")}</option>
            {teams.map((t_) => (
              <option key={t_} value={t_}>{t_}</option>
            ))}
          </select>
        </label>

        <label>
          {t("row.driver")}
          <select
            value={row.driverId || ""}
            onChange={(e) => onDriverChange(e.target.value)}
            disabled={!rowDrivers.length}
          >
            <option value="">{t("row.select")}</option>
            {rowDrivers.map((d) => (
              <option key={d.driver_id} value={d.driver_id}>
                {d.driver_code || d.driver_id}
              </option>
            ))}
          </select>
        </label>

        <div className="row-meta">
          <span className={`row-status-dot ${row.status}`} />
          <span className="row-status-text">
            {row.status === "loading" ? t("row.status.loading")
              : row.status === "error" ? t("row.status.error")
              : selected ? formatRaceDuration(selected.expected_time)
              : "—"}
          </span>
        </div>
      </aside>

      <p aria-live="polite" className="sr-only">
        {row.status === "loading"
          ? t("row.status.loading")
          : row.status === "ready"
          ? "Estrategias cargadas."
          : row.status === "error"
          ? t("row.error")
          : ""}
      </p>

      <div className="row-main">
        <StrategyStrip
          strategies={orderedStrategies}
          selectedStrategyId={selectedStrategyId}
          onSelect={onSelectStrategy}
        />

        {row.status === "loading" ? (
          <article className="strategy-curve-card featured strategy-curve-card-loading" aria-hidden="true">
            <header className="strategy-curve-head">
              <div className="loading-line w-28" />
              <div className="loading-line w-12" />
            </header>
            <div className="compound-legend loading-legend">
              <span className="loading-dot" />
              <span className="loading-dot" />
              <span className="loading-dot" />
            </div>
            <div className="curve-wrapper">
              <div className="chart-skeleton" />
            </div>
          </article>
        ) : selected ? (
          <StrategyCurveChart
            key={`featured-${row.id}`}
            strategy={selected}
            totalLaps={row.data?.context?.total_laps || 60}
            selected={true}
            onSelect={() => {}}
            yDomain={yDomain}
            bestTime={bestTime}
            featured={true}
          />
        ) : (
          <div className="row-empty">
            {row.status === "error" ? (
              <>
                <span>{t("row.error")}</span>
                {onRetry && (
                  <button className="ghost-btn" style={{ marginTop: "8px" }} onClick={() => onRetry(row.id)}>
                    {t("row.retry")}
                  </button>
                )}
              </>
            ) : row.status === "idle"
              ? t("row.idle")
              : t("row.noStrategy")}
          </div>
        )}
      </div>
    </section>
  );
}
