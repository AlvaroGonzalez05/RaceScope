import { useLang } from "../i18n/LangContext.jsx";

export default function PreRaceContextBubble({
  season,
  seasons,
  onSeasonChange,
  circuitId,
  circuits,
  onCircuitChange,
  onRun,
  canRun,
  running,
  metadataStatus,
  metadataError,
  onRetry,
  dataMode,
  staleData,
}) {
  const { t } = useLang();
  return (
    <div className="pre-race-context-wrap">
      <aside className="pre-race-context-bubble" aria-label="Contexto Pre-race">
        <div className="pre-race-context-fields">
          <select
            value={season || ""}
            onChange={(e) => onSeasonChange(e.target.value)}
            disabled={metadataStatus !== "ready" && metadataStatus !== "loading"}
            aria-label={t("ctx.season")}
          >
            <option value="">{t("ctx.season")}</option>
            {seasons.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>

          <select
            value={circuitId || ""}
            onChange={(e) => onCircuitChange(e.target.value)}
            disabled={metadataStatus !== "ready"}
            aria-label={t("ctx.circuit")}
          >
            <option value="">{t("ctx.circuit")}</option>
            {circuits.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>

          <button className="cta" onClick={onRun} disabled={!canRun || running}>
            {running ? t("ctx.running") : t("ctx.run")}
          </button>
        </div>
        {dataMode && (
          <span className={`data-mode-chip ${staleData ? "snapshot" : "live"}`}>
            {staleData ? "SNAPSHOT" : "LIVE"}
          </span>
        )}
      </aside>

      {metadataStatus === "loading" && <p className="status-note">{t("ctx.loading")}</p>}
      {metadataStatus === "error" && (
        <div className="status-error compact">
          <p>{metadataError}</p>
          <button className="ghost-btn" onClick={onRetry}>{t("ctx.retry")}</button>
        </div>
      )}
      {metadataStatus === "empty" && <p className="status-note">{metadataError}</p>}
    </div>
  );
}
