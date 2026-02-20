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
}) {
  return (
    <div className="pre-race-context-wrap">
      <aside className="pre-race-context-bubble" aria-label="Contexto Pre-race">
        <div className="pre-race-context-fields">
          <select
            value={season || ""}
            onChange={(e) => onSeasonChange(e.target.value)}
            disabled={metadataStatus !== "ready" && metadataStatus !== "loading"}
            aria-label="Temporada"
          >
            <option value="">Temporada</option>
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
            aria-label="Circuito"
          >
            <option value="">Circuito</option>
            {circuits.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>

          <button className="cta" onClick={onRun} disabled={!canRun || running}>
            {running ? "Calculando" : "Calcular"}
          </button>
        </div>
      </aside>

      {metadataStatus === "loading" && <p className="status-note">Cargando metadata...</p>}
      {metadataStatus === "error" && (
        <div className="status-error compact">
          <p>{metadataError}</p>
          <button className="ghost-btn" onClick={onRetry}>Reintentar</button>
        </div>
      )}
      {metadataStatus === "empty" && <p className="status-note">{metadataError}</p>}
    </div>
  );
}
