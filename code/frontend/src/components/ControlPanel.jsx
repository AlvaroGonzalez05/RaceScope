export default function ControlPanel({
  seasons,
  season,
  onSeasonChange,
  circuits,
  circuitId,
  onCircuitChange,
  teams,
  selectedTeam,
  onTeamChange,
  drivers,
  driverId,
  onDriverChange,
  teammateId,
  onTeammateChange,
  onGenerate,
  loading,
}) {
  return (
    <aside className="panel">
      <h2>Control</h2>

      <label>
        Temporada
        <select value={season} onChange={(e) => onSeasonChange(Number(e.target.value))}>
          {seasons.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>

      <label>
        Circuito
        <select value={circuitId} onChange={(e) => onCircuitChange(e.target.value)}>
          {circuits.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>

      <label>
        Equipo
        <select value={selectedTeam} onChange={(e) => onTeamChange(e.target.value)}>
          {teams.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </label>

      <label>
        Piloto
        <select value={driverId} onChange={(e) => onDriverChange(e.target.value)}>
          {drivers.map((d) => (
            <option key={d.driver_id} value={d.driver_id}>
              {d.driver_code || d.driver_id}
            </option>
          ))}
        </select>
      </label>

      <label>
        Compañero
        <select value={teammateId} onChange={(e) => onTeammateChange(e.target.value)}>
          {drivers.map((d) => (
            <option key={d.driver_id} value={d.driver_id}>
              {d.driver_code || d.driver_id}
            </option>
          ))}
        </select>
      </label>

      <button className="cta" onClick={onGenerate} disabled={loading}>
        {loading ? "Calculando" : "Generar"}
      </button>
    </aside>
  );
}
