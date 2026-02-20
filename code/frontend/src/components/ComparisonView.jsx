import StrategyCard from "./StrategyCard.jsx";

export default function ComparisonView({ result }) {
  const driver = result.driver || result;
  const teammate = result.teammate;

  return (
    <div className="comparison">
      <header className="comparison-head">
        <div>
          <p className="label">Estrategias</p>
          <h2>{result.circuit_id} · {result.year}</h2>
        </div>
        <div className="context">
          <span>Vueltas: {driver.context?.total_laps}</span>
          <span>Temp pista: {driver.context?.track_temp?.toFixed?.(1)}°C</span>
          <span>SC prob: {Math.round((driver.context?.sc_probability || 0) * 100)}%</span>
        </div>
      </header>

      <div className={`cards ${teammate ? "two" : "one"}`}>
        <StrategyCard title="Piloto" payload={driver} />
        {teammate && <StrategyCard title="Compañero" payload={teammate} />}
      </div>
    </div>
  );
}
