import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const colors = {
  SOFT: { stroke: "#ff4b4b", fill: "rgba(255, 75, 75, 0.25)" },
  MEDIUM: { stroke: "#f2c94c", fill: "rgba(242, 201, 76, 0.25)" },
  HARD: { stroke: "#d7dbe0", fill: "rgba(215, 219, 224, 0.25)" },
};

export default function DegradationChart({ degradation }) {
  const compounds = Object.keys(degradation || {});
  if (!compounds.length) {
    return <div className="degradation-empty">Sin curvas de degradación disponibles.</div>;
  }

  return (
    <div className="degradation">
      <h4>Curvas de degradación</h4>
      <div className="degradation-grid">
        {compounds.map((compound) => {
          const curve = degradation[compound]?.curve || [];
          const data = curve.map((value, idx) => ({ lap: idx + 1, value }));
          const palette = colors[compound] || { stroke: "#9aa3ad", fill: "rgba(154,163,173,0.2)" };
          return (
            <div key={compound} className="degradation-card">
              <div className="degradation-meta">
                <span className="compound-pill" data-compound={compound}>{compound}</span>
                <span>Lap time</span>
              </div>
              <ResponsiveContainer width="100%" height={120}>
                <AreaChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                  <XAxis dataKey="lap" stroke="var(--muted)" tick={{ fontSize: 10 }} />
                  <YAxis stroke="var(--muted)" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)" }} />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke={palette.stroke}
                    fill={palette.fill}
                    strokeWidth={2}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
    </div>
  );
}
