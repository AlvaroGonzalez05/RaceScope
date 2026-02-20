import { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { formatLapTime, formatRaceDuration } from "../utils/time.js";

const COMPOUND_COLORS = {
  SOFT: "#ff4b4b",
  MEDIUM: "#f2c94c",
  HARD: "#b8bec6",
};

const PIT_WINDOW_COLOR = "rgba(255, 107, 46, 0.18)";
const PADDING = { top: 14, right: 14, bottom: 22, left: 42 };
const FALLBACK_DOMAIN = { min: 89, max: 95 };
const MIN_VALID_SIZE = 50;

function useResizeObserver(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const rafRef = useRef(null);

  useEffect(() => {
    if (!ref.current) return undefined;
    const node = ref.current;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const nextWidth = Math.floor(entry.contentRect.width);
        const nextHeight = Math.floor(entry.contentRect.height);
        setSize((prev) => {
          if (prev.width === nextWidth && prev.height === nextHeight) return prev;
          return { width: nextWidth, height: nextHeight };
        });
      }
    });

    observer.observe(node);
    return () => {
      observer.disconnect();
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [ref]);

  useEffect(() => {
    if (size.width >= MIN_VALID_SIZE && size.height >= MIN_VALID_SIZE) return undefined;
    if (!ref.current) return undefined;

    let active = true;
    const retry = () => {
      if (!active || !ref.current) return;
      const rect = ref.current.getBoundingClientRect();
      const nextWidth = Math.floor(rect.width);
      const nextHeight = Math.floor(rect.height);
      setSize((prev) => {
        if (prev.width === nextWidth && prev.height === nextHeight) return prev;
        return { width: nextWidth, height: nextHeight };
      });
      if (nextWidth < MIN_VALID_SIZE || nextHeight < MIN_VALID_SIZE) {
        rafRef.current = requestAnimationFrame(retry);
      } else {
        rafRef.current = null;
      }
    };

    rafRef.current = requestAnimationFrame(retry);
    return () => {
      active = false;
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [size.width, size.height, ref]);

  return size;
}

function buildFallbackCurves(strategy, totalLaps) {
  let lap = 1;
  const baseline = Number(strategy.expected_time) > 0 && totalLaps > 0 ? Number(strategy.expected_time) / totalLaps : 92;

  return (strategy.stints || [])
    .map((len, idx) => {
      const stintLen = Number(len) || 0;
      const start = lap;
      const end = lap + stintLen - 1;
      lap = end + 1;

      const lapTimeData = Array.from({ length: stintLen }, (_, i) => baseline + i * 0.05);
      const tyreLifeData = Array.from({ length: stintLen }, (_, i) => {
        if (stintLen <= 1) return 100;
        return 100 - (i / (stintLen - 1)) * 100;
      });

      return {
        compound: strategy.compounds?.[idx] || "MEDIUM",
        start_lap: start,
        end_lap: end,
        lap_time_data: lapTimeData,
        tyre_life_data: tyreLifeData,
      };
    })
    .filter((stint) => stint.start_lap <= totalLaps);
}

function normalizeDomain(domain, fallback = FALLBACK_DOMAIN) {
  if (!domain || !Number.isFinite(domain.min) || !Number.isFinite(domain.max)) return fallback;
  if (domain.max <= domain.min) return fallback;
  return domain;
}

function computeDomain(values, fallback = FALLBACK_DOMAIN) {
  if (!values.length) return fallback;
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return fallback;
  const span = Math.max(0.5, max - min);
  const margin = span * 0.05;
  return { min: min - margin, max: max + margin };
}

function StrategyCurveChart({ strategy, totalLaps, selected, onSelect, yDomain }) {
  const wrapperRef = useRef(null);
  const clipId = useId().replace(/:/g, "_");
  const { width, height } = useResizeObserver(wrapperRef);
  const [hover, setHover] = useState(null);

  const layoutValid = width >= MIN_VALID_SIZE && height >= MIN_VALID_SIZE;

  const dataModel = useMemo(() => {
    const source = strategy.stint_curves?.length ? strategy.stint_curves : buildFallbackCurves(strategy, totalLaps);
    const stints = source.map((stint) => {
      const lapTimesRaw = stint.lap_time_data || stint.lapTimeData || [];
      const lapTimes = lapTimesRaw.map(Number).filter(Number.isFinite);
      const tyreLife = stint.tyre_life_data || stint.tyreLifeData || stint.degradation_data || stint.degradationData || [];
      const points = lapTimes.map((lapTime, idx) => ({
        lap: Number(stint.start_lap) + idx,
        compound: stint.compound,
        lapTime,
        tyreLife: Number(tyreLife[idx]),
      }));
      return { ...stint, points };
    });
    const flatPoints = stints
      .flatMap((stint) => stint.points)
      .filter((p) => Number.isFinite(p.lap) && Number.isFinite(p.lapTime) && p.lap >= 1 && p.lap <= totalLaps);
    const allLapTimes = flatPoints.map((p) => p.lapTime);
    const boundedDomain = yDomain ? normalizeDomain(yDomain, computeDomain(allLapTimes)) : computeDomain(allLapTimes);
    return {
      stints,
      points: flatPoints,
      domain: boundedDomain,
    };
  }, [strategy, totalLaps, yDomain]);

  const chart = useMemo(() => {
    if (!layoutValid) return null;
    if (!dataModel.points.length) return null;

    const innerWidth = Math.max(1, width - PADDING.left - PADDING.right);
    const innerHeight = Math.max(1, height - PADDING.top - PADDING.bottom);
    const xForLap = (lap) => {
      if (totalLaps <= 1) return PADDING.left;
      return PADDING.left + ((lap - 1) / (totalLaps - 1)) * innerWidth;
    };

    const yForTime = (lapTime) => {
      if (!Number.isFinite(lapTime)) return PADDING.top + innerHeight / 2;
      const ratio = (lapTime - dataModel.domain.min) / (dataModel.domain.max - dataModel.domain.min);
      return PADDING.top + innerHeight - ratio * innerHeight;
    };

    const stints = dataModel.stints.map((stint) => ({
      ...stint,
      points: stint.points.map((p) => ({
        ...p,
        x: xForLap(p.lap),
        y: yForTime(p.lapTime),
      })),
      path: "",
    })).map((stint) => ({
      ...stint,
      path: stint.points.map((p) => `${p.x},${p.y}`).join(" "),
    }));
    const points = stints.flatMap((stint) => stint.points).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));

    const pitWindows = (strategy.pit_windows || []).map((win, idx) => {
      const minLap = Math.max(1, Number(win.lap_min || 1));
      const maxLap = Math.max(minLap, Number(win.lap_max || minLap));
      const x1 = xForLap(minLap);
      const x2 = xForLap(maxLap);
      return {
        id: `${idx}-${minLap}-${maxLap}`,
        lapMin: minLap,
        lapMax: maxLap,
        x: Math.min(x1, x2),
        width: Math.max(2, Math.abs(x2 - x1)),
      };
    });

    const yTicks = [dataModel.domain.min, (dataModel.domain.min + dataModel.domain.max) / 2, dataModel.domain.max];

    return {
      stints,
      points,
      pitWindows,
      yTicks,
      innerHeight,
      xForLap,
      yForTime,
    };
  }, [strategy.pit_windows, totalLaps, width, height, layoutValid, dataModel]);

  const handleMove = useCallback((event) => {
    if (!chart?.points?.length || !layoutValid) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width) return;

    const relX = ((event.clientX - rect.left) / rect.width) * width;

    let nearest = chart.points[0];
    let minDist = Math.abs(nearest.x - relX);
    for (let i = 1; i < chart.points.length; i += 1) {
      const dist = Math.abs(chart.points[i].x - relX);
      if (dist < minDist) {
        minDist = dist;
        nearest = chart.points[i];
      }
    }

    const activeWindow = chart.pitWindows.find((win) => nearest.lap >= win.lapMin && nearest.lap <= win.lapMax) || null;
    setHover({ ...nearest, activeWindow });
  }, [chart, layoutValid, width]);

  if (!dataModel.points.length) {
    return <div className="strategy-curve-empty">Sin datos de degradación.</div>;
  }

  if (!layoutValid) {
    return (
      <article className={`strategy-curve-card ${selected ? "is-selected" : ""}`}>
        <div ref={wrapperRef} className="curve-wrapper">
          <div className="chart-skeleton" />
        </div>
      </article>
    );
  }

  return (
    <article
      className={`strategy-curve-card ${selected ? "is-selected" : ""}`}
      onClick={(e) => {
        e.stopPropagation();
        onSelect?.();
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          e.stopPropagation();
          onSelect?.();
        }
      }}
    >
      <header className="strategy-curve-head">
        <div>
          <p className="strategy-kind">{strategy.type}</p>
          <p className="strategy-time">{formatRaceDuration(strategy.expected_time)}</p>
        </div>
        <p className="strategy-risk">var {Number(strategy.variance || 0).toFixed(1)}</p>
      </header>

      <div className="compound-legend">
        {Object.entries(COMPOUND_COLORS).map(([compound, color]) => (
          <span key={compound} className="legend-item">
            <i style={{ background: color }} />
            {compound}
          </span>
        ))}
        <span className="legend-item">
          <i className="pit-window-swatch" style={{ background: PIT_WINDOW_COLOR }} />
          Ventana de parada
        </span>
      </div>

      <div ref={wrapperRef} className="curve-wrapper" onMouseMove={handleMove} onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="100%" className="curve-svg">
          <defs>
            <clipPath id={clipId}>
              <rect x={PADDING.left} y={PADDING.top} width={Math.max(1, width - PADDING.left - PADDING.right)} height={Math.max(1, height - PADDING.top - PADDING.bottom)} />
            </clipPath>
          </defs>

          <g className="curve-grid">
            {chart.yTicks.map((tick, idx) => (
              <line
                key={`y-${idx}`}
                x1={PADDING.left}
                x2={width - PADDING.right}
                y1={chart.yForTime(tick)}
                y2={chart.yForTime(tick)}
              />
            ))}
            {[1, Math.floor(totalLaps * 0.33), Math.floor(totalLaps * 0.66), totalLaps]
              .filter((lap, idx, arr) => lap > 0 && arr.indexOf(lap) === idx)
              .map((lap) => (
                <line
                  key={`x-${lap}`}
                  y1={PADDING.top}
                  y2={height - PADDING.bottom}
                  x1={chart.xForLap(lap)}
                  x2={chart.xForLap(lap)}
                />
              ))}
          </g>

          <line x1={PADDING.left} x2={PADDING.left} y1={PADDING.top} y2={height - PADDING.bottom} className="curve-axis" />
          <line x1={PADDING.left} x2={width - PADDING.right} y1={height - PADDING.bottom} y2={height - PADDING.bottom} className="curve-axis" />

          <g clipPath={`url(#${clipId})`}>
            {chart.pitWindows.map((win) => (
              <rect key={`pit-window-${win.id}`} x={win.x} y={PADDING.top} width={win.width} height={chart.innerHeight} className="pit-window-band" />
            ))}

            {chart.stints.map((stint, idx) => (
              <polyline
                key={`${strategy.strategy_id || strategy.type}-${stint.compound}-${idx}`}
                points={stint.path}
                fill="none"
                stroke={COMPOUND_COLORS[stint.compound] || "#aab2bc"}
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ))}
          </g>

          {hover && <circle cx={hover.x} cy={hover.y} r="4" className="hover-dot" />}

          <text x={PADDING.left + 4} y={PADDING.top - 2} className="axis-caption">Lap time (s)</text>
          <text x={width - PADDING.right - 4} y={height - 6} className="axis-caption" textAnchor="end">Lap</text>
        </svg>

        {hover && (
          <div
            className="curve-tooltip"
            style={{ left: `${((hover.x / width) * 100).toFixed(2)}%`, top: `${((hover.y / height) * 100).toFixed(2)}%` }}
          >
            <span>V{hover.lap}</span>
            <span>{hover.compound}</span>
            <span>{formatLapTime(hover.lapTime)}</span>
            {Number.isFinite(hover.tyreLife) && <span>Tyre {hover.tyreLife.toFixed(1)}%</span>}
            {hover.activeWindow
              ? <span>Ventana de parada: {hover.activeWindow.lapMin}-{hover.activeWindow.lapMax}</span>
              : <span>Fuera de ventana</span>}
          </div>
        )}
      </div>
    </article>
  );
}

export default memo(StrategyCurveChart);
