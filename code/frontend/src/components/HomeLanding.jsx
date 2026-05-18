import { memo, useEffect, useRef, useState, useMemo } from "react";
import { motion, useReducedMotion, useInView } from "framer-motion";
import { useLang } from "../i18n/LangContext.jsx";

/* ─── Static data keys ────────────────────────────────── */
const FEATURE_KEYS = ["mc", "tyre", "cmp", "pit"];
const PIPELINE_LABELS = ["OpenF1", "Feature Store", "Transformer", "Monte Carlo"];
const STAT_VALUES = ["200", "Top-5", "25+", "3"];

/* ─── Animated mini-visualizations ───────────────────── */

function VisMC({ running, label }) {
  const curves = useMemo(() => [
    "M 8 88 C 30 80, 55 72, 80 78 C 105 84, 128 92, 152 96",
    "M 8 88 C 30 76, 55 68, 80 74 C 105 80, 128 88, 152 84",
    "M 8 88 C 30 84, 55 78, 80 85 C 105 92, 128 100, 152 106",
    "M 8 88 C 30 72, 55 64, 80 70 C 105 76, 128 80, 152 76",
    "M 8 88 C 30 88, 55 82, 80 90 C 105 98, 128 108, 152 114",
  ], []);

  return (
    <svg viewBox="0 0 160 130" className="home-feat-svg" aria-hidden="true">
      <defs>
        <linearGradient id="mc-grad" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.8" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.25" />
        </linearGradient>
      </defs>
      {curves.map((d, i) => (
        <motion.path
          key={i}
          d={d}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="1.5"
          strokeLinecap="round"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={running ? { pathLength: 1, opacity: 0.28 + i * 0.08 } : { pathLength: 0, opacity: 0 }}
          transition={{ duration: 1.2, delay: i * 0.15, ease: "easeOut" }}
        />
      ))}
      {/* Mean line */}
      <motion.path
        d="M 8 88 C 30 78, 55 70, 80 76 C 105 82, 128 90, 152 92"
        fill="none"
        stroke="var(--accent)"
        strokeWidth="2.5"
        strokeLinecap="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={running ? { pathLength: 1, opacity: 1 } : { pathLength: 0, opacity: 0 }}
        transition={{ duration: 1.4, delay: 0.8, ease: "easeOut" }}
      />
      {/* pit markers */}
      {running && [68, 108].map((x) => (
        <motion.rect key={x} x={x} y={18} width={5} height={94} fill="var(--accent)" opacity={0.18}
          initial={{ scaleY: 0 }} animate={{ scaleY: 1 }} transition={{ delay: 1.6, duration: 0.4 }}
          style={{ transformOrigin: "center bottom" }}
        />
      ))}
      <text x="8" y="12" fontSize="7" fill="var(--muted)" opacity="0.7">{label}</text>
    </svg>
  );
}

function VisTyre({ running, label }) {
  return (
    <svg viewBox="0 0 160 130" className="home-feat-svg" aria-hidden="true">
      <defs>
        <linearGradient id="tyre-grad" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor="#ff4b4b" />
          <stop offset="50%" stopColor="#f2c94c" />
          <stop offset="100%" stopColor="#b8bec6" />
        </linearGradient>
      </defs>
      {/* Degradation background band */}
      <rect x="8" y="18" width="144" height="94" rx="4" fill="var(--panel-raised)" opacity="0.5" />
      {/* SOFT curve */}
      <motion.path
        d="M 8 24 C 40 30, 70 52, 90 72 C 110 90, 130 102, 152 110"
        fill="none" stroke="#ff4b4b" strokeWidth="2.5" strokeLinecap="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={running ? { pathLength: 1, opacity: 1 } : { pathLength: 0, opacity: 0 }}
        transition={{ duration: 1.3, ease: "easeOut" }}
      />
      {/* MEDIUM curve */}
      <motion.path
        d="M 8 40 C 40 44, 70 58, 90 72 C 110 86, 130 96, 152 102"
        fill="none" stroke="#f2c94c" strokeWidth="2" strokeLinecap="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={running ? { pathLength: 1, opacity: 0.85 } : { pathLength: 0, opacity: 0 }}
        transition={{ duration: 1.3, delay: 0.2, ease: "easeOut" }}
      />
      {/* HARD curve */}
      <motion.path
        d="M 8 52 C 40 54, 70 62, 90 72 C 110 82, 130 90, 152 94"
        fill="none" stroke="#b8bec6" strokeWidth="2" strokeLinecap="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={running ? { pathLength: 1, opacity: 0.7 } : { pathLength: 0, opacity: 0 }}
        transition={{ duration: 1.3, delay: 0.4, ease: "easeOut" }}
      />
      {/* Labels */}
      {running && (
        <>
          <motion.text x="120" y="16" fontSize="6.5" fill="#ff4b4b" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.4 }}>Soft</motion.text>
          <motion.text x="120" y="46" fontSize="6.5" fill="#f2c94c" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.6 }}>Medium</motion.text>
          <motion.text x="120" y="72" fontSize="6.5" fill="#b8bec6" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.8 }}>Hard</motion.text>
        </>
      )}
      <text x="8" y="12" fontSize="7" fill="var(--muted)" opacity="0.7">{label}</text>
    </svg>
  );
}

function VisCompare({ running, label, bestLabel }) {
  const bars = [
    { label: "VER", time: 1.0, color: "#3671c6" },
    { label: "HAM", time: 1.035, color: "#27f4d2" },
    { label: "LEC", time: 1.048, color: "#e8002d" },
    { label: "SAI", time: 1.061, color: "#e8002d" },
  ];
  return (
    <svg viewBox="0 0 160 130" className="home-feat-svg" aria-hidden="true">
      {bars.map((bar, i) => {
        const w = bar.time * 108;
        const y = 22 + i * 26;
        return (
          <g key={bar.label}>
            <text x="8" y={y + 10} fontSize="7" fill="var(--muted)">{bar.label}</text>
            <rect x="34" y={y} width="118" height="14" rx="3" fill="var(--panel-raised)" opacity="0.5" />
            <motion.rect
              x="34" y={y} height="14" rx="3"
              fill={bar.color} opacity="0.75"
              initial={{ width: 0 }}
              animate={running ? { width: w } : { width: 0 }}
              transition={{ duration: 0.8, delay: i * 0.12, ease: "easeOut" }}
            />
            {i === 0 && running && (
              <motion.text x={36 + w} y={y + 10} fontSize="6" fill="var(--accent)"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }}>
                {bestLabel}
              </motion.text>
            )}
          </g>
        );
      })}
      <text x="8" y="12" fontSize="7" fill="var(--muted)" opacity="0.7">{label}</text>
    </svg>
  );
}

function VisPit({ running, label }) {
  const strategies = [
    { label: "S-M", stops: [40], color: "#ff6b2e" },
    { label: "M-H", stops: [35], color: "#b8bec6" },
    { label: "S-M-H", stops: [20, 42], color: "#ff4b4b" },
    { label: "M-M", stops: [38], color: "#f2c94c" },
  ];
  return (
    <svg viewBox="0 0 160 130" className="home-feat-svg" aria-hidden="true">
      {strategies.map((s, i) => {
        const y = 20 + i * 25;
        return (
          <g key={s.label}>
            <text x="8" y={y + 9} fontSize="6.5" fill="var(--muted)">{s.label}</text>
            {/* Race line */}
            <motion.rect x="38" y={y + 3} height="8" rx="2"
              fill={s.color} opacity={i === 0 ? 0.75 : 0.4}
              initial={{ width: 0 }}
              animate={running ? { width: 110 } : { width: 0 }}
              transition={{ duration: 0.8, delay: i * 0.1 }}
            />
            {/* Pit windows */}
            {s.stops.map((stop, j) => {
              const x = 38 + (stop / 57) * 110;
              return (
                <motion.rect key={j} x={x - 3} y={y} width={6} height={14} rx="1"
                  fill="var(--bg)" stroke="var(--accent)" strokeWidth="1"
                  initial={{ opacity: 0 }} animate={running ? { opacity: 1 } : { opacity: 0 }}
                  transition={{ delay: 0.9 + i * 0.1 }}
                />
              );
            })}
          </g>
        );
      })}
      <text x="8" y="12" fontSize="7" fill="var(--muted)" opacity="0.7">{label}</text>
    </svg>
  );
}

/* ─── Animated counter ───────────────────────────────── */
function StatCounter({ value, inView }) {
  const [display, setDisplay] = useState("0");
  useEffect(() => {
    if (!inView) return;
    const num = parseInt(value, 10);
    if (isNaN(num)) { setDisplay(value); return; }
    let start = 0;
    const step = Math.ceil(num / 28);
    const timer = setInterval(() => {
      start += step;
      if (start >= num) { setDisplay(String(num) + (value.includes("+") ? "+" : "")); clearInterval(timer); }
      else setDisplay(String(start));
    }, 40);
    return () => clearInterval(timer);
  }, [inView, value]);
  return <>{display}</>;
}

/* ─── Feature card ───────────────────────────────────── */
function FeatureCard({ featureKey, index, reduceMotion }) {
  const { t } = useLang();
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, amount: 0.35 });
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (inView) {
      const timer = setTimeout(() => setRunning(true), 200 + index * 80);
      return () => clearTimeout(timer);
    }
  }, [inView, index]);

  const visLabel = {
    mc: t("vis.sims"),
    tyre: t("vis.degrad"),
    cmp: t("vis.pace"),
    pit: t("vis.strats"),
  }[featureKey];

  const Visual = featureKey === "mc" ? VisMC
    : featureKey === "tyre" ? VisTyre
    : featureKey === "cmp" ? VisCompare
    : VisPit;

  return (
    <motion.article
      ref={ref}
      className="home-feat-card"
      initial={reduceMotion ? false : { opacity: 0, y: 28 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.08 }}
      whileHover={reduceMotion ? {} : { y: -3, transition: { duration: 0.2 } }}
    >
      <div className="home-feat-vis">
        <Visual running={running} label={visLabel} bestLabel={t("vis.best")} />
      </div>
      <div className="home-feat-body">
        <span className="home-feat-eyebrow">{t(`home.feat.${featureKey}.ey`)}</span>
        <h3 className="home-feat-title">{t(`home.feat.${featureKey}.t`)}</h3>
        <p className="home-feat-desc">{t(`home.feat.${featureKey}.b`)}</p>
      </div>
    </motion.article>
  );
}

/* ─── Main component ─────────────────────────────────── */
function HomeLanding({ onEnterPreRace }) {
  const { t } = useLang();
  const reduceMotion = useReducedMotion();
  const shellRef = useRef(null);
  const statsRef = useRef(null);
  const statsInView = useInView(statsRef, { once: true, amount: 0.5 });

  const onShellMove = (event) => {
    if (reduceMotion || !shellRef.current) return;
    const rect = shellRef.current.getBoundingClientRect();
    const mx = ((event.clientX - rect.left) / rect.width) * 100;
    const my = ((event.clientY - rect.top) / rect.height) * 100;
    shellRef.current.style.setProperty("--mx", `${mx.toFixed(2)}%`);
    shellRef.current.style.setProperty("--my", `${my.toFixed(2)}%`);
  };

  /* Hero stagger */
  const container = {
    hidden: {},
    show: { transition: { staggerChildren: 0.1 } },
  };
  const item = {
    hidden: reduceMotion ? {} : { opacity: 0, y: 24 },
    show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
  };

  return (
    <section
      ref={shellRef}
      className="home-shell home-v2"
      aria-label="RaceScope home"
      onPointerMove={onShellMove}
    >
      <div className="home-noise" aria-hidden="true" />

      {/* ── Ambient decorations ── */}
      {!reduceMotion && (
        <>
          <motion.div
            className="home-spotlight"
            animate={{ opacity: [0.18, 0.32, 0.18] }}
            transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div className="home-orbit home-orbit-1" animate={{ rotate: 360 }} transition={{ duration: 38, repeat: Infinity, ease: "linear" }} />
          <motion.div className="home-orbit home-orbit-2" animate={{ rotate: -360 }} transition={{ duration: 62, repeat: Infinity, ease: "linear" }} />
        </>
      )}

      {/* ═══════════════════════════════════
           HERO
      ══════════════════════════════════════ */}
      <motion.header
        className="home-v2-hero"
        variants={container}
        initial="hidden"
        animate="show"
      >
        <motion.div variants={item}>
          <span className="home-v2-eyebrow">
            <span className="home-v2-dot" aria-hidden="true" />
            {t("home.eyebrow")}
          </span>
        </motion.div>

        <motion.h2 className="home-v2-headline" variants={item}>
          {t("home.headline1")}<br />
          <span className="home-v2-accent">{t("home.headline2")}</span>
        </motion.h2>

        <motion.p className="home-v2-sub" variants={item}>
          {t("home.sub")}
        </motion.p>

        <motion.div className="home-v2-actions" variants={item}>
          <button className="cta home-v2-cta" onClick={onEnterPreRace}>
            {t("home.cta")}
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </motion.div>
      </motion.header>

      {/* ── Hero visualization ── */}
      <motion.div
        className="home-v2-herovis"
        initial={reduceMotion ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 0.6 }}
        aria-hidden="true"
      >
        <div className="holo-grid" />
        <svg viewBox="0 0 1100 200" className="home-v2-curves" preserveAspectRatio="none">
          {/* SOFT stint */}
          <motion.path d="M 0 100 C 120 88, 220 80, 340 92 C 400 98, 430 108, 460 118"
            fill="none" stroke="#ff4b4b" strokeWidth="2.5" strokeLinecap="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.85 }}
            transition={{ duration: 2, delay: 0.9, ease: "easeOut" }}
          />
          {/* MEDIUM stint */}
          <motion.path d="M 460 78 C 550 72, 660 76, 780 88 C 860 96, 920 104, 1000 110"
            fill="none" stroke="#f2c94c" strokeWidth="2.5" strokeLinecap="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.85 }}
            transition={{ duration: 2.2, delay: 1.4, ease: "easeOut" }}
          />
          {/* Pit window */}
          <motion.rect x="440" y="0" width="26" height="200" fill="var(--accent)" opacity="0"
            animate={{ opacity: [0, 0.12, 0.12] }}
            transition={{ duration: 0.4, delay: 1.3 }}
          />
          {/* Second driver */}
          <motion.path d="M 0 130 C 120 115, 240 108, 360 118 C 440 126, 480 138, 530 148"
            fill="none" stroke="#3671c6" strokeWidth="1.8" strokeLinecap="round" strokeDasharray="5 4"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.55 }}
            transition={{ duration: 2, delay: 1.1, ease: "easeOut" }}
          />
          <motion.path d="M 530 110 C 640 106, 760 112, 890 122 C 960 128, 1020 132, 1100 136"
            fill="none" stroke="#27f4d2" strokeWidth="1.8" strokeLinecap="round" strokeDasharray="5 4"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.55 }}
            transition={{ duration: 2.2, delay: 1.6, ease: "easeOut" }}
          />
          {/* Axis labels */}
          <text x="10" y="195" fontSize="11" fill="var(--muted)" opacity="0.5">{t("vis.lap1")}</text>
          <text x="990" y="195" fontSize="11" fill="var(--muted)" opacity="0.5" textAnchor="end">{t("vis.end")}</text>
          <text x="446" y="14" fontSize="9" fill="var(--accent)" opacity="0.8">{t("vis.pit")}</text>
        </svg>
        <div className="home-v2-herovis-legend">
          <span><i style={{ background: "#ff4b4b" }} />{t("row.slot", { n: 1 })} — Soft</span>
          <span><i style={{ background: "#f2c94c" }} />{t("row.slot", { n: 1 })} — Medium</span>
          <span className="dashed"><i style={{ background: "#3671c6" }} />{t("row.slot", { n: 2 })} — Soft</span>
          <span className="dashed"><i style={{ background: "#27f4d2" }} />{t("row.slot", { n: 2 })} — Medium</span>
        </div>
      </motion.div>

      {/* ═══════════════════════════════════
           FEATURE GRID
      ══════════════════════════════════════ */}
      <section className="home-v2-section">
        <motion.div
          className="home-v2-section-header"
          initial={reduceMotion ? false : { opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.5 }}
          transition={{ duration: 0.5 }}
        >
          <span className="home-v2-section-label">{t("home.feat.label")}</span>
          <h3 className="home-v2-section-title">{t("home.feat.title")}</h3>
        </motion.div>

        <div className="home-feat-grid">
          {FEATURE_KEYS.map((key, i) => (
            <FeatureCard key={key} featureKey={key} index={i} reduceMotion={reduceMotion} />
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════
           PIPELINE
      ══════════════════════════════════════ */}
      <section className="home-v2-section">
        <motion.div
          className="home-v2-section-header"
          initial={reduceMotion ? false : { opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.5 }}
          transition={{ duration: 0.5 }}
        >
          <span className="home-v2-section-label">{t("home.pipe.label")}</span>
          <h3 className="home-v2-section-title">{t("home.pipe.title")}</h3>
        </motion.div>

        <div className="home-pipeline">
          {PIPELINE_LABELS.map((label, idx) => (
            <motion.div
              key={label}
              className="home-pipeline-step"
              initial={reduceMotion ? false : { opacity: 0, x: -12 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.5 }}
              transition={{ duration: 0.45, delay: idx * 0.09 }}
            >
              <div className="home-pipeline-index">{String(idx + 1).padStart(2, "0")}</div>
              <div className="home-pipeline-content">
                <span className="home-pipeline-label">{label}</span>
                <p className="home-pipeline-desc">{t(`home.pipe.${idx}.desc`)}</p>
              </div>
              {idx < PIPELINE_LABELS.length - 1 && (
                <motion.div
                  className="home-pipeline-arrow"
                  initial={reduceMotion ? false : { opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: idx * 0.09 + 0.3 }}
                  aria-hidden="true"
                />
              )}
            </motion.div>
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════
           STATS
      ══════════════════════════════════════ */}
      <div ref={statsRef} className="home-v2-stats">
        {STAT_VALUES.map((value, i) => (
          <motion.div
            key={i}
            className="home-v2-stat"
            initial={reduceMotion ? false : { opacity: 0, y: 12 }}
            animate={statsInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.45, delay: i * 0.08 }}
          >
            <span className="home-v2-stat-value">
              <StatCounter value={value} inView={statsInView} />
            </span>
            <span className="home-v2-stat-label">{t(`home.stat.${i}`)}</span>
          </motion.div>
        ))}
      </div>

      {/* ═══════════════════════════════════
           CTA FOOTER
      ══════════════════════════════════════ */}
      <motion.div
        className="home-v2-cta-footer"
        initial={reduceMotion ? false : { opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.6 }}
        transition={{ duration: 0.6 }}
      >
        <h3 className="home-v2-cta-headline">{t("home.cta2.headline")}</h3>
        <p className="home-v2-cta-sub">{t("home.cta2.sub")}</p>
        <button className="cta home-v2-cta home-v2-cta-big" onClick={onEnterPreRace}>
          {t("home.cta2.btn")}
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </motion.div>
    </section>
  );
}

export default memo(HomeLanding);
