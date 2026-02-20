import { memo, useMemo, useRef } from "react";
import { motion, useReducedMotion } from "framer-motion";

const CAPABILITIES = [
  "Monte-Carlo táctico",
  "Perfil de conducción",
  "Pit windows robustas",
  "Comparativa instantánea",
  "Curvas por stint",
  "Contexto térmico",
];

function HomeLanding({ onEnterPreRace }) {
  const reduceMotion = useReducedMotion();
  const shellRef = useRef(null);

  const heroMotion = useMemo(
    () => ({
      hidden: { opacity: 0, y: reduceMotion ? 0 : 26 },
      show: { opacity: 1, y: 0, transition: { duration: 0.58 } },
    }),
    [reduceMotion],
  );

  const onShellMove = (event) => {
    if (reduceMotion || !shellRef.current) return;
    const rect = shellRef.current.getBoundingClientRect();
    const mx = ((event.clientX - rect.left) / rect.width) * 100;
    const my = ((event.clientY - rect.top) / rect.height) * 100;
    shellRef.current.style.setProperty("--mx", `${mx.toFixed(2)}%`);
    shellRef.current.style.setProperty("--my", `${my.toFixed(2)}%`);
  };

  return (
    <section
      ref={shellRef}
      className="home-shell home-innovation"
      aria-label="RaceScope home"
      onPointerMove={onShellMove}
    >
      <div className="home-noise" aria-hidden="true" />
      {!reduceMotion && (
        <>
          <motion.div
            className="home-spotlight"
            animate={{ opacity: [0.22, 0.36, 0.22] }}
            transition={{ duration: 4.4, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div
            className="home-orbit"
            animate={{ rotate: 360 }}
            transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
          />
        </>
      )}

      <motion.header
        className="home-hero"
        variants={heroMotion}
        initial="hidden"
        animate="show"
      >
        <h2>Race Intelligence Interface</h2>
        <p>
          Una experiencia estratégica inmersiva: señales en vivo, simulación visual y decisiones de carrera que se sienten reales.
        </p>

        <div className="home-actions">
          <button className="cta" onClick={onEnterPreRace}>Entrar en Pre-race</button>
          <button className="ghost-btn" onClick={() => window.scrollTo({ top: document.body.scrollHeight, behavior: reduceMotion ? "auto" : "smooth" })}>
            Explorar escena
          </button>
        </div>

        <div className="home-kpi-row" role="list" aria-label="Indicadores RaceScope">
          <div className="home-kpi" role="listitem">
            <span>60 Hz</span>
            <small>Render UX</small>
          </div>
          <div className="home-kpi" role="listitem">
            <span>Top-5</span>
            <small>Estrategias</small>
          </div>
          <div className="home-kpi" role="listitem">
            <span>MC + LSTM</span>
            <small>Motor híbrido</small>
          </div>
        </div>
      </motion.header>

      <motion.section
        className="home-lab"
        initial={reduceMotion ? false : { opacity: 0, y: 24 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.6 }}
      >
        <article className="command-card">
          <h3>Command Feed</h3>
          <p>
            Sistema preparado para fans: explica ritmo, degradación y ventanas de parada con narrativa visual inmediata.
          </p>
          <ul>
            {CAPABILITIES.map((cap) => (
              <li key={cap}>{cap}</li>
            ))}
          </ul>
        </article>

        <article className="hologram-card" aria-label="Escena táctica">
          {!reduceMotion && (
            <motion.div
              className="sweep-beam"
              animate={{ x: ["-10%", "112%"] }}
              transition={{ duration: 5.5, repeat: Infinity, ease: "linear" }}
            />
          )}
          <div className="holo-grid" />
          <svg viewBox="0 0 1000 300" className="holo-lines" aria-hidden="true">
            <motion.path
              d="M40 78 C 200 70, 300 118, 430 142"
              className="holo-soft"
              initial={{ pathLength: 0.15, opacity: 0.72 }}
              animate={reduceMotion ? { pathLength: 1 } : { pathLength: [0.2, 1, 0.7], opacity: [0.55, 1, 0.68] }}
              transition={{ duration: 3.2, repeat: Infinity, ease: "easeInOut" }}
            />
            <motion.path
              d="M430 120 C 580 122, 700 168, 830 198"
              className="holo-hard"
              initial={{ pathLength: 0.2, opacity: 0.72 }}
              animate={reduceMotion ? { pathLength: 1 } : { pathLength: [0.25, 1, 0.76], opacity: [0.55, 1, 0.68] }}
              transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut", delay: 0.25 }}
            />
            <motion.path
              d="M830 170 C 900 172, 940 190, 980 216"
              className="holo-medium"
              initial={{ pathLength: 0.25, opacity: 0.72 }}
              animate={reduceMotion ? { pathLength: 1 } : { pathLength: [0.3, 1, 0.8], opacity: [0.55, 1, 0.68] }}
              transition={{ duration: 3.7, repeat: Infinity, ease: "easeInOut", delay: 0.45 }}
            />
          </svg>
          <div className="holo-pit pit-a" />
          <div className="holo-pit pit-b" />
          <div className="holo-label">Telemetry Mesh</div>
        </article>
      </motion.section>

      <section className="home-capability-tape" aria-label="Capacidades">
        <motion.div
          className="tape-track"
          animate={reduceMotion ? false : { x: ["0%", "-50%"] }}
          transition={{ duration: 16, repeat: Infinity, ease: "linear" }}
        >
          {[...CAPABILITIES, ...CAPABILITIES].map((item, idx) => (
            <span key={`${item}-${idx}`}>{item}</span>
          ))}
        </motion.div>
      </section>
    </section>
  );
}

export default memo(HomeLanding);
