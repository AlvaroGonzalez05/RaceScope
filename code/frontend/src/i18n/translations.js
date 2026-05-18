const T = {
  es: {
    /* ── Tabs ────────────────────────────────────── */
    "tabs.home":     "Inicio",
    "tabs.pre-race": "Pre-carrera",
    "tabs.live":     "En vivo",
    "tabs.rewatch":  "Repetición",

    /* ── Header / header-bar ─────────────────────── */
    "theme.dark":  "Oscuro",
    "theme.light": "Claro",
    "lang.toggle": "Idioma",

    /* ── Pre-race context bubble ─────────────────── */
    "ctx.season":      "Temporada",
    "ctx.circuit":     "Circuito",
    "ctx.run":         "Calcular",
    "ctx.running":     "Calculando",
    "ctx.loading":     "Cargando datos…",
    "ctx.retry":       "Reintentar",

    /* ── Driver row ──────────────────────────────── */
    "row.slot":          "Piloto {n}",
    "row.team":          "Equipo",
    "row.driver":        "Piloto",
    "row.select":        "Seleccionar",
    "row.status.loading":"Calculando…",
    "row.status.error":  "Error",
    "row.idle":          "Selecciona piloto y pulsa Calcular.",
    "row.error":         "Error al calcular estrategias.",
    "row.noStrategy":    "Sin estrategia disponible.",
    "row.retry":         "Reintentar",

    /* ── Strategy strip ──────────────────────────── */
    "strip.empty": "Sin estrategias.",

    /* ── Curve chart ─────────────────────────────── */
    "chart.laptime":       "Tiempo de vuelta (s)",
    "chart.lap":           "Vuelta",
    "chart.lap_abbrev":    "V",
    "chart.pit":           "Pit",
    "chart.outsideWindow": "Fuera de ventana",
    "chart.pitWindow":     "Ventana de parada",
    "chart.noData":        "Sin datos de degradación.",
    "chart.tyre":          "Neumático",

    /* ── Placeholder tabs ────────────────────────── */
    "placeholder.pending": "Pendiente de implementación.",

    /* ── Home landing ────────────────────────────── */
    "home.eyebrow":    "Motor de Simulación Estratégica",
    "home.headline1":  "Estrategia de carrera",
    "home.headline2":  "Inteligente",
    "home.sub":        "Simulación táctica pre-carrera con modelo de degradación por piloto, Monte Carlo y comparativa dual en tiempo real.",
    "home.cta":        "Comenzar análisis",

    "home.feat.label": "Funcionalidades",
    "home.feat.title": "Diseñado para la táctica real",

    "home.feat.mc.ey":   "Simulación",
    "home.feat.mc.t":    "Monte Carlo",
    "home.feat.mc.b":    "200 trayectorias por estrategia. Safety Car, tráfico y variabilidad de ritmo modelados probabilísticamente.",

    "home.feat.tyre.ey": "Neumáticos",
    "home.feat.tyre.t":  "Degradación real",
    "home.feat.tyre.b":  "Modelo Transformer por piloto: aprende de secuencias reales de stint para predecir el desgaste futuro.",

    "home.feat.cmp.ey":  "Análisis dual",
    "home.feat.cmp.t":   "Head-to-head",
    "home.feat.cmp.b":   "Compara dos pilotos en paralelo. Mismo circuito, distintos perfiles de ritmo y ventanas óptimas.",

    "home.feat.pit.ey":  "Timing",
    "home.feat.pit.t":   "Pit windows",
    "home.feat.pit.b":   "Ranking de 5 estrategias diversas con ventanas de parada óptimas calculadas por simulación.",

    "home.pipe.label": "Arquitectura",
    "home.pipe.title": "Pipeline de datos end-to-end",
    "home.pipe.0.desc": "Sesiones FP1–FP3, Race y Sprint en tiempo real.",
    "home.pipe.1.desc": "Stint age, compuesto, brecha al coche delantero y temperatura.",
    "home.pipe.2.desc": "Predicción de ritmo per-driver con degradación autoregresiva.",
    "home.pipe.3.desc": "200 simulaciones refinan el Top-5 con SC y tráfico.",

    "home.stat.0": "simulaciones / estrategia",
    "home.stat.1": "estrategias diversas",
    "home.stat.2": "modelos por piloto",
    "home.stat.3": "temporadas de datos",

    "home.cta2.headline": "Listo para analizar la próxima carrera",
    "home.cta2.sub":      "Selecciona circuito, piloto y lanza la simulación.",
    "home.cta2.btn":      "Abrir Pre-carrera",

    /* ── Vis labels ──────────────────────────────── */
    "vis.sims":  "200 simulaciones",
    "vis.degrad":"Curvas de degradación",
    "vis.pace":  "Ritmo relativo",
    "vis.strats":"Estrategias Top-5",
    "vis.best":  "MEJOR",
    "vis.lap1":  "Vuelta 1",
    "vis.end":   "Fin",
    "vis.pit":   "PIT",
  },

  en: {
    /* ── Tabs ────────────────────────────────────── */
    "tabs.home":     "Home",
    "tabs.pre-race": "Pre-race",
    "tabs.live":     "Live",
    "tabs.rewatch":  "Rewatch",

    /* ── Header ──────────────────────────────────── */
    "theme.dark":  "Dark",
    "theme.light": "Light",
    "lang.toggle": "Language",

    /* ── Pre-race context bubble ─────────────────── */
    "ctx.season":  "Season",
    "ctx.circuit": "Circuit",
    "ctx.run":     "Calculate",
    "ctx.running": "Calculating",
    "ctx.loading": "Loading data…",
    "ctx.retry":   "Retry",

    /* ── Driver row ──────────────────────────────── */
    "row.slot":           "Driver {n}",
    "row.team":           "Team",
    "row.driver":         "Driver",
    "row.select":         "Select",
    "row.status.loading": "Calculating…",
    "row.status.error":   "Error",
    "row.idle":           "Select driver and press Calculate.",
    "row.error":          "Error calculating strategies.",
    "row.noStrategy":     "No strategy available.",
    "row.retry":          "Retry",

    /* ── Strategy strip ──────────────────────────── */
    "strip.empty": "No strategies.",

    /* ── Curve chart ─────────────────────────────── */
    "chart.laptime":       "Lap time (s)",
    "chart.lap":           "Lap",
    "chart.lap_abbrev":    "L",
    "chart.pit":           "Pit",
    "chart.outsideWindow": "Outside window",
    "chart.pitWindow":     "Pit window",
    "chart.noData":        "No degradation data.",
    "chart.tyre":          "Tyre",

    /* ── Placeholder tabs ────────────────────────── */
    "placeholder.pending": "Pending implementation.",

    /* ── Home landing ────────────────────────────── */
    "home.eyebrow":    "Strategy Simulation Engine",
    "home.headline1":  "Race Strategy",
    "home.headline2":  "Intelligence",
    "home.sub":        "Pre-race tactical simulation with per-driver degradation model, Monte Carlo refinement, and real-time dual comparison.",
    "home.cta":        "Start analysis",

    "home.feat.label": "Features",
    "home.feat.title": "Built for real tactics",

    "home.feat.mc.ey":   "Simulation",
    "home.feat.mc.t":    "Monte Carlo",
    "home.feat.mc.b":    "200 trajectories per strategy. Safety Car, traffic, and pace variability modelled probabilistically.",

    "home.feat.tyre.ey": "Tyres",
    "home.feat.tyre.t":  "Real degradation",
    "home.feat.tyre.b":  "Per-driver Transformer model: learns from real stint sequences to predict future tyre wear.",

    "home.feat.cmp.ey":  "Dual analysis",
    "home.feat.cmp.t":   "Head-to-head",
    "home.feat.cmp.b":   "Compare two drivers in parallel. Same circuit, different pace profiles and optimal windows.",

    "home.feat.pit.ey":  "Timing",
    "home.feat.pit.t":   "Pit windows",
    "home.feat.pit.b":   "Ranking of 5 diverse strategies with optimal pit windows calculated by simulation.",

    "home.pipe.label": "Architecture",
    "home.pipe.title": "End-to-end data pipeline",
    "home.pipe.0.desc": "FP1–FP3, Race and Sprint sessions in real time.",
    "home.pipe.1.desc": "Stint age, compound, gap to car ahead and track temperature.",
    "home.pipe.2.desc": "Per-driver pace prediction with autoregressive degradation.",
    "home.pipe.3.desc": "200 simulations refine Top-5 with SC and traffic.",

    "home.stat.0": "simulations / strategy",
    "home.stat.1": "diverse strategies",
    "home.stat.2": "per-driver models",
    "home.stat.3": "seasons of data",

    "home.cta2.headline": "Ready to analyse the next race",
    "home.cta2.sub":      "Select circuit, driver and launch the simulation.",
    "home.cta2.btn":      "Open Pre-race",

    /* ── Vis labels ──────────────────────────────── */
    "vis.sims":  "200 simulations",
    "vis.degrad":"Degradation curves",
    "vis.pace":  "Relative pace",
    "vis.strats":"Top-5 strategies",
    "vis.best":  "BEST",
    "vis.lap1":  "Lap 1",
    "vis.end":   "End",
    "vis.pit":   "PIT",
  },
};

export default T;
