import { useEffect, useMemo, useRef, useCallback, useReducer } from "react";
import axios from "axios";
import ThemeToggle from "./components/ThemeToggle.jsx";
import LangToggle from "./components/LangToggle.jsx";
import TopTabs from "./components/TopTabs.jsx";
import DriverRow from "./components/DriverRow.jsx";
import HomeLanding from "./components/HomeLanding.jsx";
import PreRaceContextBubble from "./components/PreRaceContextBubble.jsx";
import { LangProvider } from "./i18n/LangContext.jsx";
import {
  appReducer,
  initialState,
  SET_THEME,
  SET_ACTIVE_TAB,
  SET_METADATA_STATUS,
  SET_METADATA_ERROR,
  SET_SEASONS,
  SET_SEASON,
  SET_CIRCUITS,
  SET_CIRCUIT_ID,
  SET_DRIVERS,
  SET_TEAMS,
  SET_ROWS,
  SET_RUNNING,
  SET_MOBILE_LAYOUT,
} from "./state/appReducer.js";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const api = axios.create({ baseURL: API_BASE || "/" });

const getPreferredTheme = () => {
  const stored = localStorage.getItem("theme");
  if (stored) return stored;
  const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
};

const mkRow = (id, team = "", driverCode = "") => ({
  id,
  team,
  driverCode,
  data: null,
  status: "idle",
  selectedStrategyId: null,
});

const ensureTwoRows = (rows) => {
  const base = [rows?.[0] || mkRow(1), rows?.[1] || mkRow(2)];
  return base.map((row, idx) => ({
    ...row,
    id: idx + 1,
  }));
};

const hydrateRowsFromMetadata = (rows, teamsData = [], driversData = []) => {
  const normalized = ensureTwoRows(rows);
  return normalized.map((row, idx) => {
    const defaultTeam = teamsData[idx] || teamsData[0] || "";
    const team = row.team || defaultTeam;
    const teamDrivers = team ? driversData.filter((d) => d.team_name === team) : driversData;
    const fallbackDriver = teamDrivers[0] || driversData[idx] || driversData[0];
    return {
      ...row,
      id: idx + 1,
      team,
      driverCode: row.driverCode || fallbackDriver?.driver_code || "",
      data: null,
      status: "idle",
      selectedStrategyId: null,
    };
  });
};

export default function App() {
  const shellRef = useRef(null);
  const [state, dispatch] = useReducer(appReducer, {
    ...initialState,
    rows: [mkRow(1), mkRow(2)],
  });
  const {
    theme, activeTab, metadataStatus, metadataError,
    seasons, season, circuits, circuitId,
    drivers, teams, rows, running, isMobileLayout,
  } = state;

  useEffect(() => {
    const initial = getPreferredTheme();
    dispatch({ type: SET_THEME, payload: initial });
    document.documentElement.dataset.theme = initial;
  }, []);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 920px)");
    const update = () => dispatch({ type: SET_MOBILE_LAYOUT, payload: media.matches });
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    dispatch({ type: SET_THEME, payload: next });
    localStorage.setItem("theme", next);
    document.documentElement.dataset.theme = next;
  };

  const onGlobalPointerMove = (event) => {
    if (!shellRef.current) return;
    const rect = shellRef.current.getBoundingClientRect();
    const mx = ((event.clientX - rect.left) / rect.width) * 100;
    const my = ((event.clientY - rect.top) / rect.height) * 100;
    shellRef.current.style.setProperty("--global-mx", `${mx.toFixed(2)}%`);
    shellRef.current.style.setProperty("--global-my", `${my.toFixed(2)}%`);
    shellRef.current.style.setProperty("--global-spot-opacity", "1");
  };

  const onGlobalPointerLeave = () => {
    if (!shellRef.current) return;
    shellRef.current.style.setProperty("--global-spot-opacity", "0");
  };

  const loadInitialMetadata = async () => {
    try {
      dispatch({ type: SET_METADATA_STATUS, payload: "loading" });
      dispatch({ type: SET_METADATA_ERROR, payload: "" });
      const seasonsResp = await api.get("/api/metadata/seasons");
      const years = seasonsResp.data || [];
      dispatch({ type: SET_SEASONS, payload: years });
      if (!years.length) {
        dispatch({ type: SET_METADATA_STATUS, payload: "empty" });
        dispatch({ type: SET_METADATA_ERROR, payload: "No hay temporadas disponibles." });
        return;
      }
      dispatch({ type: SET_SEASON, payload: years[years.length - 1] });
      dispatch({ type: SET_METADATA_STATUS, payload: "ready" });
    } catch {
      dispatch({ type: SET_METADATA_STATUS, payload: "error" });
      dispatch({ type: SET_METADATA_ERROR, payload: "No se pudieron cargar las temporadas." });
    }
  };

  const loadSeasonMetadata = async (targetSeason) => {
    if (!targetSeason) return;
    try {
      dispatch({ type: SET_METADATA_STATUS, payload: "loading" });
      const [circuitsResp, driversResp, teamsResp] = await Promise.all([
        api.get("/api/metadata/circuits", { params: { season: targetSeason } }),
        api.get("/api/metadata/drivers", { params: { season: targetSeason } }),
        api.get("/api/metadata/teams", { params: { season: targetSeason } }),
      ]);

      const circuitsData = circuitsResp.data || [];
      const driversData = driversResp.data || [];
      const teamsData = teamsResp.data || [];

      dispatch({ type: SET_CIRCUITS, payload: circuitsData });
      dispatch({ type: SET_CIRCUIT_ID, payload: circuitsData[0] || "" });
      dispatch({ type: SET_DRIVERS, payload: driversData });
      dispatch({ type: SET_TEAMS, payload: teamsData });

      const newRows = ensureTwoRows(hydrateRowsFromMetadata(rows, teamsData, driversData));
      dispatch({ type: SET_ROWS, payload: newRows });

      if (!circuitsData.length || !driversData.length || !teamsData.length) {
        dispatch({ type: SET_METADATA_STATUS, payload: "empty" });
        dispatch({ type: SET_METADATA_ERROR, payload: "Faltan datos de metadata para esta temporada." });
      } else {
        dispatch({ type: SET_METADATA_STATUS, payload: "ready" });
        dispatch({ type: SET_METADATA_ERROR, payload: "" });
      }
    } catch {
      dispatch({ type: SET_METADATA_STATUS, payload: "error" });
      dispatch({ type: SET_METADATA_ERROR, payload: "No se pudo cargar circuitos, pilotos o equipos." });
    }
  };

  useEffect(() => {
    loadInitialMetadata();
  }, []);

  useEffect(() => {
    if (season) loadSeasonMetadata(season);
  }, [season]);

  const updateRow = (id, patch) => {
    const newRows = ensureTwoRows(rows.map((r) => (r.id === id ? { ...r, ...patch } : r)));
    dispatch({ type: SET_ROWS, payload: newRows });
  };

  const canRun = useMemo(() => {
    if (!season || !circuitId || metadataStatus !== "ready") return false;
    return rows.some((r) => r.driverCode);
  }, [season, circuitId, metadataStatus, rows]);

  const dataQuality = useMemo(() => {
    const metas = rows.map((row) => row.data?.compute_meta).filter(Boolean);
    if (!metas.length) return { mode: null, stale: false };
    const stale = metas.some((meta) => Boolean(meta.stale_data));
    if (stale) return { mode: "snapshot", stale: true };
    const mode = metas[0]?.data_mode || "live";
    return { mode, stale: false };
  }, [rows]);

  const runPreRace = async () => {
    if (!canRun) return;
    dispatch({ type: SET_RUNNING, payload: true });
    dispatch({ type: SET_ROWS, payload: rows.map((row) => ({ ...row, status: row.driverCode ? "loading" : "idle" })) });

    const work = rows.map(async (row) => {
      if (!row.driverCode) return { id: row.id, status: "idle", data: null };
      try {
        const res = await api.post("/api/strategy", {
          year: Number(season),
          circuit_id: circuitId,
          driver_code: row.driverCode,
          force_recompute: true,
        });
        return { id: row.id, status: "ready", data: res.data, selectedStrategyId: null };
      } catch {
        return { id: row.id, status: "error", data: null };
      }
    });

    const updates = await Promise.all(work);
    dispatch({
      type: SET_ROWS,
      payload: rows.map((row) => {
        const match = updates.find((u) => u.id === row.id);
        return match ? { ...row, ...match } : row;
      }),
    });
    dispatch({ type: SET_RUNNING, payload: false });
  };

  const retryRow = useCallback(async (rowId) => {
    const row = rows.find((r) => r.id === rowId);
    if (!row?.driverCode) return;
    dispatch({ type: SET_ROWS, payload: rows.map((r) => r.id === rowId ? { ...r, status: "loading" } : r) });
    try {
      const res = await api.post("/api/strategy", {
        year: Number(season),
        circuit_id: circuitId,
        driver_code: row.driverCode,
        force_recompute: true,
      });
      dispatch({ type: SET_ROWS, payload: rows.map((r) =>
        r.id === rowId ? { ...r, status: "ready", data: res.data, selectedStrategyId: null } : r
      )});
    } catch {
      dispatch({ type: SET_ROWS, payload: rows.map((r) =>
        r.id === rowId ? { ...r, status: "error" } : r
      )});
    }
  }, [rows, season, circuitId]);

  const placeholder = (
    <section className="placeholder-view">
      <h2>{activeTab}</h2>
      <p>Pending implementation.</p>
    </section>
  );

  return (
    <LangProvider>
    <div
      ref={shellRef}
      className="app-shell"
      onPointerMove={onGlobalPointerMove}
      onPointerLeave={onGlobalPointerLeave}
    >
      <header className="header-bar">
        <div className="brand-block">
          <h1>RaceScope</h1>
        </div>
        <div className="header-actions">
          <LangToggle />
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
      </header>

      <div className="nav-context-row">
        <TopTabs activeTab={activeTab} onChange={(value) => dispatch({ type: SET_ACTIVE_TAB, payload: value })} />
        {activeTab === "pre-race" && (
          <PreRaceContextBubble
            season={season}
            seasons={seasons}
            onSeasonChange={(value) => dispatch({ type: SET_SEASON, payload: Number(value) })}
            circuitId={circuitId}
            circuits={circuits}
            onCircuitChange={(value) => dispatch({ type: SET_CIRCUIT_ID, payload: value })}
            onRun={runPreRace}
            canRun={canRun}
            running={running}
            metadataStatus={metadataStatus}
            metadataError={metadataError}
            onRetry={loadInitialMetadata}
            dataMode={dataQuality.mode}
            staleData={dataQuality.stale}
          />
        )}
      </div>

      {activeTab === "home" ? (
        <HomeLanding onEnterPreRace={() => dispatch({ type: SET_ACTIVE_TAB, payload: "pre-race" })} />
      ) : activeTab !== "pre-race" ? (
        placeholder
      ) : (
        <main className="pre-race-main">
          <section
            className={`rows-panel ${!isMobileLayout ? "two-fixed" : ""}`}
          >
            {rows.slice(0, 2).map((row) => (
              <DriverRow
                key={row.id}
                row={row}
                teams={teams}
                drivers={drivers}
                onRowChange={updateRow}
                onRetry={retryRow}
              />
            ))}
          </section>
        </main>
      )}
    </div>
    </LangProvider>
  );
}
