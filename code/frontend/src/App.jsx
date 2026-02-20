import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import ThemeToggle from "./components/ThemeToggle.jsx";
import TopTabs from "./components/TopTabs.jsx";
import DriverRow from "./components/DriverRow.jsx";
import HomeLanding from "./components/HomeLanding.jsx";
import PreRaceContextBubble from "./components/PreRaceContextBubble.jsx";

const API_BASE = import.meta.env.VITE_API_BASE || "";
const api = axios.create({ baseURL: API_BASE || "/" });

const getPreferredTheme = () => {
  const stored = localStorage.getItem("theme");
  if (stored) return stored;
  const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
};

const mkRow = (id, team = "", driverId = "") => ({
  id,
  team,
  driverId,
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
      driverId: row.driverId || fallbackDriver?.driver_id || "",
      data: null,
      status: "idle",
      selectedStrategyId: null,
    };
  });
};

export default function App() {
  const shellRef = useRef(null);
  const [theme, setTheme] = useState("dark");
  const [activeTab, setActiveTab] = useState("home");
  const [metadataStatus, setMetadataStatus] = useState("loading");
  const [metadataError, setMetadataError] = useState("");
  const [seasons, setSeasons] = useState([]);
  const [season, setSeason] = useState(0);
  const [circuits, setCircuits] = useState([]);
  const [circuitId, setCircuitId] = useState("");
  const [drivers, setDrivers] = useState([]);
  const [teams, setTeams] = useState([]);
  const [rows, setRows] = useState([mkRow(1), mkRow(2)]);
  const [running, setRunning] = useState(false);
  const [isMobileLayout, setIsMobileLayout] = useState(false);

  useEffect(() => {
    const initial = getPreferredTheme();
    setTheme(initial);
    document.documentElement.dataset.theme = initial;
  }, []);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 920px)");
    const update = () => setIsMobileLayout(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
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
      setMetadataStatus("loading");
      setMetadataError("");
      const seasonsResp = await api.get("/api/metadata/seasons");
      const years = seasonsResp.data || [];
      setSeasons(years);
      if (!years.length) {
        setMetadataStatus("empty");
        setMetadataError("No hay temporadas disponibles.");
        return;
      }
      setSeason(years[years.length - 1]);
      setMetadataStatus("ready");
    } catch {
      setMetadataStatus("error");
      setMetadataError("No se pudieron cargar las temporadas.");
    }
  };

  const loadSeasonMetadata = async (targetSeason) => {
    if (!targetSeason) return;
    try {
      setMetadataStatus("loading");
      const [circuitsResp, driversResp, teamsResp] = await Promise.all([
        api.get("/api/metadata/circuits", { params: { season: targetSeason } }),
        api.get("/api/metadata/drivers", { params: { season: targetSeason } }),
        api.get("/api/metadata/teams", { params: { season: targetSeason } }),
      ]);

      const circuitsData = circuitsResp.data || [];
      const driversData = driversResp.data || [];
      const teamsData = teamsResp.data || [];

      setCircuits(circuitsData);
      setCircuitId(circuitsData[0] || "");
      setDrivers(driversData);
      setTeams(teamsData);

      setRows((prev) => hydrateRowsFromMetadata(prev, teamsData, driversData));

      if (!circuitsData.length || !driversData.length || !teamsData.length) {
        setMetadataStatus("empty");
        setMetadataError("Faltan datos de metadata para esta temporada.");
      } else {
        setMetadataStatus("ready");
        setMetadataError("");
      }
    } catch {
      setMetadataStatus("error");
      setMetadataError("No se pudo cargar circuitos, pilotos o equipos.");
    }
  };

  useEffect(() => {
    loadInitialMetadata();
  }, []);

  useEffect(() => {
    if (season) loadSeasonMetadata(season);
  }, [season]);

  const updateRow = (id, patch) => {
    setRows((prev) => ensureTwoRows(prev.map((r) => (r.id === id ? { ...r, ...patch } : r))));
  };

  const canRun = useMemo(() => {
    if (!season || !circuitId || metadataStatus !== "ready") return false;
    return rows.some((r) => r.driverId);
  }, [season, circuitId, metadataStatus, rows]);

  const runPreRace = async () => {
    if (!canRun) return;
    setRunning(true);
    setRows((prev) => prev.map((row) => ({ ...row, status: row.driverId ? "loading" : "idle" })));

    const work = rows.map(async (row) => {
      if (!row.driverId) return { id: row.id, status: "idle", data: null };
      try {
        const res = await api.post("/api/strategy", {
          year: Number(season),
          circuit_id: circuitId,
          driver_id: Number(row.driverId),
        });
        return { id: row.id, status: "ready", data: res.data, selectedStrategyId: null };
      } catch {
        return { id: row.id, status: "error", data: null };
      }
    });

    const updates = await Promise.all(work);
    setRows((prev) =>
      prev.map((row) => {
        const match = updates.find((u) => u.id === row.id);
        return match ? { ...row, ...match } : row;
      }),
    );
    setRunning(false);
  };

  const fixedDesktopRowHeight = !isMobileLayout ? "350px" : "auto";

  const placeholder = (
    <section className="placeholder-view">
      <h2>{activeTab}</h2>
      <p>Pendiente de implementación en siguiente fase.</p>
    </section>
  );

  return (
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
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </div>
      </header>

      <div className="nav-context-row">
        <TopTabs activeTab={activeTab} onChange={setActiveTab} />
        {activeTab === "pre-race" && (
          <PreRaceContextBubble
            season={season}
            seasons={seasons}
            onSeasonChange={(value) => setSeason(Number(value))}
            circuitId={circuitId}
            circuits={circuits}
            onCircuitChange={setCircuitId}
            onRun={runPreRace}
            canRun={canRun}
            running={running}
            metadataStatus={metadataStatus}
            metadataError={metadataError}
            onRetry={loadInitialMetadata}
          />
        )}
      </div>

      {activeTab === "home" ? (
        <HomeLanding onEnterPreRace={() => setActiveTab("pre-race")} />
      ) : activeTab !== "pre-race" ? (
        placeholder
      ) : (
        <main className="pre-race-main">
          <section
            className={`rows-panel ${!isMobileLayout ? "two-fixed" : ""}`}
            style={{ "--row-height": fixedDesktopRowHeight }}
          >
            {rows.slice(0, 2).map((row) => (
              <DriverRow
                key={row.id}
                row={row}
                teams={teams}
                drivers={drivers}
                onRowChange={updateRow}
                rowHeight={fixedDesktopRowHeight}
              />
            ))}
          </section>
        </main>
      )}
    </div>
  );
}
