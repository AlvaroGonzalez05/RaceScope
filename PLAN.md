# PLAN.md — RaceScope Agent Execution Plan

This document is the authoritative instruction manual for a multi-agent Claude Code team tasked with delivering the improvements catalogued in `TO_DO.md`. Each agent has a defined scope, input conditions, output contracts, and interaction rules with the QA Agent.

**All agents must read this file in full before beginning work.**
Local copies with narrowed scope exist at:
- `code/backend_fastapi/PLAN.md` — backend agents only
- `code/frontend/PLAN.md` — frontend agents only

---

## Agent Roster

| ID | Name | Scope |
|----|------|-------|
| `BE-CORE` | Backend Core Agent | Async HTTP client migration, live ingestion API endpoint, cache invalidation |
| `BE-DATA` | Backend Data Agent | pit_loss/sc_probability from real data, circuit_id normalization, preprocess hardening |
| `BE-API` | Backend API Agent | Pydantic response models, route validation, CORS env config, legacy route deprecation, structured logging |
| `FE-LAYOUT` | Frontend Layout Agent | Viewport overflow, breakpoints, mobile padding, tooltip clamp, strategy-strip scroll affordance |
| `FE-A11Y` | Frontend Accessibility & Legibility Agent | Font sizes, contrast, ARIA roles, keyboard navigation, touch target sizes |
| `FE-SYSTEM` | Frontend Design System Agent | Compound color unification, TEAM_COLORS externalization, styles.css split, dead component cleanup, App.jsx state refactor |
| `QA` | QA Agent | Smoke tests, regression checks, code review, feedback annotation, iteration gating |

---

## Execution Phases

Dependencies flow strictly in phase order. **No agent may merge work from a later phase until the QA Agent has signed off on the current phase.**

```
Phase 0 — Baseline
  └─ QA: record baseline behaviour (response shapes, UI screenshots notes, failing states)

Phase 1 — Backend Foundation (BE-CORE + BE-DATA in parallel)
  └─ QA: smoke-test after each agent completes

Phase 2 — Backend Surface (BE-API, depends on Phase 1)
  └─ QA: full API contract review

Phase 3 — Frontend Plumbing (FE-LAYOUT + FE-A11Y in parallel, independent of backend phases)
  └─ QA: visual/interaction review

Phase 4 — Frontend Architecture (FE-SYSTEM, depends on Phase 3)
  └─ QA: full component review

Phase 5 — Integration (all agents available for cross-cutting fixes flagged by QA)
  └─ QA: end-to-end sign-off
```

---

## QA Agent Protocol

The QA Agent is the sole authority on whether a phase is complete. Every other agent must treat QA feedback as blocking until resolved.

### QA Responsibilities

1. **Phase 0 — Baseline Capture**
   - Start the backend (`uvicorn app.main:app --reload --port 8000`) and confirm it serves the smoke-test response: `curl -s -X POST http://localhost:8000/api/strategy -H "Content-Type: application/json" -d '{"year":2023,"circuit_id":"Sakhir","driver_id":14}'`
   - Record the exact JSON shape of `/api/strategy`, `/api/compare`, and all `/api/metadata/*` responses
   - Note which frontend components render on the Home and Pre-race tabs
   - Save baseline notes to `qa/baseline.md`

2. **Per-phase Review Checklist**
   After each agent declares work complete, QA runs the following:
   - **Backend phases**: re-run the smoke-test curl; run `scripts/benchmark_strategy.py` and compare cold/warm/hot latencies to baseline; check that no existing response field has been removed or renamed; verify the OpenAPI schema at `/docs` is coherent
   - **Frontend phases**: load the app in a browser, check Home and Pre-race tabs; verify no visual regression against baseline notes; check browser console for errors/warnings; test at three viewport widths: 1440px, 1024px, 375px
   - **All phases**: read the diff of changed files; flag any introduced security issues (hardcoded secrets, unvalidated inputs, SQL/command injection vectors), performance regressions, or broken imports

3. **Feedback Format**
   QA writes feedback to `qa/phase-N-review.md` using this structure:
   ```
   ## Phase N QA Review — [date]

   ### PASS / FAIL / PARTIAL

   ### Issues (blocking)
   - [AGENT-ID] file:line — description of problem

   ### Observations (non-blocking)
   - description

   ### Re-test after fixes
   - checklist of items QA will re-verify
   ```

4. **Sign-off**
   QA writes `LGTM — Phase N complete` at the bottom of the review file and updates the phase status table in this PLAN.md before the next phase begins.

---

## Phase Status

| Phase | Status | QA Sign-off |
|-------|--------|-------------|
| 0 — Baseline | pending | — |
| 1 — Backend Foundation | pending | — |
| 2 — Backend Surface | pending | — |
| 3 — Frontend Plumbing | pending | — |
| 4 — Frontend Architecture | pending | — |
| 5 — Integration | pending | — |

---

## General Rules for All Agents

1. **Read before writing.** Read every file you will modify. Do not patch code you have not read.
2. **One concern per change.** Do not bundle unrelated fixes in the same edit.
3. **No breaking changes to response contracts.** If a response shape must change, add the new field alongside the old one and mark the old field deprecated; removal is done only after QA confirms no consumer depends on it.
4. **No new dependencies without justification.** If you add a package, state the reason in a code comment and update `requirements.txt` or `package.json` accordingly.
5. **Do not touch files outside your scope** unless QA has explicitly delegated a cross-cutting concern to you in a review file.
6. **After completing work**, write a short summary in `qa/agent-[ID]-summary.md` describing what was changed and why, listing every file touched.

---
---

# Phase 0 — QA Agent: Baseline Capture

**Agent:** `QA`
**Precondition:** clean working tree, venv active, frontend built

## Steps

1. Start the backend:
   ```bash
   cd code/backend_fastapi
   source .venv_demo/bin/activate
   uvicorn app.main:app --reload --port 8000
   ```

2. Run and record the smoke-test response:
   ```bash
   curl -s -X POST http://localhost:8000/api/strategy \
     -H "Content-Type: application/json" \
     -d '{"year":2023,"circuit_id":"Sakhir","driver_id":14}' | python3 -m json.tool > qa/baseline_strategy_response.json
   ```

3. Record all metadata endpoint responses:
   ```bash
   curl -s http://localhost:8000/api/metadata/seasons > qa/baseline_seasons.json
   curl -s "http://localhost:8000/api/metadata/circuits?season=2023" > qa/baseline_circuits.json
   curl -s "http://localhost:8000/api/metadata/drivers?season=2023" > qa/baseline_drivers.json
   curl -s "http://localhost:8000/api/metadata/teams?season=2023" > qa/baseline_teams.json
   ```

4. Run the benchmark and save output:
   ```bash
   .venv_demo/bin/python scripts/benchmark_strategy.py
   cp benchmark_report.json qa/baseline_benchmark.json
   ```

5. Open the frontend at `http://localhost:5173` and note in `qa/baseline.md`:
   - which components render on Home tab
   - which components render on Pre-race tab after a "Calcular" run
   - any console errors or warnings already present

6. Create `qa/` directory and commit all baseline artefacts. Update Phase 0 status to `complete` in the table above.

---

# Phase 1 — Backend Foundation

**Agents:** `BE-CORE`, `BE-DATA` (run in parallel — no shared files)

---

## BE-CORE: Async HTTP Client & Live Ingestion Trigger

**Files in scope:**
- `code/backend_fastapi/app/openf1_client.py`
- `code/backend_fastapi/app/ingest.py`
- `code/backend_fastapi/app/main.py`
- `code/backend_fastapi/app/data_store.py`
- `code/backend_fastapi/requirements.txt`

**Precondition:** Phase 0 baseline captured.

### Task 1 — Replace `requests` with `httpx` async client

1. Read `app/openf1_client.py` in full.
2. Add `httpx` to `requirements.txt` (keep `requests` temporarily until ingest.py script callers are confirmed migrated).
3. Rewrite `OpenF1Client` as an async class:
   - `__init__` becomes synchronous (no I/O there)
   - `_fetch_token` becomes `async def _fetch_token`
   - `get` becomes `async def get` using `httpx.AsyncClient`
   - Replace `time.sleep` with `await asyncio.sleep`
   - All `requests.post`/`requests.get` become `httpx.AsyncClient` calls within an `async with` block
   - Rate-limiting: replace `time.sleep(wait)` with `await asyncio.sleep(wait)`
   - Backoff: replace `time.sleep(self.backoff_base ** attempt)` with `await asyncio.sleep(...)`
4. Keep the public interface (`get(endpoint, params, use_cache)`) identical in signature. Return type remains `list[dict]`.
5. For the CLI scripts (`scripts/ingest_season.py`) that call sync code, add a thin sync wrapper at the bottom of `openf1_client.py`:
   ```python
   def get_sync(self, endpoint, params=None, use_cache=True):
       import asyncio
       return asyncio.run(self.get(endpoint, params, use_cache))
   ```
   Update `ingest.py` to call `client.get_sync(...)` in its synchronous context.
6. Verify no circular imports were introduced. Run `python -c "from app.openf1_client import OpenF1Client; print('ok')"` inside the venv.

### Task 2 — Token-bucket rate limiter (6 req/s burst, 60 req/min sustained)

**Context:** The OpenF1 premium API enforces two independent limits:
- **Burst**: 6 requests per second — minimum 167 ms between consecutive calls
- **Sustained**: 60 requests per minute — no more than 1 req/s on average

A simple `min_interval` sleep only enforces the burst floor. Without a sustained-rate guard, a fast burst of 6 calls in the first second and silence for the rest of a minute is fine, but back-to-back bursts would blow the 60/min cap. Implement a **token-bucket** algorithm.

1. Read `app/openf1_client.py` and `app/config.py` in full.
2. Add two new instance variables to `OpenF1Client.__init__`:
   ```python
   self._rate_per_second: int   = OPENF1_RATE_PER_SECOND   # 6
   self._rate_per_minute: int   = OPENF1_RATE_PER_MINUTE   # 60
   self._minute_tokens: float   = float(OPENF1_RATE_PER_MINUTE)
   self._minute_refill_time: float = time.monotonic()
   ```
3. Add an `async def _acquire_token(self)` method that:
   - Refills `_minute_tokens` based on elapsed seconds since `_minute_refill_time` (rate: `RATE_PER_MINUTE / 60` tokens per second, capped at `RATE_PER_MINUTE`).
   - If `_minute_tokens < 1`, sleeps for the time needed to accumulate 1 token, then refills.
   - Decrements `_minute_tokens` by 1.
   - Also enforces `MIN_INTERVAL` between calls (burst floor) using `_last_request` as before.
4. Call `await self._acquire_token()` at the top of the `async def get(...)` method, replacing the existing `min_interval` sleep block.
5. Import `OPENF1_RATE_PER_SECOND` and `OPENF1_RATE_PER_MINUTE` from `config.py` in `openf1_client.py`.

### Task 3 — Live ingestion trigger endpoint

1. Read `app/main.py` and `app/data_store.py` in full.
2. In `main.py`, add a new route:
   ```
   POST /api/admin/ingest
   Body: {"year": int}
   ```
   - Accept only valid 4-digit years; return 422 if invalid.
   - Use `fastapi.BackgroundTasks` to run `ingest_season(year)` and `build_features_for_year(year)` sequentially in the background.
   - After the background task completes, call `load_features.cache_clear()` from `data_store` so the next strategy request picks up fresh data.
   - Return immediately with `{"status": "ingestion_started", "year": year}`.
3. Add a companion route `GET /api/admin/ingest/status` that returns the contents of `snapshot_state.json` — this lets callers poll for completion.
4. Do not add authentication to these admin routes now, but add a `# TODO: protect with API key before public deployment` comment.

### Task 3 — Cache invalidation hook in data_store

1. In `data_store.py`, expose a function `invalidate_features_cache()` that calls `load_features.cache_clear()` and logs the invalidation with `logging.getLogger(__name__).info(...)`.
2. Call this function at the end of the background ingestion task in `main.py`, replacing any direct `.cache_clear()` call.

**Completion criteria:**
- `python -c "from app.openf1_client import OpenF1Client; print('ok')"` passes
- `uvicorn app.main:app --reload` starts without error
- `POST /api/admin/ingest` with `{"year": 2023}` returns 200 with `ingestion_started`
- `GET /api/admin/ingest/status` returns the snapshot state JSON
- The smoke-test curl still returns a valid strategy response (no regression)

**Write summary to:** `qa/agent-BE-CORE-summary.md`

---

## BE-DATA: Pit Loss, SC Probability, Circuit ID Normalization

**Files in scope:**
- `code/backend_fastapi/app/strategy_engine.py`
- `code/backend_fastapi/app/preprocess.py`
- `code/backend_fastapi/app/config.py`

**Precondition:** Phase 0 baseline captured.

### Task 1 — Derive per-circuit pit_loss from feature data

1. Read `app/strategy_engine.py` (`_context` method) and `app/preprocess.py` in full.
2. In `strategy_engine.py`, modify `_context` to compute `pit_loss` from the features DataFrame instead of using the hardcoded 22.5:
   - Filter features for `circuit_id` and `session_type == "RACE"`.
   - Use stint boundaries to identify lap transitions where `stint_number` increments for the same driver — the difference in `lap_time` at that lap vs the driver's median clean lap time approximates pit delta.
   - Take the median of all such deltas for the circuit, clamped to the range `[18.0, 35.0]`.
   - Fall back to `22.5` if fewer than 5 pit events are found.
3. Add a constant `PIT_LOSS_FALLBACK = 22.5` to `config.py` and `PIT_LOSS_MIN = 18.0`, `PIT_LOSS_MAX = 35.0`.

### Task 2 — Derive sc_probability from session data

1. In `strategy_engine.py`, modify `_context` to estimate `sc_probability`:
   - Count the number of historical race sessions for this `circuit_id` in the features.
   - In the raw data structure, SC/VSC events are not directly stored — use a proxy: laps where `lap_time` for multiple drivers simultaneously exceeds `median_lap_time * 1.35` (characteristic of SC neutralization) within a 3-lap window.
   - `sc_probability = (number of sessions with at least one such event) / (total sessions)`, clamped to `[0.05, 0.55]`.
   - Fall back to `0.20` if fewer than 2 historical sessions are found.
2. Add `SC_PROBABILITY_FALLBACK = 0.20`, `SC_PROBABILITY_MIN = 0.05`, `SC_PROBABILITY_MAX = 0.55` to `config.py`.

### Task 3 — Fix circuit_id normalization

1. Read `app/preprocess.py` lines 117 and surrounding context in full.
2. In `preprocess.py`, create a module-level `CIRCUIT_ID_CANONICAL` dict mapping known OpenF1 variants to a canonical stable string. Start with:
   ```python
   CIRCUIT_ID_CANONICAL = {
       "bahrain": "Sakhir",
       "sakhir": "Sakhir",
       "jeddah": "Jeddah",
       "saudi arabia": "Jeddah",
       "albert park": "Melbourne",
       "melbourne": "Melbourne",
       "suzuka": "Suzuka",
       "baku": "Baku",
       "miami": "Miami",
       "monte carlo": "Monaco",
       "monaco": "Monaco",
       "barcelona": "Barcelona",
       "montmelo": "Barcelona",
       "red bull ring": "Spielberg",
       "spielberg": "Spielberg",
       "silverstone": "Silverstone",
       "hungaroring": "Budapest",
       "budapest": "Budapest",
       "spa": "Spa",
       "spa-francorchamps": "Spa",
       "monza": "Monza",
       "marina bay": "Singapore",
       "singapore": "Singapore",
       "suzuka": "Suzuka",
       "lusail": "Lusail",
       "qatar": "Lusail",
       "cota": "Austin",
       "austin": "Austin",
       "rodriguez": "Mexico City",
       "mexico city": "Mexico City",
       "interlagos": "São Paulo",
       "são paulo": "São Paulo",
       "las vegas": "Las Vegas",
       "yas marina": "Abu Dhabi",
       "abu dhabi": "Abu Dhabi",
   }
   ```
3. Replace the fragile fallback chain at line 117 with:
   ```python
   raw_id = (session.get("circuit_short_name") or session.get("location") or str(session.get("meeting_key", ""))).strip().lower()
   circuit_id = CIRCUIT_ID_CANONICAL.get(raw_id) or raw_id.title()
   if raw_id not in CIRCUIT_ID_CANONICAL:
       logger.warning("circuit_id not in canonical map: %r — using %r", raw_id, circuit_id)
   ```
4. Add `import logging; logger = logging.getLogger(__name__)` at the top of `preprocess.py`.

**Completion criteria:**
- `_context()` no longer returns `pit_loss=22.5` for circuits with sufficient data
- `sc_probability` is no longer always 0.20 when historical race sessions exist
- Running `build_features_for_year(2023)` with the existing raw data completes without error
- The smoke-test curl still returns a valid strategy response

**Write summary to:** `qa/agent-BE-DATA-summary.md`

---

# Phase 2 — Backend Surface

**Agent:** `BE-API`
**Precondition:** Phase 1 complete and QA signed off.

**Files in scope:**
- `code/backend_fastapi/app/main.py`
- `code/backend_fastapi/app/data_store.py`
- `code/backend_fastapi/app/config.py`
- `code/backend_fastapi/app/ingest.py`
- `code/backend_fastapi/app/preprocess.py`

### Task 1 — Pydantic response models

1. Read `main.py` in full. Identify every route that returns a `dict` or `JSONResponse`.
2. Create a new file `app/schemas.py` and define response models:
   - `DriverOut(BaseModel)`: `driver_id: int`, `driver_code: str | None`, `team_name: str | None`
   - `TeamOut(BaseModel)`: `team_name: str`
   - `CircuitOut(BaseModel)`: `circuit_id: str`
   - `StintCurve(BaseModel)`: `compound: str`, `start_lap: int`, `end_lap: int`, `lap_time_data: list[float]`, `tyre_life_data: list[float]`
   - `StrategyOut(BaseModel)`: `strategy_id: str`, `type: str`, `compounds: list[str]`, `stints: list[int]`, `stint_curves: list[StintCurve]`, `pit_windows: list[dict]`, `stop_laps: list[int]`, `expected_time: float`, `variance: float`, `risk_score: float`
   - `ContextOut(BaseModel)`: `total_laps: int`, `track_temp: float`, `air_temp: float`, `pit_loss: float`, `sc_probability: float`
   - `StrategyResponse(BaseModel)`: `context: ContextOut`, `strategies: list[StrategyOut]`, `degradation: dict`, `compute_meta: dict`
   - `CompareResponse(BaseModel)`: `driver_a: StrategyResponse`, `driver_b: StrategyResponse`
3. Add `response_model=` to each route in `main.py`. Use `response_model_exclude_none=True` on all routes.
4. Do NOT remove any existing fields from responses. If the engine returns extra fields not in the model, use `model_config = ConfigDict(extra="allow")` to pass them through rather than silently dropping them.

### Task 2 — Query param validation

1. In `main.py`, for `GET /api/metadata/circuits`, `drivers`, `teams`: after reading `season`, call `seasons_available()` and return `HTTPException(status_code=404, detail=f"Season {season} not found")` if season is not present.
2. For `GET /api/metadata/seasons`: return `HTTPException(status_code=503, detail="No data available — run the pipeline first")` if the list is empty, instead of a 200 with an empty array.

### Task 3 — CORS from env

1. In `config.py`, read `CORS_ORIGINS` from env: `CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8000")`. Parse as a comma-separated list.
2. In `main.py`, replace the hardcoded `allow_origins` list with `config.CORS_ORIGINS.split(",")`.
3. Add `CORS_ORIGINS=http://localhost:5173,http://localhost:8000` to `.env.example`.

### Task 4 — Structured logging

1. At the top of `ingest.py` and `preprocess.py`, replace `print(...)` calls with `logger.info(...)` / `logger.warning(...)` using `logging.getLogger(__name__)`.
2. In `main.py`, add a startup log:
   ```python
   logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
   ```
3. Add a request-ID middleware to `main.py` using `starlette.middleware.base.BaseHTTPMiddleware`:
   - Generate `request_id = str(uuid.uuid4())[:8]` per request
   - Add it as a response header `X-Request-ID`

### Task 5 — Legacy route deprecation

1. In `main.py`, find all routes without the `/api/` prefix.
2. Add a response header `Deprecation: true` and a log warning to each legacy route handler.
3. Do NOT remove them yet — just mark them.

**Completion criteria:**
- `/docs` (OpenAPI) renders all routes with correct request/response schemas
- All `/api/metadata/*` routes return 404 for non-existent seasons
- `curl -H "Origin: http://localhost:5173"` receives correct CORS headers
- All `print()` calls in ingest and preprocess are gone; structured logs appear in uvicorn output
- Smoke-test curl still returns valid strategy response

**Write summary to:** `qa/agent-BE-API-summary.md`

---

# Phase 3 — Frontend Plumbing

**Agents:** `FE-LAYOUT`, `FE-A11Y` (run in parallel — work on different files)

---

## FE-LAYOUT: Viewport, Responsive Breakpoints, Tooltip Clamping

**Files in scope:**
- `code/frontend/src/styles.css`
- `code/frontend/src/App.jsx`

**Precondition:** Phase 0 baseline captured. (Frontend phases are independent of backend phases but must not begin until Phase 0 is signed off.)

### Task 1 — Fix vertical overflow on short screens

1. Read `styles.css` in full. Find `.rows-panel.two-fixed` and `.app-shell`.
2. Change `.rows-panel.two-fixed`:
   ```css
   /* Before: grid-template-rows: 350px 350px; */
   grid-template-rows: repeat(2, minmax(280px, 1fr));
   ```
3. Remove the `overflow: hidden` from `.rows-panel` and instead add `overflow-y: auto` so that on constrained screens, the panel scrolls rather than clips.
4. In `App.jsx`, remove the `fixedDesktopRowHeight` variable and the `--row-height` style prop passed to `<DriverRow>` — the CSS grid handles sizing now.
5. In `styles.css`, update `.driver-row` to remove `height: var(--row-height, 350px)` and use `min-height: 280px` instead.

### Task 2 — Strategy-strip scroll affordance

1. Find `.strategy-strip` in `styles.css`.
2. Add fade-out gradients using a `::after` pseudo-element to indicate horizontal overflow:
   ```css
   .strategy-strip-wrapper {
     position: relative;
     overflow: hidden;
   }
   .strategy-strip-wrapper::after {
     content: "";
     position: absolute;
     top: 0; right: 0; bottom: 2px;
     width: 32px;
     background: linear-gradient(to right, transparent, var(--panel));
     pointer-events: none;
   }
   ```
3. In `StrategyStrip.jsx` (read it first), wrap the `strategy-strip` div in a `<div className="strategy-strip-wrapper">`.
   **Note:** `StrategyStrip.jsx` is outside this agent's primary scope — make this single, minimal wrapper change only. Flag to `FE-SYSTEM` agent in your summary if wider refactoring is needed.

### Task 3 — Clamp tooltip at chart edges

1. Find `.curve-tooltip` in `styles.css`. It currently uses `transform: translate(-50%, -120%)`.
2. Do not change the CSS class — the clamping must be done in JS where the pixel position is known.
3. Read `StrategyCurveChart.jsx`. In the tooltip rendering block (around line 346), replace the static `left` style calculation with a clamped version:
   ```jsx
   const tooltipLeftPct = (hover.x / width) * 100;
   const clampedLeft = Math.min(Math.max(tooltipLeftPct, 8), 92);
   // style={{ left: `${clampedLeft.toFixed(2)}%`, top: ... }}
   ```
   **Note:** `StrategyCurveChart.jsx` is in `FE-A11Y`'s scope for aria changes. Coordinate: apply only the tooltip-clamp change here; `FE-A11Y` handles aria. Use a comment `// FE-LAYOUT: clamped tooltip` to mark the change.

### Task 4 — Intermediate breakpoint for the rail

1. In `styles.css`, add a breakpoint between the existing `@media (max-width: 1200px)` and `@media (max-width: 920px)`:
   ```css
   @media (max-width: 1080px) {
     .driver-row {
       grid-template-columns: 1fr;
     }
     .driver-rail {
       border-right: 0;
       border-bottom: 1px solid var(--border);
     }
     .curve-wrapper {
       height: 160px;
       min-height: 160px;
     }
   }
   ```

### Task 5 — Mobile padding on the context bubble

1. In the `@media (max-width: 920px)` block in `styles.css`, add to `.pre-race-context-bubble`:
   ```css
   padding: 12px;
   ```
2. Add `.pre-race-context-fields .cta { width: 100%; }` inside the same media query so the Calcular button is full-width on mobile.

**Completion criteria:**
- On a simulated 768px-tall desktop viewport, both driver rows are visible without clipping
- Hovering near lap 1 or the final lap on any chart does not produce a clipped tooltip
- On 375px viewport width, the context bubble has adequate padding and the CTA is full-width
- Strategy strip shows fade indicator when cards overflow

**Write summary to:** `qa/agent-FE-LAYOUT-summary.md`

---

## FE-A11Y: Accessibility, Legibility, and Touch

**Files in scope:**
- `code/frontend/src/styles.css`
- `code/frontend/src/components/TopTabs.jsx`
- `code/frontend/src/components/StrategyStrip.jsx`
- `code/frontend/src/components/StrategyCurveChart.jsx`
- `code/frontend/src/components/DriverRow.jsx`

**Precondition:** Phase 0 baseline captured.

### Task 1 — Fix font sizes below 12px

1. Read `styles.css` in full. Find and update these rules:
   - `.axis-caption` (SVG): `font-size: 9px` → `font-size: 11px`
   - `.data-mode-chip`: `font-size: 10px` → `font-size: 11px`
   - `.strategy-kind`: `font-size: 10px` → `font-size: 11px`
   - `.insight-item small`: `font-size: 10px` → `font-size: 11px`
   - `.legend-item`: `font-size: 10px` → `font-size: 11px`
   - `.holo-label`: 11px — leave as-is

### Task 2 — Darken muted text in dark theme

1. In `:root` (dark theme default), change `--muted` from `#9ea6b1` to `#b0bac6` to improve contrast ratio from ~4.5:1 to ~5.5:1 against the dark panel background.
2. Leave the light theme `--muted` unchanged (already ~5.3:1).

### Task 3 — Fix TopTabs ARIA

1. Read `TopTabs.jsx` in full.
2. Add `role="tablist"` to the `<nav>` element.
3. Add `role="tab"` and `aria-selected={activeTab === tab}` to each `<button>`.
4. The `<nav>` already has `aria-label="Secciones"` — leave it.

### Task 4 — Fix StrategyStrip ARIA

1. Read `StrategyStrip.jsx` in full.
2. Change `role="listbox"` on the strip div to `role="list"`.
3. Add `role="listitem"` to each `<button>` wrapper. Since buttons are the direct children and can't be both `listitem` and a button semantically, wrap each button in a `<div role="listitem">`:
   ```jsx
   <div role="listitem" key={strategyId}>
     <button ...>...</button>
   </div>
   ```
4. Remove `role="listbox"` from the container (replaced by `role="list"`).
5. Add `aria-pressed={selectedStrategyId === strategyId}` to each button.

### Task 5 — Add aria-live for loading/error states in DriverRow

1. Read `DriverRow.jsx` in full.
2. Add a visually-hidden `aria-live="polite"` region inside the component, below the `driver-rail`, that announces the status:
   ```jsx
   <p aria-live="polite" className="sr-only">
     {row.status === "loading" ? "Calculando estrategias..." :
      row.status === "ready" ? "Estrategias cargadas." :
      row.status === "error" ? "Error al calcular estrategias." : ""}
   </p>
   ```
3. Add `.sr-only` to `styles.css`:
   ```css
   .sr-only {
     position: absolute;
     width: 1px;
     height: 1px;
     padding: 0;
     margin: -1px;
     overflow: hidden;
     clip: rect(0, 0, 0, 0);
     white-space: nowrap;
     border: 0;
   }
   ```

### Task 6 — Touch support for chart tooltip

1. Read `StrategyCurveChart.jsx` in full.
2. On the `curve-wrapper` div, replace `onMouseMove={handleMove}` with `onPointerMove={handleMove}` and `onMouseLeave={() => setHover(null)}` with `onPointerLeave={() => setHover(null)}`.
   - `onPointerMove` fires for both mouse and touch, so no additional handler is needed.
3. **Coordinate with FE-LAYOUT:** FE-LAYOUT adds a tooltip clamp in this file too. Apply both changes independently — they touch different lines. If FE-LAYOUT has already modified this file, read the current state before applying your change.

### Task 7 — Minimum touch target: icon-btn

1. In `styles.css`, find `.icon-btn` (currently 28×28px). Increase to:
   ```css
   .icon-btn {
     width: 36px;
     height: 36px;
   }
   ```
   This is a compromise: 44px is ideal but may break the layout in the rail header. Flag in your summary if further increase is needed.

### Task 8 — Disambiguate DriverRow headers

1. Read `DriverRow.jsx`. The `<h3>Piloto</h3>` heading does not distinguish which row it is.
2. The `row.id` prop is available (value 1 or 2). Change:
   ```jsx
   <h3>Piloto {row.id}</h3>
   ```
3. In `row-meta`, once `row.driverId` is set, show the driver code. The `drivers` prop is available; find the matching driver:
   ```jsx
   const driverLabel = drivers.find(d => d.driver_id === row.driverId)?.driver_code || "";
   ```
   Render it in the `row-head` as a secondary badge when non-empty.

**Completion criteria:**
- No font sizes below 11px remain in `styles.css`
- `aria-selected` appears on TopTab buttons in DOM
- `aria-live` region is present in DriverRow
- Chart tooltip fires on touch (pointer events)
- DriverRow headers are differentiated as "Piloto 1" and "Piloto 2"

**Write summary to:** `qa/agent-FE-A11Y-summary.md`

---

# Phase 4 — Frontend Architecture

**Agent:** `FE-SYSTEM`
**Precondition:** Phase 3 complete and QA signed off.

**Files in scope:**
- `code/frontend/src/styles.css`
- `code/frontend/src/App.jsx`
- `code/frontend/src/components/DriverRow.jsx`
- `code/frontend/src/components/StrategyCurveChart.jsx`
- `code/frontend/src/components/StrategyStrip.jsx`
- `code/frontend/src/components/HomeLanding.jsx`
- `code/frontend/src/components/*.jsx` (dead components)
- New files as needed

### Task 1 — Unify compound colors into a single source

1. Create `code/frontend/src/constants/compounds.js`:
   ```js
   export const COMPOUND_COLORS = {
     SOFT:   "#ff4b4b",
     MEDIUM: "#f2c94c",
     HARD:   "#b8bec6",
   };
   ```
2. In `StrategyCurveChart.jsx`, replace the inline `COMPOUND_COLORS` object with an import from `../constants/compounds.js`.
3. In `styles.css`, keep `--soft`, `--medium`, `--hard` CSS variables as-is (they are used by the holo classes and cannot be driven from JS without extra tooling). Add a comment:
   ```css
   /* Compound palette — JS canonical source: src/constants/compounds.js */
   ```
4. Do not attempt to auto-sync CSS vars from JS — that requires a build step. The comment is sufficient for now.

### Task 2 — Externalize TEAM_COLORS

1. Create `code/frontend/src/constants/teams.js`:
   ```js
   export const TEAM_COLORS = {
     "Red Bull Racing": "#1f6cff",
     "Ferrari":         "#ff3b30",
     "Mercedes":        "#00d6c7",
     "McLaren":         "#ff8a00",
     "Aston":           "#2db56f",
     "Alpine":          "#ff4d94",
     "Williams":        "#4fa2ff",
     "RB":              "#3355ff",
     "Sauber":          "#6dcf38",
     "Haas":            "#bfc7d1",
   };

   export function teamTint(teamName) {
     const key = Object.keys(TEAM_COLORS).find((k) => teamName?.includes(k));
     return key ? TEAM_COLORS[key] : "#6f7a86";
   }
   ```
2. In `DriverRow.jsx`, remove the local `TEAM_COLORS` and `teamTint` definitions, and import from `../constants/teams.js`.

### Task 3 — Remove dead components

1. Read each of: `ComparisonView.jsx`, `ControlPanel.jsx`, `DegradationChart.jsx`, `StrategyCard.jsx`, `StrategyTimeline.jsx`.
2. Confirm none are imported anywhere using grep: `grep -r "ComparisonView\|ControlPanel\|DegradationChart\|StrategyCard\|StrategyTimeline" src/`.
3. If confirmed unused, delete them.
4. If any contain logic that genuinely belongs to a future feature, move that logic to a `// TODO: implement` stub comment in `App.jsx` and then delete the file.

### Task 4 — Delta-to-best display in StrategyStrip and StrategyCurveChart

1. Read `StrategyStrip.jsx`. The strategies are already sorted by `expected_time`. The first strategy is the best.
2. In `StrategyStrip`, compute `bestTime = ordered[0]?.expected_time` and pass it to each button. Show delta:
   ```jsx
   const delta = s.expected_time - bestTime;
   // In button: show "+{formatDelta(delta)}" for non-first strategies, "BEST" for first
   ```
3. Add `formatDelta(seconds)` to `code/frontend/src/utils/time.js`:
   ```js
   export function formatDelta(seconds) {
     if (!Number.isFinite(seconds) || seconds <= 0) return "BEST";
     const m = Math.floor(seconds / 60);
     const s = (seconds % 60).toFixed(1);
     return m > 0 ? `+${m}m ${s}s` : `+${s}s`;
   }
   ```
4. In `StrategyCurveChart.jsx`, in the `strategy-risk` paragraph, replace raw `variance` display with: show `BEST` or `+Xs` delta (pass `bestTime` as a prop from `DriverRow`). Keep variance as a `title` tooltip attribute for advanced users.
   Update `DriverRow.jsx` to compute `bestTime = orderedStrategies[0]?.expected_time` and pass it as `bestTime` prop to each `StrategyCurveChart`.

### Task 5 — Lift App.jsx state into a reducer

1. Read `App.jsx` in full.
2. Create `code/frontend/src/state/appReducer.js` with a reducer that manages the full app state currently spread across 15+ `useState` hooks:
   - `theme`, `activeTab`, `metadataStatus`, `metadataError`, `seasons`, `season`, `circuits`, `circuitId`, `drivers`, `teams`, `rows`, `running`, `isMobileLayout`
3. Define action types as constants (e.g., `SET_THEME`, `SET_ACTIVE_TAB`, `SET_SEASONS`, `SET_SEASON_METADATA`, `SET_ROWS`, `SET_RUNNING`, `SET_MOBILE_LAYOUT`, `SET_METADATA_ERROR`).
4. In `App.jsx`, replace all `useState` calls with a single `useReducer(appReducer, initialState)`. Keep all existing logic and callbacks — only the state storage changes.
5. Do not extract into Context yet — that is a future concern. `useReducer` in `App.jsx` is sufficient.

### Task 6 — Add per-row retry button

1. Read `DriverRow.jsx`. When `row.status === "error"`, the only feedback is a text message.
2. Add a retry callback: update the `DriverRow` prop interface to accept `onRetry: (rowId) => void`.
3. In `App.jsx`, define `handleRetryRow(rowId)` that re-runs the strategy API call for that specific row only (extract the single-row fetch logic from `runPreRace`).
4. In `DriverRow`, render a `<button className="ghost-btn">Reintentar</button>` next to the error text, calling `onRetry(row.id)`.

**Completion criteria:**
- `COMPOUND_COLORS` and `TEAM_COLORS` imported from `src/constants/`
- Dead components deleted; no import errors
- StrategyStrip shows "BEST" / "+Xs" deltas
- `App.jsx` uses `useReducer` with no `useState` for the lifted state
- Error rows show a per-row retry button

**Write summary to:** `qa/agent-FE-SYSTEM-summary.md`

---

# Phase 5 — Integration

**Agent:** All agents available for targeted fixes.
**Precondition:** Phase 4 complete and QA signed off.

The QA Agent runs a full end-to-end pass:
1. Fresh backend start with `OPENF1_AUTH_ENABLED=true` and valid credentials.
2. Call `POST /api/admin/ingest` with `{"year": 2023}` and poll `GET /api/admin/ingest/status` until complete.
3. Call the smoke-test and verify `pit_loss` and `sc_probability` differ from the Phase 0 baseline for at least one circuit.
4. Load the frontend at 1440px, 1024px, and 375px — verify all Phase 3/4 improvements are present.
5. Run Lighthouse accessibility audit (built into Chrome DevTools); target ≥90 accessibility score.
6. Run the benchmark and compare to Phase 0 baseline; flag any cold/warm latency regression >20%.

Cross-cutting issues identified by QA in `qa/phase-5-review.md` are assigned to the most appropriate agent for a focused fix. The cycle repeats until QA issues `LGTM — Phase 5 complete`.

---

## Appendix — File Ownership Map

| File | Owner agent |
|------|-------------|
| `app/openf1_client.py` | BE-CORE |
| `app/ingest.py` | BE-CORE (async wrapper), BE-API (logging) |
| `app/main.py` | BE-CORE (admin routes), BE-API (schemas, logging, CORS) |
| `app/data_store.py` | BE-CORE (cache invalidation) |
| `app/strategy_engine.py` | BE-DATA |
| `app/preprocess.py` | BE-DATA (circuit normalization), BE-API (logging) |
| `app/config.py` | BE-DATA (constants), BE-API (CORS env) |
| `app/schemas.py` | BE-API (new file) |
| `src/styles.css` | FE-LAYOUT (layout rules), FE-A11Y (font/contrast/sr-only) |
| `src/App.jsx` | FE-LAYOUT (remove fixedDesktopRowHeight), FE-SYSTEM (reducer, retry) |
| `src/components/TopTabs.jsx` | FE-A11Y |
| `src/components/StrategyStrip.jsx` | FE-LAYOUT (wrapper), FE-A11Y (aria), FE-SYSTEM (delta) |
| `src/components/StrategyCurveChart.jsx` | FE-LAYOUT (tooltip clamp), FE-A11Y (pointer events), FE-SYSTEM (delta prop) |
| `src/components/DriverRow.jsx` | FE-A11Y (aria-live, header), FE-SYSTEM (retry, bestTime) |
| `src/components/HomeLanding.jsx` | FE-A11Y (touch hologram — polish only) |
| `src/constants/compounds.js` | FE-SYSTEM (new file) |
| `src/constants/teams.js` | FE-SYSTEM (new file) |
| `src/utils/time.js` | FE-SYSTEM (formatDelta) |
| `src/state/appReducer.js` | FE-SYSTEM (new file) |

When two agents share a file, they must read the latest version before applying their change and must not overwrite each other's modifications. If a conflict arises, the later-phase agent always reads and preserves the earlier agent's changes.
