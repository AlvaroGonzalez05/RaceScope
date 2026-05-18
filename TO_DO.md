# TO_DO — RaceScope Improvements

Generated from backend evaluation (2026-03-05). Ordered by priority.

---

## Estado sesión 2026-05-18 — Transformer v2

### Completado ✅
- Arquitectura `TyreDegradationTransformerV2` en `app/models_transformer.py`
- Config: `TRANSFORMER_V2_*`, `CIRCUIT_VOCAB`, `CIRCUIT_EXPECTED_LAPS` en `app/config.py`
- Pipeline de entrenamiento: val split 85/15, early stopping (patience=12), multi-head loss, CSV logs
- Optuna HPO: `scripts/hparam_search.py`
- Evaluación held-out 2025: `scripts/evaluate_2025.py --push-sensitivity-test`
- Benchmark v1 vs v2: `scripts/benchmark_models.py`
- Visualización entrenamiento: `scripts/plot_training.py`
- 12 tests verdes: `pytest tests/test_transformer_model.py -v`
- `CLAUDE.md`, `ARCHITECTURE.md`, `COMMANDS.md` actualizados
- **Entrenamiento v2 COMPLETADO** — 25 drivers + global, 50 epochs, patience=12
  - Global: val_mae=0.097, mediana drivers=0.117, D50 outlier=0.496
  - Relanzado con `--epochs 50 --patience 12` tras análisis de plots (instabilidad con 15 epochs)
- v2 verificado como motor de simulación activo en la webapp (`isinstance(TyreDegradationTransformerV2)` dispatch)

### Próximos pasos opcionales
1. Evaluación 2025 (held-out) — requiere ingestar 2025:
   ```bash
   python -m scripts.ingest_season --year 2025 && python -m scripts.preprocess --year 2025
   python -m scripts.evaluate_2025 --push-sensitivity-test
   ```
2. HPO para afinar hiperparámetros:
   ```bash
   python -m scripts.hparam_search --n-trials 50 --timeout-hours 4 --output hparam_results/
   python -m scripts.train_models --model-version v2 --hparam-json hparam_results/best_hparams.json
   ```
3. Smoke test tras cualquier cambio:
   ```bash
   curl -s -X POST http://localhost:8000/api/strategy \
     -H "Content-Type: application/json" \
     -d '{"year":2023,"circuit_id":"Sakhir","driver_id":14}' | python -m json.tool | head -20
   ```

---

## High Priority — Unlock Premium OpenF1 Access

- [ ] **Build a live-data ingestion trigger in the API**
  - Add `POST /api/admin/ingest?year=YYYY` endpoint that calls `ingest_season` + `build_features_for_year` in a background task (`fastapi.BackgroundTasks` or `asyncio.create_task`)
  - After completion, invalidate the `lru_cache` on `load_features()` so the engine picks up new data without a restart
  - This is the only way to actually use the premium key at runtime — currently the data pipeline is entirely offline-batch

- [ ] **Replace `requests` with `httpx` async client in `openf1_client.py`**
  - `OpenF1Client` uses blocking `requests` + `time.sleep()` inside an async FastAPI framework, stalling the event loop on every fetch
  - Replace with `httpx.AsyncClient` + `asyncio.sleep` and expose async versions of `ingest_season` and API handlers
  - Prerequisite for non-blocking live fetches and for running multiple workers without I/O stalls

- [ ] **Derive `pit_loss` and `sc_probability` from real OpenF1 data**
  - Both are currently hardcoded constants in `strategy_engine.py` (`pit_loss=22.5`, `sc_probability=0.2`) for every circuit and year
  - Use the OpenF1 pit timings endpoint to compute per-circuit `pit_loss`
  - Use SC/VSC event counts from race session data to estimate `sc_probability` per circuit
  - This directly improves simulation accuracy and is only possible with premium access

---

## Medium Priority — Structural Hygiene

- [ ] **Add a cache invalidation mechanism for `load_features()`**
  - `lru_cache(maxsize=4)` on `data_store.py:load_features()` never invalidates after re-ingestion while the server is running
  - Options: add a `POST /api/admin/reload` endpoint that calls `load_features.cache_clear()`, or switch to a module-level variable with a timestamp that `write_snapshot_state` updates

- [ ] **Fix `circuit_id` normalization in `preprocess.py`**
  - Current fallback chain (`circuit_short_name` → `location` → `meeting_key`) can produce different string IDs for the same circuit across years (e.g., `"Sakhir"` vs `"Bahrain"` vs `3`)
  - Create a canonical mapping from OpenF1 field variants to a stable string
  - Log a warning when a fallback is used so silent mismatches are detectable

- [ ] **Add Pydantic response models to all routes**
  - Routes currently return raw DataFrame records or untyped dicts, leaking internal schema to the frontend
  - Define response models for at least: driver objects (`/api/metadata/drivers`), strategy objects (`/api/strategy`), compare objects (`/api/compare`)
  - Enables auto-generated, accurate OpenAPI docs

- [ ] **Bound and harden the in-memory `_cache` in `main.py`**
  - The raw dict at `main.py:33` grows unboundedly (only TTL-expired entries evicted on read, nothing proactively purges)
  - Replace with `functools.lru_cache` on handlers or `cachetools.TTLCache` with a max size
  - Also dies on process restart and is invisible to multiple workers — consider whether Redis is warranted if deploying multi-worker

- [ ] **Improve ingestion error handling and observability**
  - `ingest.py:62-67` silently skips an entire session on any `RuntimeError`, with only a `print()` — no count of skipped sessions, no structured record
  - Track skipped sessions and expose them in `snapshot_state.json` (e.g., `skipped_sessions: [...]`)

---

## Lower Priority — Polish

- [ ] **Add structured logging throughout the backend**
  - Replace all `print()` calls in `ingest.py`, `preprocess.py`, and elsewhere with `logging.getLogger(__name__)`
  - Add a request-ID middleware to `main.py` so traces across a single request are correlatable

- [ ] **Make CORS origins env-driven**
  - `allow_origins` in `main.py` is a hardcoded localhost list — fine for dev, fragile for staging/prod
  - Move to a `CORS_ORIGINS` env var read in `config.py`, comma-separated

- [ ] **Deprecate and remove legacy routes**
  - Routes without `/api/` prefix (`/metadata/seasons`, `/strategy`, `/compare`, etc.) add noise to the OpenAPI spec and invite accidental use
  - Add deprecation notices, then remove once the frontend is fully migrated to `/api/`

- [ ] **Add query param validation to metadata endpoints**
  - `GET /api/metadata/circuits?season=9999` returns an empty list with 200 — no indication the season doesn't exist
  - Validate `season` against `seasons_available()` and return 404 if not found

- [ ] **Swap MD5 for SHA-256 in `openf1_client.py` cache keying**
  - `_cache_path` uses `hashlib.md5` for cache file naming (`openf1_client.py:51`)
  - Replace with `hashlib.sha256(...).hexdigest()[:32]` — trivial change, technically more correct

---

## Open Design Questions — Backend

- **`/api/strategy/{strategy_id}` endpoint**: strategies include a `strategy_id` SHA1 in responses, implying persistence — but there is no storage or retrieval mechanism. Decide: ephemeral IDs (rename/remove) or persist strategies to disk/DB for later retrieval.

- **Multi-worker deployment**: the in-memory `_cache` and `lru_cache` instances are per-process. If uvicorn is run with multiple workers, each worker has its own cache state. Decide whether a shared cache (Redis, memcached) is needed before deploying beyond single-worker.

---
---

## Frontend Evaluation — RaceScope UI

Generated from frontend analysis (2026-03-05). Ordered by priority.

---

## Critical — Layout Collisions & Overflow

- [ ] **Fix vertical overflow on short-screen desktops (laptops)**
  - `body` has `overflow: hidden` on desktop and `.rows-panel.two-fixed` hardcodes `grid-template-rows: 350px 350px` = 700px for the rows panel alone
  - On a 768px-tall screen (common laptop), the header (~50px) + nav-context-row (~60px) + gaps (24px) + two rows (700px) = ~834px — content is cut off with no scroll and no visual indication
  - Fix: make the grid rows use `minmax(280px, 1fr)` so they share available height, or switch to `min-height` with scroll on the rows panel

- [ ] **Add a scroll/fade indicator to the horizontal `strategy-strip`**
  - The strategy carousel overflows with `overflow-x: auto` but has no scrollbar styling, no fade gradient at the edges, and no visual cue that more cards exist off-screen
  - On small desktops and mobile where only 1-2 cards fit, users have no affordance to scroll

- [ ] **Clamp the hover tooltip to prevent it clipping at chart edges**
  - `curve-tooltip` uses `transform: translate(-50%, -120%)` — at the leftmost and rightmost laps, the tooltip overflows outside the chart wrapper visually
  - Add horizontal clamping: when `hover.x / width < 0.15`, anchor tooltip to the right; when `hover.x / width > 0.85`, anchor to the left

- [ ] **Intermediate breakpoint (920px–1200px) leaves charts too narrow**
  - At these widths, `.driver-row` is `220px 1fr` — on a 1000px screen the chart panel is ~770px but also contains the strategy stack with multiple 220px+ tall cards
  - The `curve-wrapper` stays at 140px height regardless, making axis labels (9px) and chart lines very compressed
  - Consider adding a breakpoint around 1100px to collapse the rail into a top strip earlier

---

## High Priority — Mobile & Touch Experience

- [ ] **Add touch support for chart tooltips**
  - `StrategyCurveChart` uses `onMouseMove` / `onMouseLeave` for the interactive tooltip — these events do not fire on touch devices
  - The hover tooltip (lap time, compound, tyre life, pit window) is completely inaccessible on phones and tablets
  - Replace or supplement with `onPointerMove` / `onPointerLeave` (already used for the app-shell spotlight), which handles both mouse and touch

- [ ] **Make the hologram card tilt work on touch**
  - `.hologram-card` applies a 3D perspective tilt via CSS vars `--mx`/`--my` driven by `onShellMove` in `HomeLanding.jsx`, which only listens to pointer (mouse) events
  - On touch devices the card never tilts — add `onTouchMove` handler or use `onPointerMove` for unified behavior
  - If touch is not added, ensure the static state (`--mx: 50%, --my: 50%`) renders neutrally (it currently does, so this is a polish issue)

- [ ] **Minimum touch target size for small buttons**
  - `.icon-btn` is 28×28px — below the WCAG 2.5.5 recommended 44×44px and Apple/Google HIG 44pt/48dp minimums
  - `.data-mode-chip` at 26px height is tappable only with precision — too small for touch
  - The tab buttons at `padding: 8px 14px` with 13px text are borderline; verify they hit 44px height on mobile

- [ ] **Prevent `.pre-race-context-bubble` inner padding collapse on mobile**
  - At <920px, the bubble becomes a full-width rectangle (radius: 16px, width: 100%) but `padding: 6px` remains — the selects and CTA stacked vertically have only 6px breathing room on each side
  - Increase mobile padding to at least 12px and verify the CTA button is full-width

---

## High Priority — Legibility & Type Scale

- [ ] **Increase minimum font sizes for sub-12px text**
  - The following elements use font sizes below 12px, which fail WCAG 1.4.4 and are illegible on non-retina screens:
    - `.axis-caption`: 9px SVG text — increase to 11px minimum
    - `.data-mode-chip`: 10px — increase to 11px
    - `.strategy-kind`: 10px — increase to 11px
    - `.insight-item small`: 10px uppercase — increase to 11px
    - `.legend-item`: 10px — increase to 11px
    - `.holo-label`: 11px — acceptable, keep
  - At 10px, these elements have contrast ratios that can fall below WCAG AA (4.5:1) for muted colors on dark panels

- [ ] **Verify muted-text contrast ratios in both themes**
  - Dark theme: `--muted: #9ea6b1` on `--panel: #16181b` ≈ 4.5:1 contrast — passes AA for 14px+ but fails for 10–12px elements
  - Light theme: `--muted: #667587` on `--panel: #ffffff` ≈ 5.3:1 — better, but still marginal for 10px elements
  - Either darken `--muted` in dark mode to ~#b0bac6 (≈5.5:1) or ensure no informational text runs below 12px

- [ ] **Disambiguate the two `DriverRow` sections**
  - Both rows show `<h3>Piloto</h3>` as the section title and "Equipo" / "Piloto" as the labels — when two rows are visible simultaneously there is no visual indication of "Row 1" vs "Row 2" or any driver identity at the row header level
  - Add a row number badge ("A" / "B" or "1" / "2") to the `row-head`, and once a driver is selected, show their code/name prominently in the header

---

## Medium Priority — Accessibility (a11y)

- [ ] **Fix incorrect ARIA roles in `StrategyStrip`**
  - `strategy-strip` has `role="listbox"` but child `<button>` elements don't have `role="option"` — this is invalid ARIA; listbox items must be `option`
  - Either change to `role="list"` + `role="listitem"` on buttons, or use a proper `role="listbox"` + `role="option"` with `aria-selected` on each option

- [ ] **Add `role="tab"` and `aria-selected` to `TopTabs`**
  - Tab buttons in `TopTabs` use plain `<button>` with an `"active"` class — screen readers cannot identify these as tabs or know which is selected
  - Add `role="tablist"` to the nav, `role="tab"` to each button, and `aria-selected={activeTab === tab}` to each

- [ ] **Add `aria-live` regions for async state changes**
  - When strategy calculation completes or fails, there is no announcement for screen readers — the result silently appears
  - Add an `aria-live="polite"` region that announces loading/error/ready states for each driver row

- [ ] **Add keyboard navigation to the strategy strip**
  - `StrategyStrip` buttons are focusable but there is no arrow-key navigation between them — ARIA `listbox` pattern requires arrow keys to move selection
  - If using the corrected `role="list"` approach, ensure Tab moves through cards naturally; if using `listbox`, implement arrow key handlers

---

## Medium Priority — Design Consistency & Code Health

- [ ] **Unify compound color definitions — currently defined in 3 places**
  - `--soft`, `--medium`, `--hard` CSS variables in `styles.css`
  - `COMPOUND_COLORS` object in `StrategyCurveChart.jsx`
  - `.holo-soft`, `.holo-medium`, `.holo-hard` CSS classes in `styles.css`
  - Any color change requires 3 synchronized edits; create a single source of truth (CSS custom properties, consumed by both JS and CSS via `getComputedStyle` or a shared JS constant file)

- [ ] **Externalize `TEAM_COLORS` from `DriverRow.jsx` into a shared constants file**
  - Team colors are hardcoded in the component with partial name matching (`teamName?.includes(k)`) — fragile if team names change
  - This data belongs with the CSS design tokens, not in a component file

- [ ] **Split `styles.css` into component-scoped stylesheets or CSS modules**
  - At 1100+ lines, the single stylesheet will cause maintenance problems as more tabs are implemented
  - Risk of unintended style bleed: global `label`, `select`, `button` rules override browser defaults globally and will affect any future form elements
  - At minimum, scope component styles with a parent class prefix (`.home-shell .home-kpi`, etc.)

- [ ] **Remove or wire up orphaned components**
  - `ComparisonView.jsx`, `ControlPanel.jsx`, `DegradationChart.jsx`, `StrategyCard.jsx`, `StrategyTimeline.jsx` exist in `src/components/` but are not imported anywhere in the active code
  - Either integrate them or remove them to reduce confusion; having dead components makes it unclear what the intended UI is

- [ ] **Lift `App.jsx` state into context or a reducer**
  - `App.jsx` has 15+ `useState` hooks managing season, circuits, drivers, teams, rows, running state, metadata status — this will become unmanageable as Live/Rewatch/Explore tabs are implemented
  - Extract into a `useReducer` or a lightweight context (`SeasonContext`, `RowsContext`) to keep `App.jsx` as a layout shell only

---

## Lower Priority — UX Polish

- [ ] **Show delta-to-best instead of absolute time in strategy cards**
  - `strategy-strip` and `strategy-curve-card` display `formatRaceDuration(expected_time)` (e.g., "1h 35m 42s") for each strategy — users must mentally subtract to compare options
  - Display the best strategy as the baseline and show "+0:33", "+1:12" for the rest; keep absolute time in a secondary position

- [ ] **Add a per-row retry button on strategy calculation error**
  - When a row's status is `"error"`, the only recovery is to press "Calcular" again for both rows simultaneously
  - Add a small retry icon/button on the errored row's rail so the user can re-run that driver's calculation independently

- [ ] **Persist selected season/circuit across tab switches**
  - Navigating away from Pre-race and back resets nothing (state is in `App.jsx`), but switching to other tabs and back to Pre-race while a calculation is in progress shows stale UI
  - Consider `sessionStorage` for the last-used season/circuit so a page refresh also restores context

- [ ] **Add a loading spinner or progress indicator to the "Calcular" button**
  - The button text changes from "Calcular" to "Calculando" during computation, but there is no animated indicator — on slow connections users may think the click did not register
  - Add a spinner icon next to "Calculando" text, or use the existing `chart-shimmer` animation keyframes on the button

- [ ] **Display `stale_data` warning more prominently**
  - The `SNAPSHOT` chip on the context bubble is small (10px, tucked to the right) and easy to miss — stale data affects all strategy outputs and the user needs to know this clearly
  - Consider a banner/toast below the nav-context-row when `stale_data` is true

---

## Open Design Questions — Frontend

- **`force_recompute: true` always**: every "Calcular" press forces full recomputation, bypassing the server cache. This is intentional for demo freshness but means there is no way for users to get fast cached results. Decide: add a "Use cache" toggle, or remove `force_recompute` from the frontend call once live data is reliable.

- **Two-row layout vs. free-selection**: the UI is hardcoded to exactly 2 driver rows. Consider whether the design should support 1 (single analysis) or 3+ (team comparison) rows, or explicitly commit to the 2-row comparative design and add visual differentiation (color-coded row borders using `--team-tint`).

- **Unimplemented tabs (Live, Rewatch, Explore)**: all three show a placeholder. Before adding tabs to the nav that do nothing, decide whether to hide them until implemented (avoids user confusion) or keep as roadmap indicators with a clear "Coming soon" state that is more informative than a generic placeholder.
