# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RaceScope Strategy Lab** — a Formula 1 pre-race strategy explorer. Given a season, circuit, and driver, it generates and ranks pit-stop strategies with three collaborating motors: a linear `DriverProfile` solves the analytical break-even (when to pit), a Transformer v3 predicts non-linear pace and refines the Top-K via Monte Carlo, and a mean-variance heuristic orders all candidates. See `code/backend_fastapi/STRATEGY_CONSTRUCTION.md` for the full pipeline.

The repo has two runnable parts:
- `code/backend_fastapi/` — Python FastAPI backend (data pipeline + simulation API)
- `code/frontend/` — React + Vite SPA

Datasets, model artefacts, and cache are **not versioned**. They must be regenerated locally.

---

## Running the Backend

```bash
cd code/backend_fastapi
python3.11 -m venv .venv_demo
source .venv_demo/bin/activate
pip install -r requirements.txt
cp .env.example .env          # credentials already set in .env — do not overwrite if it exists
```

Start the API:
```bash
uvicorn app.main:app --reload --port 8000
```

Smoke test (caso estable: 2023 / Sakhir / VER):
```bash
curl -s -X POST http://localhost:8000/api/strategy \
  -H "Content-Type: application/json" \
  -d '{"year":2023,"circuit_id":"Sakhir","driver_code":"VER"}'
```

## Running the Frontend

```bash
cd code/frontend
npm install
npm run build  # compila dist/, el backend la sirve en localhost:8000
```

`VITE_API_BASE` está vacío. El frontend usa URLs relativas (`/api/...`).
FastAPI sirve `frontend/dist/` como archivos estáticos en el mismo origen.

---

## Data Pipeline (must run once before the API has data)

All scripts run from `code/backend_fastapi/` with the venv active:

```bash
python -m scripts.ingest_season --year 2023 --sleep-s 1.5 --min-interval 1.2
python -m scripts.preprocess --year 2023
python -m scripts.train_models --model-version v3 --epochs 30 --patience 5
python -m scripts.train_profiles --min-laps 160
```

Outputs (arquitectura medallion):
- `data/bronze/year=<YYYY>/` — JSON crudo de OpenF1 (nunca transformado)
- `data/silver/year=<YYYY>/` — Parquet limpio y tipado
- `data/gold/year=<YYYY>/features.parquet` — feature store ML
- `data/gold/metadata/` — drivers/teams/circuits parquets + `snapshot_state.json`
- `models/driver_<code>.joblib`, `models/global.joblib` — Transformer v3 por piloto
- `models/driver_profile_<code>.joblib` — perfiles paramétricos
- `models/logs/` — CSV por época de entrenamiento
- `cache/pace_curves/` — cache de curvas por petición (borrable)

**2025 evaluation** (held-out season — never used in training):
```bash
python -m scripts.evaluate_2025 --push-sensitivity-test
```

---

## Testing

Run all backend tests from `code/backend_fastapi/` with the venv active:

```bash
cd code/backend_fastapi
source .venv_demo/bin/activate
pytest                          # all tests
pytest tests/test_strategy_engine.py          # single file
pytest tests/test_strategy_engine.py::test_pit_loss_fallback  # single test
pytest -x                       # stop on first failure
pytest -v                       # verbose output
```

Tests use `pytest-asyncio` in auto mode (configured in `pytest.ini`). Test data is synthetic (no pipeline data required).

Key test names in `tests/test_transformer_model.py`:

| Test | What it checks |
|---|---|
| `test_forward_pass` | v1 model forward pass (13 features, d_model=256) |
| `test_train_synthetic_v1` | v1 training loop completes without error |
| `test_rollout_mc_v1` | v1 `rollout_mc` returns array of correct length |
| `test_lap_type_classification` | `classify_lap_type` rules |
| `test_backward_compat_lstm_load` | Old LSTM `.joblib` still loads via `model_type` absence |
| `test_v2_forward_pass` | v2 forward pass: 14 cont. features + 3 cat + circuit_idx → 3 heads |
| `test_v2_rollout_mc_push_sensitivity` | Higher `pace_intent` → steeper degradation slope |
| `test_v2_circuit_differentiation` | Same inputs, different `circuit_int` → different predictions |
| `test_v2_backward_compat` | v1 `.joblib` loads without error when v2 code is present |
| `test_v2_train_synthetic` | v2 training 3 epochs, val_mae tracked, early stopping param accepted |
| `test_build_context_seed_v2` | `build_context_seed` returns `(T, 14)` array |
| `test_practice_distribution_defaults` | `PracticeDistribution` defaults: `pace_lo > pace_hi` (raw seconds) |
| `test_v3_forward_pass` | v3 forward pass: 15 cont. features + 3 cat + circuit_idx + compound_idx_static → 4 heads |
| `test_v3_rollout_mc_push_sensitivity` | Higher `pace_intent` → steeper degradation slope in v3 batched rollout |
| `test_v3_circuit_differentiation` | Same inputs, different `circuit_int` → different v3 predictions |
| `test_v3_build_context_seed` | `build_context_seed_v3` returns `(T, 15)` array; column 14 = `stint_number_norm` |
| `test_v3_stop_profitability_positive` | `_stop_profitability` returns positive profit when fresh tyre saves time |
| `test_v3_stop_profitability_negative` | `_stop_profitability` returns negative profit when pit cost exceeds tyre gain |

Key test names in `tests/test_strategy_engine.py` (analytical redesign):

| Test | What it checks |
|---|---|
| `TestPitLossClamp15_45::test_clamp_lower_at_15` | `_derive_pit_loss` clamped at new lower bound 15 s |
| `TestPitLossClamp15_45::test_clamp_upper_at_45` | `_derive_pit_loss` clamped at new upper bound 45 s |
| `TestPaceTable::test_temperature_correction_applied` | `_pace_table` adds `track_coef · ΔT_track` to `base` |
| `TestPaceTable::test_returns_all_three_compounds` | `_pace_table` returns SOFT, MEDIUM, HARD entries |
| `TestBreakEvenAnalytic::test_1stop_optimum_matches_closed_form` | Closed-form `s*` matches numerical minimum within 1 lap |
| `TestBreakEvenAnalytic::test_2stop_optimum_close_to_numerical` | 2x2 linear system matches numerical minimum within 2 laps |
| `TestCandidateGeneration::test_unprofitable_stop_dropped` | Candidates with no profitable stop are filtered before ranking |
| `TestCandidateGeneration::test_at_least_two_compounds` | Every generated candidate uses ≥2 distinct compounds (F1 rule) |
| `TestCandidateGeneration::test_at_least_one_stop` | Every generated candidate has ≥1 stop (F1 rule) |
| `TestHardStartFilteredOut::test_no_hard_start_in_final_strategies` | HARD-start hard-filtered from payload |

---

## Architecture

### Backend (`app/`)

| File | Role |
|---|---|
| `main.py` | FastAPI app, routes, in-process cache (24-hour TTL), SPA static serving |
| `config.py` | All paths and tunable constants; reads `.env` at import time |
| `schemas.py` | Pydantic response models (`StrategyResponse`, `CompareResponse`, `DriverOut`, etc.) |
| `data_store.py` | Loads parquet features/metadata via `lru_cache`; manages `snapshot_state.json` |
| `strategy_engine.py` | Three motors: `_pace_table` (linear profile) generates candidates by closed-form break-even; `_analytical_eval` ranks all with `mean + λ·var`; `_simulate_strategy` refines Top-K via Transformer MC. HARD-start hard-filtered from output. |
| `driver_profile.py` | Parametric pace model per driver/circuit/compound with 4-level fallback (source of truth for `_pace_table`) |
| `models_lstm.py` | LSTM model wrapper (lazy-loaded at first strategy request) |
| `models_transformer.py` | Transformer pace models: `TyreTransformerNet` (v1), `TyreDegradationTransformerV2`, `TyreDegradationTransformerV3` (current, d_model=384, 8 layers, 15 features, 4 heads). Wrapper: `TransformerPaceModel`. Rollout: `rollout_mc` dispatches per model version. |
| `train.py` | Internal training logic called by `scripts/train_models.py` |
| `ingest.py` | OpenF1 HTTP client logic used by `scripts/ingest_season.py` |
| `preprocess.py` | Feature engineering logic used by `scripts/preprocess.py` |

The scientific stack (pandas, numpy, torch, scipy) is **lazy-imported** inside request handlers to keep API startup fast.

**Known schema mismatch**: `CompareResponse` in `schemas.py` declares `driver_a`/`driver_b` fields, but `_post_compare` in `main.py` returns `driver`/`teammate` keys. The response still serialises correctly because `extra="allow"` is set, but the typed fields are unused.

### Stable API routes (use `/api/` prefix)

- `GET /api/metadata/seasons`
- `GET /api/metadata/circuits?season=YYYY`
- `GET /api/metadata/drivers?season=YYYY`
- `GET /api/metadata/teams?season=YYYY`
- `POST /api/strategy` — single driver strategy ranking
- `POST /api/compare` — two-driver head-to-head comparison
- `POST /api/admin/ingest` — trigger live data ingestion for a season (background task)
- `GET /api/admin/ingest/status` — poll ingestion/snapshot state

Legacy routes without `/api/` exist for temporary compatibility but should not be extended.

### Data/snapshot modes

`OPENF1_AUTH_ENABLED` in `.env` controls whether the API tries to hit OpenF1 live:
- `false` (default) → snapshot mode; serves local parquet data, `stale_data=false`
- `true` → live mode (premium credentials configured); if OpenF1 fails and local data exists, falls back to snapshot with `stale_data=true`

**Premium API rate limits:** 6 req/s burst, 60 req/min sustained. Enforced by a token-bucket in `openf1_client.py`. Config constants: `OPENF1_RATE_PER_SECOND`, `OPENF1_RATE_PER_MINUTE` (set in `.env`).

### Frontend (`src/`)

State and routing live entirely in `App.jsx` (no router library). Tabs: `Home | Pre-race | Live | Rewatch | Explore`. Only `Home` and `Pre-race` are implemented.

All application state is managed via `useReducer` + `state/appReducer.js`. Every state field (theme, activeTab, season, circuitId, rows, running, …) is updated by dispatching named action constants exported from that file. Never add local `useState` for fields that belong to the global session — add them to `appReducer.js` instead.

`src/constants/` holds static lookup data: `compounds.js` (tyre compound colours/labels) and `teams.js` (team colour map).

Key components:
- `PreRaceContextBubble.jsx` — season/circuit selectors + "Calcular" trigger (visible only on Pre-race tab)
- `DriverRow.jsx` — per-driver row: strategy strip + curve chart
- `StrategyStrip.jsx` — horizontal carousel of ranked strategies
- `StrategyCurveChart.jsx` — recharts lap-time curve by stint
- `HomeLanding.jsx` — animated landing (framer-motion)

Design principle: the Pre-race strategy area uses full viewport width (no side panel). `Calcular` is the only simulation trigger.

---

## Key Configuration Constants (`config.py`)

| Constant | Default | Meaning |
|---|---|---|
| `MC_TOP_K` | 3 | How many strategies get Monte Carlo refinement |
| `DEFAULT_RISK_LAMBDA` | 0.15 | Default risk bias for strategy ranking |
| `DEFAULT_STRATEGY_COUNT` | 5 | Strategies returned per request |
| `PIT_LOSS_FALLBACK` | 22.5 | Default pit-loss when no data |
| `PIT_LOSS_MIN` | 15.0 | Hard lower clamp on `_derive_pit_loss` |
| `PIT_LOSS_MAX` | 45.0 | Hard upper clamp on `_derive_pit_loss` |
| `PIT_WINDOW_BIN` | 5 | Bin (laps) for `_cluster_key` strategy dedup |
| `CACHE_TTL_SECONDS` | 86400 | Disk pace-curve cache TTL |
| `RANDOM_SEED` | 42 | Reproducibility for MC simulation |
| `TRANSFORMER_V3_D_MODEL` | 384 | Transformer v3 model dimension |
| `TRANSFORMER_V3_N_HEADS` | 8 | Number of attention heads |
| `TRANSFORMER_V3_N_LAYERS` | 8 | Number of encoder layers |
| `TRANSFORMER_V3_DIM_FF` | 1536 | Feedforward dimension |
| `TRANSFORMER_V3_DROPOUT` | 0.0 | Dropout (intentional overfit per driver) |
| `TRANSFORMER_V3_CONTEXT_LAPS` | 40 | Autoregressive context window (laps) |
| `TRANSFORMER_V3_INPUT_DIM` | 15 | Continuous features per timestep (adds `stint_number_norm`) |
| `TRANSFORMER_V3_AUX_LOSS_W` | 0.10 | Weight of absolute lap-time head in loss |
| `TRANSFORMER_V3_DEG_LOSS_W` | 0.25 | Weight of degradation-rate head in loss |
| `TRANSFORMER_V3_VALUE_LOSS_W` | 0.20 | Weight of remaining-stint cost head in loss |
| `TRANSFORMER_V3_N_SIM` | 100 | MC simulations per strategy |
| `CIRCUIT_VOCAB` | dict (24 entries) | Circuit name → integer ID for circuit embedding |
| `CIRCUIT_EXPECTED_LAPS` | dict (24 entries) | Expected race laps per circuit (for `race_lap_norm`) |

---

## Troubleshooting

- **API returns 400**: no features loaded — run the pipeline first.
- **Import freeze on startup**: create a fresh `.venv_demo` and reinstall. Quick check: `.venv_demo/bin/python -c "import pandas, numpy, scipy, torch; print('ok')"`
- **Port busy**: `pkill -f "uvicorn app.main:app"`
- **Stale cache after pipeline re-run**: delete `cache/pace_curves/*` and call with `"force_recompute": true`.
- **OpenF1 401 during ingestion**: set `OPENF1_USERNAME` and `OPENF1_PASSWORD` in `.env`.
- **`KeyError` on circuit_id**: circuit not in `CIRCUIT_VOCAB` — defaults to 0 (unknown). Add the circuit key if it's a new venue.
- **Missing telemetry columns** (`mean_throttle`, `mean_brake`, `max_speed`, etc.): re-run `scripts/preprocess.py` — older feature parquets may lack these columns. The model falls back to zeros if columns are absent at training time.
- **`hparam_search` fails with `ImportError: optuna`**: `pip install "optuna>=3.0,<4.0"` (already in `requirements.txt`; ensure venv is up to date).
- **Training: `val_mae` not improving / early stopping after 5 epochs**: expected on tiny datasets; use `--patience 0` to disable early stopping for smoke tests.
- **`evaluate_2025.py` reports no 2025 data**: run `python -m scripts.ingest_season --year 2025` then `python -m scripts.preprocess --year 2025` first. 2025 data is never loaded during training.
- **v3 model `KeyError` on compound_idx_static**: compound integer out of range (vocab size 10). Check that compound names map to valid entries in `COMPOUND_VOCAB`.

## Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `ingest_season.py` | Download raw OpenF1 data for a season into `data/raw/` |
| `preprocess.py` | Build gold-layer feature parquet (includes `push_index`, `race_lap_norm`) |
| `train_models.py` | Train per-driver (+ global fallback) models. Key flags: `--model-version v3` (default), `--val-frac 0.15`, `--patience 5`, `--hparam-json`. Choices: `v1`, `v2`, `v3`. |
| `train_profiles.py` | Train lightweight parametric driver profiles (4-level fallback) |
| `hparam_search.py` | Optuna hyperparameter search for `TyreDegradationTransformerV2`. 50 trials × 8 epochs; outputs `best_hparams.json` |
| `benchmark_architectures.py` | Comparative benchmark (v1/v2/v3): params, train time/epoch, single and batched inference, peak RAM. Results in `reports/benchmark_architectures.csv` |
| `evaluate_2025.py` | Held-out evaluation on 2025 data. Reports strategy match rate, pace MAE/RMSE. `--push-sensitivity-test` flag checks degradation slope at attack vs conservation pace |
| `benchmark_strategy.py` | Latency benchmark (cold/warm/hot). Writes `benchmark_report.json` |

---

## Benchmark

```bash
cd code/backend_fastapi
.venv_demo/bin/python scripts/benchmark_strategy.py
```

Results written to `benchmark_report.json`. Metrics: `cold`, `warm`, `hot` latencies.

---

## Writing Style — Memoria del TFG (`memoria/src/`)

All prose written for the TFG LaTeX chapters must comply with the style guide in `CLAUDE_BANNED.md`. That document is a personal writing guide cataloguing constructions and vocabulary to avoid. The full list is there; the patterns most likely to appear in academic Spanish writing are flagged below.

### Banned constructions (Spanish equivalents)

| Pattern | Example to avoid | Fix |
|---|---|---|
| "in the heart of" | "en el corazón de cualquier sistema..." | State the claim directly: "La predicción del ritmo es el componente central..." |
| Trailing participle pile-up | "[clause], permitiendo..., habilitando..." (consecutive) | Break into separate sentences or restructure |
| "not only... but also" | "no solo el rendimiento, sino también los patrones" | State directly: "tanto el rendimiento como los patrones" |
| "opens the door to" | "abre la puerta a un análisis más profundo" | State what it enables concretely |
| Puffery/importance asserting | "garantiza la autenticidad y precisión", "nivel importante de detalle" | Cut the assertion; show the fact that supports it |
| AI vocabulary cluster | "fundamental", "notable", "inherente", "profundo" as vague intensifiers | Use only when the word carries specific meaning; replace generics with concrete claims |
| Promotional framing | "se caracteriza por su capacidad para", "acceso integral a" | Direct statement of what the system does |

### Content accuracy rules

- Data paths in the memoria must match the live codebase: `data/gold/` (not `data/features/`), `models/` (not other paths).
- Chapter cross-references (`\ref{cap:...}`) must point to labels that exist in compiled `.tex` files. Currently active labels: `cap:introduccion`, `cap:estado_arte`, `cap:metodologia`, `cap:caso_estudio`, `cap:resultados`, `cap:conclusiones`, `ann:ods`. There is no `cap:implementacion`.
- Chapters 5 and 6 (`chapter5.tex`, `chapter6.tex`) are commented out in `main.tex` — do not reference them as if they exist.
- `chapter7.tex` duplicates the label `cap:conclusiones` from `chapter6.tex` — one must be removed or relabelled before both are compiled.
- The architecture description in chapters 2–4 currently describes the LSTM model (v1). The live codebase uses `TyreDegradationTransformerV3` (medium). Any update to the architecture section must reflect v3 specifics: 15 continuous features, d_model=384, 8 layers, Circuit×Compound multiplicative gate, 4 output heads (`delta`, `abs`, `deg`, `value`), context_len=40, ~14.3M parameters.
