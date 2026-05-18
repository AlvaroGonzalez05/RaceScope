# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RaceScope Strategy Lab** — a Formula 1 pre-race strategy explorer. Given a season, circuit, and driver, it simulates and ranks pit-stop strategies using a two-phase engine: analytical scoring of all candidates, followed by Monte Carlo refinement of only the top-K.

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

Smoke test (stable case: 2023 / Sakhir / driver 14):
```bash
curl -s -X POST http://localhost:8000/api/strategy \
  -H "Content-Type: application/json" \
  -d '{"year":2023,"circuit_id":"Sakhir","driver_id":14}'
```

## Running the Frontend

```bash
cd code/frontend
npm install
npm run dev           # Vite dev server on :5173
npm run build         # Build dist/ (served by FastAPI on same origin)
```

`VITE_API_BASE` env var sets the backend URL (default: `http://localhost:8000`).

For a single-origin demo (no CORS issues), build first then start only the FastAPI server — it auto-serves `frontend/dist/` at `/`.

---

## Data Pipeline (must run once before the API has data)

All scripts run from `code/backend_fastapi/` with the venv active:

```bash
python -m scripts.ingest_season --year 2023 --sleep-s 1.5 --min-interval 1.2
python -m scripts.preprocess --year 2023
python -m scripts.train_models --min-laps 260 --epochs 12
python -m scripts.train_profiles --min-laps 160
```

For the Transformer v2 model (recommended), use:
```bash
python -m scripts.train_models --model-version v2 --epochs 15 --patience 5
```

Optional hyperparameter search (requires `optuna`):
```bash
python -m scripts.hparam_search --n-trials 50 --timeout-hours 4 --output hparam_results/
python -m scripts.train_models --model-version v2 --hparam-json hparam_results/best_hparams.json
```

Outputs:
- `data/raw/year=<YYYY>/.../*.parquet` — raw OpenF1 data
- `data/features/year=<YYYY>/features.parquet` — ML feature store (includes `push_index` and `race_lap_norm` computed at preprocess time)
- `data/features/metadata/` — drivers/teams/circuits parquets + `snapshot_state.json`
- `models/driver_<id>.joblib`, `models/global.joblib` — Transformer v2 (or LSTM) models
- `models/driver_profile_<id>.joblib`, `models/driver_profile_global.joblib` — parametric profiles
- `models/logs/training_<timestamp>.csv` — per-epoch train/val loss CSV (v2 training)
- `cache/pace_curves/` — per-request pace curve cache (can be deleted safely)

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

---

## Architecture

### Backend (`app/`)

| File | Role |
|---|---|
| `main.py` | FastAPI app, routes, in-process cache (24-hour TTL), SPA static serving |
| `config.py` | All paths and tunable constants; reads `.env` at import time |
| `schemas.py` | Pydantic response models (`StrategyResponse`, `CompareResponse`, `DriverOut`, etc.) |
| `data_store.py` | Loads parquet features/metadata via `lru_cache`; manages `snapshot_state.json` |
| `strategy_engine.py` | Two-phase engine: analytical scoring → MC top-K refinement |
| `driver_profile.py` | Parametric pace model per driver/circuit/compound with 4-level fallback |
| `models_lstm.py` | LSTM model wrapper (lazy-loaded at first strategy request) |
| `models_transformer.py` | Transformer pace models: legacy `TyreTransformerNet` (v1) + `TyreDegradationTransformerV2` (current). Wrapper: `TransformerPaceModel`. Rollout: `rollout_mc` dispatches per model version. |
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
| `MC_TOP_K` | 5 | How many strategies get Monte Carlo refinement |
| `DEFAULT_RISK_LAMBDA` | 0.15 | Default risk bias for strategy ranking |
| `DEFAULT_STRATEGY_COUNT` | 5 | Strategies returned per request |
| `CACHE_TTL_SECONDS` | 86400 | Disk pace-curve cache TTL |
| `RANDOM_SEED` | 42 | Reproducibility for MC simulation |
| `TRANSFORMER_V2_D_MODEL` | 256 | Transformer v2 model dimension |
| `TRANSFORMER_V2_N_HEADS` | 8 | Number of attention heads |
| `TRANSFORMER_V2_N_LAYERS` | 6 | Number of encoder layers |
| `TRANSFORMER_V2_DIM_FF` | 1024 | Feedforward dimension |
| `TRANSFORMER_V2_DROPOUT` | 0.1 | Dropout rate |
| `TRANSFORMER_V2_CONTEXT_LAPS` | 25 | Autoregressive context window (laps) |
| `TRANSFORMER_V2_INPUT_DIM` | 14 | Continuous features per timestep |
| `TRANSFORMER_V2_AUX_LOSS_W` | 0.15 | Weight of absolute lap-time head in loss |
| `TRANSFORMER_V2_DEG_LOSS_W` | 0.10 | Weight of degradation-rate head in loss |
| `TRANSFORMER_V2_N_SIM` | 500 | MC simulations per strategy (v2; v1 uses 200) |
| `CIRCUIT_VOCAB` | dict (24 entries) | Circuit name → integer ID for circuit embedding |
| `CIRCUIT_EXPECTED_LAPS` | dict (24 entries) | Expected race laps per circuit (for `race_lap_norm`) |

---

## Troubleshooting

- **API returns 400**: no features loaded — run the pipeline first.
- **Import freeze on startup**: create a fresh `.venv_demo` and reinstall. Quick check: `.venv_demo/bin/python -c "import pandas, numpy, scipy, torch; print('ok')"`
- **Port busy**: `pkill -f "uvicorn app.main:app"`
- **Stale cache after pipeline re-run**: delete `cache/pace_curves/*` and call with `"force_recompute": true`.
- **OpenF1 401 during ingestion**: set `OPENF1_USERNAME` and `OPENF1_PASSWORD` in `.env`.
- **v2 model: `KeyError` on circuit_id**: circuit not in `CIRCUIT_VOCAB` — defaults to 0 (unknown). Add the circuit key if it's a new venue.
- **v2 model: missing telemetry columns** (`mean_throttle`, `mean_brake`, `max_speed`, etc.): re-run `scripts/preprocess.py` — older feature parquets may lack these columns. The model falls back to zeros if columns are absent at training time.
- **`hparam_search` fails with `ImportError: optuna`**: `pip install "optuna>=3.0,<4.0"` (already in `requirements.txt`; ensure venv is up to date).
- **v2 training: `val_mae` not improving / early stopping after 5 epochs**: expected on tiny datasets; use `--patience 0` to disable early stopping for smoke tests.
- **`evaluate_2025.py` reports no 2025 data**: run `python -m scripts.ingest_season --year 2025` then `python -m scripts.preprocess --year 2025` first. 2025 data is never loaded during training.

## Scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `ingest_season.py` | Download raw OpenF1 data for a season into `data/raw/` |
| `preprocess.py` | Build gold-layer feature parquet (includes `push_index`, `race_lap_norm`) |
| `train_models.py` | Train per-driver (+ global fallback) Transformer v2 models. Key flags: `--model-version v2`, `--val-frac 0.15`, `--patience 5`, `--hparam-json` |
| `train_profiles.py` | Train lightweight parametric driver profiles (4-level fallback) |
| `hparam_search.py` | Optuna hyperparameter search for `TyreDegradationTransformerV2`. 50 trials × 8 epochs; outputs `best_hparams.json` |
| `evaluate_2025.py` | Held-out evaluation on 2025 data. Reports strategy match rate, pace MAE/RMSE. `--push-sensitivity-test` flag checks degradation slope at attack vs conservation pace |
| `benchmark_strategy.py` | Latency benchmark (cold/warm/hot). Writes `benchmark_report.json` |

---

## Benchmark

```bash
cd code/backend_fastapi
.venv_demo/bin/python scripts/benchmark_strategy.py
```

Results written to `benchmark_report.json`. Metrics: `cold`, `warm`, `hot` latencies.
