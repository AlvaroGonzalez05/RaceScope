# Changelog — RaceScope Strategy Lab (backend)

All notable changes to the FastAPI backend are documented here.
Format: most recent entry first.

---

## [0.4.0] — 2026-05-25

### Added — Transformer v3 medium + stop profitability engine

#### `app/models_transformer.py`
- `TyreDegradationTransformerV3` — 14.3M param architecture (d_model=384, 8 layers, dim_ff=1536).
  Key innovations vs v2: Circuit×Compound multiplicative gate, 4th output head (`head_value`),
  15th feature `stint_number_norm`, context_len=40, dropout=0.0 (intentional overfit per driver).
- `SequenceDatasetV3` — 8-element dataset for v3 training (adds `compound_static` and `y_value`).
- `build_context_seed_v3` — context seed builder with `stint_number` argument.
- `_build_sequences_v3`, `_train_v3`, `_rollout_v3`, `_rollout_v3_batched` — full v3 training
  and inference path mirroring the v2 batched structure.

#### `app/strategy_engine.py`
- Added v3 branch in `_simulate_strategy()` using `_rollout_v3_batched` and
  `build_context_seed_v3` with per-stint `stint_number` argument.
- Existing v2 branch demoted to `elif isinstance(model.model, TyreDegradationTransformerV2)`.
- `_stop_profitability(candidate, curves, pit_loss)` — per-stop marginal profitability:
  `profit = E[t_old for L laps] - E[t_fresh for L laps] - pit_loss`.
  Positive = stop is cost-effective.
- `stop_profitability` field added to each strategy dict in `generate_strategies()`.

#### `app/schemas.py`
- `StrategyOut.stop_profitability: list[float] = []` — per-stop profit in seconds.

#### `app/config.py`
- 11 new `TRANSFORMER_V3_*` constants (d_model=384, n_heads=8, n_layers=8, dim_ff=1536,
  dropout=0.0, context_laps=40, input_dim=15, aux_loss_w=0.10, deg_loss_w=0.25,
  value_loss_w=0.20, n_sim=100).

#### `app/train.py`
- v3 dispatch: defaults application, input_dim=15, `model_type="transformer_v3"`.

#### `scripts/train_models.py`
- `--model-version` choices extended to `["v1", "v2", "v3"]`.

#### `scripts/benchmark_architectures.py` (new)
- Comparative benchmark: parameters, train time/epoch, single and batched inference,
  peak RAM. Results in `reports/benchmark_architectures.csv`.

#### `tests/test_transformer_model.py`
- 6 new v3 tests: `test_v3_forward_pass`, `test_v3_rollout_mc_push_sensitivity`,
  `test_v3_circuit_differentiation`, `test_v3_build_context_seed`,
  `test_v3_stop_profitability_positive`, `test_v3_stop_profitability_negative`.

### Changed
- `vite.config.js` — Vite proxy added (`/api`, `/strategy`, `/compare` → `:8000`).
  Eliminates CORS in dev mode. `VITE_API_BASE` cleared (relative URLs in both modes).
- `driver_id` field removed from API request schemas; `driver_code` is now the sole
  driver identifier (3-letter code: VER, HAM, LEC…).
- Pre-existing test failures from `driver_id` refactor fixed:
  `test_strategy_route_returns_400_without_features`,
  `test_driver_out_optional_fields`,
  `test_legacy_post_strategy_logs_deprecation`.

---

## [Unreleased → 0.3.0] — 2026-05-25

### Performance — Inference speed optimization (~100× speedup on strategy endpoint)

**Context**  
The `/api/strategy` endpoint was taking 5–10 minutes per request. Root cause:
125 000 sequential Transformer forward passes (500 MC simulations × 5 strategies ×
~50 laps), each with batch size 1, on CPU. No batching, no parallelism.

**Changes**

#### `app/models_transformer.py`
- Added `TransformerPaceModel._rollout_v2_batched()` — vectorised Monte Carlo rollout
  that processes all *n_sim* simulations in a single batched forward pass per lap step
  (batch size = n_sim) instead of n_sim sequential forward passes per step.
  Input shapes: `ctx_seeds (n_sim, T, 14)`, `phase_as/phase_bs (n_sim,)`.
  Output: `(n_sim, stint_len)` numpy array.
  The PyTorch `TransformerEncoder` supports arbitrary batch dimension natively
  (`batch_first=True`), so no architectural change is needed.

#### `app/strategy_engine.py`
- Replaced the `for sim_i in range(n_sim):` outer loop in `_simulate_strategy()` (v2 path)
  with pre-generation of all per-simulation random values (seeded rngs, same order as
  original for statistical equivalence), followed by a single call to
  `_rollout_v2_batched()` per stint.
- Traffic noise is still drawn from per-sim seeded rngs after the batched rollout
  to maintain the original rng state progression.

#### `app/config.py`
- `TRANSFORMER_V2_N_SIM`: 500 → **100**
  100 MC simulations are statistically sufficient for mean/variance estimation
  (standard error < 0.5 s on a ~1500 s total race time). The increase 100 → 500
  adds diminishing returns with 5× cost.
- `MC_TOP_K`: 5 → **3**
  The analytical scoring phase already ranks candidates well. The 4th and 5th MC
  candidates rarely change the final ranking. Reduces MC work by 40%.

#### `app/main.py`
- Added `@app.on_event("startup")` handler `_preload_models()` that pre-warms the
  LRU model cache (`_load_model_cached`, maxsize=16) at server startup in a background
  thread. Loads the global fallback model and all per-driver models for the available
  season. Eliminates the ~250 ms cold-start latency on the first request.

#### `tests/test_transformer_model.py`
- Added three new tests for the batched rollout:
  - `test_v2_rollout_batched_shape_and_sanity` — shape `(n_sim, stint_len)`, no NaN,
    values in `[60, 300]`.
  - `test_v2_rollout_batched_independence` — sims with different `phase_a/phase_b`
    produce different trajectories; batch dimension is independent.
  - `test_v2_rollout_batched_vs_sequential` — batched mean race time is within 5 s of
    sequential `_rollout_v2` with identical pre-generated random values.

#### `pytest.ini`
- Added `addopts`: `--junit-xml=reports/test_report.xml -v --tb=short -q`.
  Every test run now writes a machine-readable JUnit XML report to `reports/`.

**Expected latency after all changes**

| Step | Before | After |
|---|---|---|
| Forward passes per strategy | ~25 000 | ~150 (50 laps × batch 100) |
| MC candidates | 5 | 3 |
| N_SIM | 500 | 100 |
| Estimated total (warm) | 5–10 min | **5–15 s** |

---

## [0.2.0] — 2026-05 (approx.)

### Added — Transformer v2 architecture (`TyreDegradationTransformerV2`)

- **Architecture**: 6 TransformerEncoder layers, 8 attention heads, d_model=256,
  dim_ff=1024, context window of 25 laps, `batch_first=True`.
- **Inputs**: 14 continuous features (stint_age_norm, delta_norm, pace_intent_norm,
  gap_norm, track_temp_norm, air_temp_norm, throttle_dev, brake_dev, speed_dev,
  rpm_norm, drs_dev, push_index, race_lap_norm, lap_speed_deviation) +
  3 categorical (compound, lap_type, session_type) + 1 static circuit embedding.
- **Outputs**: 3 heads — `head_delta` (normalised Δlap_time, primary),
  `head_abs` (absolute lap time, auxiliary), `head_deg` (degradation rate, auxiliary).
- **Training**: multi-head Huber + MSE loss, AdamW, early stopping, per-epoch CSV log.
- **MC rollout**: `_rollout_v2()` — autoregressive, lerp pace trajectory via
  `phase_a / phase_b` sampled from practice distribution bounds.
- **Practice distributions**: `PracticeDistribution` dataclass fitted per compound per
  driver from historical sessions; 4-level fallback (driver+compound → driver+global →
  global+compound → global).
- **Scripts**: `train_models.py --model-version v2`, `hparam_search.py` (Optuna,
  50 trials), `evaluate_2025.py` (held-out 2025 season evaluation).
- Full test suite in `tests/test_transformer_model.py` (12 tests, all
  `requires_torch`).

### Changed
- `strategy_engine._simulate_strategy()`: added v2 MC path alongside existing v1 path.
- `_load_model_cached()`: discriminator `model_type="transformer_v2"` for new payloads.
- `config.py`: added all `TRANSFORMER_V2_*` constants and `CIRCUIT_VOCAB` / `CIRCUIT_EXPECTED_LAPS`.

---

## [0.1.0] — initial

### Added
- FastAPI backend with `/api/strategy` and `/api/compare` endpoints.
- LSTM pace model (`LSTMPaceNet`) with per-driver training and 4-level fallback.
- Two-phase strategy engine: analytical scoring of all candidates + Monte Carlo
  refinement of top-K.
- Data pipeline: `ingest_season.py` → `preprocess.py` → `train_models.py`.
- In-process TTLCache (24 h) for strategy responses.
- Static serving of `frontend/dist/` from FastAPI root.
