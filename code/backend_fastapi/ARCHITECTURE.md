# Backend Architecture — Transformer Pace Models (v1 / v2 / v3)

> Last updated: 2026-05-25
> Model files: `models/driver_*.joblib` (trained 2026-05-25, `model_type: transformer_v3`)
> Source: `app/models_transformer.py`, `app/strategy_engine.py`, `app/config.py`

---

## 1. System Overview

```
  OpenF1 API
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Ingestion  (scripts/ingest_season.py)                               │
│  FP1 · FP2 · FP3 · Race · Sprint  →  data/bronze/year=YYYY/        │
│  + intervals endpoint → gap_to_car_ahead per lap                    │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Preprocessing  (scripts/preprocess.py → app/preprocess.py)          │
│  bronze → silver (typed parquet) → gold (feature store)             │
│                                                                      │
│  Gold layer columns (23):                                            │
│    year, session_key, session_type, circuit_id, driver_id,          │
│    driver_code, team_name, lap_number, stint_number, stint_age,     │
│    compound, lap_time, track_temp, air_temp, gap_to_car_ahead,      │
│    mean_throttle, mean_brake, max_speed, mean_speed, mean_rpm,      │
│    drs_fraction, lap_type, is_valid_train                           │
│                                                                      │
│  lap_type classification:                                            │
│    normal | pit_out | pit_in | sc | formation | outlier             │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Training  (scripts/train_models.py → app/train.py)                  │
│    Val split: last 15% of sessions (chronological)                  │
│    Per driver (≥200 clean laps) → driver_<id>.joblib                │
│    Global fallback              → global.joblib                     │
│    2025 data: NEVER loaded (held out for evaluate_2025.py)          │
│    Logs: models/logs/driver_<id>_<timestamp>.csv                    │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Strategy Engine  (app/strategy_engine.py)                           │
│  Phase 1: _pace_table → break-even s* in closed form (1-stop) and   │
│          2x2 linear system (2-stop) per compound combination;       │
│          drop candidates with no profitable stop                    │
│  Phase 2: Analytical pre-scorer (mean + λ·var) over Transformer     │
│          pace curves; rank all candidates                           │
│  Phase 3: Monte Carlo refinement of top-3                           │
│          100 simulations × batched Transformer v3 rollout           │
│  Phase 4: Hard filter — exclude HARD-start from payload             │
│  Output: ranked StrategyResponse with pit_windows + stint_curves   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Transformer v2 Architecture — `TyreDegradationTransformerV2`

### 2.1 Input Feature Set

**Continuous per timestep (14 dims):**

| # | Feature | Formula |
|---|---------|---------|
| 1 | `stint_age_norm` | `stint_age / 40` |
| 2 | `delta_norm` | `Δlap_time / delta_std` |
| 3 | `pace_intent_norm` | `(lap_time − stint_median) / lap_std` — negative = pushing |
| 4 | `gap_norm` | `clip(gap_to_car_ahead, 0, 5) / 5` |
| 5 | `track_temp_norm` | `(track_temp − track_ref) / 10` |
| 6 | `air_temp_norm` | `(air_temp − air_ref) / 8` |
| 7 | `throttle_dev` | `(mean_throttle − μ_throttle) / σ_throttle` per (circuit, compound) |
| 8 | `brake_dev` | `(mean_brake − μ_brake) / σ_brake` per (circuit, compound) |
| 9 | `speed_dev` | `(max_speed − μ_speed) / σ_speed` per (circuit, compound) |
| 10 | `rpm_norm` | `(mean_rpm − 10000) / 3000` |
| 11 | `drs_dev` | `(drs_fraction − μ_drs) / σ_drs` per (circuit, compound) |
| 12 | `push_index` | `0.4·throttle_dev + 0.3·speed_dev + 0.2·brake_dev + 0.1·drs_dev`, clipped [-3, 3] |
| 13 | `race_lap_norm` | `lap_number / CIRCUIT_EXPECTED_LAPS[circuit_id]`, clipped [0, 1.2] |
| 14 | `lap_speed_deviation` | same as `speed_dev` (circuit-normalised max speed) |

At inference time, features 7–14 (telemetry deviations) are set to 0 since live telemetry is unavailable. `push_index` (feature 12) is synthesised as `clip(-pace_intent_t, -3, 3)` during rollout.

**Categorical (3 per timestep) + static circuit context:**

| Input | Vocab size | Embedding dim |
|-------|-----------|---------------|
| `compound_int` | 10 | 8 |
| `lap_type_int` | 10 | 6 |
| `session_type_int` | 10 | 6 |
| `circuit_idx` (static, per sequence) | 28 | 16 |

### 2.2 Network Forward Pass

```
Input: x_cont (B, T, 14) · x_cat (B, T, 3) · circuit_idx (B,)
  where T = context_len = 25 laps,  B = batch size

───────────────────────────────────────────────────────────────
Step 1 — Per-timestep projections
  cont_proj:   Linear(14 → d_model//2 = 128)          → (B, T, 128)
  cat_concat:  [comp_emb(8) | lap_type_emb(6) | session_emb(6) | circuit_emb(16)]
               = (B, T, 36)
  cat_proj:    Linear(36 → d_model//2 = 128)           → (B, T, 128)
  concat:      [cont | cat]                             → (B, T, 256)

Step 2 — Circuit global prior (broadcast)
  circuit_emb(circuit_idx):  (B, 16)
  circuit_prior_proj:        Linear(16 → 256)           → (B, 256)
  broadcast add:             x + prior.unsqueeze(1)     → (B, T, 256)

Step 3 — LayerNorm + positional encoding
  input_ln:  LayerNorm(256)
  pos_enc:   Embedding(50, 256)[0..T-1]  +  x           → (B, T, 256)

Step 4 — Transformer encoder  (6 layers, Pre-LN)
  TransformerEncoderLayer(d_model=256, nhead=8, dim_ff=1024,
                          dropout=0.1, norm_first=True)
  × 6 layers → (B, T, 256)

Step 5 — Take last token
  x[:, -1, :]   →  (B, 256)

Step 6 — Three output heads
  head_delta:  Linear(256 → 1)  → normalised Δlap_time   [main head]
  head_abs:    Linear(256 → 1)  → normalised lap_time     [aux head]
  head_deg:    Linear(256 → 1)  → normalised deg rate     [aux head]
───────────────────────────────────────────────────────────────
```

### 2.3 Hyperparameters (`config.py`)

| Constant | Value | Meaning |
|---|---|---|
| `TRANSFORMER_V2_CONTEXT_LAPS` | 25 | Context window length T |
| `TRANSFORMER_V2_D_MODEL` | 256 | Hidden dimension |
| `TRANSFORMER_V2_N_HEADS` | 8 | Attention heads |
| `TRANSFORMER_V2_N_LAYERS` | 6 | Encoder layers |
| `TRANSFORMER_V2_DIM_FF` | 1024 | FFN inner dimension |
| `TRANSFORMER_V2_DROPOUT` | 0.1 | Dropout rate |
| `TRANSFORMER_V2_INPUT_DIM` | 14 | Continuous features per timestep |
| `TRANSFORMER_V2_AUX_LOSS_W` | 0.15 | Weight of absolute lap-time head |
| `TRANSFORMER_V2_DEG_LOSS_W` | 0.10 | Weight of degradation-rate head |
| `TRANSFORMER_V2_N_SIM` | 500 | MC simulations per strategy |

### 2.4 Training Setup

| Setting | Value |
|---|---|
| Optimiser | AdamW (lr=5e-4, weight_decay=1e-4) |
| Scheduler | OneCycleLR (max_lr=5e-4) |
| Loss | `Huber(Δlap) + 0.15·MSE(abs) + 0.10·MSE(deg_rate)` |
| Max epochs | 50 |
| Early stopping | patience=12 on val_mae |
| Batch size | 64 |
| Val split | last 15% of sessions (chronological) |
| Min laps/driver | 200 clean normal laps |
| Training target laps | `lap_type == "normal"` |
| Context laps | `lap_type ∈ {normal, pit_out, pit_in}` |
| Weight init | Xavier uniform (linear layers) |
| Grad clipping | norm ≤ 1.0 |
| Training log | `models/logs/<driver>_<timestamp>.csv` |

**Observed training results (2026-05-18 run):**
- Global model: val_mae = 0.107 (best checkpoint at epoch 4, 21.6K sequences)
- Per-driver median val_mae: 0.117 (25 drivers, best D55: 0.079)
- 4.76M parameters, 18.2 MB state dict

### 2.5 Parameter Count

```
Embeddings:
  compound_emb:        10 × 8  =     80
  lap_type_emb:        10 × 6  =     60
  session_emb:         10 × 6  =     60
  circuit_emb:         28 × 16 =    448
Projections:
  cont_proj:           14 × 128 + 128        =  1,920
  cat_proj:            36 × 128 + 128        =  4,736
  circuit_prior_proj:  16 × 256 + 256        =  4,352
  input_ln:            256 × 2               =    512
  pos_enc:             50 × 256              = 12,800
TransformerEncoder (6 layers, d=256, nhead=8, ff=1024):
  per layer ≈ 4×(256×256) + 2×(256×1024+1024×256) ≈ 786,432
  × 6 layers                                = 4,718,592
Output heads (3 × Linear(256→1)):           =  3 × 257 = 771
─────────────────────────────────────────────────────────────
Total ≈ 4,764,299 parameters
```

---

## 3. Transformer v3 Architecture — `TyreDegradationTransformerV3`

**Active model since 2026-05-25.** Medium-size variant: ~14M parameters.

### 3.1 Key differences vs v2

| Aspect | v2 | v3 |
|---|---|---|
| d_model | 256 | **384** |
| n_layers | 6 | **8** |
| dim_ff | 1024 | **1536** |
| n_heads | 8 | 8 |
| context_len | 25 | **40** |
| dropout | 0.1 | **0.0** (intentional overfit) |
| weight_decay | 1e-4 | **0.0** |
| Continuous features | 14 | **15** (+`stint_number_norm`) |
| Output heads | 3 | **4** (+`head_value`) |
| Circuit prior | additive bias | **multiplicative CC gate** |
| compound_idx_static | — | **yes** (per-stint scalar) |
| Parameters | ~4.8M | **~14.3M** |
| Joblib size | ~18 MB | **~56 MB** |

### 3.2 New feature: `stint_number_norm` (index 14)

```
stint_number_norm = clip((stint_number - 1) / 3.0, 0.0, 1.5)
```
- Stint 1 → 0.0  |  Stint 2 → 0.33  |  Stint 3 → 0.67  |  Stint 6+ → 1.5 (clipped)

Differentiates tyre behaviour across stints (fuel load, track rubber, thermal history).

### 3.3 Circuit×Compound multiplicative gate

The key architectural innovation. A sigmoid-gated mask over all 768 feature dimensions,
computed from the joint (circuit, compound) static embedding:

```
circ_v = circuit_emb(circuit_idx)               # (B, 48)
comp_v = compound_emb_gate(compound_idx_static)  # (B, 24)
gate   = sigmoid(LayerNorm(Linear([circ_v||comp_v] → d_model)))  # (B, d_model)
x      = x * gate.unsqueeze(1)                   # element-wise per timestep
```

Effect: dimensions relevant for degradation at Monaco+SOFT are amplified;
irrelevant dimensions suppressed. An additive prior (v2) cannot suppress dimensions.
LayerNorm before sigmoid ensures gate ≈ 0.5 at init (neutral → progressive specialisation).

### 3.4 Fourth output head: `head_value`

Predicts normalised cumulative remaining-stint cost:

```
y_value[i] = sum(lap_times[i+1 .. end_of_stint]) / (remaining_laps × lap_std)
```

Used by `_stop_profitability()` to estimate whether an additional pit stop is cost-effective.
Not used during autoregressive rollout (only `head_delta` drives lap-by-lap prediction).

### 3.5 `stop_profitability` in the strategy engine

```python
profit(stop_i) = E[t_old for L laps] - E[t_fresh for L laps] - pit_loss
```
- `profit > 0` → stopping recovers more time than is lost in the pitlane
- `profit < 0` → better to stay out

Computed from pace curves in `_stop_profitability()` and returned in `StrategyOut.stop_profitability`.

### 3.6 v3 Hyperparameters (`config.py`)

| Constant | Value |
|---|---|
| `TRANSFORMER_V3_D_MODEL` | 384 |
| `TRANSFORMER_V3_N_HEADS` | 8 |
| `TRANSFORMER_V3_N_LAYERS` | 8 |
| `TRANSFORMER_V3_DIM_FF` | 1536 |
| `TRANSFORMER_V3_DROPOUT` | 0.0 |
| `TRANSFORMER_V3_CONTEXT_LAPS` | 40 |
| `TRANSFORMER_V3_INPUT_DIM` | 15 |
| `TRANSFORMER_V3_AUX_LOSS_W` | 0.10 |
| `TRANSFORMER_V3_DEG_LOSS_W` | 0.25 |
| `TRANSFORMER_V3_VALUE_LOSS_W` | 0.20 |
| `TRANSFORMER_V3_N_SIM` | 100 |

### 3.7 v3 Training Setup

| Setting | Value |
|---|---|
| Optimiser | AdamW (lr=2e-4, weight_decay=0.0) |
| Scheduler | OneCycleLR |
| Loss | `Huber(Δlap) + 0.10·MSE(abs) + 0.25·MSE(deg) + 0.20·MSE(value)` |
| Max epochs | 30 |
| Early stopping | patience=5 on val_mae |
| Grad clipping | norm ≤ 1.0 |

---

## 4. Backward Compatibility — v1, v2 & LSTM

`_load_model_cached` in `strategy_engine.py` discriminates on `model_type`:

| `model_type` value | Class loaded | Notes |
|---|---|---|
| `"transformer_v3"` | `TyreDegradationTransformerV3` | **Current default** |
| `"transformer_v2"` | `TyreDegradationTransformerV2` | Older joblibs |
| `"transformer"` | `TyreTransformerNet` (v1) | Legacy |
| absent | `LSTMPaceModel` | Old LSTM payloads |

`_simulate_strategy` checks in order: v3 → v2 → v1/LSTM.

---

## 5. Vocabulary Encodings

### Compound (vocab size 10)
| Token | Index |
|---|---|
| SOFT | 1 |
| MEDIUM | 2 |
| HARD | 3 |
| INTERMEDIATE / INTER | 4 |
| WET | 5 |
| SUPERSOFT / ULTRASOFT / HYPERSOFT | 6 |
| Unknown / padding | 0 |

### Lap type (vocab size 10)
| Token | Index |
|---|---|
| normal | 1 |
| pit_out | 2 |
| pit_in | 3 |
| sc | 4 |
| formation | 5 |
| outlier | 6 |

### Session type (vocab size 10)
| Token | Index |
|---|---|
| FP1 | 1 | FP2 | 2 | FP3 | 3 | RACE | 4 | SPRINT | 5 | SPRINT_SHOOTOUT | 6 |

### Circuit (vocab size 28, headroom for new venues)
24 known circuits mapped in `CIRCUIT_VOCAB` (config.py). Unknown circuits default to 0.

---

## 6. Autoregressive MC Rollout (v2 / v3)

Used at inference time inside `strategy_engine._simulate_strategy()`.

```
For each simulation (500 per strategy):

  1. Sample priors from PracticeDistribution:
       phase_a, phase_b ~ Uniform(pace_hi_norm, pace_lo_norm)
       sc_event         ~ Bernoulli(sc_probability)
       track_temp, air_temp ~ N(μ, σ) from PracticeDistribution

  2. Build context seed (25-lap warm-up via build_context_seed):
       stint_age_norm = [1..25] / 40
       delta_norm     ~ N(delta_mu, delta_sigma)
       Features 7-14 (telemetry) = 0   ← not available at inference

  3. For each lap in stint:
       a. Compute pace_intent_t = lerp(phase_a, phase_b, t/stint_len)
       b. Synthesise push_index_t = clip(-pace_intent_t, -3, 3)
       c. Compute race_lap_norm_t = (race_lap_cursor + step) / exp_laps
       d. Append new feature row to sliding window
       e. Forward pass → Δlap_time_norm_pred  (scalar, from head_delta)
       f. lap_time[t+1] = clip(base + pred × delta_std, 60, 300)
       g. Slide window: drop oldest lap, append predicted lap

  4. Accumulate race time:
       total += sum(lap_times across all stints)
               + pit_loss × n_stops
               + (15.0 if sc_event and near stop lap else 0)
               + traffic_noise ~ N(0.15 × stint_len, 0.05 × √stint_len)

  5. Score = mean(totals) + risk_lambda × var(totals)  [risk-adjusted]

  6. Post-score compound penalties (first_compound == HARD → +50s, MEDIUM → +2s)
```

### PracticeDistribution (fitted per compound at training time)

```python
@dataclass
class PracticeDistribution:
    compound: str
    pace_intent_mu: float    # mean normalised pace intent
    pace_intent_sigma: float
    delta_mu: float          # mean Δlap_time within stints
    delta_sigma: float
    outlap_delta: float      # first-lap delta after pit stop (~-2s)
    track_temp_mu: float; track_temp_sigma: float
    air_temp_mu: float;   air_temp_sigma: float
    pace_lo: float = 95.0   # 75th percentile lap time (conservation)
    pace_hi: float = 88.0   # 5th percentile lap time  (max-attack)
    # Convention: pace_lo > pace_hi  (larger seconds = slower)
```

---

## 7. Joblib Payload Format (v3)

```python
{
    "bundle": ModelBundle(
        model_state = OrderedDict,   # state_dict of TyreDegradationTransformerV3
        encoders    = {
            "compound":     {str: int},
            "lap_type":     {str: int},
            "session_type": {str: int},
        },
        stats = {
            "lap_mean":               float,
            "lap_std":                float,
            "delta_std":              float,
            "track_temp_ref":         float,
            "air_temp_ref":           float,
            "circuit_compound_stats": dict,  # per (circuit, compound) telemetry μ/σ
        },
    ),
    "input_dim":    15,
    "context_len":  40,
    "model_type":   "transformer_v3",
    "model_kwargs": {
        "d_model": 384, "n_heads": 8, "n_layers": 8,
        "dim_ff": 1536, "dropout": 0.0,
    },
    "practice_dist": {compound_str: PracticeDistribution, ..., "global": PracticeDistribution},
}
```

---

## 8. Sequence Building (training)

```
For each (session_key, driver_id) group:

  Filter: lap_type ∈ {normal, pit_out, pit_in}
  Sort by lap_number

  Compute per-lap features (14 continuous):
    pace_intent_norm, delta_norm, gap_norm, track/air temp norms,
    throttle_dev, brake_dev, speed_dev, rpm_norm, drs_dev,
    push_index = 0.4·throttle_dev + 0.3·speed_dev + 0.2·brake_dev + 0.1·drs_dev,
    race_lap_norm = lap_number / CIRCUIT_EXPECTED_LAPS[circuit_id],
    lap_speed_deviation = speed_dev

  Compute deg_rate per stint via linear polyfit of lap_time vs stint_age
  (min 3 normal laps per stint; fallback = 0.05)

  Slide window of length T=25:
    For i in [T, len):
      if lap_type[i] == "normal":
        X_cont[i]    = cont[i-T : i]           ← (25, 14)
        X_cat[i]     = cat [i-T : i]           ← (25, 3)
        circuit_id[i] = CIRCUIT_VOCAB[circuit]  ← scalar
        y_delta[i]   = (lap[i] - lap[i-1]) / delta_std
        y_abs[i]     = (lap[i] - lap_mean)  / lap_std
        y_deg[i]     = deg_rate[i]           / lap_std
```

---

## 9. Benchmark (v1 / v2 / v3, measured 2026-05-25, CPU)

Generated by `scripts/benchmark_architectures.py`.

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Parameters | 26,517 | 4,764,299 | 14,287,364 |
| Joblib size | ~0.1 MB | ~18 MB | ~56 MB |
| Train time / epoch (n=512) | 0.05 s | 0.78 s | 1.08 s |
| Inference 1×60 laps | 11 ms | 69 ms | 185 ms |
| Inference 100×60 laps (batched MC) | 122 ms | 1,570 ms | 4,595 ms |
| Factor vs v1 (batched) | 1× | 13× | 38× |

v3 batched MC (100 sims × 60 laps) takes ~4.6 s per strategy on CPU.
Top-K=3 strategies → ~14 s per `/api/strategy` request (cached after first call).

---

*Update this file whenever `TyreDegradationTransformerV3`, `PracticeDistribution`, or the MC rollout logic in `strategy_engine.py` changes.*
