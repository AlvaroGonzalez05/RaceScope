from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Medallion architecture
# ┌──────────┬─────────────────────────────────────────────────────────────┐
# │  Bronze  │ Raw API responses (JSON / JSON.GZ). Never transformed.      │
# │  Silver  │ Cleaned, typed Parquet. One file per endpoint per session.  │
# │  Gold    │ Feature store ready for ML training and inference.          │
# └──────────┴─────────────────────────────────────────────────────────────┘
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR   = DATA_DIR / "gold"

# Backward-compat aliases — existing code that imports RAW_DIR / FEATURE_DIR
# continues to work without modification.
RAW_DIR      = SILVER_DIR
FEATURE_DIR  = GOLD_DIR

MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"
METADATA_DIR = GOLD_DIR / "metadata"
SNAPSHOT_STATE_PATH = METADATA_DIR / "snapshot_state.json"

OPENF1_BASE_URL = "https://api.openf1.org/v1"


def _load_dotenv() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


_load_dotenv()

OPENF1_AUTH_ENABLED = _env_bool("OPENF1_AUTH_ENABLED", False)
OPENF1_USERNAME = os.getenv("OPENF1_USERNAME", "")
OPENF1_PASSWORD = os.getenv("OPENF1_PASSWORD", "")

# Rate limiting. Premium tier: 6 req/s burst, 60 req/min sustained.
# MIN_INTERVAL enforces the burst floor between consecutive requests.
# RATE_PER_SECOND and RATE_PER_MINUTE are enforced by the token-bucket
# logic in openf1_client.py (implemented by BE-CORE agent).
OPENF1_MIN_INTERVAL = _env_float("OPENF1_MIN_INTERVAL", 0.167)
OPENF1_RATE_PER_SECOND = _env_int("OPENF1_RATE_PER_SECOND", 6)
OPENF1_RATE_PER_MINUTE = _env_int("OPENF1_RATE_PER_MINUTE", 60)

OPENF1_MAX_RETRIES = _env_int("OPENF1_MAX_RETRIES", 5)
OPENF1_BACKOFF_BASE = _env_float("OPENF1_BACKOFF_BASE", 2.0)

SESSION_NAMES = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "RACE": "Race",
    "SPRINT": "Sprint",
    "SPRINT_SHOOTOUT": "Sprint Shootout",
}

# --- Transformer v1 hyperparameters (kept for backward compatibility) ---
TRANSFORMER_CONTEXT_LAPS = 15
TRANSFORMER_D_MODEL = 32
TRANSFORMER_N_HEADS = 4
TRANSFORMER_N_LAYERS = 2
TRANSFORMER_DIM_FF = 128
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_INPUT_DIM = 13  # 6 continuous + 7 embedding dims projected

# --- Transformer v2 hyperparameters ---
TRANSFORMER_V2_D_MODEL      = 256
TRANSFORMER_V2_N_HEADS      = 8
TRANSFORMER_V2_N_LAYERS     = 6
TRANSFORMER_V2_DIM_FF       = 1024
TRANSFORMER_V2_DROPOUT      = 0.1
TRANSFORMER_V2_CONTEXT_LAPS = 25
TRANSFORMER_V2_INPUT_DIM    = 14   # 14 continuous features (see models_transformer.py)
TRANSFORMER_V2_AUX_LOSS_W   = 0.15  # weight of absolute-time auxiliary head
TRANSFORMER_V2_DEG_LOSS_W   = 0.10  # weight of degradation-rate auxiliary head
TRANSFORMER_V2_N_SIM        = 100   # MC simulations per strategy candidate

# --- Transformer v3 hyperparameters (medium — ~14M params) ---
TRANSFORMER_V3_D_MODEL      = 384   # 1.5× v2; head_dim=48 (384/8)
TRANSFORMER_V3_N_HEADS      = 8     # mismo que v2; head_dim=48
TRANSFORMER_V3_N_LAYERS     = 8     # 1.33× v2; profundidad razonable
TRANSFORMER_V3_DIM_FF       = 1536  # ratio FF/d_model=4 estándar
TRANSFORMER_V3_DROPOUT      = 0.0   # overfitting intencional (un modelo por piloto)
TRANSFORMER_V3_CONTEXT_LAPS = 40    # +15 vs v2; captura casi un stint completo
TRANSFORMER_V3_INPUT_DIM    = 15    # +1 vs v2: stint_number_norm
TRANSFORMER_V3_AUX_LOSS_W   = 0.10  # peso head tiempo absoluto
TRANSFORMER_V3_DEG_LOSS_W   = 0.25  # peso head degradación (objetivo primario)
TRANSFORMER_V3_VALUE_LOSS_W = 0.20  # peso head coste acumulado restante (nuevo)
TRANSFORMER_V3_N_SIM        = 100   # MC simulations per strategy candidate

# Lap types used as prediction targets vs allowed in context window
TRAINING_LAP_TYPES: frozenset = frozenset({"normal"})
CONTEXT_LAP_TYPES: frozenset = frozenset({"normal", "pit_out", "pit_in"})

# Circuit vocabulary: canonical circuit_id → integer index (0 = unknown/padding)
CIRCUIT_VOCAB: dict = {
    "Sakhir": 1, "Jeddah": 2, "Melbourne": 3, "Suzuka": 4, "Baku": 5,
    "Miami": 6, "Monaco": 7, "Barcelona": 8, "Spielberg": 9, "Silverstone": 10,
    "Budapest": 11, "Spa": 12, "Monza": 13, "Singapore": 14, "Lusail": 15,
    "Austin": 16, "Mexico City": 17, "São Paulo": 18, "Las Vegas": 19,
    "Abu Dhabi": 20, "Montreal": 21, "Imola": 22, "Zandvoort": 23, "Shanghai": 24,
}  # 0 = unknown / unseen circuit

# Expected race laps per circuit (used for race_lap_norm feature)
CIRCUIT_EXPECTED_LAPS: dict = {
    "Sakhir": 57, "Jeddah": 50, "Melbourne": 58, "Suzuka": 53, "Baku": 51,
    "Miami": 57, "Monaco": 78, "Barcelona": 66, "Spielberg": 71, "Silverstone": 52,
    "Budapest": 70, "Spa": 44, "Monza": 53, "Singapore": 62, "Lusail": 57,
    "Austin": 56, "Mexico City": 71, "São Paulo": 71, "Las Vegas": 50,
    "Abu Dhabi": 58, "Montreal": 70, "Imola": 63, "Zandvoort": 72, "Shanghai": 56,
}

DEFAULT_CONTEXT_LAPS = 10
DEFAULT_RISK_LAMBDA = 0.15
DEFAULT_STRATEGY_COUNT = 5

PIT_WINDOW_BIN = 5

CACHE_TTL_SECONDS = 24 * 3600

RANDOM_SEED = 42
MC_TOP_K = 3
PACE_CURVE_CACHE_DIR = CACHE_DIR / "pace_curves"

# Strategy engine fallbacks
PIT_LOSS_FALLBACK = 22.5
PIT_LOSS_MIN = 15.0
PIT_LOSS_MAX = 45.0
SC_PROBABILITY_FALLBACK = 0.20
SC_PROBABILITY_MIN = 0.05
SC_PROBABILITY_MAX = 0.55

# Pace sanity envelopes (driver_profile OLS can overfit on noisy stints; without
# these caps a single bad coefficient propagates into expected_time and the UI
# shows wildly different race durations across drivers at the same circuit).
# Real-world F1 thermal sensitivity is ≈0.05–0.15 s/°C with typical ΔT ≤ 15 °C →
# legitimate correction stays ≤ 2 s; the clamp at 3 s leaves margin for unusual
# venues while killing the 100+ s artefacts seen in the baseline diagnostic.
TEMP_CORR_CLAMP_S = 3.0

# Per-driver pace_base and the per-lap predicted curve are clamped to a window
# derived from the *actual* historical race-lap distribution at the circuit
# (Q5/Q50/Q90). This is ground-truth data from the gold parquet, completely
# decoupled from the DriverProfile OLS fit which has been observed to be
# polluted by qualifying laps and in/out laps. Without this clamp a single
# overfit `base` intercept (e.g. 188 s for SOFT in 2023 Sakhir) flows straight
# into the per-lap curve and inflates expected_time by 30+ minutes.
#
# pace_base (fresh-tyre pace) → [fast * PACE_BASE_LO_FACTOR, median * PACE_BASE_HI_FACTOR]
PACE_BASE_LO_FACTOR = 0.99
PACE_BASE_HI_FACTOR = 1.02
# per-lap on `curves` (allowed to span fresh→worn within the race envelope)
PACE_LAP_LO_FACTOR = 0.98
PACE_LAP_HI_FACTOR = 1.10

# Final sanity envelope for the total race time, anchored to
# `total_laps · circuit_median_lap_time + n_stops · pit_loss`. Real F1 race
# winners run within ≈98 % of the field median pace; backmarkers losing a lap
# (~90 s, ~1.6 %) are still within 4 % of median. The clamp keeps the UI
# defensible without dropping legitimate driver-quality variation.
EXPECTED_TIME_LO_FACTOR = 0.985
EXPECTED_TIME_HI_FACTOR = 1.04
EXPECTED_TIME_SC_MARGIN_S = 60.0

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost,http://127.0.0.1,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
)

# Admin API key — required to call /api/admin/* endpoints.
# Must be set in .env; if empty, all admin routes return 403.
# Generate a strong value with: python -c "import secrets; print(secrets.token_hex(32))"
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")
