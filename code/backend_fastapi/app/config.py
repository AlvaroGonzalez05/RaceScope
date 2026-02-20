from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"

OPENF1_BASE_URL = "https://api.openf1.org/v1"
OPENF1_MIN_INTERVAL = 0.8
OPENF1_MAX_RETRIES = 4
OPENF1_BACKOFF_BASE = 1.6

SESSION_NAMES = {
    "FP2": "Practice 2",
    "RACE": "Race",
    "SPRINT": "Sprint",
}

DEFAULT_CONTEXT_LAPS = 10
DEFAULT_RISK_LAMBDA = 0.15
DEFAULT_STRATEGY_COUNT = 5

PIT_WINDOW_BIN = 5

CACHE_TTL_SECONDS = 24 * 3600

RANDOM_SEED = 42
MC_TOP_K = 5
PACE_CURVE_CACHE_DIR = CACHE_DIR / "pace_curves"
