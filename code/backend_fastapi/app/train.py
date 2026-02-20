from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from .config import FEATURE_DIR, MODELS_DIR, DEFAULT_CONTEXT_LAPS
from .models_lstm import LSTMPaceModel, ModelBundle


def _load_features() -> pd.DataFrame:
    dfs = []
    for year_dir in FEATURE_DIR.glob("year=*/features.parquet"):
        dfs.append(pd.read_parquet(year_dir))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def train_per_driver(min_laps: int = 200, epochs: int = 8) -> Dict[int, Path]:
    df = _load_features()
    if df.empty:
        return {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    trained = {}
    for driver_id, df_driver in df.groupby("driver_id"):
        if len(df_driver) < min_laps:
            continue
        model = LSTMPaceModel(context_len=DEFAULT_CONTEXT_LAPS)
        bundle = model.train(df_driver, epochs=epochs)
        input_dim = 8
        payload = {
            "bundle": bundle,
            "input_dim": input_dim,
            "context_len": DEFAULT_CONTEXT_LAPS,
        }
        path = MODELS_DIR / f"driver_{int(driver_id)}.joblib"
        joblib.dump(payload, path)
        trained[int(driver_id)] = path

    global_model = LSTMPaceModel(context_len=DEFAULT_CONTEXT_LAPS)
    bundle = global_model.train(df, epochs=epochs)
    payload = {"bundle": bundle, "input_dim": 8, "context_len": DEFAULT_CONTEXT_LAPS}
    joblib.dump(payload, MODELS_DIR / "global.joblib")

    return trained
