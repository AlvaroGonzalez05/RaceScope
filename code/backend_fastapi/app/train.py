from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from .config import (
    FEATURE_DIR,
    MODELS_DIR,
    TRANSFORMER_CONTEXT_LAPS,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_INPUT_DIM,
    TRANSFORMER_V2_CONTEXT_LAPS,
    TRANSFORMER_V2_D_MODEL,
    TRANSFORMER_V2_DIM_FF,
    TRANSFORMER_V2_DROPOUT,
    TRANSFORMER_V2_INPUT_DIM,
    TRANSFORMER_V2_N_HEADS,
    TRANSFORMER_V2_N_LAYERS,
    TRANSFORMER_V2_AUX_LOSS_W,
    TRANSFORMER_V2_DEG_LOSS_W,
    TRANSFORMER_V3_CONTEXT_LAPS,
    TRANSFORMER_V3_D_MODEL,
    TRANSFORMER_V3_DIM_FF,
    TRANSFORMER_V3_DROPOUT,
    TRANSFORMER_V3_INPUT_DIM,
    TRANSFORMER_V3_N_HEADS,
    TRANSFORMER_V3_N_LAYERS,
    TRANSFORMER_V3_AUX_LOSS_W,
    TRANSFORMER_V3_DEG_LOSS_W,
)
from .models_transformer import (
    PracticeDistribution,
    TransformerPaceModel,
)

logger = logging.getLogger(__name__)

# Practice session types whose data feeds PracticeDistribution fitting
_PRACTICE_SESSION_TYPES = {"FP1", "FP2", "FP3"}

# Lap types excluded from training (but kept in the parquet; filtered here)
_EXCLUDE_LAP_TYPES = {"sc", "formation", "outlier"}

# Drivers that get priority training (most requested by the strategy API)
_PRIORITY_DRIVERS = {"SAI", "LEC"}


# ---------------------------------------------------------------------------
# joblib helper
# ---------------------------------------------------------------------------


def _joblib_dump(obj: object, path: Path, retries: int = 4) -> None:
    """joblib.dump with retry on OneDrive / NFS timeout errors."""
    for attempt in range(retries + 1):
        try:
            joblib.dump(obj, path)
            return
        except (TimeoutError, OSError) as exc:
            if attempt == retries:
                raise
            wait = 2.0 ** attempt
            logger.warning(
                "joblib write timeout %s (attempt %d/%d) — retry in %.0fs: %s",
                path.name, attempt + 1, retries, wait, exc,
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_features(max_year: Optional[int] = None) -> pd.DataFrame:
    """
    Load all features.parquet files.  Pass `max_year` to hold out future seasons
    (e.g. max_year=2024 prevents any 2025+ data from entering the training set).
    """
    dfs = []
    for year_path in sorted(FEATURE_DIR.glob("year=*/features.parquet")):
        try:
            year_val = int(year_path.parent.name.split("=")[1])
        except (IndexError, ValueError):
            year_val = None

        if max_year is not None and year_val is not None and year_val > max_year:
            logger.info(
                "_load_features: skipping %s (year=%s > max_year=%s)",
                year_path, year_val, max_year,
            )
            continue
        dfs.append(pd.read_parquet(year_path))

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Outlier filtering
# ---------------------------------------------------------------------------


def filter_training_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove laps unsuitable for training:
      - Any lap where is_valid_train == False   (sc, formation, outlier, pit_in)
      - Explicit lap_type membership guard for older parquets missing is_valid_train

    Keeps: "normal" and "pit_out" laps.
    """
    if df.empty:
        return df

    if "is_valid_train" in df.columns:
        df = df[df["is_valid_train"].astype(bool)].copy()
    elif "lap_type" in df.columns:
        df = df[~df["lap_type"].isin(_EXCLUDE_LAP_TYPES)].copy()

    return df


# ---------------------------------------------------------------------------
# Validation split
# ---------------------------------------------------------------------------


def _split_val(df: pd.DataFrame, val_frac: float = 0.15):
    """
    Temporal validation split based on session_key ordering.

    session_key is an incrementing OpenF1 integer → approximate chronological order.
    The last val_frac fraction of session keys are held out as validation.

    Returns (train_df, val_df).
    """
    if "session_key" not in df.columns or df.empty:
        return df, pd.DataFrame()

    keys  = sorted(df["session_key"].unique())
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[-n_val:])

    train_df = df[~df["session_key"].isin(val_keys)].copy()
    val_df   = df[df["session_key"].isin(val_keys)].copy()

    logger.info(
        "_split_val: train sessions=%d  val sessions=%d  (val_frac=%.2f)",
        len(keys) - n_val, n_val, val_frac,
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# PracticeDistribution fitting
# ---------------------------------------------------------------------------


def _fit_practice_distribution(
    df_practice: pd.DataFrame,
    compound: Optional[str],
    global_lap_std: float,
) -> PracticeDistribution:
    """
    Fit a PracticeDistribution from a (filtered) practice-session DataFrame.
    Includes pace_lo / pace_hi bounds (raw lap-time seconds).
    """
    compound_str = compound or "global"

    if compound is not None:
        df_c = (
            df_practice[df_practice["compound"].str.upper() == compound.upper()]
            if "compound" in df_practice.columns
            else df_practice
        )
    else:
        df_c = df_practice

    if df_c.empty or len(df_c) < 10:
        return PracticeDistribution(compound=compound_str)

    # per-stint pace_intent
    stint_med   = df_c.groupby(
        ["session_key", "driver_id", "stint_number"]
    )["lap_time"].transform("median")
    pace_intent = (df_c["lap_time"] - stint_med) / (global_lap_std or 1.0)

    deltas:       list = []
    outlap_deltas: list = []
    for _, grp in df_c.groupby(["session_key", "driver_id", "stint_number"]):
        grp_sorted = grp.sort_values("lap_number")
        d = grp_sorted["lap_time"].diff().dropna().tolist()
        deltas.extend(d)
        if "lap_type" in grp_sorted.columns:
            pit_out     = grp_sorted[grp_sorted["lap_type"] == "pit_out"]["lap_time"]
            normal_laps = grp_sorted[grp_sorted["lap_type"] == "normal"]["lap_time"]
            if len(pit_out) > 0 and len(normal_laps) > 0:
                outlap_deltas.append(float(pit_out.iloc[0] - normal_laps.mean()))

    delta_arr  = np.array(deltas, dtype=float)
    track_temps = df_c["track_temp"].dropna().values if "track_temp" in df_c.columns else np.array([35.0])
    air_temps   = df_c["air_temp"].dropna().values   if "air_temp"   in df_c.columns else np.array([25.0])

    # Pace bounds: fit from valid training laps
    if "is_valid_train" in df_c.columns:
        valid_times = df_c[df_c["is_valid_train"].astype(bool)]["lap_time"].dropna()
    else:
        valid_times = df_c[df_c.get("lap_type", pd.Series("normal", index=df_c.index)) == "normal"]["lap_time"].dropna()

    if len(valid_times) >= 20:
        pace_hi = float(np.percentile(valid_times, 5))   # max-attack pace (5th pct = fastest)
        pace_lo = float(np.percentile(valid_times, 75))  # conservation pace (75th pct = slower)
    else:
        pace_hi = float(valid_times.min()) if not valid_times.empty else 88.0
        pace_lo = float(valid_times.mean()) if not valid_times.empty else 95.0

    return PracticeDistribution(
        compound=compound_str,
        pace_intent_mu=float(pace_intent.mean()),
        pace_intent_sigma=max(0.01, float(pace_intent.std())),
        delta_mu=float(delta_arr.mean()) if len(delta_arr) else 0.1,
        delta_sigma=max(0.01, float(delta_arr.std())) if len(delta_arr) else 0.15,
        outlap_delta=float(np.mean(outlap_deltas)) if outlap_deltas else -2.0,
        track_temp_mu=float(track_temps.mean()) if len(track_temps) else 35.0,
        track_temp_sigma=max(0.5, float(track_temps.std())) if len(track_temps) else 3.0,
        air_temp_mu=float(air_temps.mean()) if len(air_temps) else 25.0,
        air_temp_sigma=max(0.5, float(air_temps.std())) if len(air_temps) else 2.0,
        pace_lo=pace_lo,
        pace_hi=pace_hi,
    )


def _fit_all_practice_distributions(
    df: pd.DataFrame,
    global_lap_std: float,
) -> Dict[str, PracticeDistribution]:
    """
    Return {compound_upper: PracticeDistribution, "global": PracticeDistribution}
    fitted from practice-session rows in `df`.
    """
    if "session_type" not in df.columns:
        return {}

    df_practice = df[df["session_type"].isin(_PRACTICE_SESSION_TYPES)].copy()
    dists: Dict[str, PracticeDistribution] = {}

    if "compound" in df_practice.columns:
        for compound in df_practice["compound"].dropna().unique():
            key        = str(compound).upper()
            dists[key] = _fit_practice_distribution(df_practice, str(compound), global_lap_std)

    dists["global"] = _fit_practice_distribution(df_practice, None, global_lap_std)
    return dists


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_per_driver(
    max_train_year: int = 2024,
    driver_ids: Optional[Iterable[int]] = None,
    min_laps: int = 200,
    epochs: int = 15,
    context_len: int = TRANSFORMER_V2_CONTEXT_LAPS,
    d_model: int = TRANSFORMER_V2_D_MODEL,
    n_heads: int = TRANSFORMER_V2_N_HEADS,
    n_layers: int = TRANSFORMER_V2_N_LAYERS,
    dim_ff: int = TRANSFORMER_V2_DIM_FF,
    dropout: float = TRANSFORMER_V2_DROPOUT,
    batch_size: int = 64,
    model_version: str = "v2",
    val_frac: float = 0.15,
    patience: int = 5,
    hparam_overrides: Optional[Dict] = None,
) -> Dict[int, Path]:
    """
    Train one TransformerPaceModel per driver (plus a global fallback).

    Parameters
    ----------
    max_train_year  : seasons up to and including this year are used for training.
                      2025+ data is never loaded here (held out for evaluation).
    driver_ids      : if given, only train these drivers (plus global).
    min_laps        : minimum clean "normal" laps for a driver to get their own model.
    epochs          : maximum training epochs (early stopping may stop earlier for v2/v3).
    model_version   : "v3" (large, d_model=768), "v2" (default, d_model=256), "v1" (compat).
    val_frac        : fraction of sessions held out as validation for early stopping.
    patience        : early stopping patience in epochs (0 = disabled).
    hparam_overrides: dict of hyperparameter overrides loaded from hparam_search output.
    """
    # Apply v3 defaults when model_version is "v3" and params were not overridden by caller
    if model_version == "v3":
        if context_len == TRANSFORMER_V2_CONTEXT_LAPS:
            context_len = TRANSFORMER_V3_CONTEXT_LAPS
        if d_model == TRANSFORMER_V2_D_MODEL:
            d_model = TRANSFORMER_V3_D_MODEL
        if n_heads == TRANSFORMER_V2_N_HEADS:
            n_heads = TRANSFORMER_V3_N_HEADS
        if n_layers == TRANSFORMER_V2_N_LAYERS:
            n_layers = TRANSFORMER_V3_N_LAYERS
        if dim_ff == TRANSFORMER_V2_DIM_FF:
            dim_ff = TRANSFORMER_V3_DIM_FF
        if dropout == TRANSFORMER_V2_DROPOUT:
            dropout = TRANSFORMER_V3_DROPOUT
        if epochs == 15:
            epochs = 30
    # Apply hyperparameter overrides if provided
    if hparam_overrides:
        context_len = hparam_overrides.get("context_len", context_len)
        d_model     = hparam_overrides.get("d_model",     d_model)
        n_heads     = hparam_overrides.get("n_heads",     n_heads)
        n_layers    = hparam_overrides.get("n_layers",    n_layers)
        dim_ff      = hparam_overrides.get("dim_ff",      dim_ff)
        dropout     = hparam_overrides.get("dropout",     dropout)
        batch_size  = hparam_overrides.get("batch_size",  batch_size)
        epochs      = hparam_overrides.get("epochs",      epochs)

    df_all = _load_features(max_year=max_train_year)
    if df_all.empty:
        logger.warning("train_per_driver: no feature data found (max_year=%s)", max_train_year)
        return {}

    if "lap_type" not in df_all.columns:
        df_all["lap_type"]       = "normal"
        df_all["is_valid_train"] = True

    df_clean = filter_training_laps(df_all)

    global_lap_std = float(df_clean["lap_time"].std()) or 1.0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Build id→code mapping from the clean dataset
    _id_to_code: Dict[int, str] = {}
    if "driver_code" in df_clean.columns:
        for did, grp in df_clean.groupby("driver_id"):
            codes = grp["driver_code"].dropna().unique()
            if len(codes) > 0:
                _id_to_code[int(did)] = str(codes[0])

    # Determine which driver IDs to train
    all_driver_ids = sorted(df_clean["driver_id"].dropna().unique().astype(int).tolist())
    target_ids     = [int(d) for d in driver_ids] if driver_ids is not None else all_driver_ids

    target_ids_ordered = sorted(
        target_ids,
        key=lambda d: (0 if _id_to_code.get(d, "") in _PRIORITY_DRIVERS else 1, d),
    )

    # Fit practice distributions from the full clean dataset
    practice_dists = _fit_all_practice_distributions(df_clean, global_lap_std)
    logger.info(
        "train_per_driver: practice distributions fitted for compounds=%s",
        list(practice_dists.keys()),
    )

    # Log directory for training CSVs
    log_dir = MODELS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Effective input_dim for payload
    if model_version == "v3":
        input_dim = TRANSFORMER_V3_INPUT_DIM
    elif model_version == "v2":
        input_dim = TRANSFORMER_V2_INPUT_DIM
    else:
        input_dim = TRANSFORMER_INPUT_DIM

    trained: Dict[str, Path] = {}

    for driver_id in target_ids_ordered:
        driver_code = _id_to_code.get(int(driver_id), str(driver_id))
        df_driver = df_clean[df_clean["driver_id"] == driver_id]
        n_normal  = int((df_driver["lap_type"] == "normal").sum())

        if n_normal < min_laps:
            logger.info(
                "driver_id=%s skip (only %d normal laps < min_laps=%d)",
                driver_code, n_normal, min_laps,
            )
            continue

        # Validation split (only for v2 with early stopping)
        train_df, val_df = _split_val(df_driver, val_frac=val_frac) if model_version in ("v2", "v3") else (df_driver, None)
        if train_df.empty:
            logger.warning("driver_id=%s: empty train split, skipping", driver_code)
            continue

        logger.info(
            "driver_id=%s training  normal_laps=%d  context=%d  d_model=%d  layers=%d  epochs=%d  version=%s",
            driver_code, n_normal, context_len, d_model, n_layers, epochs, model_version,
        )

        model = TransformerPaceModel(
            context_len=context_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            model_version=model_version,
        )

        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path    = str(log_dir / f"driver_{driver_code}_{ts}.csv")

        try:
            bundle = model.train(
                train_df,
                epochs=epochs,
                batch_size=batch_size,
                val_df=val_df if (val_df is not None and not val_df.empty) else None,
                patience=patience,
                log_csv_path=csv_path,
            )
        except ValueError as exc:
            logger.warning("driver_id=%s training failed: %s", driver_code, exc)
            continue

        # Driver-specific practice distributions
        driver_practice = {}
        if "session_type" in df_driver.columns:
            driver_dists = _fit_all_practice_distributions(df_driver, global_lap_std)
            for k, v in practice_dists.items():
                driver_practice[k] = driver_dists.get(k, v)
        else:
            driver_practice = practice_dists

        payload = {
            "bundle":      bundle,
            "input_dim":   input_dim,
            "context_len": context_len,
            "model_type":  f"transformer_{model_version}" if model_version in ("v2", "v3") else "transformer",
            "model_kwargs": {
                "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
                "dim_ff": dim_ff, "dropout": dropout,
            },
            "practice_dist": driver_practice,
        }
        path = MODELS_DIR / f"driver_{driver_code}.joblib"
        _joblib_dump(payload, path)
        trained[driver_code] = path
        logger.info("driver_id=%s saved → %s", driver_code, path)

    # --- Global fallback model (all clean data) ---
    train_all, val_all = _split_val(df_clean, val_frac=val_frac) if model_version in ("v2", "v3") else (df_clean, None)
    logger.info(
        "global model: normal_laps=%d  context=%d  d_model=%d  layers=%d  version=%s",
        int((df_clean["lap_type"] == "normal").sum()), context_len, d_model, n_layers, model_version,
    )
    global_model = TransformerPaceModel(
        context_len=context_len, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, dim_ff=dim_ff, dropout=dropout, model_version=model_version,
    )
    ts_g = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        global_bundle = global_model.train(
            train_all if model_version in ("v2", "v3") else df_clean,
            epochs=epochs,
            batch_size=batch_size,
            val_df=val_all if (model_version in ("v2", "v3") and not val_all.empty) else None,
            patience=patience,
            log_csv_path=str(log_dir / f"global_{ts_g}.csv"),
        )
        global_payload = {
            "bundle":      global_bundle,
            "input_dim":   input_dim,
            "context_len": context_len,
            "model_type":  f"transformer_{model_version}" if model_version in ("v2", "v3") else "transformer",
            "model_kwargs": {
                "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
                "dim_ff": dim_ff, "dropout": dropout,
            },
            "practice_dist": practice_dists,
        }
        _joblib_dump(global_payload, MODELS_DIR / "global.joblib")
        logger.info("global model saved → %s", MODELS_DIR / "global.joblib")
    except ValueError as exc:
        logger.error("global model training failed: %s", exc)

    return trained
