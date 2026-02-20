from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR


@dataclass
class ProfileParams:
    base: float
    slope: float
    track_coef: float
    air_coef: float
    track_ref: float
    air_ref: float


@dataclass
class DriverProfile:
    driver_id: int
    profiles: Dict[Tuple[str, str], ProfileParams]
    driver_defaults: Dict[str, ProfileParams]
    global_defaults: Dict[str, ProfileParams]


def _fit_params(df: pd.DataFrame) -> ProfileParams:
    track_ref = float(df["track_temp"].mean()) if df["track_temp"].notna().any() else 30.0
    air_ref = float(df["air_temp"].mean()) if df["air_temp"].notna().any() else 22.0
    temps_track = df["track_temp"].fillna(track_ref).values
    temps_air = df["air_temp"].fillna(air_ref).values
    stint_age = df["stint_age"].values

    X = np.column_stack([
        np.ones(len(df)),
        stint_age,
        temps_track - track_ref,
        temps_air - air_ref,
    ])
    y = df["lap_time"].values

    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        base, slope, track_coef, air_coef = coef
    except Exception:
        base = float(np.median(y))
        slope = float(np.polyfit(stint_age, y, 1)[0]) if len(df) > 3 else 0.04
        track_coef = 0.0
        air_coef = 0.0

    return ProfileParams(
        base=float(base),
        slope=float(slope),
        track_coef=float(track_coef),
        air_coef=float(air_coef),
        track_ref=track_ref,
        air_ref=air_ref,
    )


def _build_global_defaults(df: pd.DataFrame) -> Dict[str, ProfileParams]:
    defaults = {}
    for compound, cdf in df.groupby("compound"):
        if compound is None or compound != compound:
            continue
        key = str(compound).upper()
        if key not in {"SOFT", "MEDIUM", "HARD"}:
            continue
        defaults[key] = _fit_params(cdf)
    return defaults


def train_driver_profiles(df: pd.DataFrame, min_laps: int = 120) -> Dict[int, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trained = {}

    global_defaults = _build_global_defaults(df)
    joblib.dump(global_defaults, MODELS_DIR / "driver_profile_global.joblib")

    for driver_id, df_driver in df.groupby("driver_id"):
        if len(df_driver) < min_laps:
            continue

        profiles = {}
        driver_defaults = {}

        for compound, cdf in df_driver.groupby("compound"):
            if compound is None or compound != compound:
                continue
            compound_key = str(compound).upper()
            if compound_key not in {"SOFT", "MEDIUM", "HARD"}:
                continue
            driver_defaults[compound_key] = _fit_params(cdf)

        for (circuit_id, compound), cdf in df_driver.groupby(["circuit_id", "compound"]):
            if compound is None or compound != compound:
                continue
            compound_key = str(compound).upper()
            if compound_key not in {"SOFT", "MEDIUM", "HARD"}:
                continue
            if len(cdf) < min_laps / 3:
                continue
            profiles[(str(circuit_id), compound_key)] = _fit_params(cdf)

        profile = DriverProfile(
            driver_id=int(driver_id),
            profiles=profiles,
            driver_defaults=driver_defaults,
            global_defaults=global_defaults,
        )
        path = MODELS_DIR / f"driver_profile_{int(driver_id)}.joblib"
        joblib.dump(profile, path)
        trained[int(driver_id)] = path

    return trained


@lru_cache(maxsize=64)
def load_driver_profile(driver_id: int) -> DriverProfile:
    profile_path = MODELS_DIR / f"driver_profile_{int(driver_id)}.joblib"
    if profile_path.exists():
        return joblib.load(profile_path)

    global_path = MODELS_DIR / "driver_profile_global.joblib"
    if global_path.exists():
        global_defaults = joblib.load(global_path)
    else:
        global_defaults = {"SOFT": ProfileParams(90.0, 0.05, 0.0, 0.0, 30.0, 22.0)}
    return DriverProfile(
        driver_id=int(driver_id),
        profiles={},
        driver_defaults={},
        global_defaults=global_defaults,
    )


def resolve_profile_params(profile: DriverProfile, circuit_id: str, compound: str) -> ProfileParams:
    compound_key = compound.upper()
    key = (str(circuit_id), compound_key)

    if key in profile.profiles:
        return profile.profiles[key]
    if compound_key in profile.driver_defaults:
        return profile.driver_defaults[compound_key]
    if compound_key in profile.global_defaults:
        return profile.global_defaults[compound_key]

    # Fallback default
    return ProfileParams(base=90.0, slope=0.05, track_coef=0.0, air_coef=0.0, track_ref=30.0, air_ref=22.0)
