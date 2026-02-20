from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import RAW_DIR, FEATURE_DIR, SESSION_NAMES


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _session_type(session_name: str) -> str:
    mapping = {v: k for k, v in SESSION_NAMES.items()}
    return mapping.get(session_name, session_name.upper().replace(" ", "_"))


def build_features_for_year(year: int) -> pd.DataFrame:
    sessions_path = RAW_DIR / f"year={year}" / "sessions.parquet"
    sessions = _read_parquet(sessions_path)
    if sessions.empty:
        return pd.DataFrame()

    features = []
    metadata_drivers = {}
    metadata_teams = set()
    metadata_circuits = set()

    for _, session in sessions.iterrows():
        session_key = session["session_key"]
        session_dir = RAW_DIR / f"year={year}" / f"session_key={session_key}"

        laps = _read_parquet(session_dir / "laps.parquet")
        stints = _read_parquet(session_dir / "stints.parquet")
        weather = _read_parquet(session_dir / "weather.parquet")
        drivers = _read_parquet(session_dir / "drivers.parquet")

        if laps.empty:
            continue

        laps = laps.copy()
        has_stint_number = "stint_number" in laps.columns
        laps["lap_time"] = _safe_float(laps.get("lap_duration"))
        laps["lap_number"] = _safe_float(laps.get("lap_number"))
        laps["stint_number"] = _safe_float(laps.get("stint_number"))
        laps["driver_number"] = _safe_float(laps.get("driver_number"))
        laps = laps.dropna(subset=["lap_time", "lap_number", "driver_number"])

        if not stints.empty:
            stints = stints.copy()
            stints["driver_number"] = _safe_float(stints.get("driver_number"))
            stints["stint_number"] = _safe_float(stints.get("stint_number"))
            stints["lap_start"] = _safe_float(stints.get("lap_start"))
            stints["lap_end"] = _safe_float(stints.get("lap_end"))
            stints["compound"] = stints.get("compound")

            if has_stint_number and laps["stint_number"].notna().any():
                laps = laps.merge(
                    stints[["driver_number", "stint_number", "compound", "lap_start", "lap_end"]],
                    on=["driver_number", "stint_number"],
                    how="left",
                )
            else:
                laps = laps.reset_index().rename(columns={"index": "lap_index"})
                merged = laps.merge(
                    stints[["driver_number", "stint_number", "compound", "lap_start", "lap_end"]],
                    on="driver_number",
                    how="left",
                )
                mask = (merged["lap_number"] >= merged["lap_start"]) & (merged["lap_number"] <= merged["lap_end"])
                merged = merged[mask | merged["lap_start"].isna()]
                merged = merged.sort_values(["lap_index", "lap_start"]).drop_duplicates("lap_index", keep="first")
                laps = merged.drop(columns=["lap_index"])
        else:
            laps["compound"] = np.nan
            laps["lap_start"] = np.nan
            laps["lap_end"] = np.nan
            laps["stint_number"] = np.nan

        if not drivers.empty:
            drivers = drivers.copy()
            drivers["driver_number"] = _safe_float(drivers.get("driver_number"))
            laps = laps.merge(
                drivers[["driver_number", "name_acronym", "team_name"]],
                on="driver_number",
                how="left",
            )
        else:
            laps["name_acronym"] = None
            laps["team_name"] = None

        track_temp = None
        air_temp = None
        if not weather.empty:
            track_temp = _safe_float(weather.get("track_temperature")).mean()
            air_temp = _safe_float(weather.get("air_temperature")).mean()

        laps["track_temp"] = track_temp
        laps["air_temp"] = air_temp

        laps["stint_age"] = laps["lap_number"] - laps["lap_start"].fillna(laps["lap_number"]) + 1
        if "stint_number" not in laps.columns:
            laps["stint_number"] = np.nan
        laps["stint_number"] = laps["stint_number"].fillna(1)
        laps["session_key"] = session_key
        laps["session_type"] = _session_type(session.get("session_name", ""))
        laps["circuit_id"] = session.get("circuit_short_name") or session.get("location") or session.get("meeting_key")
        laps["year"] = year

        laps = laps.rename(columns={
            "driver_number": "driver_id",
            "name_acronym": "driver_code",
        })

        for _, row in laps[["driver_id", "driver_code", "team_name"]].dropna().drop_duplicates().iterrows():
            metadata_drivers[int(row["driver_id"])] = {
                "driver_id": int(row["driver_id"]),
                "driver_code": row.get("driver_code"),
                "team_name": row.get("team_name"),
            }
            if row.get("team_name"):
                metadata_teams.add(row.get("team_name"))

        if laps["circuit_id"].notna().any():
            metadata_circuits.update(laps["circuit_id"].dropna().unique().tolist())

        features.append(
            laps[[
                "year",
                "session_key",
                "session_type",
                "circuit_id",
                "driver_id",
                "driver_code",
                "team_name",
                "lap_number",
                "stint_number",
                "stint_age",
                "compound",
                "lap_time",
                "track_temp",
                "air_temp",
            ]]
        )

    if not features:
        return pd.DataFrame()

    df = pd.concat(features, ignore_index=True)
    out_path = FEATURE_DIR / f"year={year}" / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    metadata_path = FEATURE_DIR / "metadata"
    metadata_path.mkdir(parents=True, exist_ok=True)

    drivers_df = pd.DataFrame(metadata_drivers.values())
    drivers_df.to_parquet(metadata_path / f"drivers_{year}.parquet", index=False)

    teams_df = pd.DataFrame(sorted(metadata_teams), columns=["team_name"])
    teams_df.to_parquet(metadata_path / f"teams_{year}.parquet", index=False)

    circuits_df = pd.DataFrame(sorted(metadata_circuits), columns=["circuit_id"])
    circuits_df.to_parquet(metadata_path / f"circuits_{year}.parquet", index=False)

    return df


def build_features_range(start_year: int, end_year: int) -> None:
    for year in range(start_year, end_year + 1):
        build_features_for_year(year)
