from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import RAW_DIR, SESSION_NAMES
from .openf1_client import OpenF1Client


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def ingest_season(year: int, session_names: Iterable[str] | None = None, sleep_s: float = 1.0) -> None:
    client = OpenF1Client()
    sessions = client.get("sessions", params={"year": year})
    if not sessions:
        return

    df_sessions = pd.DataFrame(sessions)
    if session_names is None:
        session_names = list(SESSION_NAMES.values())

    df_sessions = df_sessions[df_sessions["session_name"].isin(session_names)]
    _to_parquet(df_sessions, RAW_DIR / f"year={year}" / "sessions.parquet")

    for _, session in df_sessions.iterrows():
        session_key = session["session_key"]
        session_dir = RAW_DIR / f"year={year}" / f"session_key={session_key}"

        laps = client.get("laps", params={"session_key": session_key})
        stints = client.get("stints", params={"session_key": session_key})
        weather = client.get("weather", params={"session_key": session_key})
        drivers = client.get("drivers", params={"session_key": session_key})

        _to_parquet(pd.DataFrame(laps), session_dir / "laps.parquet")
        _to_parquet(pd.DataFrame(stints), session_dir / "stints.parquet")
        _to_parquet(pd.DataFrame(weather), session_dir / "weather.parquet")
        _to_parquet(pd.DataFrame(drivers), session_dir / "drivers.parquet")

        time.sleep(sleep_s)


def ingest_range(start_year: int, end_year: int, sleep_s: float = 0.3) -> None:
    for year in range(start_year, end_year + 1):
        ingest_season(year, sleep_s=sleep_s)
