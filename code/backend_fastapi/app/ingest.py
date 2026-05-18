from __future__ import annotations

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Iterable

_PARQUET_RETRIES = 4
_PARQUET_RETRY_BASE = 2.0  # seconds

import pandas as pd

from .config import BRONZE_DIR, SILVER_DIR, SESSION_NAMES
from .openf1_client import OpenF1Client

logger = logging.getLogger(__name__)

# Endpoints whose raw responses are large enough to warrant gzip compression
# in the bronze layer (~3.7 Hz telemetry, hundreds of thousands of rows/session).
_BRONZE_GZIP_ENDPOINTS = frozenset({"car_data", "location"})


# ---------------------------------------------------------------------------
# Bronze layer — raw JSON preservation
# ---------------------------------------------------------------------------

def _save_bronze(data: list[dict], bronze_dir: Path, filename: str) -> None:
    """Persist raw API response to bronze layer.

    Small endpoints are stored as plain JSON (human-readable).
    Large high-frequency endpoints (car_data, location) use gzip to save space.
    """
    bronze_dir.mkdir(parents=True, exist_ok=True)
    endpoint = filename.replace(".json.gz", "").replace(".json", "")
    if endpoint in _BRONZE_GZIP_ENDPOINTS:
        path = bronze_dir / f"{endpoint}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
    else:
        path = bronze_dir / f"{endpoint}.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Silver layer — cleaned Parquet
# ---------------------------------------------------------------------------

def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(_PARQUET_RETRIES + 1):
        try:
            df.to_parquet(path, index=False)
            return
        except (TimeoutError, OSError) as exc:
            if attempt == _PARQUET_RETRIES:
                raise
            wait = _PARQUET_RETRY_BASE ** attempt
            logger.warning("parquet write timeout %s (attempt %d/%d) — retry in %.0fs: %s",
                           path.name, attempt + 1, _PARQUET_RETRIES, wait, exc)
            time.sleep(wait)


def _session_silver_complete(session_dir: Path) -> bool:
    """Silver is complete when the four mandatory endpoint Parquets exist."""
    required = [
        session_dir / "laps.parquet",
        session_dir / "stints.parquet",
        session_dir / "weather.parquet",
        session_dir / "drivers.parquet",
    ]
    return all(p.exists() for p in required)


def _session_bronze_complete(bronze_dir: Path) -> bool:
    """Bronze is complete when the four mandatory raw JSON files exist."""
    required = [
        bronze_dir / "laps.json",
        bronze_dir / "stints.json",
        bronze_dir / "weather.json",
        bronze_dir / "drivers.json",
    ]
    return all(p.exists() for p in required)


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def _fetch_and_save(
    client: OpenF1Client,
    endpoint: str,
    params: dict,
    session_dir: Path,
    bronze_dir: Path,
    *,
    required: bool = True,
    numeric_cols: tuple[str, ...] = (),
    bronze_only: bool = False,
) -> list[dict] | None:
    """Fetch one endpoint, save to bronze (JSON) and optionally silver (Parquet).

    bronze_only=True skips the silver write (use when silver already exists).
    If required=True, raises RuntimeError on failure so the caller skips the session.
    """
    try:
        data = client.get_sync(endpoint, params=params)
    except RuntimeError as exc:
        if required:
            raise
        logger.debug("session_key=%s %s unavailable: %s", params.get("session_key"), endpoint, exc)
        return None

    # --- Bronze: raw JSON ---
    _save_bronze(data, bronze_dir, endpoint)

    # --- Silver: typed Parquet (skip if caller says silver is already complete) ---
    if not bronze_only:
        df = pd.DataFrame(data)
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        _to_parquet(df, session_dir / f"{endpoint}.parquet")

    logger.info("session_key=%s %s=%d", params.get("session_key"), endpoint, len(data))
    return data


# ---------------------------------------------------------------------------
# Optional-endpoint helpers
# ---------------------------------------------------------------------------

# Maps endpoint name → expected bronze filename
_OPTIONAL_ENDPOINTS: list[tuple[str, str, tuple[str, ...]]] = [
    ("intervals",    "intervals.json",    ("gap_to_leader", "interval")),
    ("car_data",     "car_data.json.gz",  ()),
    ("location",     "location.json.gz",  ()),
    ("race_control", "race_control.json", ()),
    ("pit",          "pit.json",          ()),
]


def _fetch_optional_endpoints(
    client: OpenF1Client,
    params: dict,
    session_dir: Path,
    bronze_dir: Path,
    *,
    only_missing: bool = False,
) -> None:
    """Fetch all optional endpoints, optionally skipping those already on disk."""
    for endpoint, bronze_filename, numeric_cols in _OPTIONAL_ENDPOINTS:
        bronze_path = bronze_dir / bronze_filename
        if only_missing and bronze_path.exists():
            logger.debug("session_key=%s %s already in bronze — skip", params.get("session_key"), endpoint)
            continue
        _fetch_and_save(
            client, endpoint, params, session_dir, bronze_dir,
            required=False,
            numeric_cols=numeric_cols,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_season(
    year: int,
    session_names: Iterable[str] | None = None,
    sleep_s: float = 1.5,
    min_interval: float = 1.2,
    force_optional: bool = False,
) -> None:
    """Ingest one season from OpenF1 into bronze and silver layers.

    force_optional=True: for sessions where silver+bronze mandatory files are
    already complete, re-fetch only the optional bronze files that are missing
    (car_data, location, race_control, pit, intervals).  Mandatory endpoints
    are never re-downloaded in this mode.
    """
    client = OpenF1Client(min_interval=min_interval)
    logger.info(
        "season=%s min_interval=%.2fs sleep_s=%.2fs force_optional=%s",
        year, min_interval, sleep_s, force_optional,
    )

    sessions = client.get_sync("sessions", params={"year": year})
    if not sessions:
        logger.warning("season=%s no sessions returned", year)
        return

    df_sessions = pd.DataFrame(sessions)
    if session_names is None:
        session_names = list(SESSION_NAMES.values())

    df_sessions = df_sessions[df_sessions["session_name"].isin(session_names)]

    # Bronze + Silver for the sessions index itself.
    bronze_year = BRONZE_DIR / f"year={year}"
    silver_year = SILVER_DIR / f"year={year}"
    _save_bronze(sessions, bronze_year, "sessions")
    _to_parquet(df_sessions, silver_year / "sessions.parquet")
    logger.info("season=%s sessions=%d filtered", year, len(df_sessions))

    for _, session in df_sessions.iterrows():
        session_key = int(session["session_key"])
        params = {"session_key": session_key}
        session_dir  = silver_year / f"session_key={session_key}"
        bronze_dir   = bronze_year / f"session_key={session_key}"

        silver_done = _session_silver_complete(session_dir)
        bronze_done = _session_bronze_complete(bronze_dir)

        # ------------------------------------------------------------------
        # force_optional mode: only fill in missing optional bronze files for
        # sessions that already have mandatory silver+bronze complete.
        # ------------------------------------------------------------------
        if force_optional and silver_done and bronze_done:
            missing = [
                ep for ep, fn, _ in _OPTIONAL_ENDPOINTS
                if not (bronze_dir / fn).exists()
            ]
            if not missing:
                logger.info("session_key=%s force-optional: all optional bronze present — skip", session_key)
                continue
            logger.info(
                "session_key=%s force-optional: fetching missing optional %s",
                session_key, missing,
            )
            _fetch_optional_endpoints(
                client, params, session_dir, bronze_dir, only_missing=True,
            )
            time.sleep(sleep_s)
            continue

        # ------------------------------------------------------------------
        # Normal mode
        # ------------------------------------------------------------------
        if silver_done and bronze_done:
            logger.info("session_key=%s skip (silver+bronze complete)", session_key)
            continue
        # When silver is already complete, only write bronze (avoid re-writing
        # silver parquets on OneDrive which can cause timeout errors).
        write_silver = not silver_done

        if silver_done and not bronze_done:
            logger.info("session_key=%s silver complete — fetching bronze only", session_key)

        logger.info("session_key=%s fetching endpoints", session_key)

        # ---- Mandatory endpoints (skip whole session on failure) ----
        try:
            laps    = _fetch_and_save(client, "laps",    params, session_dir, bronze_dir, required=True, bronze_only=not write_silver)
            stints  = _fetch_and_save(client, "stints",  params, session_dir, bronze_dir, required=True, bronze_only=not write_silver)
            weather = _fetch_and_save(client, "weather", params, session_dir, bronze_dir, required=True, bronze_only=not write_silver)
            drivers = _fetch_and_save(client, "drivers", params, session_dir, bronze_dir, required=True, bronze_only=not write_silver)
        except RuntimeError as exc:
            logger.warning("session_key=%s skipped — fetch error: %s", session_key, exc)
            time.sleep(sleep_s)
            continue

        # ---- Optional endpoints ----
        _fetch_optional_endpoints(client, params, session_dir, bronze_dir, only_missing=False)

        logger.info(
            "session_key=%s done laps=%d stints=%d weather=%d drivers=%d",
            session_key, len(laps), len(stints), len(weather), len(drivers),
        )
        time.sleep(sleep_s)


def ingest_range(
    start_year: int,
    end_year: int,
    sleep_s: float = 1.5,
    min_interval: float = 1.2,
    force_optional: bool = False,
) -> None:
    for year in range(start_year, end_year + 1):
        ingest_season(year, sleep_s=sleep_s, min_interval=min_interval, force_optional=force_optional)
