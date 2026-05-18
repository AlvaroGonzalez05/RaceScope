from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import FEATURE_DIR, SNAPSHOT_STATE_PATH


@lru_cache(maxsize=4)
def load_features() -> pd.DataFrame:
    dfs = []
    for path in FEATURE_DIR.glob("year=*/features.parquet"):
        dfs.append(pd.read_parquet(path))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def metadata_for_year(year: int) -> Dict[str, pd.DataFrame]:
    metadata_path = FEATURE_DIR / "metadata"
    drivers = metadata_path / f"drivers_{year}.parquet"
    teams = metadata_path / f"teams_{year}.parquet"
    circuits = metadata_path / f"circuits_{year}.parquet"

    return {
        "drivers": pd.read_parquet(drivers) if drivers.exists() else pd.DataFrame(),
        "teams": pd.read_parquet(teams) if teams.exists() else pd.DataFrame(),
        "circuits": pd.read_parquet(circuits) if circuits.exists() else pd.DataFrame(),
    }


def seasons_available() -> List[int]:
    years = []
    for path in FEATURE_DIR.glob("year=*/features.parquet"):
        try:
            years.append(int(path.parent.name.replace("year=", "")))
        except ValueError:
            continue
    if years:
        return sorted(set(years))
    snapshot = load_snapshot_state()
    snapshot_years = snapshot.get("years_available", [])
    return sorted({int(y) for y in snapshot_years if str(y).isdigit()})


def load_snapshot_state() -> Dict:
    if not SNAPSHOT_STATE_PATH.exists():
        return {
            "last_successful_ingest_at": None,
            "years_available": [],
            "stale_mode": False,
            "last_error": None,
        }
    try:
        return json.loads(SNAPSHOT_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "last_successful_ingest_at": None,
            "years_available": [],
            "stale_mode": True,
            "last_error": "invalid_snapshot_state_json",
        }


def write_snapshot_state(*, years_available: List[int], stale_mode: bool, last_error: str | None = None) -> None:
    SNAPSHOT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_successful_ingest_at": datetime.now(timezone.utc).isoformat(),
        "years_available": sorted(set(int(y) for y in years_available)),
        "stale_mode": bool(stale_mode),
        "last_error": last_error,
    }
    SNAPSHOT_STATE_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def refresh_snapshot_state(stale_mode: bool, last_error: str | None = None) -> None:
    years = []
    for path in FEATURE_DIR.glob("year=*/features.parquet"):
        try:
            years.append(int(path.parent.name.replace("year=", "")))
        except ValueError:
            continue
    write_snapshot_state(years_available=years, stale_mode=stale_mode, last_error=last_error)


import logging as _logging
_ds_logger = _logging.getLogger(__name__)


def invalidate_features_cache() -> None:
    """Clear the in-memory features LRU cache. Call after re-ingestion."""
    load_features.cache_clear()
    _ds_logger.info("features LRU cache cleared")
