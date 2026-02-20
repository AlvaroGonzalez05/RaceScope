from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import FEATURE_DIR


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
    return sorted(set(years))
