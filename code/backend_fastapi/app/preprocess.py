from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import RAW_DIR, FEATURE_DIR, SESSION_NAMES
from .data_store import refresh_snapshot_state

logger = logging.getLogger(__name__)

CIRCUIT_ID_CANONICAL: dict[str, str] = {
    "bahrain": "Sakhir",
    "sakhir": "Sakhir",
    "jeddah": "Jeddah",
    "saudi arabia": "Jeddah",
    "albert park": "Melbourne",
    "melbourne": "Melbourne",
    "suzuka": "Suzuka",
    "baku": "Baku",
    "azerbaijan": "Baku",
    "miami": "Miami",
    "monte carlo": "Monaco",
    "monaco": "Monaco",
    "barcelona": "Barcelona",
    "montmelo": "Barcelona",
    "catalunya": "Barcelona",
    "red bull ring": "Spielberg",
    "spielberg": "Spielberg",
    "austria": "Spielberg",
    "silverstone": "Silverstone",
    "hungaroring": "Budapest",
    "budapest": "Budapest",
    "hungary": "Budapest",
    "spa": "Spa",
    "spa-francorchamps": "Spa",
    "francorchamps": "Spa",
    "monza": "Monza",
    "marina bay": "Singapore",
    "singapore": "Singapore",
    "lusail": "Lusail",
    "qatar": "Lusail",
    "cota": "Austin",
    "austin": "Austin",
    "circuit of the americas": "Austin",
    "rodriguez": "Mexico City",
    "mexico city": "Mexico City",
    "mexico": "Mexico City",
    "interlagos": "São Paulo",
    "são paulo": "São Paulo",
    "sao paulo": "São Paulo",
    "brazil": "São Paulo",
    "las vegas": "Las Vegas",
    "yas marina": "Abu Dhabi",
    "yas marina circuit": "Abu Dhabi",
    "abu dhabi": "Abu Dhabi",
    "montreal": "Montreal",
    "circuit gilles villeneuve": "Montreal",
    "canada": "Montreal",
    "imola": "Imola",
    "zandvoort": "Zandvoort",
    "netherlands": "Zandvoort",
    "shanghai": "Shanghai",
    "china": "Shanghai",
}


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _session_type(session_name: str) -> str:
    mapping = {v: k for k, v in SESSION_NAMES.items()}
    return mapping.get(session_name, session_name.upper().replace(" ", "_"))


def _load_intervals(session_dir: Path, laps_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Join the intervals endpoint (gap to car ahead) to laps via a backward merge_asof on
    lap-end timestamp.  Returns a DataFrame with columns
    [driver_number, lap_number, gap_to_car_ahead] (float seconds; NaN when unavailable).

    The intervals parquet is optional — if absent this returns an empty DataFrame with
    the expected columns so callers can unconditionally merge.
    """
    empty = pd.DataFrame(columns=["driver_number", "lap_number", "gap_to_car_ahead"])

    intervals_path = session_dir / "intervals.parquet"
    if not intervals_path.exists() or laps_raw.empty:
        return empty

    intervals = _read_parquet(intervals_path)
    if intervals.empty or "interval" not in intervals.columns:
        return empty

    intervals = intervals.copy()
    # "+1 LAP" and any other non-numeric value → NaN
    intervals["gap_to_car_ahead"] = pd.to_numeric(intervals["interval"], errors="coerce")
    intervals["driver_number"] = _safe_float(intervals.get("driver_number"))
    intervals["date"] = pd.to_datetime(intervals["date"], utc=True, errors="coerce")
    intervals = intervals.dropna(subset=["driver_number", "date"])
    if intervals.empty:
        return empty

    if "date_start" not in laps_raw.columns or "lap_duration" not in laps_raw.columns:
        return empty

    laps = laps_raw.copy()
    laps["driver_number"] = _safe_float(laps.get("driver_number"))
    laps["lap_number"] = _safe_float(laps.get("lap_number"))
    laps["_date_start"] = pd.to_datetime(laps["date_start"], utc=True, errors="coerce")
    laps["_lap_duration"] = _safe_float(laps.get("lap_duration"))
    laps["_lap_end"] = laps["_date_start"] + pd.to_timedelta(
        laps["_lap_duration"].fillna(0), unit="s"
    )
    laps = laps.dropna(subset=["driver_number", "lap_number", "_lap_end"])
    if laps.empty:
        return empty

    intervals = intervals.sort_values(["driver_number", "date"])
    result_rows: list[dict] = []

    for drv_num, grp_laps in laps.groupby("driver_number"):
        drv_ivs = (
            intervals[intervals["driver_number"] == drv_num][["date", "gap_to_car_ahead"]]
            .copy()
        )
        grp_sorted = grp_laps[["lap_number", "_lap_end"]].sort_values("_lap_end")

        if drv_ivs.empty:
            for lap_n in grp_sorted["lap_number"]:
                result_rows.append(
                    {"driver_number": drv_num, "lap_number": lap_n, "gap_to_car_ahead": np.nan}
                )
            continue

        merged = pd.merge_asof(
            grp_sorted,
            drv_ivs,
            left_on="_lap_end",
            right_on="date",
            direction="backward",
        )
        for _, row in merged.iterrows():
            result_rows.append(
                {
                    "driver_number": drv_num,
                    "lap_number": row["lap_number"],
                    "gap_to_car_ahead": row.get("gap_to_car_ahead", np.nan),
                }
            )

    if not result_rows:
        return empty
    return pd.DataFrame(result_rows)


def _aggregate_car_data(session_dir: Path, laps_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 3.7 Hz car telemetry to per-lap features.

    Returns a DataFrame with columns:
        driver_number, lap_number,
        mean_throttle, mean_brake, max_speed, mean_speed, mean_rpm, drs_fraction

    All float32.  Empty DataFrame if car_data.parquet is absent.
    """
    car_path = session_dir / "car_data.parquet"
    if not car_path.exists() or laps_raw.empty:
        return pd.DataFrame()

    car = _read_parquet(car_path)
    if car.empty:
        return pd.DataFrame()

    required = {"date", "driver_number", "speed", "rpm", "throttle", "brake", "drs"}
    if not required.issubset(car.columns):
        return pd.DataFrame()

    car = car.copy()
    car["date"] = pd.to_datetime(car["date"], utc=True, errors="coerce")
    car["driver_number"] = _safe_float(car["driver_number"])
    for col in ("speed", "rpm", "throttle", "brake", "drs"):
        car[col] = _safe_float(car[col])
    car = car.dropna(subset=["date", "driver_number"])

    if "date_start" not in laps_raw.columns or "lap_duration" not in laps_raw.columns:
        return pd.DataFrame()

    laps = laps_raw.copy()
    laps["driver_number"] = _safe_float(laps["driver_number"])
    laps["lap_number"] = _safe_float(laps["lap_number"])
    laps["_t_start"] = pd.to_datetime(laps["date_start"], utc=True, errors="coerce")
    laps["_t_end"] = laps["_t_start"] + pd.to_timedelta(
        _safe_float(laps["lap_duration"]).fillna(0), unit="s"
    )
    laps = laps.dropna(subset=["driver_number", "lap_number", "_t_start"])

    results: list[pd.DataFrame] = []
    for drv, drv_laps in laps.groupby("driver_number"):
        drv_car = car[car["driver_number"] == drv].sort_values("date")
        if drv_car.empty:
            continue
        drv_laps = drv_laps[["lap_number", "_t_start", "_t_end"]].sort_values("_t_start")

        # Assign each telemetry sample to a lap via backward merge_asof then
        # filter out points that fall after the lap's end timestamp.
        merged = pd.merge_asof(
            drv_car[["date", "speed", "rpm", "throttle", "brake", "drs"]],
            drv_laps,
            left_on="date",
            right_on="_t_start",
            direction="backward",
        )
        merged = merged[merged["date"] <= merged["_t_end"]]
        if merged.empty:
            continue

        # DRS: values >= 10 indicate DRS flap is open (8 = eligible but not open).
        merged["drs_open"] = (merged["drs"].fillna(0) >= 10).astype(float)

        agg = merged.groupby("lap_number").agg(
            mean_throttle=("throttle", "mean"),
            mean_brake=("brake", "mean"),
            max_speed=("speed", "max"),
            mean_speed=("speed", "mean"),
            mean_rpm=("rpm", "mean"),
            drs_fraction=("drs_open", "mean"),
        ).reset_index()
        agg["driver_number"] = drv
        results.append(agg)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    for col in ("mean_throttle", "mean_brake", "max_speed", "mean_speed", "mean_rpm", "drs_fraction"):
        out[col] = out[col].astype("float32")
    return out


def _load_race_control_sc_laps(session_dir: Path, laps_raw: pd.DataFrame) -> frozenset[float]:
    """Return the set of lap_numbers confirmed as SC or VSC from race_control events.

    Uses actual flag data when available; returns empty frozenset if unavailable
    so the statistical fallback in _classify_lap_type takes over.

    Recognised flags (case-insensitive match): SAFETY CAR, VIRTUAL SAFETY CAR, VSC.
    """
    rc_path = session_dir / "race_control.parquet"
    if not rc_path.exists() or laps_raw.empty:
        return frozenset()

    rc = _read_parquet(rc_path)
    if rc.empty or "flag" not in rc.columns:
        return frozenset()

    sc_flags = {"safety car", "virtual safety car", "vsc"}
    rc = rc.copy()
    rc["flag_lower"] = rc["flag"].astype(str).str.lower().str.strip()
    rc["date"] = pd.to_datetime(rc.get("date"), utc=True, errors="coerce")

    # Build lap time windows for timestamp → lap_number mapping.
    laps = laps_raw.copy()
    laps["lap_number"] = _safe_float(laps["lap_number"])
    laps["_t_start"] = pd.to_datetime(laps.get("date_start"), utc=True, errors="coerce")
    laps["_t_end"] = laps["_t_start"] + pd.to_timedelta(
        _safe_float(laps.get("lap_duration")).fillna(0), unit="s"
    )
    laps = laps.dropna(subset=["lap_number", "_t_start"]).sort_values("_t_start")

    # Find SC/VSC deployment and GREEN (end) events.
    sc_events = rc[rc["flag_lower"].isin(sc_flags)].dropna(subset=["date"])
    green_events = rc[rc["flag_lower"] == "green"].dropna(subset=["date"])

    sc_lap_numbers: set[float] = set()

    for _, ev in sc_events.iterrows():
        # Find next GREEN event after this SC deployment.
        later_greens = green_events[green_events["date"] > ev["date"]]
        sc_end = later_greens["date"].min() if not later_greens.empty else pd.NaT

        # Collect all laps whose start falls within [sc_start, sc_end].
        window = laps[laps["_t_start"] >= ev["date"]]
        if pd.notna(sc_end):
            window = window[window["_t_start"] < sc_end]
        sc_lap_numbers.update(window["lap_number"].dropna().unique())

    return frozenset(sc_lap_numbers)


def _classify_lap_type(df: pd.DataFrame, session_name: str,
                       rc_sc_laps: frozenset[float] | None = None) -> pd.Series:
    """
    Classify each lap into one of: formation | pit_out | pit_in | sc | outlier | normal.

    Classification precedence (lower wins, higher-priority labels are applied last):
        normal → outlier → sc → pit_in → pit_out → formation

    SC detection: a lap_number is flagged as SC if ≥50 % of active drivers have a lap_time
    above 1.28× the session median *and* that condition holds for ≥3 consecutive lap numbers.

    Requires columns: lap_number, lap_time, stint_number, stint_age, driver_id.
    Returns a string Series aligned to df.index.
    """
    if df.empty:
        return pd.Series(dtype=object)

    session_type = _session_type(session_name)
    lap_type = pd.Series("normal", index=df.index, dtype=object)

    session_median = df["lap_time"].median()
    session_min = df["lap_time"].min()
    if pd.isna(session_median) or session_median <= 0:
        return lap_type

    slow_threshold = 1.28 * session_median
    fast_threshold = 0.97 * session_min
    is_slow = df["lap_time"] > slow_threshold

    # --- SC window detection ---
    # Prefer official race_control flags when available; fall back to statistical.
    if rc_sc_laps:
        sc_confirmed: set = set(rc_sc_laps)
    else:
        total_per_lap = df.groupby("lap_number")["driver_id"].nunique()
        slow_per_lap = df[is_slow].groupby("lap_number")["driver_id"].nunique()
        slow_frac = (slow_per_lap / total_per_lap).fillna(0)
        candidate_sc = set(slow_frac[slow_frac >= 0.5].index.tolist())

        sc_confirmed = set()
        if candidate_sc:
            run: list = []
            for lap_n in sorted(candidate_sc):
                if not run or lap_n == run[-1] + 1:
                    run.append(lap_n)
                else:
                    if len(run) >= 3:
                        sc_confirmed.update(run)
                    run = [lap_n]
            if len(run) >= 3:
                sc_confirmed.update(run)

    # --- outlier: isolated slow or suspiciously fast (base classification) ---
    outlier_mask = is_slow | (df["lap_time"] < fast_threshold)
    lap_type[outlier_mask] = "outlier"

    # --- SC: slow AND in confirmed SC window (overrides outlier) ---
    if sc_confirmed:
        sc_mask = is_slow & df["lap_number"].isin(sc_confirmed)
        lap_type[sc_mask] = "sc"

    # --- pit_in: last lap of a stint before the driver pits for another stint ---
    stint_last_lap = df.groupby(["driver_id", "stint_number"])["lap_number"].transform("max")
    driver_max_stint = df.groupby("driver_id")["stint_number"].transform("max")
    pit_in_mask = (df["lap_number"] == stint_last_lap) & (df["stint_number"] < driver_max_stint)
    lap_type[pit_in_mask] = "pit_in"

    # --- pit_out: first lap of any non-opening stint ---
    pit_out_mask = (df["stint_age"] == 1) & (df["stint_number"] > 1)
    lap_type[pit_out_mask] = "pit_out"

    # --- formation / rolling-start laps (highest priority for early race laps) ---
    if session_type in ("RACE", "SPRINT"):
        lap_type[df["lap_number"] <= 2] = "formation"

    return lap_type


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
        # Preserve original parquet columns (including date_start) for intervals join.
        laps_raw_parquet = laps

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
        _raw_circuit = (
            session.get("circuit_short_name") or session.get("location") or str(session.get("meeting_key", ""))
        ).strip().lower()
        circuit_id = CIRCUIT_ID_CANONICAL.get(_raw_circuit)
        if circuit_id is None:
            circuit_id = _raw_circuit.title() if _raw_circuit else "Unknown"
            logger.warning("circuit_id not in canonical map: %r — using %r", _raw_circuit, circuit_id)
        laps["circuit_id"] = circuit_id
        laps["year"] = year

        laps = laps.rename(columns={
            "driver_number": "driver_id",
            "name_acronym": "driver_code",
        })

        # --- gap_to_car_ahead (from intervals endpoint) ---
        gap_df = _load_intervals(session_dir, laps_raw_parquet)
        if not gap_df.empty:
            gap_df = gap_df.rename(columns={"driver_number": "driver_id"})
            laps = laps.merge(
                gap_df[["driver_id", "lap_number", "gap_to_car_ahead"]],
                on=["driver_id", "lap_number"],
                how="left",
            )
        else:
            laps["gap_to_car_ahead"] = np.nan
        laps["gap_to_car_ahead"] = laps["gap_to_car_ahead"].astype("float32")

        # --- car telemetry features (3.7 Hz aggregated to per-lap) ---
        car_feats = _aggregate_car_data(session_dir, laps_raw_parquet)
        if not car_feats.empty:
            car_feats = car_feats.rename(columns={"driver_number": "driver_id"})
            laps = laps.merge(
                car_feats[["driver_id", "lap_number",
                            "mean_throttle", "mean_brake", "max_speed",
                            "mean_speed", "mean_rpm", "drs_fraction"]],
                on=["driver_id", "lap_number"],
                how="left",
            )
        else:
            for col in ("mean_throttle", "mean_brake", "max_speed",
                        "mean_speed", "mean_rpm", "drs_fraction"):
                laps[col] = np.nan
        for col in ("mean_throttle", "mean_brake", "max_speed",
                    "mean_speed", "mean_rpm", "drs_fraction"):
            laps[col] = laps[col].astype("float32")

        # --- SC/VSC laps from race_control (falls back to statistical if absent) ---
        rc_sc_laps = _load_race_control_sc_laps(session_dir, laps_raw_parquet)

        # --- lap_type classification ---
        laps["lap_type"] = _classify_lap_type(
            laps, session.get("session_name", ""), rc_sc_laps=rc_sc_laps
        )

        # --- is_valid_train: "normal" and "pit_out" laps are valid training targets ---
        laps["is_valid_train"] = laps["lap_type"].isin({"normal", "pit_out"})

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
                "gap_to_car_ahead",
                # 3.7 Hz car telemetry aggregated to per-lap
                "mean_throttle",
                "mean_brake",
                "max_speed",
                "mean_speed",
                "mean_rpm",
                "drs_fraction",
                "lap_type",
                "is_valid_train",
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

    # Successful feature build means snapshot is fresh for available local years.
    refresh_snapshot_state(stale_mode=False, last_error=None)

    return df


def build_features_range(start_year: int, end_year: int) -> None:
    for year in range(start_year, end_year + 1):
        build_features_for_year(year)
