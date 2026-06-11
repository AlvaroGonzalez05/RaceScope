"""Diagnose the spread of `expected_time` across drivers in a single race.

Background
----------
Two drivers in the SAME race were showing total race durations that differed
by ~45 minutes. Physically, a driver lapped once is ~90 s behind the leader.
This script measures the spread end-to-end so each mitigation phase can be
verified against the previous baseline.

For every driver who appears in the race (year+circuit_id, session_type=RACE),
the script captures:
  * raw `params.base` from each profile fallback level (SOFT/MEDIUM/HARD)
  * the temperature-corrected pace seen by `_pace_table`
  * the fallback level reached (1: circuit+compound, 2: driver+compound,
    3: global compound, 4: hardcoded literal)
  * the analytical `expected_time` of the *best* generated strategy
  * `sum(stints) / total_laps` (sanity: must equal 1.0)

Output
------
* `reports/pace_spread_<label>.csv`: one row per (driver, compound) for the
  per-compound facts, plus one row per driver flagged with `compound="best"`
  for the per-driver `expected_time`.
* stdout summary: median, IQR, min, max, max−min of `expected_time`; list of
  drivers in fallback level ≥ 3; list of drivers whose base diverges from the
  circuit median by >5 s.

Usage
-----
    python -m scripts.diagnose_pace_spread \
        --year 2023 --circuit Sakhir --label baseline

Re-run after each mitigation phase with a different `--label`; the script is
read-only against the model code path, so any improvement is fully attributable
to the changes in `app/`.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

# Allow `python -m scripts.diagnose_pace_spread` and direct `python scripts/...`.
THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = THIS_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.config import TEMP_CORR_CLAMP_S  # noqa: E402
from app.data_store import load_features  # noqa: E402
from app.driver_profile import (  # noqa: E402
    load_driver_profile,
    resolve_profile_params_with_level,
)
from app.strategy_engine import StrategyEngine  # noqa: E402


COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def _drivers_in_race(features: pd.DataFrame, year: int, circuit_id: str) -> List[str]:
    """Return the list of driver_codes who logged at least one race lap."""
    race = features[
        (features["year"] == year)
        & (features["circuit_id"] == circuit_id)
        & (features["session_type"] == "RACE")
    ]
    if race.empty:
        return []
    codes = (
        race["driver_code"].dropna().astype(str).str.upper().unique().tolist()
    )
    return sorted(codes)


def _per_compound_facts(
    engine: StrategyEngine,
    driver_code: str,
    circuit_id: str,
) -> List[Dict[str, object]]:
    """For each compound, return level/base/post-temp-corr/slope/coefs."""
    context = engine._context(engine.features["year"].iloc[0], circuit_id)
    # Re-derive context with the real year:
    context = engine._context(_active_year(engine.features), circuit_id)

    profile = load_driver_profile(driver_code)
    rows: List[Dict[str, object]] = []
    for compound in COMPOUNDS:
        params, level = resolve_profile_params_with_level(profile, circuit_id, compound)
        track_corr_raw = params.track_coef * (context.track_temp - params.track_ref)
        air_corr_raw = params.air_coef * (context.air_temp - params.air_ref)
        track_corr_eff = float(
            np.clip(track_corr_raw, -TEMP_CORR_CLAMP_S, TEMP_CORR_CLAMP_S)
        )
        air_corr_eff = float(
            np.clip(air_corr_raw, -TEMP_CORR_CLAMP_S, TEMP_CORR_CLAMP_S)
        )
        pace_raw = params.base + track_corr_raw + air_corr_raw
        pace_eff = params.base + track_corr_eff + air_corr_eff
        rows.append(
            {
                "driver_code": driver_code,
                "compound": compound,
                "fallback_level": level,
                "base_s": round(float(params.base), 3),
                "slope_s_per_lap": round(float(params.slope), 5),
                "track_coef": round(float(params.track_coef), 5),
                "air_coef": round(float(params.air_coef), 5),
                "track_ref": round(float(params.track_ref), 2),
                "air_ref": round(float(params.air_ref), 2),
                "track_corr_raw_s": round(float(track_corr_raw), 3),
                "air_corr_raw_s": round(float(air_corr_raw), 3),
                "track_corr_eff_s": round(float(track_corr_eff), 3),
                "air_corr_eff_s": round(float(air_corr_eff), 3),
                "pace_raw_s": round(float(pace_raw), 3),
                "pace_eff_s": round(float(pace_eff), 3),
            }
        )
    return rows


def _active_year(features: pd.DataFrame) -> int:
    """Used internally to recover year inside helpers that only have engine.features."""
    return int(features["year"].iloc[0])


def _per_driver_best(
    engine: StrategyEngine,
    year: int,
    circuit_id: str,
    driver_code: str,
) -> Dict[str, object]:
    """Run the engine end-to-end and return the best-strategy expected_time."""
    try:
        payload = engine.generate_strategies(year, circuit_id, driver_code)
    except Exception as exc:  # noqa: BLE001 — diagnostic must keep going
        return {
            "driver_code": driver_code,
            "compound": "best",
            "error": f"{type(exc).__name__}: {exc}",
            "expected_time_s": None,
            "total_laps": None,
            "stints_sum": None,
            "stints_match_total_laps": None,
            "best_strategy_type": None,
            "best_compounds": None,
            "best_stop_laps": None,
        }
    strategies = payload.get("strategies", [])
    if not strategies:
        return {
            "driver_code": driver_code,
            "compound": "best",
            "error": "no_strategies",
            "expected_time_s": None,
            "total_laps": int(payload["context"]["total_laps"]),
            "stints_sum": None,
            "stints_match_total_laps": None,
            "best_strategy_type": None,
            "best_compounds": None,
            "best_stop_laps": None,
        }
    best = min(strategies, key=lambda s: s["expected_time"])
    total_laps = int(payload["context"]["total_laps"])
    stints_sum = int(sum(best["stints"]))
    return {
        "driver_code": driver_code,
        "compound": "best",
        "error": "",
        "expected_time_s": round(float(best["expected_time"]), 2),
        "total_laps": total_laps,
        "stints_sum": stints_sum,
        "stints_match_total_laps": stints_sum == total_laps,
        "best_strategy_type": best["type"],
        "best_compounds": "|".join(best["compounds"]),
        "best_stop_laps": "|".join(str(s) for s in best["stop_laps"]),
    }


def _summarise(per_driver: List[Dict[str, object]], total_laps: int) -> Dict[str, float]:
    times = [
        float(r["expected_time_s"])
        for r in per_driver
        if r.get("expected_time_s") is not None
    ]
    if not times:
        return {}
    arr = np.array(times, dtype=float)
    return {
        "n_drivers": int(len(arr)),
        "median_s": float(np.median(arr)),
        "iqr_s": float(np.subtract(*np.percentile(arr, [75, 25]))),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "spread_s": float(arr.max() - arr.min()),
        "spread_min": float((arr.max() - arr.min()) / 60.0),
        "median_per_lap_s": float(np.median(arr) / total_laps) if total_laps else math.nan,
    }


def _format_hms(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "--:--:--"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:01d}:{m:02d}:{sec:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--circuit", required=True, help="circuit_id (e.g. Sakhir)")
    parser.add_argument(
        "--label",
        required=True,
        help="Tag for output file (e.g. baseline, phase1, phase2).",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(BACKEND_DIR / "reports"),
        help="Directory where the CSV summary is written.",
    )
    args = parser.parse_args()

    features = load_features()
    if features.empty:
        print("ERROR: no features available — run the pipeline first.", file=sys.stderr)
        return 2

    drivers = _drivers_in_race(features, args.year, args.circuit)
    if not drivers:
        print(
            f"ERROR: no race-session drivers found for {args.year}/{args.circuit}.",
            file=sys.stderr,
        )
        return 2

    engine = StrategyEngine(features)
    context = engine._context(args.year, args.circuit)
    total_laps = int(context.total_laps)

    print(
        f"Diagnostic: year={args.year} circuit={args.circuit} "
        f"drivers={len(drivers)} total_laps={total_laps} "
        f"track_temp={context.track_temp:.1f} air_temp={context.air_temp:.1f} "
        f"pit_loss={context.pit_loss:.2f}"
    )

    rows: List[Dict[str, object]] = []
    per_driver_rows: List[Dict[str, object]] = []
    for code in drivers:
        rows.extend(_per_compound_facts(engine, code, args.circuit))
        best = _per_driver_best(engine, args.year, args.circuit, code)
        rows.append(best)
        per_driver_rows.append(best)

    summary = _summarise(per_driver_rows, total_laps)

    # Write CSV
    out_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"pace_spread_{args.label}.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {out_csv}")

    if not summary:
        print("No expected_time values collected — nothing to summarise.")
        return 1

    print("")
    print("=== Summary of expected_time across drivers ===")
    print(f"  n_drivers        = {summary['n_drivers']}")
    print(
        "  median           = "
        f"{summary['median_s']:.1f} s  ({_format_hms(summary['median_s'])})"
    )
    print(f"  iqr              = {summary['iqr_s']:.1f} s")
    print(f"  min              = {summary['min_s']:.1f} s  ({_format_hms(summary['min_s'])})")
    print(f"  max              = {summary['max_s']:.1f} s  ({_format_hms(summary['max_s'])})")
    print(
        f"  spread (max-min) = {summary['spread_s']:.1f} s "
        f"({summary['spread_min']:.1f} min)"
    )
    print(
        f"  median/lap       = {summary['median_per_lap_s']:.2f} s/lap "
        f"over {total_laps} laps"
    )

    print("")
    flagged_levels = [
        r
        for r in rows
        if r.get("compound") in COMPOUNDS and int(r.get("fallback_level", 0)) >= 3
    ]
    if flagged_levels:
        print(f"Drivers with fallback level >= 3 ({len(flagged_levels)} rows):")
        for r in flagged_levels:
            print(
                f"  {r['driver_code']:>4s}  {r['compound']:<6s}  "
                f"level={r['fallback_level']}  base={r['base_s']:.2f}s  "
                f"pace_with_temp={r['pace_with_temp_s']:.2f}s"
            )

    print("")
    pace_eff_by_compound: Dict[str, List[float]] = {c: [] for c in COMPOUNDS}
    pace_raw_by_compound: Dict[str, List[float]] = {c: [] for c in COMPOUNDS}
    for r in rows:
        if r.get("compound") in COMPOUNDS:
            pace_eff_by_compound[str(r["compound"])].append(float(r["pace_eff_s"]))
            pace_raw_by_compound[str(r["compound"])].append(float(r["pace_raw_s"]))
    print("Per-compound effective pace (post-clamp, what the engine uses):")
    for compound, vals in pace_eff_by_compound.items():
        if not vals:
            continue
        med = float(np.median(vals))
        lo = float(min(vals))
        hi = float(max(vals))
        print(
            f"  {compound:<6s} median={med:.2f}s  min={lo:.2f}s  max={hi:.2f}s  spread={hi - lo:.2f}s"
        )
    print("Per-compound raw pace (pre-clamp, OLS as-is — forensics only):")
    for compound, vals in pace_raw_by_compound.items():
        if not vals:
            continue
        med = float(np.median(vals))
        lo = float(min(vals))
        hi = float(max(vals))
        print(
            f"  {compound:<6s} median={med:.2f}s  min={lo:.2f}s  max={hi:.2f}s  spread={hi - lo:.2f}s"
        )

    print("")
    print("Outlier drivers (best expected_time more than 90 s from median):")
    median_t = summary["median_s"]
    outliers = sorted(
        [
            r
            for r in per_driver_rows
            if r.get("expected_time_s") is not None
            and abs(float(r["expected_time_s"]) - median_t) > 90.0
        ],
        key=lambda r: float(r["expected_time_s"]),  # type: ignore[arg-type]
    )
    if not outliers:
        print("  (none)")
    for r in outliers:
        delta = float(r["expected_time_s"]) - median_t  # type: ignore[arg-type]
        print(
            f"  {r['driver_code']:>4s}  expected_time={_format_hms(float(r['expected_time_s']))} "  # type: ignore[arg-type]
            f"({float(r['expected_time_s']):.1f}s)  Δmedian={delta:+.1f}s  "  # type: ignore[arg-type]
            f"strategy={r['best_compounds']}"
        )

    print("")
    bad_sum = [
        r for r in per_driver_rows if r.get("stints_match_total_laps") is False
    ]
    if bad_sum:
        print(f"WARNING: {len(bad_sum)} driver(s) had stint_sum != total_laps:")
        for r in bad_sum:
            print(
                f"  {r['driver_code']}: stints_sum={r['stints_sum']} "
                f"total_laps={r['total_laps']} compounds={r['best_compounds']}"
            )

    print("")
    print(f"Wrote CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
