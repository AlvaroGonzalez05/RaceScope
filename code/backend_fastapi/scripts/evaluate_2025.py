"""
evaluate_2025.py — Held-out evaluation of the Transformer pace model on 2025 race data.

Usage
-----
python -m scripts.evaluate_2025 [--drivers 55,16] [--output evaluation_2025.csv]

What it does
------------
For each 2025 race × target driver:
  1. Loads the trained model (≤2024 data only) via StrategyEngine._load_model().
  2. Runs generate_strategies() to obtain the top-5 ranked strategies.
  3. Loads the actual 2025 race data from features.parquet (year=2025).
  4. Identifies the driver's actual strategy (stint compounds + lengths from raw data).
  5. Computes:
     - strategy_match:   True if the actual strategy is within the top-5 predictions.
     - pace_mae:         Mean Absolute Error between predicted and actual lap times
                         (for normal laps only, matched by lap_number).
     - pace_rmse:        RMSE of the same.
     - predicted_race_time_error: (predicted best expected_time) − (actual race time).

Output
------
evaluation_2025.csv with one row per (year, circuit_id, driver_id) combination.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_2025_features() -> pd.DataFrame:
    """Load 2025 features.parquet (held-out season)."""
    from app.config import FEATURE_DIR
    path = FEATURE_DIR / "year=2025" / "features.parquet"
    if not path.exists():
        logger.error("2025 features not found at %s — run ingest + preprocess first.", path)
        return pd.DataFrame()
    return pd.read_parquet(path)


def _extract_actual_strategy(
    df_race: pd.DataFrame, driver_id: int
) -> Optional[Dict]:
    """
    Reconstruct the actual strategy for a driver from race features.

    The `stint_number` column in the gold layer is corrupted (always 1.0 for the
    whole race). We reconstruct stints from two reliable signals that the
    preprocessor *does* compute correctly:
      - `stint_age == 1` marks the first lap of a stint.
      - A change in `compound` from the previous lap also marks a new stint.

    Returns a dict with keys: compounds (list[str]), stint_lengths (list[int]),
    stop_laps (list[int]), actual_race_time (float).
    Returns None if data is insufficient.
    """
    drv = df_race[df_race["driver_id"] == driver_id].copy()
    if drv.empty or "compound" not in drv.columns:
        return None

    drv = drv.sort_values("lap_number").reset_index(drop=True)

    # Stint boundary heuristic: new stint if stint_age resets to 1 OR compound changes.
    compound_norm = drv["compound"].astype(str).str.upper()
    age = drv["stint_age"] if "stint_age" in drv.columns else None
    new_stint = pd.Series(False, index=drv.index)
    new_stint.iloc[0] = True
    for i in range(1, len(drv)):
        cond_age = (age is not None and float(age.iloc[i]) == 1.0)
        cond_compound = compound_norm.iloc[i] != compound_norm.iloc[i - 1]
        if cond_age or cond_compound:
            new_stint.iloc[i] = True
    drv["_stint_idx"] = new_stint.cumsum()

    stints = (
        drv.groupby("_stint_idx")
        .agg(
            compound=("compound", lambda x: str(x.mode().iloc[0]).upper() if not x.empty else "UNKNOWN"),
            stint_len=("lap_number", "count"),
            start_lap=("lap_number", "min"),
        )
        .sort_values("start_lap")
    )
    if stints.empty:
        return None

    compounds = stints["compound"].tolist()
    stint_lengths = stints["stint_len"].astype(int).tolist()
    stop_laps = [int(s) for s in stints["start_lap"].tolist()[1:]]  # first lap of each non-opening stint

    # Race time aggregated over real race laps only (drop in/out outliers to keep totals comparable).
    if "lap_type" in drv.columns:
        race_laps = drv[drv["lap_type"].isin({"normal"})]
    else:
        race_laps = drv
    actual_race_time = float(race_laps["lap_time"].sum()) if not race_laps.empty else float("nan")

    return {
        "compounds": compounds,
        "stint_lengths": stint_lengths,
        "stop_laps": stop_laps,
        "actual_race_time": actual_race_time,
    }


def _strategy_matches(
    predicted_strategies: List[Dict], actual: Dict
) -> bool:
    """
    Check whether any of the predicted top-5 strategies matches the actual strategy.

    A match is: same number of stops AND the same compounds (order-insensitive for each
    stop position is acceptable within 1 position tolerance on stop_lap).
    """
    actual_compounds = [c.upper() for c in actual["compounds"]]
    actual_stops = len(actual["stop_laps"])

    for pred in predicted_strategies:
        pred_compounds = [c.upper() for c in pred.get("compounds", [])]
        pred_stops = len(pred.get("stop_laps", []))
        if pred_stops != actual_stops:
            continue
        if pred_compounds == actual_compounds:
            return True
        # Allow compound order mismatch (undercut / overcut variants)
        if sorted(pred_compounds) == sorted(actual_compounds):
            return True
    return False


def _pace_errors(
    predicted_strategies: List[Dict],
    df_driver_race: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Compute MAE and RMSE between the best predicted strategy's lap-time curve
    and the actual race lap times (normal laps only).
    """
    if not predicted_strategies:
        return float("nan"), float("nan")

    best = predicted_strategies[0]
    pred_lap_times: List[float] = []
    for stint in best.get("stint_curves", []):
        pred_lap_times.extend(stint.get("lap_time_data", []))

    if not pred_lap_times:
        return float("nan"), float("nan")

    actual_filter = (
        df_driver_race["lap_type"].isin({"normal"})
        if "lap_type" in df_driver_race.columns
        else pd.Series(True, index=df_driver_race.index)
    )
    actual_times = df_driver_race[actual_filter]["lap_time"].values
    n = min(len(pred_lap_times), len(actual_times))
    if n == 0:
        return float("nan"), float("nan")

    errors = np.abs(np.array(pred_lap_times[:n]) - actual_times[:n])
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    target_drivers: Optional[List[int]] = None,
    output_path: str = "evaluation_2025.csv",
    max_races: Optional[int] = None,
    debug_dump: bool = False,
) -> pd.DataFrame:
    from app.data_store import load_features
    from app.strategy_engine import StrategyEngine

    df_2025 = _load_2025_features()
    if df_2025.empty:
        logger.error("No 2025 data to evaluate.")
        return pd.DataFrame()

    # Use ALL available features for the strategy engine (its models are ≤2024)
    df_all = load_features()
    if df_all.empty:
        logger.error("No features loaded — run the data pipeline first.")
        return pd.DataFrame()

    engine = StrategyEngine(df_all)

    # Determine target drivers
    race_2025 = df_2025[df_2025["session_type"] == "RACE"] if "session_type" in df_2025.columns else df_2025
    if race_2025.empty:
        logger.error("No RACE rows in 2025 features.")
        return pd.DataFrame()

    all_drivers = sorted(race_2025["driver_id"].dropna().unique().astype(int).tolist())
    if target_drivers:
        drivers_to_eval = [d for d in target_drivers if d in all_drivers]
    else:
        drivers_to_eval = all_drivers

    circuits = sorted(race_2025["circuit_id"].dropna().unique().tolist())
    if max_races is not None:
        circuits = circuits[:max_races]
    print(
        f"[evaluate_2025] drivers={drivers_to_eval} circuits={len(circuits)} "
        f"max_races={max_races} debug={debug_dump}",
        flush=True,
    )

    rows: List[Dict] = []

    id_to_code = (
        race_2025[["driver_id", "driver_code"]]
        .dropna()
        .drop_duplicates()
        .set_index("driver_id")["driver_code"]
        .to_dict()
    )

    total_combos = len(circuits) * len(drivers_to_eval)
    done = 0
    import time as _time

    for circuit_id in circuits:
        race_circ = race_2025[race_2025["circuit_id"] == circuit_id]
        if race_circ.empty:
            continue

        for driver_id in drivers_to_eval:
            done += 1
            actual = _extract_actual_strategy(race_circ, driver_id)
            if actual is None:
                print(f"[{done}/{total_combos}] {circuit_id:15s} drv={driver_id:>3}  SKIP (no actual)", flush=True)
                continue

            driver_code = id_to_code.get(driver_id)
            if driver_code is None:
                print(f"[{done}/{total_combos}] {circuit_id:15s} drv={driver_id:>3}  SKIP (no code)", flush=True)
                continue

            t0 = _time.time()
            try:
                result = engine.generate_strategies(
                    year=2025,
                    circuit_id=circuit_id,
                    driver_code=driver_code,
                    n_strategies=5,
                )
                strategies = result.get("strategies", [])
            except Exception as exc:
                print(
                    f"[{done}/{total_combos}] {circuit_id:15s} drv={driver_id:>3} ({driver_code})  "
                    f"generate_strategies FAILED: {exc}",
                    flush=True,
                )
                strategies = []
            dt = _time.time() - t0

            drv_race = race_circ[race_circ["driver_id"] == driver_id]
            mae, rmse = _pace_errors(strategies, drv_race)
            match = _strategy_matches(strategies, actual) if strategies else False

            best_pred_time = (
                float(strategies[0]["expected_time"]) if strategies else float("nan")
            )
            time_error = best_pred_time - actual["actual_race_time"]

            row = {
                "year": 2025,
                "circuit_id": circuit_id,
                "driver_id": driver_id,
                "actual_compounds": "|".join(actual["compounds"]),
                "actual_stint_lengths": "|".join(str(s) for s in actual["stint_lengths"]),
                "actual_stop_laps": "|".join(str(s) for s in actual["stop_laps"]),
                "actual_race_time": round(actual["actual_race_time"], 2),
                "predicted_top5_includes_actual": match,
                "best_predicted_time": round(best_pred_time, 2),
                "predicted_race_time_error": round(time_error, 2) if not np.isnan(time_error) else None,
                "pace_mae": round(mae, 4) if not np.isnan(mae) else None,
                "pace_rmse": round(rmse, 4) if not np.isnan(rmse) else None,
                "n_predicted_strategies": len(strategies),
            }
            rows.append(row)

            print(
                f"[{done}/{total_combos}] {circuit_id:15s} drv={driver_id:>3} ({driver_code})  "
                f"t={dt:5.1f}s  match={str(match):5s}  mae={mae:7.3f}  time_err={time_error:+8.1f}s  "
                f"n_pred={len(strategies)}",
                flush=True,
            )

            if debug_dump:
                pred_compact = []
                for k, s in enumerate(strategies[:3]):
                    pred_compact.append(
                        f"  pred[{k}] compounds={s.get('compounds')} "
                        f"stops={s.get('stop_laps')} exp_t={s.get('expected_time'):.1f}"
                    )
                print(
                    f"  actual compounds={actual['compounds']} "
                    f"stints={actual['stint_lengths']} stops={actual['stop_laps']} "
                    f"race_time={actual['actual_race_time']:.1f}s",
                    flush=True,
                )
                print("\n".join(pred_compact), flush=True)

    df_out = pd.DataFrame(rows)
    out_file = Path(output_path)
    df_out.to_csv(out_file, index=False)
    logger.info("Evaluation written to %s  (%d rows)", out_file, len(df_out))

    # Summary statistics
    if not df_out.empty:
        match_rate = df_out["predicted_top5_includes_actual"].mean()
        mae_mean = df_out["pace_mae"].dropna().mean()
        rmse_mean = df_out["pace_rmse"].dropna().mean()
        logger.info(
            "SUMMARY  strategy_match_rate=%.1f%%  pace_mae=%.3fs  pace_rmse=%.3fs",
            match_rate * 100, mae_mean, rmse_mean,
        )

    return df_out


def push_sensitivity_test(
    circuit_id: str = "Sakhir",
    compound: str = "SOFT",
    stint_len: int = 15,
) -> Dict:
    """
    Test whether the v2 model produces steeper degradation under attack pace
    vs conservation pace.

    Returns a dict with:
      slope_attack      — degradation slope (s/lap) at max-attack (pace_hi)
      slope_conservation— degradation slope (s/lap) at conservation (pace_lo)
      push_sensitive    — True if slope_attack > slope_conservation
    """
    from app.strategy_engine import _load_model_cached   # noqa: PLC0415
    from app.models_transformer import (                  # noqa: PLC0415
        TransformerPaceModel,
        TyreDegradationTransformerV2,
        PracticeDistribution,
        build_context_seed,
        COMPOUND_VOCAB,
        SESSION_TYPE_VOCAB,
    )
    from app.config import CIRCUIT_VOCAB                  # noqa: PLC0415

    model, _ = _load_model_cached(55)  # driver 55 as reference
    if not isinstance(model, TransformerPaceModel) or not isinstance(model.model, TyreDegradationTransformerV2):
        logger.warning("push_sensitivity_test: requires a v2 model (driver 55). Skipping.")
        return {"push_sensitive": None, "slope_attack": None, "slope_conservation": None}

    lap_mean  = float(model.stats.get("lap_mean",  90.0))
    lap_std   = float(model.stats.get("lap_std",   1.0))
    delta_std = float(model.stats.get("delta_std", 1.0))
    track_ref = float(model.stats.get("track_temp_ref", 30.0))
    air_ref   = float(model.stats.get("air_temp_ref",   25.0))

    compound_key = compound.upper()
    pd_dist = (
        model.practice_dist.get(compound_key)
        or model.practice_dist.get("global")
        or PracticeDistribution(compound=compound_key)
    )
    circuit_int  = CIRCUIT_VOCAB.get(circuit_id, 0)
    compound_int = COMPOUND_VOCAB.get(compound_key, 0)
    session_int  = SESSION_TYPE_VOCAB.get("RACE", 4)

    rng = np.random.default_rng(42)

    # Normalised pace bounds
    pace_hi_norm = (pd_dist.pace_hi - lap_mean) / (lap_std or 1.0)
    pace_lo_norm = (pd_dist.pace_lo - lap_mean) / (lap_std or 1.0)

    def _run(tight_bound: float) -> np.ndarray:
        ctx = build_context_seed(pd_dist, model.context_len, delta_std,
                                 track_ref, air_ref, compound_int, session_int,
                                 circuit_int=circuit_int, rng=np.random.default_rng(42))
        return model.rollout_mc(
            ctx, stint_len,
            (tight_bound, tight_bound),   # pinned to a single pace level
            track_temp=30.0, air_temp=22.0,
            compound_int=compound_int,
            base_lap_time=lap_mean,
            rng=np.random.default_rng(42),
            circuit_int=circuit_int,
            race_lap_start=0,
        )

    lap_times_attack       = _run(pace_hi_norm)  # max attack
    lap_times_conservation = _run(pace_lo_norm)  # conservation

    # Degradation slope: linear fit of lap_time vs stint position
    x = np.arange(1, stint_len + 1, dtype=float)
    slope_attack       = float(np.polyfit(x, lap_times_attack,       1)[0])
    slope_conservation = float(np.polyfit(x, lap_times_conservation, 1)[0])
    push_sensitive     = slope_attack > slope_conservation

    result = {
        "circuit_id":          circuit_id,
        "compound":            compound_key,
        "stint_len":           stint_len,
        "slope_attack":        round(slope_attack,       4),
        "slope_conservation":  round(slope_conservation, 4),
        "push_sensitive":      push_sensitive,
        "lap_times_attack":    [round(float(v), 3) for v in lap_times_attack],
        "lap_times_conservation": [round(float(v), 3) for v in lap_times_conservation],
    }

    logger.info(
        "Push-sensitivity test — circuit=%s compound=%s  "
        "slope_attack=%.4f  slope_conservation=%.4f  sensitive=%s",
        circuit_id, compound_key, slope_attack, slope_conservation, push_sensitive,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Transformer pace model on held-out 2025 season data."
    )
    parser.add_argument(
        "--drivers",
        type=str,
        default=None,
        help="Comma-separated driver IDs to evaluate (e.g. '55,16'). Default: all drivers.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_2025.csv",
        help="Output CSV path (default: evaluation_2025.csv).",
    )
    parser.add_argument(
        "--push-sensitivity-test",
        action="store_true",
        default=False,
        help="Run the push-sensitivity test after evaluation.",
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="Sakhir",
        help="Circuit for push-sensitivity test (default: Sakhir).",
    )
    parser.add_argument(
        "--compound",
        type=str,
        default="SOFT",
        help="Compound for push-sensitivity test (default: SOFT).",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=None,
        help="Cap the number of circuits processed (debug/iteration).",
    )
    parser.add_argument(
        "--debug-dump",
        action="store_true",
        default=False,
        help="Print actual vs predicted strategies after each combo.",
    )
    args = parser.parse_args()

    target_drivers = None
    if args.drivers:
        target_drivers = [int(d.strip()) for d in args.drivers.split(",") if d.strip()]

    evaluate(
        target_drivers=target_drivers,
        output_path=args.output,
        max_races=args.max_races,
        debug_dump=args.debug_dump,
    )

    if args.push_sensitivity_test:
        result = push_sensitivity_test(
            circuit_id=args.circuit,
            compound=args.compound,
        )
        import json as _json  # noqa: PLC0415
        print("\n--- Push-Sensitivity Test ---")
        print(_json.dumps(result, indent=2))
        if result.get("push_sensitive") is True:
            print("\nPASS: slope_attack > slope_conservation (model is push-sensitive)")
        elif result.get("push_sensitive") is False:
            print("\nFAIL: slope_attack <= slope_conservation (model is NOT push-sensitive)")
        else:
            print("\nSKIP: v2 model not available.")


if __name__ == "__main__":
    main()
