from __future__ import annotations

import math
import random
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
from functools import lru_cache
import numpy as np
import pandas as pd

from .config import (
    DEFAULT_RISK_LAMBDA,
    DEFAULT_STRATEGY_COUNT,
    MODELS_DIR,
    PIT_WINDOW_BIN,
    RANDOM_SEED,
    MC_TOP_K,
    PACE_CURVE_CACHE_DIR,
)
from .models_lstm import LSTMPaceModel, ModelBundle
from .driver_profile import load_driver_profile, resolve_profile_params


@dataclass
class RaceContext:
    year: int
    total_laps: int
    track_temp: float
    air_temp: float
    pit_loss: float
    sc_probability: float


@dataclass
class StrategyCandidate:
    strategy_type: str
    compounds: List[str]
    stint_lengths: List[int]
    pit_windows: List[Dict[str, int]]
    stop_laps: List[int]


class StrategyEngine:
    def __init__(self, features: pd.DataFrame):
        self.features = features
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        self.valid_compounds = {"SOFT", "MEDIUM", "HARD"}

    def _context(self, year: int, circuit_id: str) -> RaceContext:
        df = self.features
        df = df[(df["year"] == year) & (df["circuit_id"] == circuit_id)]
        if df.empty:
            return RaceContext(year=year, total_laps=55, track_temp=30.0, air_temp=22.0, pit_loss=22.5, sc_probability=0.2)

        race = df[df["session_type"] == "RACE"]
        total_laps = int(race["lap_number"].max()) if not race.empty else int(df["lap_number"].max())
        track_temp = float(df["track_temp"].mean()) if df["track_temp"].notna().any() else 30.0
        air_temp = float(df["air_temp"].mean()) if df["air_temp"].notna().any() else 22.0

        return RaceContext(
            year=year,
            total_laps=total_laps,
            track_temp=track_temp,
            air_temp=air_temp,
            pit_loss=22.5,
            sc_probability=0.2,
        )

    def _compound_stats(self, driver_id: int, year: int, circuit_id: str) -> Dict[str, Dict[str, float]]:
        df = self.features
        df = df[(df["year"] == year) & (df["circuit_id"] == circuit_id)]
        driver_df = df[df["driver_id"] == driver_id]
        if driver_df.empty:
            driver_df = df

        stats = {}
        for compound, cdf in driver_df.groupby("compound"):
            if compound is None or compound != compound:
                continue
            compound_key = str(compound).upper()
            if compound_key not in self.valid_compounds:
                continue
            base = cdf["lap_time"].median()
            if len(cdf) > 3:
                slope = np.polyfit(cdf["stint_age"], cdf["lap_time"], 1)[0]
            else:
                slope = 0.04
            stats[compound_key] = {"base": float(base), "slope": float(slope)}
        return stats

    def _tyre_life_bounds(self, year: int, circuit_id: str) -> Dict[str, Tuple[int, int]]:
        df = self.features
        df = df[(df["year"] == year) & (df["circuit_id"] == circuit_id)]
        if df.empty:
            return {"SOFT": (12, 18), "MEDIUM": (18, 26), "HARD": (24, 34)}

        stint_lengths = (
            df.groupby(["driver_id", "session_key", "stint_number", "compound"])  
            ["stint_age"].max().reset_index()
        )
        bounds = {}
        for compound, sdf in stint_lengths.groupby("compound"):
            if compound is None or compound != compound:
                continue
            compound_key = str(compound).upper()
            if compound_key not in self.valid_compounds:
                continue
            q1 = int(np.quantile(sdf["stint_age"], 0.2))
            q9 = int(np.quantile(sdf["stint_age"], 0.8))
            bounds[compound_key] = (max(q1, 8), max(q9, q1 + 4))
        if not bounds:
            bounds = {"SOFT": (12, 18), "MEDIUM": (18, 26), "HARD": (24, 34)}
        return bounds

    def _load_model(self, driver_id: int) -> Tuple[LSTMPaceModel, int]:
        return _load_model_cached(driver_id)

    def _predict_stint(self, model: LSTMPaceModel, driver_id: int, compound: str, stint_len: int, context: RaceContext, base: float, slope: float, circuit_id: str) -> np.ndarray:
        return _predict_stint_cached(
            driver_id,
            compound,
            stint_len,
            context.track_temp,
            context.air_temp,
            base,
            slope,
            circuit_id,
        )

    def _candidate_strategies(self, total_laps: int, bounds: Dict[str, Tuple[int, int]]) -> List[StrategyCandidate]:
        compounds = [c for c in bounds.keys() if c in self.valid_compounds] or ["SOFT", "MEDIUM", "HARD"]
        candidates = []

        for c1 in compounds:
            for c2 in compounds:
                if c1 == c2:
                    continue
                min1, max1 = bounds.get(c1, (10, 20))
                min2, max2 = bounds.get(c2, (10, 20))
                min_stop = max(min1, total_laps - max2)
                max_stop = min(max1, total_laps - min2)
                if min_stop >= max_stop:
                    continue
                stop_lap = int((min_stop + max_stop) / 2)
                candidates.append(
                    StrategyCandidate(
                        strategy_type="1-stop",
                        compounds=[c1, c2],
                        stint_lengths=[stop_lap, total_laps - stop_lap],
                        pit_windows=[{"lap_min": min_stop, "lap_max": max_stop}],
                        stop_laps=[stop_lap],
                    )
                )

        for c1 in compounds:
            for c2 in compounds:
                for c3 in compounds:
                    if len({c1, c2, c3}) < 2:
                        continue
                    min1, max1 = bounds.get(c1, (8, 16))
                    min2, max2 = bounds.get(c2, (8, 16))
                    min3, max3 = bounds.get(c3, (8, 16))

                    for stop1 in range(min1, max1 + 1, 2):
                        remaining = total_laps - stop1
                        min_stop2 = max(min2, remaining - max3)
                        max_stop2 = min(max2, remaining - min3)
                        if min_stop2 >= max_stop2:
                            continue
                        stop2 = stop1 + int((min_stop2 + max_stop2) / 2)
                        len2 = stop2 - stop1
                        len3 = total_laps - stop2
                        if len3 < min3 or len3 > max3:
                            continue
                        candidates.append(
                            StrategyCandidate(
                                strategy_type="2-stop",
                                compounds=[c1, c2, c3],
                                stint_lengths=[stop1, len2, len3],
                                pit_windows=[
                                    {"lap_min": stop1 - 2, "lap_max": stop1 + 2},
                                    {"lap_min": stop2 - 2, "lap_max": stop2 + 2},
                                ],
                                stop_laps=[stop1, stop2],
                            )
                        )

        return candidates

    def _cluster_key(self, candidate: StrategyCandidate) -> Tuple[int, ...]:
        return tuple(int(stop / PIT_WINDOW_BIN) for stop in candidate.stop_laps)

    def _pace_curve_path(self, year: int, circuit_id: str, driver_id: int, context: RaceContext) -> Path:
        PACE_CURVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        circuit_safe = str(circuit_id).replace(" ", "_")
        key = f"{year}_{circuit_safe}_{driver_id}_{context.track_temp:.1f}_{context.air_temp:.1f}.parquet"
        return PACE_CURVE_CACHE_DIR / key

    def _precompute_pace_curves(self, year: int, circuit_id: str, driver_id: int, context: RaceContext) -> Dict[str, np.ndarray]:
        path = self._pace_curve_path(year, circuit_id, driver_id, context)
        if path.exists():
            return _load_pace_curves_cached(str(path))

        profile = load_driver_profile(driver_id)
        model, _ = self._load_model(driver_id)
        total_laps = context.total_laps

        curves = {}
        for compound in self.valid_compounds:
            params = resolve_profile_params(profile, circuit_id, compound)
            laps = np.arange(1, total_laps + 1)
            base_series = (
                params.base
                + params.slope * (laps - 1)
                + params.track_coef * (context.track_temp - params.track_ref)
                + params.air_coef * (context.air_temp - params.air_ref)
            )
            df = pd.DataFrame({
                "lap_number": laps,
                "stint_age": laps,
                "compound": compound,
                "session_type": "RACE",
                "circuit_id": circuit_id,
                "track_temp": context.track_temp,
                "air_temp": context.air_temp,
                "lap_time": base_series,
            })
            curves[compound] = model.predict_stint(df)

        rows = []
        for compound, series in curves.items():
            for lap_idx, lap_time in enumerate(series, start=1):
                rows.append({"lap": lap_idx, "compound": compound, "lap_time": float(lap_time)})
        pd.DataFrame(rows).to_parquet(path, index=False)

        return curves

    def _analytical_eval(
        self,
        candidate: StrategyCandidate,
        curves: Dict[str, np.ndarray],
        context: RaceContext,
        traffic_mu: float,
        traffic_sigma: float,
    ) -> Tuple[float, float]:
        total_laps = context.total_laps
        total_mean = 0.0
        total_var = 0.0

        for stint_len, compound in zip(candidate.stint_lengths, candidate.compounds):
            series = curves.get(compound.upper())
            if series is None or len(series) < stint_len:
                series = curves.get("MEDIUM", np.full(stint_len, 90.0))
            total_mean += float(np.sum(series[:stint_len]))
            total_mean += traffic_mu * stint_len
            total_var += (traffic_sigma ** 2) * stint_len

        if candidate.stop_laps:
            sc_range = max(1, context.total_laps - 9)
            for stop in candidate.stop_laps:
                p_window = min(1.0, 5 / sc_range)
                p_sc = context.sc_probability * p_window
                normal = context.pit_loss
                reduced = max(12.0, context.pit_loss - 8.0)
                mean = p_sc * reduced + (1 - p_sc) * normal
                var = p_sc * reduced**2 + (1 - p_sc) * normal**2 - mean**2
                total_mean += mean
                total_var += var
        else:
            total_mean += 0.0

        sc_mean = context.sc_probability * 15.0
        sc_var = context.sc_probability * (15.0**2) - sc_mean**2
        total_mean += sc_mean
        total_var += sc_var

        return total_mean, total_var

    def _simulate_strategy(
        self,
        model: LSTMPaceModel,
        driver_id: int,
        candidate: StrategyCandidate,
        context: RaceContext,
        stats: Dict[str, Dict[str, float]],
        circuit_id: str,
        n_sim: int = 200,
    ) -> Tuple[float, float, List[float]]:
        n_sim = max(n_sim, 1)
        sc_events = np.random.rand(n_sim) < context.sc_probability
        sc_laps = np.random.randint(5, max(6, context.total_laps - 5), size=n_sim)

        pit_loss = np.full(n_sim, context.pit_loss, dtype=float)
        if candidate.stop_laps:
            stops = np.array(candidate.stop_laps, dtype=int)
            near_stop = (np.abs(sc_laps[:, None] - stops[None, :]) <= 2).any(axis=1)
            pit_loss = np.where(sc_events & near_stop, np.maximum(12.0, pit_loss - 8.0), pit_loss)

        totals = np.zeros(n_sim, dtype=float)
        traffic_mu = 0.15
        traffic_sigma = 0.05

        curves = self._precompute_pace_curves(context.year, circuit_id, driver_id, context)
        for stint_len, compound in zip(candidate.stint_lengths, candidate.compounds):
            series = curves.get(compound.upper())
            if series is None or len(series) < stint_len:
                base = stats.get(compound.upper(), {}).get("base", 90.0)
                slope = stats.get(compound.upper(), {}).get("slope", 0.05)
                series = self._predict_stint(model, driver_id, compound.upper(), stint_len, context, base, slope, circuit_id)
            base_sum = float(np.sum(series[:stint_len]))
            noise = np.random.normal(
                traffic_mu * stint_len,
                traffic_sigma * math.sqrt(stint_len),
                size=n_sim,
            )
            totals += base_sum + noise

        totals += pit_loss * len(candidate.stop_laps)
        totals += np.where(sc_events, 15.0, 0.0)

        return float(np.mean(totals)), float(np.var(totals)), totals.tolist()

    def _stint_curves(
        self,
        candidate: StrategyCandidate,
        curves: Dict[str, np.ndarray],
    ) -> List[Dict]:
        stint_payload: List[Dict] = []
        current_lap = 1

        for stint_len, compound in zip(candidate.stint_lengths, candidate.compounds):
            series = curves.get(compound.upper())
            if series is None or len(series) < stint_len:
                series = np.linspace(90.0, 92.0, stint_len)
            raw = np.asarray(series[:stint_len], dtype=float)
            if raw.size == 0:
                continue

            # Enforce monotonic degradation for stable UX curves.
            monotonic = np.maximum.accumulate(raw)
            total_deg = float(monotonic[-1] - monotonic[0])
            if total_deg <= 1e-6:
                life = np.linspace(100.0, 90.0, raw.size)
            else:
                life = 100.0 - ((monotonic - monotonic[0]) / total_deg) * 100.0
            life = np.clip(life, 0.0, 100.0)

            end_lap = current_lap + stint_len - 1
            stint_payload.append(
                {
                    "compound": compound,
                    "start_lap": current_lap,
                    "end_lap": end_lap,
                    "lap_time_data": [float(round(v, 4)) for v in raw.tolist()],
                    "tyre_life_data": [float(round(v, 4)) for v in life.tolist()],
                    # Legacy field kept for compatibility with previous UI versions.
                    "degradation_data": [float(round(v, 4)) for v in life.tolist()],
                }
            )
            current_lap = end_lap + 1

        return stint_payload

    def generate_strategies(
        self,
        year: int,
        circuit_id: str,
        driver_id: int,
        risk_bias: float = DEFAULT_RISK_LAMBDA,
        n_strategies: int = DEFAULT_STRATEGY_COUNT,
        opponent_id: int | None = None,
        debug_profile: bool = False,
    ) -> Dict:
        context = self._context(year, circuit_id)
        stats = self._compound_stats(driver_id, year, circuit_id)
        bounds = self._tyre_life_bounds(year, circuit_id)
        model, _ = self._load_model(driver_id)
        curves = self._precompute_pace_curves(year, circuit_id, driver_id, context)

        candidates = self._candidate_strategies(context.total_laps, bounds)
        opponent_best = None
        if opponent_id is not None:
            opp_curves = self._precompute_pace_curves(year, circuit_id, opponent_id, context)
            opp_candidates = self._candidate_strategies(context.total_laps, bounds)
            opp_scores = []
            for opp_candidate in opp_candidates:
                mean, var = self._analytical_eval(
                    opp_candidate,
                    opp_curves,
                    context,
                    traffic_mu=0.15,
                    traffic_sigma=0.05,
                )
                opp_scores.append(mean + risk_bias * var)
            if opp_scores:
                opponent_best = float(min(opp_scores))
        ranked = []

        for candidate in candidates:
            mean, var = self._analytical_eval(
                candidate,
                curves,
                context,
                traffic_mu=0.15,
                traffic_sigma=0.05,
            )
            score = mean + risk_bias * var
            if opponent_best is not None and mean > opponent_best:
                score += (mean - opponent_best) * 0.25
            ranked.append((score, mean, var, candidate))

        ranked.sort(key=lambda x: x[0])

        final = []
        seen = set()
        topk = ranked[:MC_TOP_K]
        refined = {}
        for score, mean, var, candidate in topk:
            mc_mean, mc_var, _ = self._simulate_strategy(
                model,
                driver_id,
                candidate,
                context,
                stats,
                circuit_id,
                n_sim=200,
            )
            refined[id(candidate)] = (mc_mean, mc_var)

        for score, mean, var, candidate in ranked:
            if id(candidate) in refined:
                mean, var = refined[id(candidate)]
                score = mean + risk_bias * var
            key = self._cluster_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            strategy_fingerprint = {
                "year": year,
                "circuit_id": circuit_id,
                "driver_id": driver_id,
                "type": candidate.strategy_type,
                "compounds": candidate.compounds,
                "stints": candidate.stint_lengths,
                "stop_laps": candidate.stop_laps,
                "pit_windows": candidate.pit_windows,
            }
            strategy_id = hashlib.sha1(
                json.dumps(strategy_fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16]
            final.append({
                "strategy_id": strategy_id,
                "type": candidate.strategy_type,
                "compounds": candidate.compounds,
                "stints": candidate.stint_lengths,
                "stint_curves": self._stint_curves(candidate, curves),
                "pit_windows": candidate.pit_windows,
                "stop_laps": candidate.stop_laps,
                "expected_time": mean,
                "variance": var,
                "risk_score": score,
            })
            if len(final) >= n_strategies:
                break

        degradation = {}
        for compound, vals in stats.items():
            base = vals["base"]
            slope = vals["slope"]
            degradation[compound] = {
                "base": base,
                "slope": slope,
                "curve": [base + slope * i for i in range(1, min(30, context.total_laps) + 1)],
            }

        response = {
            "context": {
                "total_laps": context.total_laps,
                "track_temp": context.track_temp,
                "air_temp": context.air_temp,
                "pit_loss": context.pit_loss,
                "sc_probability": context.sc_probability,
            },
            "strategies": final,
            "degradation": degradation,
        }
        if debug_profile:
            profile = load_driver_profile(driver_id)
            response["driver_profile"] = {
                "driver_id": driver_id,
                "defaults": {k: vars(v) for k, v in profile.driver_defaults.items()},
            }
        return response


@lru_cache(maxsize=16)
def _load_model_cached(driver_id: int) -> Tuple[LSTMPaceModel, int]:
    path = MODELS_DIR / f"driver_{driver_id}.joblib"
    if not path.exists():
        path = MODELS_DIR / "global.joblib"
    payload = joblib.load(path)
    bundle: ModelBundle = payload["bundle"]
    input_dim = payload["input_dim"]
    model = LSTMPaceModel(context_len=payload["context_len"])
    model.load(bundle, input_dim)
    return model, input_dim


@lru_cache(maxsize=32)
def _load_pace_curves_cached(path_str: str) -> Dict[str, np.ndarray]:
    df = pd.read_parquet(path_str)
    curves = {}
    for compound, cdf in df.groupby("compound"):
        curves[str(compound).upper()] = cdf.sort_values("lap")["lap_time"].to_numpy()
    return curves


@lru_cache(maxsize=512)
def _predict_stint_cached(
    driver_id: int,
    compound: str,
    stint_len: int,
    track_temp: float,
    air_temp: float,
    base: float,
    slope: float,
    circuit_id: str,
) -> np.ndarray:
    laps = np.arange(1, stint_len + 1)
    base_series = base + slope * (laps - 1)
    model, _ = _load_model_cached(driver_id)
    df = pd.DataFrame({
        "lap_number": laps,
        "stint_age": laps,
        "compound": compound,
        "session_type": "RACE",
        "circuit_id": circuit_id,
        "track_temp": track_temp,
        "air_temp": air_temp,
        "lap_time": base_series,
    })
    return model.predict_stint(df)
