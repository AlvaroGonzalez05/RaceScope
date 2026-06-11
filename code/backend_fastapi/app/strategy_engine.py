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
    CIRCUIT_VOCAB,
    DEFAULT_CONTEXT_LAPS,
    DEFAULT_RISK_LAMBDA,
    DEFAULT_STRATEGY_COUNT,
    MODELS_DIR,
    MC_TOP_K,
    PACE_CURVE_CACHE_DIR,
    EXPECTED_TIME_HI_FACTOR, EXPECTED_TIME_LO_FACTOR, EXPECTED_TIME_SC_MARGIN_S,
    PACE_BASE_HI_FACTOR, PACE_BASE_LO_FACTOR,
    PACE_LAP_HI_FACTOR, PACE_LAP_LO_FACTOR,
    PIT_LOSS_FALLBACK, PIT_LOSS_MIN, PIT_LOSS_MAX,
    PIT_WINDOW_BIN,
    RANDOM_SEED,
    SC_PROBABILITY_FALLBACK, SC_PROBABILITY_MIN, SC_PROBABILITY_MAX,
    TEMP_CORR_CLAMP_S,
    TRANSFORMER_V2_N_SIM,
)
import logging

from .driver_profile import load_driver_profile, resolve_profile_params

logger = logging.getLogger(__name__)

# torch-dependent model imports are done lazily inside _load_model_cached and
# _simulate_strategy so that importing this module never triggers torch
# initialisation (which can block on macOS MPS at collection time in tests).


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
        # (year, circuit_id) → {compound: {"fast", "median", "slow"}} cache.
        # Built lazily on first access; rebuilt only if `features` is swapped.
        self._lap_envelope_cache: Dict[
            Tuple[int, str], Dict[str, Dict[str, float]]
        ] = {}

    def _context(self, year: int, circuit_id: str) -> RaceContext:
        df = self.features
        df = df[(df["year"] == year) & (df["circuit_id"] == circuit_id)]
        if df.empty:
            return RaceContext(year=year, total_laps=55, track_temp=30.0, air_temp=22.0, pit_loss=PIT_LOSS_FALLBACK, sc_probability=SC_PROBABILITY_FALLBACK)

        race = df[df["session_type"] == "RACE"]
        total_laps = int(race["lap_number"].max()) if not race.empty else int(df["lap_number"].max())
        track_temp = float(df["track_temp"].mean()) if df["track_temp"].notna().any() else 30.0
        air_temp = float(df["air_temp"].mean()) if df["air_temp"].notna().any() else 22.0

        return RaceContext(
            year=year,
            total_laps=total_laps,
            track_temp=track_temp,
            air_temp=air_temp,
            pit_loss=self._derive_pit_loss(year, circuit_id),
            sc_probability=self._derive_sc_probability(year, circuit_id),
        )

    def _derive_pit_loss(self, year: int, circuit_id: str) -> float:
        """Estimate pit-stop time loss from historical stint transitions."""
        df = self.features
        race = df[(df["year"] == year) & (df["circuit_id"] == circuit_id) & (df["session_type"] == "RACE")]
        if race.empty:
            return PIT_LOSS_FALLBACK
        pit_deltas = []
        for driver_id, driver_df in race.groupby("driver_id"):
            driver_df = driver_df.sort_values("lap_number")
            stints = driver_df["stint_number"].dropna().unique()
            if len(stints) < 2:
                continue
            median_pace = driver_df["lap_time"].median()
            for stint_num in sorted(stints)[1:]:
                prev_stint = driver_df[driver_df["stint_number"] == stint_num - 1]
                curr_stint = driver_df[driver_df["stint_number"] == stint_num]
                if prev_stint.empty or curr_stint.empty:
                    continue
                # First lap of a new stint is the in-lap or slow outlap — skip outlap, use 2nd lap
                if len(curr_stint) >= 2:
                    outlap_time = curr_stint.iloc[0]["lap_time"]
                    delta = outlap_time - median_pace
                    if 5.0 <= delta <= 60.0:  # sanity bounds
                        pit_deltas.append(delta)
        if len(pit_deltas) < 5:
            return PIT_LOSS_FALLBACK
        result = float(np.median(pit_deltas))
        return float(np.clip(result, PIT_LOSS_MIN, PIT_LOSS_MAX))

    def _derive_sc_probability(self, year: int, circuit_id: str) -> float:
        """Estimate safety-car probability from historical lap-time anomalies."""
        df = self.features
        race = df[(df["year"] == year) & (df["circuit_id"] == circuit_id) & (df["session_type"] == "RACE")]
        if race.empty:
            return SC_PROBABILITY_FALLBACK
        sessions = race["session_key"].unique()
        if len(sessions) < 2:
            return SC_PROBABILITY_FALLBACK
        sc_sessions = 0
        for session_key in sessions:
            sess_df = race[race["session_key"] == session_key]
            if sess_df.empty:
                continue
            median_lap = sess_df["lap_time"].median()
            if not (median_lap > 0):
                continue
            threshold = median_lap * 1.35
            # Check for ≥3 consecutive laps where most drivers exceed threshold
            laps_grouped = sess_df.groupby("lap_number")["lap_time"].median()
            slow_laps = (laps_grouped > threshold).astype(int)
            # Detect runs of 3+ slow laps
            run = 0
            for v in slow_laps.values:
                if v:
                    run += 1
                    if run >= 3:
                        sc_sessions += 1
                        break
                else:
                    run = 0
        probability = sc_sessions / len(sessions)
        return float(np.clip(probability, SC_PROBABILITY_MIN, SC_PROBABILITY_MAX))

    def _compound_stats(self, driver_code: str, year: int, circuit_id: str) -> Dict[str, Dict[str, float]]:
        df = self.features
        df = df[(df["year"] == year) & (df["circuit_id"] == circuit_id)]
        driver_df = df[df["driver_code"] == driver_code]
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

    # Race-context upper-bound floors: practice/qualifying stints are short by design,
    # but in a full race a driver can extend a soft to 25+ laps and a medium to 35+.
    # Without these floors, SOFT-start strategies are never generated for long races
    # because the observed upper bound (e.g. 18 laps) is smaller than the required
    # stint length (e.g. total_laps − MEDIUM_lower ≈ 39).
    _RACE_MAX_FLOOR: Dict[str, int] = {"SOFT": 25, "MEDIUM": 35, "HARD": 45}

    def _tyre_life_bounds(self, year: int, circuit_id: str) -> Dict[str, Tuple[int, int]]:
        df = self.features
        df = df[(df["year"] == year) & (df["circuit_id"] == circuit_id)]
        if df.empty:
            return {"SOFT": (10, 25), "MEDIUM": (16, 35), "HARD": (22, 45)}

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
            hi = max(q9, q1 + 4, self._RACE_MAX_FLOOR.get(compound_key, q9))
            bounds[compound_key] = (max(q1, 8), hi)
        # Fill any compounds missing from data with race-context defaults
        for compound, (default_lo, default_hi) in [("SOFT", (10, 25)), ("MEDIUM", (16, 35)), ("HARD", (22, 45))]:
            if compound not in bounds:
                bounds[compound] = (default_lo, default_hi)
        return bounds

    def _circuit_lap_envelope(
        self, year: int, circuit_id: str
    ) -> Dict[str, Dict[str, float]]:
        """Return ground-truth race-lap percentiles per compound for a circuit.

        The DriverProfile OLS is unreliable (the 2023 Sakhir baseline diagnostic
        recorded MEDIUM-compound `base` ranging from 84 s to 188 s and SOFT-
        compound `base` up to 188 s). Real race-lap times at a given circuit
        do not move 80 s across drivers in the same race, so we derive a hard
        sanity envelope from the actual lap-time distribution stored in the
        gold parquet and use it to clamp the per-driver pace_base and the
        per-lap predicted curve.

        Returns ``{compound: {"fast": Q5, "median": Q50, "slow": Q90}}`` in
        raw seconds. Falls back to the same compound in any other year for
        the same circuit if the requested year has too few laps, and finally
        to a default envelope hard-coded by compound. Cached per (year,
        circuit_id) — the underlying parquets only change when the pipeline
        re-runs.
        """
        key = (int(year), str(circuit_id))
        cached = self._lap_envelope_cache.get(key)
        if cached is not None:
            return cached

        df = self.features
        race = df[
            (df["circuit_id"] == circuit_id)
            & (df["session_type"] == "RACE")
        ]
        same_year = race[race["year"] == year]
        # Prefer same-year data, broaden to all years if scarce.
        source = same_year if len(same_year) >= 400 else race

        envelope: Dict[str, Dict[str, float]] = {}
        for compound, cdf in source.groupby("compound"):
            if compound is None or compound != compound:
                continue
            compound_key = str(compound).upper()
            if compound_key not in self.valid_compounds:
                continue
            lap_times = cdf["lap_time"].dropna()
            if lap_times.empty:
                continue
            # Drop in-/out-/SC-laps using the same heuristic as `_compound_stats`:
            # anything more than 2× the median is structural noise, not pace.
            med0 = float(lap_times.median())
            clean = lap_times[(lap_times < 2.0 * med0) & (lap_times > 0.5 * med0)]
            if len(clean) < 5:
                continue
            envelope[compound_key] = {
                "fast": float(np.quantile(clean, 0.05)),
                "median": float(np.quantile(clean, 0.50)),
                "slow": float(np.quantile(clean, 0.90)),
            }

        # Fallback per compound: hard floor so callers can rely on the dict.
        # These are conservative averages across modern F1 circuits.
        default = {
            "SOFT":   {"fast": 86.0, "median": 92.0, "slow": 99.0},
            "MEDIUM": {"fast": 87.0, "median": 93.0, "slow": 101.0},
            "HARD":   {"fast": 88.0, "median": 94.0, "slow": 103.0},
        }
        for compound, fallback in default.items():
            envelope.setdefault(compound, fallback)

        self._lap_envelope_cache[key] = envelope
        return envelope

    def _race_envelope(
        self, year: int, circuit_id: str, context: "RaceContext"
    ) -> Tuple[float, float, float]:
        """Per-race plausible total-time envelope ``(lo, hi, anchor)`` in seconds.

        Returns the lower bound, upper bound, and the "anchor" median race time
        from which both are derived. Used as a last-resort clamp on
        ``expected_time`` plus a warning if any strategy escapes the window.

        The anchor is ``total_laps · median_lap`` where ``median_lap`` is the
        median (across SOFT/MEDIUM/HARD) of the 50th-percentile lap time at
        this circuit. Adding ``n_stops · pit_loss`` is left to the caller so
        the envelope can be tightened per candidate.
        """
        envelope = self._circuit_lap_envelope(year, circuit_id)
        medians = [
            v["median"] for k, v in envelope.items() if k in self.valid_compounds
        ]
        median_lap = float(np.median(medians)) if medians else 95.0
        anchor = float(context.total_laps) * median_lap
        lo = anchor * EXPECTED_TIME_LO_FACTOR
        hi = anchor * EXPECTED_TIME_HI_FACTOR + EXPECTED_TIME_SC_MARGIN_S
        return lo, hi, anchor

    @staticmethod
    def _apply_temperature_correction(params, context: "RaceContext") -> float:
        """Return the clamped temperature delta to add on top of ``params.base``.

        ``DriverProfile.params`` are fitted by OLS on noisy stint data, and the
        track/air coefficients are routinely overfit (the baseline diagnostic
        recorded |track_corr| > 3 s on 53/60 driver-compound rows for 2023
        Sakhir). We clamp each component to ±``TEMP_CORR_CLAMP_S`` to keep the
        effective pace inside a physically defensible window. Shared between
        ``_pace_table`` (analytical candidate generation) and
        ``_precompute_pace_curves`` (model input curve) so the cap cannot drift
        out of sync.
        """
        track_corr = params.track_coef * (context.track_temp - params.track_ref)
        air_corr = params.air_coef * (context.air_temp - params.air_ref)
        track_corr = float(np.clip(track_corr, -TEMP_CORR_CLAMP_S, TEMP_CORR_CLAMP_S))
        air_corr = float(np.clip(air_corr, -TEMP_CORR_CLAMP_S, TEMP_CORR_CLAMP_S))
        return track_corr + air_corr

    def _pace_table(
        self,
        driver_code: str,
        circuit_id: str,
        context: RaceContext,
    ) -> Dict[str, Tuple[float, float]]:
        """Para cada compuesto: (pace_base_s, deg_rate_s_per_lap).

        Fuente única de ritmo y degradación para la fase analítica de
        generación de candidatas. Se apoya en `DriverProfile` (lineal
        por driver/circuit/compound con 3 niveles de fallback) y aplica
        la corrección de temperatura del contexto actual, clampada para
        evitar coeficientes OLS sobreajustados.
        """
        profile = load_driver_profile(driver_code)
        envelope = self._circuit_lap_envelope(context.year, circuit_id)
        table: Dict[str, Tuple[float, float]] = {}
        for compound in self.valid_compounds:
            params = resolve_profile_params(profile, circuit_id, compound)
            pace_base = params.base + self._apply_temperature_correction(params, context)
            env = envelope.get(compound)
            if env is not None:
                lo = env["fast"] * PACE_BASE_LO_FACTOR
                hi = env["median"] * PACE_BASE_HI_FACTOR
                pace_base = float(np.clip(pace_base, lo, hi))
            deg_rate = max(float(params.slope), 0.0)
            table[compound] = (float(pace_base), float(deg_rate))
        return table

    def _load_model(self, driver_code: str):
        return _load_model_cached(driver_code)

    def _predict_stint(self, model, driver_code: str, compound: str, stint_len: int, context: RaceContext, base: float, slope: float, circuit_id: str) -> np.ndarray:
        return _predict_stint_cached(
            driver_code,
            compound,
            stint_len,
            context.track_temp,
            context.air_temp,
            base,
            slope,
            circuit_id,
        )

    _MIN_STINT_LEN = 5  # vueltas mínimas para considerar un stint físicamente razonable

    @staticmethod
    def _candidate_profit(
        compounds: List[str],
        stint_lengths: List[int],
        pace_table: Dict[str, Tuple[float, float]],
        pit_loss: float,
    ) -> List[float]:
        """Profit analítico por cada parada de la candidata.

        Para la parada i (entre stint i y stint i+1, ambos contados a
        partir de 0): ¿cuánto tiempo perderíamos respecto a pararnos si
        siguiéramos con el neumático viejo durante las vueltas del
        stint i+1?

        profit_i = (tiempo_con_viejo - tiempo_con_fresco) - pit_loss

        Positivo → parar es rentable; negativo → mejor seguir.
        """
        profits: List[float] = []
        for i in range(len(compounds) - 1):
            so_far    = sum(stint_lengths[: i + 1])
            remaining = stint_lengths[i + 1]
            pace_old, deg_old = pace_table.get(compounds[i].upper(),     (90.0, 0.05))
            pace_new, deg_new = pace_table.get(compounds[i + 1].upper(), (90.0, 0.05))
            t_old = sum(
                pace_old + deg_old * (so_far + k) for k in range(remaining)
            )
            t_new = sum(
                pace_new + deg_new * k for k in range(remaining)
            )
            profits.append((t_old - t_new) - pit_loss)
        return profits

    def _candidate_strategies(
        self,
        total_laps: int,
        bounds: Dict[str, Tuple[int, int]],
        pace_table: Dict[str, Tuple[float, float]],
        pit_loss: float,
    ) -> List[StrategyCandidate]:
        """Genera candidatas resolviendo analíticamente la vuelta de parada óptima.

        Fórmula 1-stop A→B sobre L vueltas (suma cerrada de aritméticas):
            T(s) = s·pace_A + deg_A·s(s-1)/2 + pit_loss
                 + (L-s)·pace_B + deg_B·(L-s)(L-s-1)/2

            dT/ds = 0  →
            s* = ((pace_B − pace_A) + (deg_A − deg_B)/2 + deg_B·L) / (deg_A + deg_B)

        Fórmula 2-stop A→B→C: sistema lineal 2×2 en (s1, s2).
        """
        compounds = [c for c in bounds.keys() if c in self.valid_compounds] or ["SOFT", "MEDIUM", "HARD"]
        candidates: List[StrategyCandidate] = []
        L = int(total_laps)

        def clamp_stop(stop: float, c_a: str, c_b: str) -> int:
            min_a, max_a = bounds.get(c_a, (self._MIN_STINT_LEN, L))
            min_b, max_b = bounds.get(c_b, (self._MIN_STINT_LEN, L))
            lo = max(min_a, L - max_b, self._MIN_STINT_LEN)
            hi = min(max_a, L - min_b, L - self._MIN_STINT_LEN)
            if lo >= hi:
                return -1
            return int(round(min(max(stop, lo), hi)))

        # ---------------------------------------------------------------
        # 1-stop: forma cerrada para cada par de compuestos
        # ---------------------------------------------------------------
        for c1 in compounds:
            for c2 in compounds:
                if c1 == c2:
                    continue
                pace_a, deg_a = pace_table.get(c1, (90.0, 0.05))
                pace_b, deg_b = pace_table.get(c2, (90.0, 0.05))
                denom = deg_a + deg_b
                if denom <= 1e-6:
                    # Sin degradación neta: optimum no está definido,
                    # uso el centro físico de la ventana viable.
                    min_a, max_a = bounds.get(c1, (8, L))
                    min_b, max_b = bounds.get(c2, (8, L))
                    lo = max(min_a, L - max_b)
                    hi = min(max_a, L - min_b)
                    if lo >= hi:
                        continue
                    s_star = (lo + hi) / 2.0
                else:
                    s_star = ((pace_b - pace_a) + (deg_a - deg_b) / 2.0 + deg_b * L) / denom

                stop = clamp_stop(s_star, c1, c2)
                if stop < 0:
                    continue
                len1, len2 = stop, L - stop
                if len1 < self._MIN_STINT_LEN or len2 < self._MIN_STINT_LEN:
                    continue

                stint_lengths = [len1, len2]
                profits = self._candidate_profit([c1, c2], stint_lengths, pace_table, pit_loss)
                if all(p < 0.0 for p in profits):
                    continue  # parada no rentable

                min_a, max_a = bounds.get(c1, (self._MIN_STINT_LEN, L))
                min_b, _ = bounds.get(c2, (self._MIN_STINT_LEN, L))
                _, max_b = bounds.get(c2, (self._MIN_STINT_LEN, L))
                pit_min = max(min_a, L - max_b, self._MIN_STINT_LEN)
                pit_max = min(max_a, L - min_b, L - self._MIN_STINT_LEN)
                candidates.append(
                    StrategyCandidate(
                        strategy_type="1-stop",
                        compounds=[c1, c2],
                        stint_lengths=stint_lengths,
                        pit_windows=[{"lap_min": int(pit_min), "lap_max": int(pit_max)}],
                        stop_laps=[stop],
                    )
                )

        # ---------------------------------------------------------------
        # 2-stop: sistema 2x2 para cada triple con ≥2 compuestos
        # ---------------------------------------------------------------
        for c1 in compounds:
            for c2 in compounds:
                for c3 in compounds:
                    if len({c1, c2, c3}) < 2:
                        continue
                    pace_a, deg_a = pace_table.get(c1, (90.0, 0.05))
                    pace_b, deg_b = pace_table.get(c2, (90.0, 0.05))
                    pace_c, deg_c = pace_table.get(c3, (90.0, 0.05))

                    A = np.array([
                        [deg_a + deg_b, -deg_b],
                        [-deg_b,        deg_b + deg_c],
                    ], dtype=float)
                    rhs = np.array([
                        (pace_b - pace_a) + (deg_a - deg_b) / 2.0,
                        (pace_c - pace_b) + (deg_b - deg_c) / 2.0 + deg_c * L,
                    ], dtype=float)

                    if abs(np.linalg.det(A)) < 1e-6:
                        # Sistema degenerado: reparto uniforme como fallback
                        s1_star = L / 3.0
                        s2_star = 2.0 * L / 3.0
                    else:
                        s1_star, s2_star = np.linalg.solve(A, rhs)

                    min_a, max_a = bounds.get(c1, (self._MIN_STINT_LEN, L))
                    min_b, max_b = bounds.get(c2, (self._MIN_STINT_LEN, L))
                    min_c, max_c = bounds.get(c3, (self._MIN_STINT_LEN, L))

                    s1 = int(round(min(max(s1_star, min_a), max_a)))
                    s2_lo = max(s1 + min_b, L - max_c, self._MIN_STINT_LEN + s1)
                    s2_hi = min(s1 + max_b, L - min_c, L - self._MIN_STINT_LEN)
                    if s2_lo >= s2_hi:
                        continue
                    s2 = int(round(min(max(s2_star, s2_lo), s2_hi)))

                    len1 = s1
                    len2 = s2 - s1
                    len3 = L - s2
                    if min(len1, len2, len3) < self._MIN_STINT_LEN:
                        continue

                    stint_lengths = [len1, len2, len3]
                    profits = self._candidate_profit([c1, c2, c3], stint_lengths, pace_table, pit_loss)
                    if all(p < 0.0 for p in profits):
                        continue

                    candidates.append(
                        StrategyCandidate(
                            strategy_type="2-stop",
                            compounds=[c1, c2, c3],
                            stint_lengths=stint_lengths,
                            pit_windows=[
                                {"lap_min": int(s1 - 2), "lap_max": int(s1 + 2)},
                                {"lap_min": int(s2 - 2), "lap_max": int(s2 + 2)},
                            ],
                            stop_laps=[s1, s2],
                        )
                    )

        return candidates

    def _cluster_key(self, candidate: StrategyCandidate) -> Tuple[int, ...]:
        return tuple(int(stop / PIT_WINDOW_BIN) for stop in candidate.stop_laps)

    def _pace_curve_path(self, year: int, circuit_id: str, driver_code: str, context: RaceContext) -> Path:
        PACE_CURVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        circuit_safe = str(circuit_id).replace(" ", "_")
        key = f"{year}_{circuit_safe}_{driver_code}_{context.track_temp:.1f}_{context.air_temp:.1f}.parquet"
        return PACE_CURVE_CACHE_DIR / key

    def _precompute_pace_curves(self, year: int, circuit_id: str, driver_code: str, context: RaceContext) -> Dict[str, np.ndarray]:
        path = self._pace_curve_path(year, circuit_id, driver_code, context)
        if path.exists():
            return _load_pace_curves_cached(str(path))

        profile = load_driver_profile(driver_code)
        model, _ = self._load_model(driver_code)
        total_laps = context.total_laps

        envelope = self._circuit_lap_envelope(context.year, circuit_id)
        curves = {}
        for compound in self.valid_compounds:
            params = resolve_profile_params(profile, circuit_id, compound)
            laps = np.arange(1, total_laps + 1)
            base_series = (
                params.base
                + params.slope * (laps - 1)
                + self._apply_temperature_correction(params, context)
            )
            env = envelope.get(compound)
            if env is not None:
                # Clamp the input context the model sees so a polluted profile
                # cannot leak into the autoregressive rollout.
                lap_lo = env["fast"] * PACE_LAP_LO_FACTOR
                lap_hi = env["slow"] * PACE_LAP_HI_FACTOR
                base_series = np.clip(base_series, lap_lo, lap_hi)
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
            predicted = np.asarray(model.predict_stint(df), dtype=float)
            if env is not None:
                # Hard cap on each output lap: the transformer is overfit per
                # driver and routinely emits 60 s and 200 s laps when the
                # input context is unfamiliar. Anchoring to the circuit's real
                # lap-time envelope keeps expected_time inside a defensible
                # window regardless of model version (v1/v2/v3).
                predicted = np.clip(predicted, lap_lo, lap_hi)
            curves[compound] = predicted

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
        model,
        driver_code: str,
        candidate: StrategyCandidate,
        context: RaceContext,
        stats: Dict[str, Dict[str, float]],
        circuit_id: str,
        n_sim: int = 200,
    ) -> Tuple[float, float, List[float]]:
        n_sim = max(n_sim, 1)
        sc_events = np.random.rand(n_sim) < context.sc_probability
        sc_laps   = np.random.randint(5, max(6, context.total_laps - 5), size=n_sim)

        pit_loss = np.full(n_sim, context.pit_loss, dtype=float)
        if candidate.stop_laps:
            stops     = np.array(candidate.stop_laps, dtype=int)
            near_stop = (np.abs(sc_laps[:, None] - stops[None, :]) <= 2).any(axis=1)
            pit_loss  = np.where(sc_events & near_stop, np.maximum(12.0, pit_loss - 8.0), pit_loss)

        totals        = np.zeros(n_sim, dtype=float)
        traffic_mu    = 0.15
        traffic_sigma = 0.05

        # Per-lap clamp envelope derived from real race data for this circuit.
        # The MC transformer rollout (v1/v2/v3) is overfit per driver and emits
        # 60 s and 200 s laps when the input context is unfamiliar, so we cap
        # each predicted lap before it accumulates into `totals`.
        envelope = self._circuit_lap_envelope(context.year, circuit_id)

        def _lap_bounds_for(compound: str) -> Tuple[float, float] | None:
            env = envelope.get(compound.upper())
            if env is None:
                return None
            return (env["fast"] * PACE_LAP_LO_FACTOR, env["slow"] * PACE_LAP_HI_FACTOR)

        # Lazy imports keep startup torch-free
        from .models_transformer import (  # noqa: PLC0415
            TransformerPaceModel,
            TyreDegradationTransformerV2,
            TyreDegradationTransformerV3,
            PracticeDistribution,
            build_context_seed,
            build_context_seed_v3,
            _build_context_seed_v1,
            COMPOUND_VOCAB,
            SESSION_TYPE_VOCAB,
        )

        circuit_int = CIRCUIT_VOCAB.get(circuit_id, 0)

        if isinstance(model, TransformerPaceModel) and model.practice_dist:

            if isinstance(model.model, TyreDegradationTransformerV3):
                # -------------------------------------------------------
                # v3 path — BATCHED MC rollout with 15-feature context,
                # Circuit×Compound gate, and stint_number_norm feature
                # -------------------------------------------------------
                from .config import TRANSFORMER_V3_N_SIM  # noqa: PLC0415
                n_sim = max(n_sim, TRANSFORMER_V3_N_SIM)
                totals = np.zeros(n_sim, dtype=float)
                sc_events = np.random.rand(n_sim) < context.sc_probability
                sc_laps   = np.random.randint(5, max(6, context.total_laps - 5), size=n_sim)
                pit_loss  = np.full(n_sim, context.pit_loss, dtype=float)
                if candidate.stop_laps:
                    near_stop = (np.abs(sc_laps[:, None] - stops[None, :]) <= 2).any(axis=1)
                    pit_loss  = np.where(sc_events & near_stop, np.maximum(12.0, pit_loss - 8.0), pit_loss)

                lap_mean  = float(model.stats.get("lap_mean",       90.0))
                lap_std   = float(model.stats.get("lap_std",         1.0))
                delta_std = float(model.stats.get("delta_std",       1.0))
                track_ref = float(model.stats.get("track_temp_ref", 30.0))
                air_ref   = float(model.stats.get("air_temp_ref",   25.0))

                rngs = [np.random.default_rng(RANDOM_SEED + sim_i) for sim_i in range(n_sim)]

                race_lap_cursor = 0
                for stint_idx, (stint_len, compound) in enumerate(
                    zip(candidate.stint_lengths, candidate.compounds)
                ):
                    compound_key   = compound.upper()
                    pd_dist = (
                        model.practice_dist.get(compound_key)
                        or model.practice_dist.get("global")
                        or PracticeDistribution(compound=compound_key)
                    )

                    pace_hi_norm   = (pd_dist.pace_hi - lap_mean) / (lap_std or 1.0)
                    pace_lo_norm   = (pd_dist.pace_lo - lap_mean) / (lap_std or 1.0)
                    compound_int_c = COMPOUND_VOCAB.get(compound_key, 0)
                    session_int    = SESSION_TYPE_VOCAB.get("RACE", 4)
                    stint_number   = stint_idx + 1

                    t_tracks  = np.empty(n_sim, dtype=np.float32)
                    t_airs    = np.empty(n_sim, dtype=np.float32)
                    ctx_seeds = np.empty((n_sim, model.context_len, 15), dtype=np.float32)
                    phase_as  = np.empty(n_sim, dtype=np.float32)
                    phase_bs  = np.empty(n_sim, dtype=np.float32)

                    for i in range(n_sim):
                        t_tracks[i]  = rngs[i].normal(pd_dist.track_temp_mu, pd_dist.track_temp_sigma)
                        t_airs[i]    = rngs[i].normal(pd_dist.air_temp_mu,   pd_dist.air_temp_sigma)
                        ctx_seeds[i] = build_context_seed_v3(
                            pd_dist, model.context_len, delta_std, track_ref, air_ref,
                            compound_int_c, session_int,
                            circuit_int=circuit_int, rng=rngs[i],
                            stint_number=stint_number,
                        )
                        phase_as[i]  = rngs[i].uniform(pace_hi_norm, pace_lo_norm)
                        phase_bs[i]  = rngs[i].uniform(pace_hi_norm, pace_lo_norm)

                    track_temp_norms = (t_tracks - track_ref) / 10.0
                    air_temp_norms   = (t_airs   - air_ref)   / 8.0

                    lap_times_batch = model._rollout_v3_batched(
                        ctx_seeds, phase_as, phase_bs, stint_len,
                        track_temp_norms, air_temp_norms,
                        compound_int_c,
                        lap_type_int=1,
                        session_type_int=session_int,
                        base_lap_time=lap_mean,
                        circuit_int=circuit_int,
                        race_lap_start=race_lap_cursor,
                        stint_number=stint_number,
                    )  # (n_sim, stint_len)
                    bounds = _lap_bounds_for(compound_key)
                    if bounds is not None:
                        lap_times_batch = np.clip(lap_times_batch, bounds[0], bounds[1])
                    race_lap_cursor += stint_len

                    traffic_noises = np.array([
                        rngs[i].normal(
                            traffic_mu * stint_len,
                            traffic_sigma * math.sqrt(max(1, stint_len)),
                        )
                        for i in range(n_sim)
                    ], dtype=np.float64)

                    totals += np.sum(lap_times_batch, axis=1) + traffic_noises

            elif isinstance(model.model, TyreDegradationTransformerV2):
                # -------------------------------------------------------
                # v2 path — BATCHED MC rollout (one fwd pass per lap step,
                # batch_size = n_sim instead of n_sim sequential calls)
                # -------------------------------------------------------
                n_sim = max(n_sim, TRANSFORMER_V2_N_SIM)
                totals = np.zeros(n_sim, dtype=float)
                sc_events = np.random.rand(n_sim) < context.sc_probability
                sc_laps   = np.random.randint(5, max(6, context.total_laps - 5), size=n_sim)
                pit_loss  = np.full(n_sim, context.pit_loss, dtype=float)
                if candidate.stop_laps:
                    near_stop = (np.abs(sc_laps[:, None] - stops[None, :]) <= 2).any(axis=1)
                    pit_loss  = np.where(sc_events & near_stop, np.maximum(12.0, pit_loss - 8.0), pit_loss)

                lap_mean  = float(model.stats.get("lap_mean",       90.0))
                lap_std   = float(model.stats.get("lap_std",         1.0))
                delta_std = float(model.stats.get("delta_std",       1.0))
                track_ref = float(model.stats.get("track_temp_ref", 30.0))
                air_ref   = float(model.stats.get("air_temp_ref",   25.0))

                rngs = [np.random.default_rng(RANDOM_SEED + sim_i) for sim_i in range(n_sim)]

                race_lap_cursor = 0
                for stint_len, compound in zip(candidate.stint_lengths, candidate.compounds):
                    compound_key   = compound.upper()
                    pd_dist = (
                        model.practice_dist.get(compound_key)
                        or model.practice_dist.get("global")
                        or PracticeDistribution(compound=compound_key)
                    )

                    pace_hi_norm   = (pd_dist.pace_hi - lap_mean) / (lap_std or 1.0)
                    pace_lo_norm   = (pd_dist.pace_lo - lap_mean) / (lap_std or 1.0)
                    compound_int_c = COMPOUND_VOCAB.get(compound_key, 0)
                    session_int    = SESSION_TYPE_VOCAB.get("RACE", 4)

                    t_tracks  = np.empty(n_sim, dtype=np.float32)
                    t_airs    = np.empty(n_sim, dtype=np.float32)
                    ctx_seeds = np.empty((n_sim, model.context_len, 14), dtype=np.float32)
                    phase_as  = np.empty(n_sim, dtype=np.float32)
                    phase_bs  = np.empty(n_sim, dtype=np.float32)

                    for i in range(n_sim):
                        t_tracks[i]  = rngs[i].normal(pd_dist.track_temp_mu, pd_dist.track_temp_sigma)
                        t_airs[i]    = rngs[i].normal(pd_dist.air_temp_mu,   pd_dist.air_temp_sigma)
                        ctx_seeds[i] = build_context_seed(
                            pd_dist, model.context_len, delta_std, track_ref, air_ref,
                            compound_int_c, session_int,
                            circuit_int=circuit_int, rng=rngs[i],
                        )
                        phase_as[i]  = rngs[i].uniform(pace_hi_norm, pace_lo_norm)
                        phase_bs[i]  = rngs[i].uniform(pace_hi_norm, pace_lo_norm)

                    track_temp_norms = (t_tracks - track_ref) / 10.0
                    air_temp_norms   = (t_airs   - air_ref)   / 8.0

                    lap_times_batch = model._rollout_v2_batched(
                        ctx_seeds, phase_as, phase_bs, stint_len,
                        track_temp_norms, air_temp_norms,
                        compound_int_c,
                        lap_type_int=1,
                        session_type_int=session_int,
                        base_lap_time=lap_mean,
                        circuit_int=circuit_int,
                        race_lap_start=race_lap_cursor,
                    )  # (n_sim, stint_len)
                    bounds = _lap_bounds_for(compound_key)
                    if bounds is not None:
                        lap_times_batch = np.clip(lap_times_batch, bounds[0], bounds[1])
                    race_lap_cursor += stint_len

                    traffic_noises = np.array([
                        rngs[i].normal(
                            traffic_mu * stint_len,
                            traffic_sigma * math.sqrt(max(1, stint_len)),
                        )
                        for i in range(n_sim)
                    ], dtype=np.float64)

                    totals += np.sum(lap_times_batch, axis=1) + traffic_noises

            else:
                # -------------------------------------------------------
                # v1 transformer path — original autoregressive MC
                # -------------------------------------------------------
                base_lap_time = float(model.stats.get("lap_mean", 90.0))
                delta_std     = model.stats.get("delta_std",       1.0)
                track_ref     = model.stats.get("track_temp_ref", 30.0)
                air_ref       = model.stats.get("air_temp_ref",   25.0)

                for sim_i in range(n_sim):
                    rng_i     = np.random.default_rng(RANDOM_SEED + sim_i)
                    sim_total = 0.0

                    for stint_len, compound in zip(candidate.stint_lengths, candidate.compounds):
                        compound_key     = compound.upper()
                        pd_dist          = (
                            model.practice_dist.get(compound_key)
                            or model.practice_dist.get("global")
                            or PracticeDistribution(compound=compound_key)
                        )
                        pace_intent      = float(rng_i.normal(pd_dist.pace_intent_mu, pd_dist.pace_intent_sigma))
                        t_track          = float(rng_i.normal(pd_dist.track_temp_mu,   pd_dist.track_temp_sigma))
                        t_air            = float(rng_i.normal(pd_dist.air_temp_mu,     pd_dist.air_temp_sigma))
                        compound_int_c   = COMPOUND_VOCAB.get(compound_key, 0)
                        session_type_int = SESSION_TYPE_VOCAB.get("RACE", 4)

                        ctx_seed = _build_context_seed_v1(
                            pd_dist,
                            model.context_len,
                            delta_std,
                            track_ref,
                            air_ref,
                            compound_int_c,
                            session_type_int,
                            rng_i,
                        )
                        lap_times = model.rollout_mc(
                            ctx_seed,
                            stint_len,
                            pace_intent,
                            t_track,
                            t_air,
                            compound_int_c,
                            base_lap_time=base_lap_time,
                            rng=rng_i,
                        )
                        bounds = _lap_bounds_for(compound_key)
                        if bounds is not None:
                            lap_times = np.clip(np.asarray(lap_times, dtype=float), bounds[0], bounds[1])
                        traffic_noise = float(rng_i.normal(
                            traffic_mu * stint_len,
                            traffic_sigma * math.sqrt(max(1, stint_len)),
                        ))
                        sim_total += float(np.sum(lap_times)) + traffic_noise

                    totals[sim_i] = sim_total

        else:
            # --- LSTM / legacy path: vectorised (deterministic base curve + noise) ---
            curves = self._precompute_pace_curves(context.year, circuit_id, driver_code, context)
            for stint_len, compound in zip(candidate.stint_lengths, candidate.compounds):
                series = curves.get(compound.upper())
                if series is None or len(series) < stint_len:
                    base  = stats.get(compound.upper(), {}).get("base",  90.0)
                    slope = stats.get(compound.upper(), {}).get("slope",  0.05)
                    series = self._predict_stint(
                        model, driver_code, compound.upper(), stint_len, context, base, slope, circuit_id
                    )
                base_sum = float(np.sum(series[:stint_len]))
                noise    = np.random.normal(
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

    @staticmethod
    def _stop_profitability(
        candidate: "StrategyCandidate",
        curves: Dict[str, np.ndarray],
        pit_loss: float,
    ) -> List[float]:
        """Per-stop marginal profitability in seconds.

        profit > 0 → stopping is worth it (time saved on fresh tyre > pit loss).
        profit < 0 → better to stay out.
        """
        profits = []
        for i, _stop_lap in enumerate(candidate.stop_laps):
            old_compound  = candidate.compounds[i].upper()
            new_compound  = candidate.compounds[i + 1].upper()
            remaining     = candidate.stint_lengths[i + 1]
            so_far        = candidate.stint_lengths[i]

            old_curve   = curves.get(old_compound)
            fresh_curve = curves.get(new_compound)

            if old_curve is None or len(old_curve) < so_far + remaining:
                # Fallback: assume 0.04 s/lap degradation per lap on old tyre
                base = old_curve[0] if old_curve is not None and len(old_curve) > 0 else 90.0
                t_old = float(sum(
                    base + 0.04 * (so_far + k) for k in range(remaining)
                ))
            else:
                t_old = float(np.sum(old_curve[so_far : so_far + remaining]))

            if fresh_curve is None or len(fresh_curve) < remaining:
                base = fresh_curve[0] if fresh_curve is not None and len(fresh_curve) > 0 else 90.0
                t_fresh = float(sum(
                    base + 0.04 * k for k in range(remaining)
                ))
            else:
                t_fresh = float(np.sum(fresh_curve[:remaining]))

            profit = (t_old - t_fresh) - pit_loss
            profits.append(round(profit, 2))
        return profits

    def generate_strategies(
        self,
        year: int,
        circuit_id: str,
        driver_code: str,
        risk_bias: float = DEFAULT_RISK_LAMBDA,
        n_strategies: int = DEFAULT_STRATEGY_COUNT,
        opponent_code: str | None = None,
        debug_profile: bool = False,
    ) -> Dict:
        context = self._context(year, circuit_id)
        stats = self._compound_stats(driver_code, year, circuit_id)
        bounds = self._tyre_life_bounds(year, circuit_id)
        pace_table = self._pace_table(driver_code, circuit_id, context)
        model, _ = self._load_model(driver_code)
        curves = self._precompute_pace_curves(year, circuit_id, driver_code, context)

        candidates = self._candidate_strategies(context.total_laps, bounds, pace_table, context.pit_loss)
        opponent_best = None
        if opponent_code is not None:
            opp_curves = self._precompute_pace_curves(year, circuit_id, opponent_code, context)
            opp_pace_table = self._pace_table(opponent_code, circuit_id, context)
            opp_candidates = self._candidate_strategies(context.total_laps, bounds, opp_pace_table, context.pit_loss)
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
            # Soft nudge: medium-start válido pero no por defecto.
            # HARD-start no se penaliza aquí; se filtra duro a la salida.
            first_compound = candidate.compounds[0].upper() if candidate.compounds else ""
            if first_compound == "MEDIUM":
                score += 1.0
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
                driver_code,
                candidate,
                context,
                stats,
                circuit_id,
                n_sim=200,
            )
            refined[id(candidate)] = (mc_mean, mc_var)

        env_lo, env_hi, env_anchor = self._race_envelope(year, circuit_id, context)

        for score, mean, var, candidate in ranked:
            if id(candidate) in refined:
                mean, var = refined[id(candidate)]
                score = mean + risk_bias * var
            # Last-resort sanity clamp. If M1–M3 mitigations work this should
            # never trigger; the warning gives operational visibility of any
            # regression where a driver or model produces totals outside the
            # circuit's physical envelope.
            n_stops = len(candidate.stop_laps)
            stops_floor = n_stops * context.pit_loss * 0.85
            stops_ceil = n_stops * context.pit_loss * 1.15
            cand_lo = env_lo + stops_floor
            cand_hi = env_hi + stops_ceil
            expected_time_clamped = False
            if not cand_lo <= mean <= cand_hi:
                logger.warning(
                    "strategy_envelope_clamp: driver=%s circuit=%s type=%s "
                    "expected_time=%.1fs envelope=[%.1f, %.1f] "
                    "(anchor=%.1f, n_stops=%d)",
                    driver_code, circuit_id, candidate.strategy_type,
                    mean, cand_lo, cand_hi, env_anchor, n_stops,
                )
                mean = float(np.clip(mean, cand_lo, cand_hi))
                score = mean + risk_bias * var
                expected_time_clamped = True
            key = self._cluster_key(candidate)
            if key in seen:
                continue
            # Filtro duro: HARD-start no llega al payload final.
            # Es competitivamente irreal en F1 moderna y el usuario no
            # quiere ver estrategias que en la práctica nunca se ven.
            if candidate.compounds and candidate.compounds[0].upper() == "HARD":
                continue
            seen.add(key)
            strategy_fingerprint = {
                "year": year,
                "circuit_id": circuit_id,
                "driver_code": driver_code,
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
                "expected_time_clamped": expected_time_clamped,
                "variance": var,
                "risk_score": score,
                "stop_profitability": self._stop_profitability(
                    candidate, curves, context.pit_loss
                ),
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
            profile = load_driver_profile(driver_code)
            response["driver_profile"] = {
                "driver_code": driver_code,
                "defaults": {k: vars(v) for k, v in profile.driver_defaults.items()},
            }
        return response


@lru_cache(maxsize=16)
def _load_model_cached(driver_code: str):
    """
    Load a pace model from disk.  Supports both the new Transformer format
    (model_type="transformer") and old LSTM payloads (no model_type key).
    """
    path = MODELS_DIR / f"driver_{driver_code}.joblib"
    if not path.exists():
        path = MODELS_DIR / "global.joblib"
    payload = joblib.load(path)
    bundle = payload["bundle"]
    input_dim = payload.get("input_dim", 8)
    context_len = payload.get("context_len", DEFAULT_CONTEXT_LAPS)
    model_type = payload.get("model_type", "lstm")  # backward-compat discriminator

    if model_type == "transformer_v3":
        from .models_transformer import TransformerPaceModel  # noqa: PLC0415
        mkwargs = payload.get("model_kwargs", {})
        model = TransformerPaceModel(
            context_len=context_len,
            model_version="v3",
            **mkwargs,
        )
        model.load(bundle, input_dim)
        model.practice_dist = payload.get("practice_dist")
    elif model_type == "transformer_v2":
        from .models_transformer import TransformerPaceModel  # noqa: PLC0415
        mkwargs = payload.get("model_kwargs", {})
        model = TransformerPaceModel(
            context_len=context_len,
            model_version="v2",
            **mkwargs,
        )
        model.load(bundle, input_dim)
        model.practice_dist = payload.get("practice_dist")
    elif model_type == "transformer":
        from .models_transformer import TransformerPaceModel  # noqa: PLC0415
        model = TransformerPaceModel(context_len=context_len, model_version="v1")
        model.load(bundle, input_dim)
        model.practice_dist = payload.get("practice_dist")
    else:
        # Legacy LSTM payload
        from .models_lstm import LSTMPaceModel  # noqa: PLC0415
        model = LSTMPaceModel(context_len=context_len)
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
    driver_code: str,
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
    model, _ = _load_model_cached(driver_code)
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
