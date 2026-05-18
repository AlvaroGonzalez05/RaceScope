"""Tests for StrategyEngine._derive_pit_loss and _derive_sc_probability."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from app.strategy_engine import StrategyEngine
from app.config import (
    PIT_LOSS_FALLBACK, PIT_LOSS_MIN, PIT_LOSS_MAX,
    SC_PROBABILITY_FALLBACK, SC_PROBABILITY_MIN, SC_PROBABILITY_MAX,
)


def make_engine(df):
    return StrategyEngine(df)


def empty_df():
    return pd.DataFrame(columns=[
        "year", "session_key", "session_type", "circuit_id",
        "driver_id", "lap_number", "stint_number", "lap_time",
        "track_temp", "air_temp", "compound", "stint_age",
        "driver_code", "team_name",
    ])


def make_race_df(n_drivers=3, laps_per_stint=20, n_stints=2, median_lap=90.0, outlap_delta=25.0, n_sessions=3):
    """Build a synthetic race DataFrame with pit events."""
    rng = np.random.default_rng(42)
    rows = []
    year = 2023
    circuit_id = "TestCircuit"
    for sess_idx in range(n_sessions):
        session_key = 1000 + sess_idx
        for driver_id in range(1, n_drivers + 1):
            for stint in range(1, n_stints + 1):
                for lap_in_stint in range(laps_per_stint):
                    lap_number = (stint - 1) * laps_per_stint + lap_in_stint + 1
                    if lap_in_stint == 0 and stint > 1:
                        lap_time = median_lap + outlap_delta  # pit outlap
                    else:
                        lap_time = median_lap + rng.normal(0, 0.3)
                    rows.append({
                        "year": year,
                        "session_key": session_key,
                        "session_type": "RACE",
                        "circuit_id": circuit_id,
                        "driver_id": driver_id,
                        "lap_number": lap_number,
                        "stint_number": float(stint),
                        "lap_time": lap_time,
                        "track_temp": 30.0,
                        "air_temp": 22.0,
                        "compound": "MEDIUM",
                        "stint_age": float(lap_in_stint + 1),
                        "driver_code": f"D{driver_id:02d}",
                        "team_name": "TestTeam",
                    })
    return pd.DataFrame(rows)


class TestDerivePitLoss:
    def test_returns_fallback_for_empty_features(self):
        engine = make_engine(empty_df())
        result = engine._derive_pit_loss(2023, "TestCircuit")
        assert result == PIT_LOSS_FALLBACK

    def test_returns_fallback_when_no_circuit_data(self):
        df = make_race_df()
        engine = make_engine(df)
        result = engine._derive_pit_loss(2023, "NonExistentCircuit")
        assert result == PIT_LOSS_FALLBACK

    def test_returns_float_in_valid_range(self):
        df = make_race_df(n_drivers=5, outlap_delta=25.0)
        engine = make_engine(df)
        result = engine._derive_pit_loss(2023, "TestCircuit")
        assert isinstance(result, float)
        assert PIT_LOSS_MIN <= result <= PIT_LOSS_MAX

    def test_pit_loss_close_to_outlap_delta(self):
        """Pit loss estimate should be in the ballpark of the synthetic outlap delta."""
        df = make_race_df(n_drivers=10, laps_per_stint=25, outlap_delta=28.0, n_sessions=1)
        engine = make_engine(df)
        result = engine._derive_pit_loss(2023, "TestCircuit")
        # Should be within a few seconds of 28.0 given the synthetic data
        assert abs(result - 28.0) < 8.0, f"Expected ~28.0, got {result}"

    def test_clamps_to_min(self):
        """If derived value is below PIT_LOSS_MIN, should be clamped."""
        df = make_race_df(n_drivers=10, outlap_delta=2.0)  # outlap barely above median
        engine = make_engine(df)
        result = engine._derive_pit_loss(2023, "TestCircuit")
        assert result >= PIT_LOSS_MIN


class TestDeriveScProbability:
    def test_returns_fallback_for_empty_features(self):
        engine = make_engine(empty_df())
        result = engine._derive_sc_probability(2023, "TestCircuit")
        assert result == SC_PROBABILITY_FALLBACK

    def test_returns_fallback_with_only_one_session(self):
        df = make_race_df(n_sessions=1)
        engine = make_engine(df)
        result = engine._derive_sc_probability(2023, "TestCircuit")
        assert result == SC_PROBABILITY_FALLBACK

    def test_returns_float_in_valid_range(self):
        df = make_race_df(n_sessions=4)
        engine = make_engine(df)
        result = engine._derive_sc_probability(2023, "TestCircuit")
        assert isinstance(result, float)
        assert SC_PROBABILITY_MIN <= result <= SC_PROBABILITY_MAX

    def test_high_sc_probability_with_slow_laps(self):
        """Sessions where many laps are >1.35x median should yield higher SC probability."""
        rows = []
        year, circuit_id = 2023, "TestCircuit"
        for sess_idx in range(5):
            session_key = 2000 + sess_idx
            for driver_id in range(1, 4):
                for lap_number in range(1, 56):
                    # Laps 20-25: SC period (lap_time = 1.5x median)
                    lap_time = 130.0 if 20 <= lap_number <= 25 else 90.0
                    rows.append({
                        "year": year, "session_key": session_key,
                        "session_type": "RACE", "circuit_id": circuit_id,
                        "driver_id": driver_id, "lap_number": lap_number,
                        "stint_number": 1.0, "lap_time": lap_time,
                        "track_temp": 30.0, "air_temp": 22.0,
                        "compound": "MEDIUM", "stint_age": float(lap_number),
                        "driver_code": f"D{driver_id}", "team_name": "T",
                    })
        df = pd.DataFrame(rows)
        engine = make_engine(df)
        result = engine._derive_sc_probability(2023, "TestCircuit")
        assert result >= 0.5, f"Expected high SC probability, got {result}"

    def test_zero_sc_probability_with_clean_laps(self):
        """Sessions with no slow lap clusters should yield low SC probability."""
        df = make_race_df(n_sessions=4, median_lap=90.0, outlap_delta=5.0)
        engine = make_engine(df)
        result = engine._derive_sc_probability(2023, "TestCircuit")
        # With only small outlap deltas, no lap should exceed 1.35 * median
        assert result <= 0.3


class TestContextUsesHelpers:
    def test_context_uses_pit_loss_fallback_for_no_data(self):
        engine = make_engine(empty_df())
        ctx = engine._context(2023, "TestCircuit")
        assert ctx.pit_loss == PIT_LOSS_FALLBACK
        assert ctx.sc_probability == SC_PROBABILITY_FALLBACK

    def test_context_returns_valid_values(self):
        df = make_race_df(n_drivers=5, n_sessions=3)
        engine = make_engine(df)
        ctx = engine._context(2023, "TestCircuit")
        assert ctx.pit_loss >= PIT_LOSS_MIN
        assert SC_PROBABILITY_MIN <= ctx.sc_probability <= SC_PROBABILITY_MAX
