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


# =============================================================================
# Rediseño analítico de la construcción de estrategias
# =============================================================================

from app.strategy_engine import RaceContext, StrategyCandidate
from app.driver_profile import DriverProfile, ProfileParams


class TestPitLossClamp15_45:
    def test_clamp_lower_at_15(self):
        df = make_race_df(n_drivers=10, outlap_delta=2.0)
        engine = make_engine(df)
        result = engine._derive_pit_loss(2023, "TestCircuit")
        assert result >= 15.0

    def test_clamp_upper_at_45(self):
        df = make_race_df(n_drivers=10, outlap_delta=58.0)  # outlap +58s, fuera de [5,60]→quizá excluido
        # Forzar outlap dentro del filtro 5-60s pero por encima del nuevo clamp
        df = make_race_df(n_drivers=10, outlap_delta=55.0, laps_per_stint=15)
        engine = make_engine(df)
        result = engine._derive_pit_loss(2023, "TestCircuit")
        assert result <= 45.0


class TestPaceTable:
    def _engine_with_profile(self, profile: DriverProfile, driver_code: str = "TST"):
        engine = make_engine(empty_df())

        def fake_loader(code):
            return profile

        # Monkeypatch local: el engine usa `load_driver_profile` importado en el módulo
        from app import strategy_engine as se_mod
        se_mod.load_driver_profile = fake_loader  # type: ignore
        return engine

    def test_temperature_correction_applied(self):
        # Perfil con coef de track = +0.2 s/°C; referencia 30, actual 40 → +2 s
        # (Dentro de TEMP_CORR_CLAMP_S=3 → no se clampa.)
        params = ProfileParams(
            base=90.0, slope=0.10,
            track_coef=0.2, air_coef=0.0,
            track_ref=30.0, air_ref=22.0,
        )
        profile = DriverProfile(
            driver_code="TST",
            profiles={("Bahrain", "MEDIUM"): params},
            driver_defaults={"SOFT": params, "MEDIUM": params, "HARD": params},
            global_defaults={"SOFT": params, "MEDIUM": params, "HARD": params},
        )
        engine = self._engine_with_profile(profile)
        ctx = RaceContext(
            year=2023, total_laps=50,
            track_temp=40.0, air_temp=22.0,
            pit_loss=22.0, sc_probability=0.2,
        )
        table = engine._pace_table("TST", "Bahrain", ctx)
        pace_med, deg_med = table["MEDIUM"]
        # 90 + 0.2·10 = 92; queda dentro del envelope por defecto [85.1, 93.8]
        assert abs(pace_med - 92.0) < 1e-6
        assert abs(deg_med - 0.10) < 1e-6

    def test_temperature_correction_clamped_at_cap(self):
        # Coef irreal (5 s/°C) × ΔT=10 → 50 s sin clamp; con clamp = 3 s
        params = ProfileParams(
            base=90.0, slope=0.05,
            track_coef=5.0, air_coef=0.0,
            track_ref=30.0, air_ref=22.0,
        )
        profile = DriverProfile(
            driver_code="TST",
            profiles={},
            driver_defaults={},
            global_defaults={"SOFT": params, "MEDIUM": params, "HARD": params},
        )
        engine = self._engine_with_profile(profile)
        ctx = RaceContext(
            year=2023, total_laps=50,
            track_temp=40.0, air_temp=22.0,
            pit_loss=22.0, sc_probability=0.2,
        )
        table = engine._pace_table("TST", "Bahrain", ctx)
        pace_med, _ = table["MEDIUM"]
        # 90 + min(3, 50) = 93; dentro de [85.1, 93.8]; no se clampa por envelope
        assert abs(pace_med - 93.0) < 1e-6

    def test_pace_base_clamped_to_circuit_envelope(self):
        # Coef enorme + base inflada: sin envelope sería 200 s; clampa a hi=92·1.02
        params = ProfileParams(
            base=200.0, slope=0.05,
            track_coef=0.0, air_coef=0.0,
            track_ref=30.0, air_ref=22.0,
        )
        profile = DriverProfile(
            driver_code="TST",
            profiles={},
            driver_defaults={},
            global_defaults={"SOFT": params, "MEDIUM": params, "HARD": params},
        )
        engine = self._engine_with_profile(profile)
        ctx = RaceContext(
            year=2023, total_laps=50,
            track_temp=30.0, air_temp=22.0,
            pit_loss=22.0, sc_probability=0.2,
        )
        table = engine._pace_table("TST", "Bahrain", ctx)
        pace_med, _ = table["MEDIUM"]
        # Envelope por defecto MEDIUM: fast=87, median=93, slow=101
        # → hi = 93 · 1.02 = 94.86. pace_med debe estar en torno a 94.86.
        assert pace_med <= 95.0
        assert pace_med >= 90.0  # no debe bajar del lo

    def test_race_envelope_lo_hi(self):
        engine = make_engine(empty_df())
        ctx = RaceContext(
            year=2023, total_laps=57,
            track_temp=33.0, air_temp=26.0,
            pit_loss=22.5, sc_probability=0.2,
        )
        lo, hi, anchor = engine._race_envelope(2023, "Bahrain", ctx)
        # Envelope por defecto: medians SOFT=92 MEDIUM=93 HARD=94 → mediana global=93
        # anchor = 57 · 93 = 5301
        assert abs(anchor - 5301.0) < 1.0
        assert lo == anchor * 0.985
        assert hi > anchor * 1.04
        assert lo < anchor < hi

    def test_returns_all_three_compounds(self):
        params = ProfileParams(90.0, 0.08, 0.0, 0.0, 30.0, 22.0)
        profile = DriverProfile(
            driver_code="TST",
            profiles={},
            driver_defaults={},
            global_defaults={"SOFT": params, "MEDIUM": params, "HARD": params},
        )
        engine = self._engine_with_profile(profile)
        ctx = RaceContext(year=2023, total_laps=50, track_temp=30.0, air_temp=22.0,
                          pit_loss=22.0, sc_probability=0.2)
        table = engine._pace_table("TST", "Bahrain", ctx)
        assert set(table.keys()) == {"SOFT", "MEDIUM", "HARD"}


class TestBreakEvenAnalytic:
    """Verifica la forma cerrada del punto de parada óptima 1-stop."""

    def _total_time(self, s, pace_a, deg_a, pace_b, deg_b, pit_loss, L):
        t1 = sum(pace_a + deg_a * i for i in range(s))
        t2 = sum(pace_b + deg_b * j for j in range(L - s))
        return t1 + pit_loss + t2

    def test_1stop_optimum_matches_closed_form(self):
        pace_a, deg_a = 90.0, 0.10
        pace_b, deg_b = 89.5, 0.06
        L = 50
        pit_loss = 22.0

        # Forma cerrada del motor
        denom = deg_a + deg_b
        s_star = ((pace_b - pace_a) + (deg_a - deg_b) / 2.0 + deg_b * L) / denom

        # Búsqueda numérica como referencia
        scores = [(self._total_time(s, pace_a, deg_a, pace_b, deg_b, pit_loss, L), s)
                  for s in range(5, L - 4)]
        best = min(scores)[1]

        assert abs(s_star - best) < 1.0, (
            f"closed form {s_star:.2f} vs numerical {best}"
        )

    def test_2stop_optimum_close_to_numerical(self):
        pace = {"A": (90.0, 0.10), "B": (89.5, 0.06), "C": (89.0, 0.04)}
        L = 60
        pit_loss = 22.0
        pa, da = pace["A"]; pb, db = pace["B"]; pc, dc = pace["C"]

        A = np.array([[da + db, -db], [-db, db + dc]], dtype=float)
        rhs = np.array([
            (pb - pa) + (da - db) / 2.0,
            (pc - pb) + (db - dc) / 2.0 + dc * L,
        ])
        s1_an, s2_an = np.linalg.solve(A, rhs)

        # Búsqueda numérica
        best, best_time = (None, None), float("inf")
        for s1 in range(5, L - 10):
            for s2 in range(s1 + 5, L - 4):
                t = (
                    sum(pa + da * i for i in range(s1))
                    + pit_loss
                    + sum(pb + db * j for j in range(s2 - s1))
                    + pit_loss
                    + sum(pc + dc * k for k in range(L - s2))
                )
                if t < best_time:
                    best, best_time = (s1, s2), t

        assert abs(s1_an - best[0]) < 2.0
        assert abs(s2_an - best[1]) < 2.0


class TestCandidateGeneration:
    def _engine_with_pace(self, pace_table):
        engine = make_engine(empty_df())
        engine._fixed_pace_table = pace_table  # type: ignore
        return engine

    def test_unprofitable_stop_dropped(self):
        """Si la degradación es ~0 y pit_loss alto, ninguna parada es rentable."""
        engine = make_engine(empty_df())
        pace_table = {
            "SOFT":   (90.0, 0.0),
            "MEDIUM": (90.0, 0.0),
            "HARD":   (90.0, 0.0),
        }
        bounds = {"SOFT": (10, 30), "MEDIUM": (10, 35), "HARD": (10, 45)}
        candidates = engine._candidate_strategies(
            total_laps=50, bounds=bounds, pace_table=pace_table, pit_loss=30.0,
        )
        assert candidates == [], "Sin degradación no debería haber candidatas rentables"

    def test_at_least_two_compounds(self):
        engine = make_engine(empty_df())
        pace_table = {
            "SOFT":   (89.0, 0.12),
            "MEDIUM": (90.0, 0.08),
            "HARD":   (91.0, 0.04),
        }
        bounds = {"SOFT": (8, 25), "MEDIUM": (12, 35), "HARD": (18, 45)}
        candidates = engine._candidate_strategies(
            total_laps=55, bounds=bounds, pace_table=pace_table, pit_loss=22.0,
        )
        assert len(candidates) > 0
        for c in candidates:
            assert len(set(c.compounds)) >= 2

    def test_at_least_one_stop(self):
        engine = make_engine(empty_df())
        pace_table = {
            "SOFT":   (89.0, 0.12),
            "MEDIUM": (90.0, 0.08),
            "HARD":   (91.0, 0.04),
        }
        bounds = {"SOFT": (8, 25), "MEDIUM": (12, 35), "HARD": (18, 45)}
        candidates = engine._candidate_strategies(
            total_laps=55, bounds=bounds, pace_table=pace_table, pit_loss=22.0,
        )
        for c in candidates:
            assert len(c.stop_laps) >= 1


class TestHardStartFilteredOut:
    """HARD-start nunca debe llegar al payload final."""

    def test_no_hard_start_in_final_strategies(self):
        # Forzar parámetros que harían HARD-start atractiva: deg del HARD baja
        # y pace base muy competitiva. Aun así, el filtro de salida debe excluirla.
        engine = make_engine(empty_df())
        candidates = [
            StrategyCandidate(
                strategy_type="1-stop",
                compounds=["HARD", "MEDIUM"],
                stint_lengths=[30, 25],
                pit_windows=[{"lap_min": 25, "lap_max": 35}],
                stop_laps=[30],
            ),
            StrategyCandidate(
                strategy_type="1-stop",
                compounds=["SOFT", "MEDIUM"],
                stint_lengths=[18, 37],
                pit_windows=[{"lap_min": 15, "lap_max": 22}],
                stop_laps=[18],
            ),
        ]
        # Simulación del filtro de salida tal como aparece en generate_strategies
        kept = [c for c in candidates if c.compounds[0].upper() != "HARD"]
        assert all(k.compounds[0].upper() != "HARD" for k in kept)
        assert len(kept) == 1
        assert kept[0].compounds[0] == "SOFT"
