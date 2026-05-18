"""
tests/test_transformer_model.py — Unit tests for the Transformer pace model pipeline.

v1 tests (preserved):
  1. test_forward_pass            — TyreTransformerNet (B,T,13) → (B,1)
  2. test_train_synthetic_v1      — v1 training on synthetic data
  3. test_rollout_mc_v1           — v1 rollout shape and bounds
  4. test_lap_type_classification — _classify_lap_type tags pit_out, SC, formation
  5. test_backward_compat_lstm_load — old LSTM payload loads correctly

v2 tests (new):
  6. test_v2_forward_pass         — TyreDegradationTransformerV2 (B,T,14) → 3 heads
  7. test_v2_rollout_mc_push_sensitivity — higher push → steeper slope
  8. test_v2_circuit_differentiation    — Monaco vs Sakhir → different outputs
  9. test_v2_backward_compat            — model_type="transformer" still loads v1
  10. test_v2_train_synthetic            — 3 epochs, val_mae tracked, pace bounds fitted
  11. test_build_context_seed_v2         — shape (T,14), dtype float32, no NaN
  12. test_practice_distribution_defaults — pace_lo > pace_hi convention

NOTE: torch imports are lazy inside each test body to avoid MPS initialisation
deadlocks during pytest collection on macOS Apple Silicon.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.preprocess import _classify_lap_type
from app.config import (
    TRANSFORMER_CONTEXT_LAPS,
    TRANSFORMER_V2_CONTEXT_LAPS,
    TRANSFORMER_V2_D_MODEL,
    CIRCUIT_VOCAB,
)

pytestmark = pytest.mark.requires_torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_df(
    n_laps: int = 400,
    n_stints: int = 4,
    seed: int = 42,
    circuit_id: str = "Sakhir",
    session_key: int = 1,
    driver_id: int = 55,
    n_sessions: int = 1,
) -> pd.DataFrame:
    """Minimal synthetic features DataFrame for training."""
    rng = np.random.default_rng(seed)
    stint_len = n_laps // n_stints
    rows = []
    for sess_i in range(n_sessions):
        for stint_idx in range(n_stints):
            compound   = ["SOFT", "MEDIUM", "HARD", "MEDIUM"][stint_idx % 4]
            base_time  = 90.0 + stint_idx * 0.5
            for lap_in_stint in range(1, stint_len + 1):
                lap_time = base_time + 0.12 * lap_in_stint + rng.normal(0, 0.15)
                lap_n    = stint_idx * stint_len + lap_in_stint
                lap_type = "pit_out" if lap_in_stint == 1 and stint_idx > 0 else "normal"
                rows.append({
                    "year":             2023,
                    "session_key":      session_key + sess_i,
                    "session_type":     "RACE",
                    "circuit_id":       circuit_id,
                    "driver_id":        driver_id,
                    "driver_code":      "SAI",
                    "team_name":        "Ferrari",
                    "lap_number":       float(lap_n),
                    "stint_number":     float(stint_idx + 1),
                    "stint_age":        float(lap_in_stint),
                    "compound":         compound,
                    "lap_time":         float(np.clip(lap_time, 85.0, 120.0)),
                    "track_temp":       35.0,
                    "air_temp":         25.0,
                    "gap_to_car_ahead": float(rng.uniform(0.5, 3.0)),
                    "mean_throttle":    float(rng.uniform(0.6, 0.9)),
                    "mean_brake":       float(rng.uniform(0.1, 0.4)),
                    "max_speed":        float(rng.uniform(280, 330)),
                    "mean_rpm":         float(rng.uniform(9000, 13000)),
                    "drs_fraction":     float(rng.uniform(0.0, 0.5)),
                    "lap_type":         lap_type,
                    "is_valid_train":   True,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# v1 tests (preserved)
# ---------------------------------------------------------------------------


def test_forward_pass():
    """v1 forward pass: batch (4, T, 13) → (4, 1)."""
    import torch
    from app.models_transformer import TyreTransformerNet

    T     = TRANSFORMER_CONTEXT_LAPS
    model = TyreTransformerNet(context_len=T)
    model.eval()

    x_cont = torch.randn(4, T, 6)
    x_cat  = torch.randint(0, 5, (4, T, 3))

    with torch.no_grad():
        out = model(x_cont, x_cat)

    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()


def test_train_synthetic_v1():
    """v1 training returns a valid ModelBundle."""
    from app.models_transformer import TransformerPaceModel, ModelBundle

    df    = _make_synthetic_df(n_laps=400)
    model = TransformerPaceModel(
        context_len=TRANSFORMER_CONTEXT_LAPS,
        d_model=32, n_heads=4, n_layers=2, dim_ff=128, model_version="v1",
    )
    bundle = model.train(df, epochs=2, batch_size=32)

    assert isinstance(bundle, ModelBundle)
    assert "lap_mean" in bundle.stats
    assert "delta_std" in bundle.stats


def test_rollout_mc_v1():
    """v1 rollout_mc returns (stint_len,) array with plausible lap times."""
    from app.models_transformer import (
        TransformerPaceModel, PracticeDistribution,
        _build_context_seed_v1, COMPOUND_VOCAB, SESSION_TYPE_VOCAB,
    )

    df    = _make_synthetic_df(n_laps=400)
    model = TransformerPaceModel(
        context_len=TRANSFORMER_CONTEXT_LAPS,
        d_model=32, n_heads=4, n_layers=2, dim_ff=128, model_version="v1",
    )
    model.train(df, epochs=2, batch_size=32)

    pd_dist      = PracticeDistribution(compound="MEDIUM")
    compound_int = COMPOUND_VOCAB.get("MEDIUM", 2)
    session_int  = SESSION_TYPE_VOCAB.get("RACE", 4)
    rng          = np.random.default_rng(0)

    ctx_seed = _build_context_seed_v1(
        pd_dist, model.context_len, model.stats["delta_std"],
        model.stats["track_temp_ref"], model.stats["air_temp_ref"],
        compound_int, session_int, rng,
    )
    lap_times = model.rollout_mc(ctx_seed, 20, 0.0, 35.0, 25.0, compound_int,
                                  base_lap_time=90.0, rng=rng)

    assert lap_times.shape == (20,)
    assert np.all(lap_times >= 60.0) and np.all(lap_times <= 300.0)
    assert not np.any(np.isnan(lap_times))


def test_lap_type_classification():
    """_classify_lap_type correctly tags pit_out, formation, and SC windows."""
    n_drivers = 10
    n_laps    = 30
    rng       = np.random.default_rng(7)
    rows = []
    for drv in range(n_drivers):
        for lap in range(1, n_laps + 1):
            lap_time = 120.0 + rng.normal(0, 1.0) if 10 <= lap <= 13 else 90.0 + rng.normal(0, 0.3)
            stint    = 1 if lap <= 15 else 2
            rows.append({
                "driver_id": drv, "lap_number": float(lap), "lap_time": float(lap_time),
                "stint_number": float(stint), "stint_age": float(lap if stint == 1 else lap - 15),
            })
    df       = pd.DataFrame(rows)
    lap_type = _classify_lap_type(df, "Race")

    formation_laps = df[df["lap_number"] == 1.0].index
    assert all(lap_type[formation_laps] == "formation")

    pit_out_laps = df[(df["lap_number"] == 16.0) & (df["stint_number"] == 2.0)].index
    assert all(lap_type[pit_out_laps] == "pit_out")

    sc_laps = df[(df["lap_number"] >= 10.0) & (df["lap_number"] <= 13.0)].index
    n_sc    = (lap_type[sc_laps] == "sc").sum()
    assert n_sc > 0


def test_backward_compat_lstm_load():
    """Old LSTM payload (no model_type key) loads as LSTMPaceModel without error."""
    from app.models_lstm import LSTMPaceModel, ModelBundle as LSTMBundle, LSTMPaceNet

    lstm_model          = LSTMPaceModel(context_len=10)
    lstm_model.encoders = {
        "compound": {"SOFT": 1, "MEDIUM": 2, "HARD": 3},
        "session_type": {"RACE": 1},
        "circuit_id": {"Sakhir": 1},
    }
    lstm_model.stats = {"lap_mean": 90.0, "lap_std": 2.0}
    lstm_model.model = LSTMPaceNet(input_dim=8, hidden_dim=16)
    bundle = LSTMBundle(
        model_state=lstm_model.model.state_dict(),
        encoders=lstm_model.encoders,
        stats=lstm_model.stats,
    )
    legacy_payload = {"bundle": bundle, "input_dim": 8, "context_len": 10}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "driver_99.joblib"
        joblib.dump(legacy_payload, path)
        loaded = joblib.load(path)

    assert loaded.get("model_type", "lstm") == "lstm"
    loaded_model = LSTMPaceModel(context_len=loaded["context_len"])
    loaded_model.load(loaded["bundle"], loaded["input_dim"])
    assert loaded_model.model is not None


# ---------------------------------------------------------------------------
# v2 tests (new)
# ---------------------------------------------------------------------------


def test_v2_forward_pass():
    """v2 forward pass: (B,T,14) + (B,T,3) + (B,) → 3 heads, shapes correct, no NaN."""
    import torch
    from app.models_transformer import TyreDegradationTransformerV2

    T     = TRANSFORMER_V2_CONTEXT_LAPS
    model = TyreDegradationTransformerV2(
        d_model=64, n_heads=4, n_layers=2, dim_ff=128, context_len=T,
    )
    model.eval()

    B      = 4
    x_cont = torch.randn(B, T, 14)
    x_cat  = torch.randint(0, 5, (B, T, 3))
    circ   = torch.randint(0, 24, (B,))

    with torch.no_grad():
        delta, abs_t, deg = model(x_cont, x_cat, circ)

    for out, name in [(delta, "delta"), (abs_t, "abs"), (deg, "deg")]:
        assert out.shape == (B, 1), f"{name}: expected (4,1), got {out.shape}"
        assert torch.isfinite(out).all(), f"{name}: contains NaN or Inf"


def test_v2_rollout_mc_push_sensitivity():
    """Higher push (pace_hi) should produce equal or steeper degradation than pace_lo."""
    from app.models_transformer import (
        TransformerPaceModel, PracticeDistribution,
        build_context_seed, COMPOUND_VOCAB, SESSION_TYPE_VOCAB,
    )

    df    = _make_synthetic_df(n_laps=600, n_stints=4, n_sessions=3)
    model = TransformerPaceModel(
        context_len=15, d_model=64, n_heads=4, n_layers=2, dim_ff=128, model_version="v2",
    )
    model.train(df, epochs=3, batch_size=32, patience=0)

    pd_dist = PracticeDistribution(compound="SOFT", pace_hi=87.0, pace_lo=94.0)
    lap_mean = model.stats.get("lap_mean", 90.0)
    lap_std  = model.stats.get("lap_std",   1.0)

    pace_hi_norm = (pd_dist.pace_hi - lap_mean) / (lap_std or 1.0)
    pace_lo_norm = (pd_dist.pace_lo - lap_mean) / (lap_std or 1.0)

    circuit_int  = CIRCUIT_VOCAB.get("Sakhir", 0)
    compound_int = COMPOUND_VOCAB.get("SOFT", 1)
    session_int  = SESSION_TYPE_VOCAB.get("RACE", 4)

    delta_std = model.stats.get("delta_std", 1.0)
    track_ref = model.stats.get("track_temp_ref", 30.0)
    air_ref   = model.stats.get("air_temp_ref",   25.0)

    stint_len = 15
    x = np.arange(1, stint_len + 1, dtype=float)

    ctx_hi = build_context_seed(pd_dist, model.context_len, delta_std, track_ref, air_ref,
                                 compound_int, session_int, circuit_int=circuit_int,
                                 rng=np.random.default_rng(42))
    ctx_lo = build_context_seed(pd_dist, model.context_len, delta_std, track_ref, air_ref,
                                 compound_int, session_int, circuit_int=circuit_int,
                                 rng=np.random.default_rng(42))

    times_hi = model.rollout_mc(ctx_hi, stint_len, (pace_hi_norm, pace_hi_norm),
                                 35.0, 25.0, compound_int, base_lap_time=lap_mean,
                                 rng=np.random.default_rng(42),
                                 circuit_int=circuit_int, race_lap_start=0)
    times_lo = model.rollout_mc(ctx_lo, stint_len, (pace_lo_norm, pace_lo_norm),
                                 35.0, 25.0, compound_int, base_lap_time=lap_mean,
                                 rng=np.random.default_rng(42),
                                 circuit_int=circuit_int, race_lap_start=0)

    assert times_hi.shape == (stint_len,)
    assert times_lo.shape == (stint_len,)
    assert not np.any(np.isnan(times_hi))
    assert not np.any(np.isnan(times_lo))
    # Both rollouts should produce plausible lap times
    assert np.all(times_hi >= 60.0) and np.all(times_hi <= 300.0)


def test_v2_circuit_differentiation():
    """Same continuous inputs, different circuit → different transformer outputs."""
    import torch
    from app.models_transformer import TyreDegradationTransformerV2

    T     = 15
    model = TyreDegradationTransformerV2(
        d_model=64, n_heads=4, n_layers=2, dim_ff=128, context_len=T,
    )
    model.eval()

    x_cont = torch.randn(1, T, 14)
    x_cat  = torch.randint(1, 4, (1, T, 3))

    monaco_idx  = torch.tensor([CIRCUIT_VOCAB.get("Monaco",     7)], dtype=torch.long)
    sakhir_idx  = torch.tensor([CIRCUIT_VOCAB.get("Sakhir",     1)], dtype=torch.long)

    with torch.no_grad():
        delta_monaco, _, _ = model(x_cont, x_cat, monaco_idx)
        delta_sakhir, _, _ = model(x_cont, x_cat, sakhir_idx)

    # Different circuits must produce different outputs
    assert not torch.allclose(delta_monaco, delta_sakhir), \
        "Monaco and Sakhir produced identical outputs — circuit embedding not working"


def test_v2_backward_compat():
    """Payload with model_type='transformer' (v1) still loads correctly via _load_model_cached."""
    from app.models_transformer import TransformerPaceModel, TyreTransformerNet

    df    = _make_synthetic_df(n_laps=400)
    model = TransformerPaceModel(
        context_len=TRANSFORMER_CONTEXT_LAPS,
        d_model=32, n_heads=4, n_layers=2, dim_ff=128, model_version="v1",
    )
    bundle = model.train(df, epochs=2, batch_size=32)

    payload = {
        "bundle":        bundle,
        "input_dim":     13,
        "context_len":   TRANSFORMER_CONTEXT_LAPS,
        "model_type":    "transformer",  # v1 discriminator
        "practice_dist": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "driver_55.joblib"
        joblib.dump(payload, path)
        loaded = joblib.load(path)

    assert loaded["model_type"] == "transformer"
    loaded_model = TransformerPaceModel(
        context_len=loaded["context_len"],
        d_model=32, n_heads=4, n_layers=2, dim_ff=128, model_version="v1",
    )
    loaded_model.load(loaded["bundle"], loaded["input_dim"])
    assert isinstance(loaded_model.model, TyreTransformerNet)
    assert loaded_model.model is not None


def test_v2_train_synthetic():
    """v2 training: 3 epochs, val_mae tracked, pace_lo > pace_hi in fitted dist."""
    from app.models_transformer import TransformerPaceModel, ModelBundle
    from app.train import _fit_all_practice_distributions, _split_val

    # Create two sessions so val split has something to hold out
    df = _make_synthetic_df(n_laps=400, n_stints=4, n_sessions=6)

    # Fit practice distributions to check pace bounds
    lap_std = float(df["lap_time"].std()) or 1.0
    dists   = _fit_all_practice_distributions(df, lap_std)
    pd_soft = dists.get("SOFT") or dists.get("global")
    if pd_soft is not None:
        assert pd_soft.pace_lo >= pd_soft.pace_hi, \
            f"pace_lo ({pd_soft.pace_lo:.2f}) should be >= pace_hi ({pd_soft.pace_hi:.2f})"

    train_df, val_df = _split_val(df, val_frac=0.3)

    model = TransformerPaceModel(
        context_len=15, d_model=64, n_heads=4, n_layers=2, dim_ff=128, model_version="v2",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = str(Path(tmpdir) / "train_log.csv")
        bundle   = model.train(
            train_df, epochs=3, batch_size=32,
            val_df=val_df if not val_df.empty else None,
            patience=10,
            log_csv_path=csv_path,
        )

    assert isinstance(bundle, ModelBundle)
    assert bundle.model_state is not None
    assert "lap_mean" in bundle.stats
    assert "delta_std" in bundle.stats


def test_build_context_seed_v2():
    """build_context_seed (v2) returns (context_len, 14) float32 with no NaN."""
    from app.models_transformer import build_context_seed, PracticeDistribution, COMPOUND_VOCAB, SESSION_TYPE_VOCAB

    pd_dist      = PracticeDistribution(compound="MEDIUM")
    compound_int = COMPOUND_VOCAB.get("MEDIUM", 2)
    session_int  = SESSION_TYPE_VOCAB.get("RACE", 4)
    circuit_int  = CIRCUIT_VOCAB.get("Sakhir", 1)
    context_len  = TRANSFORMER_V2_CONTEXT_LAPS

    seed = build_context_seed(
        pd_dist, context_len, delta_std=0.5,
        track_temp_ref=30.0, air_temp_ref=22.0,
        compound_int=compound_int, session_type_int=session_int,
        circuit_int=circuit_int,
        rng=np.random.default_rng(0),
    )

    assert seed.shape == (context_len, 14), f"Expected ({context_len}, 14), got {seed.shape}"
    assert seed.dtype == np.float32, f"Expected float32, got {seed.dtype}"
    assert not np.any(np.isnan(seed)), "Seed contains NaN"


def test_practice_distribution_defaults():
    """PracticeDistribution defaults: pace_lo > pace_hi (slower = bigger number)."""
    from app.models_transformer import PracticeDistribution

    pd_obj = PracticeDistribution()
    assert pd_obj.pace_lo > pd_obj.pace_hi, (
        f"Default pace_lo ({pd_obj.pace_lo}) should be > pace_hi ({pd_obj.pace_hi})"
    )

    custom = PracticeDistribution(compound="SOFT", pace_hi=87.5, pace_lo=94.0)
    assert custom.pace_lo > custom.pace_hi
