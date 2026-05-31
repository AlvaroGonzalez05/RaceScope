"""
models_transformer.py — Transformer-based tyre degradation & lap-time model.

Three architectures are supported:
  TyreTransformerNet           (v1) — 6 cont. features, 3 categoricals, 2 encoder layers.
                                      Kept for backward compatibility with existing .joblib files.
  TyreDegradationTransformerV2 (v2) — 14 cont. features, circuit embedding, 3 output heads,
                                       6 encoder layers.  Default for new training.
  TyreDegradationTransformerV3 (v3) — 15 cont. features, Circuit×Compound multiplicative gate,
                                       4 output heads (delta, abs, deg, value), 12 encoder layers,
                                       d_model=768.  Designed for intentional per-driver memorisation.

Public API
----------
  TransformerPaceModel          — unified wrapper (train / load / rollout_mc / predict_stint)
  PracticeDistribution          — per-compound statistics for MC context seeding
  build_context_seed            — v2 context-window seed → np.ndarray (T, 14)
  build_context_seed_v3         — v3 context-window seed → np.ndarray (T, 15)
  TyreTransformerNet            — v1 network (keep for compat)
  TyreDegradationTransformerV2  — v2 network
  TyreDegradationTransformerV3  — v3 network
  COMPOUND_VOCAB, SESSION_TYPE_VOCAB — exported for strategy_engine

Internal
--------
  _build_context_seed_v1 — old v1 signature, returns (x_cont, x_cat) tuple
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .config import (
    CIRCUIT_EXPECTED_LAPS,
    CIRCUIT_VOCAB,
    CONTEXT_LAP_TYPES,
    RANDOM_SEED,
    TRAINING_LAP_TYPES,
    TRANSFORMER_CONTEXT_LAPS,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DIM_FF,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_INPUT_DIM,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_N_LAYERS,
    TRANSFORMER_V2_AUX_LOSS_W,
    TRANSFORMER_V2_CONTEXT_LAPS,
    TRANSFORMER_V2_D_MODEL,
    TRANSFORMER_V2_DEG_LOSS_W,
    TRANSFORMER_V2_DIM_FF,
    TRANSFORMER_V2_DROPOUT,
    TRANSFORMER_V2_INPUT_DIM,
    TRANSFORMER_V2_N_HEADS,
    TRANSFORMER_V2_N_LAYERS,
    TRANSFORMER_V3_AUX_LOSS_W,
    TRANSFORMER_V3_CONTEXT_LAPS,
    TRANSFORMER_V3_D_MODEL,
    TRANSFORMER_V3_DEG_LOSS_W,
    TRANSFORMER_V3_DIM_FF,
    TRANSFORMER_V3_DROPOUT,
    TRANSFORMER_V3_INPUT_DIM,
    TRANSFORMER_V3_N_HEADS,
    TRANSFORMER_V3_N_LAYERS,
    TRANSFORMER_V3_VALUE_LOSS_W,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed categorical vocabularies  (0 = unknown / padding)
# ---------------------------------------------------------------------------

COMPOUND_VOCAB: Dict[str, int] = {
    "SOFT": 1,
    "MEDIUM": 2,
    "HARD": 3,
    "INTERMEDIATE": 4,
    "INTER": 4,
    "WET": 5,
    "SUPERSOFT": 6,
    "ULTRASOFT": 6,
    "HYPERSOFT": 6,
}
LAP_TYPE_VOCAB: Dict[str, int] = {
    "normal": 1,
    "pit_out": 2,
    "pit_in": 3,
    "sc": 4,
    "formation": 5,
    "outlier": 6,
}
SESSION_TYPE_VOCAB: Dict[str, int] = {
    "FP1": 1,
    "FP2": 2,
    "FP3": 3,
    "RACE": 4,
    "SPRINT": 5,
    "SPRINT_SHOOTOUT": 6,
}

# Vocabulary sizes for nn.Embedding (leave headroom for unseen values)
_COMPOUND_EMB_VOCAB    = 10
_LAP_TYPE_EMB_VOCAB    = 10
_SESSION_TYPE_EMB_VOCAB = 10
_CIRCUIT_EMB_VOCAB     = 28   # 24 known circuits + 0 + headroom


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PracticeDistribution:
    """
    Per-compound statistics fitted from practice sessions (FP1/FP2/FP3).

    Used in two roles:
      1. Seeding the MC context window (build_context_seed) with realistic warm-up laps.
      2. Sampling priors for pace_intent, track_temp and air_temp in each MC simulation.

    pace_lo / pace_hi (raw seconds):
      pace_hi = 5th-percentile lap time ≈ max-attack pace (lower is faster → smaller number)
      pace_lo = 75th-percentile lap time ≈ conservation pace (larger number)
    Convention: pace_lo > pace_hi in seconds (slower = bigger number).
    """

    compound: str = "global"
    pace_intent_mu: float = 0.0
    pace_intent_sigma: float = 0.05
    delta_mu: float = 0.1
    delta_sigma: float = 0.15
    outlap_delta: float = -2.0
    track_temp_mu: float = 35.0
    track_temp_sigma: float = 3.0
    air_temp_mu: float = 25.0
    air_temp_sigma: float = 2.0
    # Pace bounds in raw seconds (fitted during training)
    pace_lo: float = 95.0   # conservation / management pace
    pace_hi: float = 88.0   # max-attack pace  (pace_lo > pace_hi)


@dataclass
class ModelBundle:
    model_state: Dict
    encoders: Dict[str, Dict]
    stats: Dict


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """v1 dataset: (x_cont, x_cat, y) triplets."""

    def __init__(self, X_cont: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.X_cat  = torch.tensor(X_cat,  dtype=torch.long)
        self.y      = torch.tensor(y,      dtype=torch.float32).reshape(-1, 1)

    def __len__(self)  -> int: return len(self.y)
    def __getitem__(self, idx): return self.X_cont[idx], self.X_cat[idx], self.y[idx]


class SequenceDatasetV2(Dataset):
    """v2 dataset: (x_cont (T,14), x_cat (T,3), circuit_idx, y_delta, y_abs, y_deg)."""

    def __init__(
        self,
        X_cont: np.ndarray,      # (N, T, 14)
        X_cat:  np.ndarray,      # (N, T, 3)
        circuit_ids: np.ndarray, # (N,)
        y_delta: np.ndarray,     # (N,)
        y_abs:   np.ndarray,     # (N,)
        y_deg:   np.ndarray,     # (N,)
    ):
        self.X_cont      = torch.tensor(X_cont,      dtype=torch.float32)
        self.X_cat       = torch.tensor(X_cat,       dtype=torch.long)
        self.circuit_ids = torch.tensor(circuit_ids, dtype=torch.long)
        self.y_delta     = torch.tensor(y_delta,     dtype=torch.float32).reshape(-1, 1)
        self.y_abs       = torch.tensor(y_abs,       dtype=torch.float32).reshape(-1, 1)
        self.y_deg       = torch.tensor(y_deg,       dtype=torch.float32).reshape(-1, 1)

    def __len__(self): return len(self.y_delta)

    def __getitem__(self, idx):
        return (
            self.X_cont[idx], self.X_cat[idx], self.circuit_ids[idx],
            self.y_delta[idx], self.y_abs[idx], self.y_deg[idx],
        )


class SequenceDatasetV3(Dataset):
    """v3 dataset: same as v2 but adds compound_static and y_value targets.

    compound_static: (N,) int64 — compound index for the stint (used in CC gate).
    y_value: (N,) float32 — normalised cumulative remaining-stint time target.
    """

    def __init__(
        self,
        X_cont:         np.ndarray,  # (N, T, 15)
        X_cat:          np.ndarray,  # (N, T, 3)
        circuit_ids:    np.ndarray,  # (N,)
        compound_static: np.ndarray, # (N,)
        y_delta:        np.ndarray,  # (N,)
        y_abs:          np.ndarray,  # (N,)
        y_deg:          np.ndarray,  # (N,)
        y_value:        np.ndarray,  # (N,)
    ):
        self.X_cont          = torch.tensor(X_cont,          dtype=torch.float32)
        self.X_cat           = torch.tensor(X_cat,           dtype=torch.long)
        self.circuit_ids     = torch.tensor(circuit_ids,     dtype=torch.long)
        self.compound_static = torch.tensor(compound_static, dtype=torch.long)
        self.y_delta         = torch.tensor(y_delta,         dtype=torch.float32).reshape(-1, 1)
        self.y_abs           = torch.tensor(y_abs,           dtype=torch.float32).reshape(-1, 1)
        self.y_deg           = torch.tensor(y_deg,           dtype=torch.float32).reshape(-1, 1)
        self.y_value         = torch.tensor(y_value,         dtype=torch.float32).reshape(-1, 1)

    def __len__(self): return len(self.y_delta)

    def __getitem__(self, idx):
        return (
            self.X_cont[idx], self.X_cat[idx], self.circuit_ids[idx],
            self.compound_static[idx],
            self.y_delta[idx], self.y_abs[idx], self.y_deg[idx], self.y_value[idx],
        )


# ---------------------------------------------------------------------------
# v1 Neural network (kept for backward compatibility)
# ---------------------------------------------------------------------------


class TyreTransformerNet(nn.Module):
    """
    v1 Transformer encoder — predicts Δlap_time (residual degradation).

    Input : x_cont (B,T,6), x_cat (B,T,3)
    Output: (B,1) normalised Δlap_time[t+1]
    """

    def __init__(
        self,
        d_model: int = TRANSFORMER_D_MODEL,
        n_heads: int = TRANSFORMER_N_HEADS,
        n_layers: int = TRANSFORMER_N_LAYERS,
        dim_ff: int = TRANSFORMER_DIM_FF,
        dropout: float = TRANSFORMER_DROPOUT,
        context_len: int = TRANSFORMER_CONTEXT_LAPS,
        compound_vocab: int = _COMPOUND_EMB_VOCAB,
        lap_type_vocab: int = _LAP_TYPE_EMB_VOCAB,
        session_type_vocab: int = _SESSION_TYPE_EMB_VOCAB,
    ):
        super().__init__()
        self.compound_emb     = nn.Embedding(compound_vocab,     4, padding_idx=0)
        self.lap_type_emb     = nn.Embedding(lap_type_vocab,     4, padding_idx=0)
        self.session_type_emb = nn.Embedding(session_type_vocab, 4, padding_idx=0)
        self.compound_proj    = nn.Linear(4, 3, bias=False)
        self.lap_type_proj    = nn.Linear(4, 2, bias=False)
        self.session_type_proj = nn.Linear(4, 2, bias=False)
        self.input_proj = nn.Linear(TRANSFORMER_INPUT_DIM, d_model)
        self.pos_enc    = nn.Embedding(context_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.fc_out = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_cont.shape
        comp_emb = torch.relu(self.compound_proj(self.compound_emb(x_cat[:, :, 0])))
        lt_emb   = torch.relu(self.lap_type_proj(self.lap_type_emb(x_cat[:, :, 1])))
        st_emb   = torch.relu(self.session_type_proj(self.session_type_emb(x_cat[:, :, 2])))
        x = torch.cat([x_cont, comp_emb, lt_emb, st_emb], dim=-1)
        x = self.input_proj(x)
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_enc(pos_ids)
        x = self.transformer(x)
        return self.fc_out(x[:, -1, :])


# ---------------------------------------------------------------------------
# v2 Neural network
# ---------------------------------------------------------------------------


class TyreDegradationTransformerV2(nn.Module):
    """
    v2 Transformer — ambitious architecture for tyre degradation & lap time prediction.

    Features (per timestep)
    -----------------------
    Continuous (14):  stint_age_norm, delta_norm, pace_intent_norm, gap_norm,
                      track_temp_norm, air_temp_norm, throttle_dev, brake_dev,
                      speed_dev, rpm_norm, drs_dev, push_index, race_lap_norm,
                      lap_speed_deviation
    Categorical (3):  compound_int, lap_type_int, session_type_int

    Static context:   circuit_idx (one per sequence, broadcast as prior)

    Output heads (applied to last-token representation)
    ---------------------------------------------------
    head_delta  → normalised Δlap_time        [Huber loss]
    head_abs    → normalised absolute lap_time [MSE × aux_w]
    head_deg    → normalised degradation rate  [MSE × deg_w]
    """

    def __init__(
        self,
        d_model:    int   = TRANSFORMER_V2_D_MODEL,
        n_heads:    int   = TRANSFORMER_V2_N_HEADS,
        n_layers:   int   = TRANSFORMER_V2_N_LAYERS,
        dim_ff:     int   = TRANSFORMER_V2_DIM_FF,
        dropout:    float = TRANSFORMER_V2_DROPOUT,
        context_len: int  = TRANSFORMER_V2_CONTEXT_LAPS,
        n_cont:     int   = TRANSFORMER_V2_INPUT_DIM,  # 14
    ):
        super().__init__()
        self.d_model = d_model

        # Per-timestep categorical embeddings
        # compound: 7 vocab → 8 dims; lap_type: 7 → 6; session: 7 → 6; circuit: 28 → 16
        self.compound_emb     = nn.Embedding(_COMPOUND_EMB_VOCAB,     8,  padding_idx=0)
        self.lap_type_emb     = nn.Embedding(_LAP_TYPE_EMB_VOCAB,     6,  padding_idx=0)
        self.session_emb      = nn.Embedding(_SESSION_TYPE_EMB_VOCAB, 6,  padding_idx=0)
        self.circuit_emb      = nn.Embedding(_CIRCUIT_EMB_VOCAB,      16, padding_idx=0)

        # Per-timestep projections: continuous → d_model//2, cat_concat → d_model//2
        cat_dim = 8 + 6 + 6 + 16  # 36
        self.cont_proj = nn.Linear(n_cont, d_model // 2)
        self.cat_proj  = nn.Linear(cat_dim, d_model // 2)

        # Circuit global prior (static per sequence, broadcast to all T positions)
        self.circuit_prior_proj = nn.Linear(16, d_model)

        self.input_ln = nn.LayerNorm(d_model)

        # Positional encoding (up to 50 positions)
        self.pos_enc = nn.Embedding(50, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Three output heads
        self.head_delta = nn.Linear(d_model, 1)
        self.head_abs   = nn.Linear(d_model, 1)
        self.head_deg   = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_cont:     torch.Tensor,  # (B, T, 14)
        x_cat:      torch.Tensor,  # (B, T, 3) [compound, lap_type, session_type]
        circuit_idx: torch.Tensor, # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x_cont.shape

        # Per-timestep categorical embeddings
        comp = self.compound_emb(x_cat[:, :, 0])   # (B, T, 8)
        lt   = self.lap_type_emb(x_cat[:, :, 1])   # (B, T, 6)
        st   = self.session_emb(x_cat[:, :, 2])    # (B, T, 6)
        circ_seq = self.circuit_emb(circuit_idx).unsqueeze(1).expand(-1, T, -1)  # (B, T, 16)

        cat_concat = torch.cat([comp, lt, st, circ_seq], dim=-1)  # (B, T, 36)

        x_c = self.cont_proj(x_cont)     # (B, T, d_model//2)
        x_k = self.cat_proj(cat_concat)  # (B, T, d_model//2)
        x   = torch.cat([x_c, x_k], dim=-1)  # (B, T, d_model)

        # Circuit global prior: embed circuit once, broadcast to all positions
        circ_emb   = self.circuit_emb(circuit_idx)          # (B, 16)
        circ_prior = self.circuit_prior_proj(circ_emb)       # (B, d_model)
        x = x + circ_prior.unsqueeze(1)                      # (B, T, d_model)

        x = self.input_ln(x)

        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_enc(pos_ids)

        x = self.transformer(x)  # (B, T, d_model)
        x = x[:, -1, :]          # (B, d_model)  last-token rep

        return self.head_delta(x), self.head_abs(x), self.head_deg(x)


# ---------------------------------------------------------------------------
# v3 Neural network
# ---------------------------------------------------------------------------


class TyreDegradationTransformerV3(nn.Module):
    """
    v3 Transformer — significantly larger architecture for per-driver memorisation.

    Key improvements over v2
    ------------------------
    * d_model=384 (1.5×), 8 layers (1.33×), dim_ff=1536 (1.5×): ~14M params vs ~5M in v2.
    * Circuit×Compound multiplicative gate: a sigmoid-gated modulator computed from the
      joint (circuit, compound) embedding applied element-wise to the full sequence.
      Allows the model to specialise each of the 768 feature dimensions per
      (circuit, compound) pair without relying on additive priors.
    * 4th output head `head_value`: predicts normalised cumulative remaining-stint time.
      Trained with direct supervision, enabling the strategy engine to estimate
      whether an additional pit stop is cost-effective.
    * 15 continuous features: adds `stint_number_norm` (feature index 14).
    * dropout=0.0, weight_decay=0.0: intentional overfitting — one model per driver,
      circuits/compounds are fixed.
    * pos_enc: 100 positions (vs 50 in v2), supporting context_len=40 + long rollouts.

    Continuous features (15):
      0  stint_age_norm          7  brake_dev
      1  delta_norm              8  speed_dev
      2  pace_intent_norm        9  rpm_norm
      3  gap_norm               10  drs_dev
      4  track_temp_norm        11  push_index
      5  air_temp_norm          12  race_lap_norm
      6  throttle_dev           13  lap_speed_deviation
                                14  stint_number_norm  ← NEW

    Categorical (3): compound_int, lap_type_int, session_type_int
    Static:          circuit_idx, compound_idx_static (both per-sequence scalars)
    """

    def __init__(
        self,
        d_model:     int   = TRANSFORMER_V3_D_MODEL,
        n_heads:     int   = TRANSFORMER_V3_N_HEADS,
        n_layers:    int   = TRANSFORMER_V3_N_LAYERS,
        dim_ff:      int   = TRANSFORMER_V3_DIM_FF,
        dropout:     float = TRANSFORMER_V3_DROPOUT,
        context_len: int   = TRANSFORMER_V3_CONTEXT_LAPS,
        n_cont:      int   = TRANSFORMER_V3_INPUT_DIM,  # 15
    ):
        super().__init__()
        self.d_model = d_model

        # Per-timestep categorical embeddings (larger dims than v2)
        self.compound_emb  = nn.Embedding(_COMPOUND_EMB_VOCAB,     24, padding_idx=0)
        self.lap_type_emb  = nn.Embedding(_LAP_TYPE_EMB_VOCAB,      8, padding_idx=0)
        self.session_emb   = nn.Embedding(_SESSION_TYPE_EMB_VOCAB,  8, padding_idx=0)
        self.circuit_emb   = nn.Embedding(_CIRCUIT_EMB_VOCAB,      48, padding_idx=0)

        # Separate gate embedding for compound (distinct role from per-timestep embedding)
        self.compound_emb_gate = nn.Embedding(_COMPOUND_EMB_VOCAB, 24, padding_idx=0)

        # Per-timestep projections: continuous → d_model//2, cat_concat → d_model//2
        cat_dim = 24 + 8 + 8 + 48  # 88
        self.cont_proj = nn.Linear(n_cont, d_model // 2)
        self.cat_proj  = nn.Linear(cat_dim, d_model // 2)

        # Circuit×Compound multiplicative gate
        # [circuit_emb(48) || compound_emb_gate(24)] → d_model, then sigmoid(LayerNorm)
        self.cc_gate_proj = nn.Linear(48 + 24, d_model)
        self.cc_gate_ln   = nn.LayerNorm(d_model)

        self.input_ln = nn.LayerNorm(d_model)

        # Positional encoding: 100 positions (supports context_len=40 + long rollouts)
        self.pos_enc = nn.Embedding(100, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
        )

        # Four output heads (applied to last-token representation)
        self.head_delta = nn.Linear(d_model, 1)  # normalised Δlap_time  [Huber]
        self.head_abs   = nn.Linear(d_model, 1)  # normalised abs lap_time [MSE × aux_w]
        self.head_deg   = nn.Linear(d_model, 1)  # normalised deg rate     [MSE × deg_w]
        self.head_value = nn.Linear(d_model, 1)  # normalised cum. remaining cost [MSE × val_w]

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_cont:              torch.Tensor,  # (B, T, 15)
        x_cat:               torch.Tensor,  # (B, T, 3) [compound, lap_type, session_type]
        circuit_idx:         torch.Tensor,  # (B,)
        compound_idx_static: torch.Tensor,  # (B,)  compound for this stint (scalar per seq)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x_cont.shape

        # -- Per-timestep categorical embeddings --
        comp     = self.compound_emb(x_cat[:, :, 0])                         # (B, T, 24)
        lt       = self.lap_type_emb(x_cat[:, :, 1])                         # (B, T, 8)
        st       = self.session_emb(x_cat[:, :, 2])                          # (B, T, 8)
        circ_seq = self.circuit_emb(circuit_idx).unsqueeze(1).expand(-1, T, -1)  # (B, T, 48)
        cat_concat = torch.cat([comp, lt, st, circ_seq], dim=-1)             # (B, T, 88)

        # -- Dual projection and concat --
        x_c = self.cont_proj(x_cont)     # (B, T, d_model//2)
        x_k = self.cat_proj(cat_concat)  # (B, T, d_model//2)
        x   = torch.cat([x_c, x_k], dim=-1)  # (B, T, d_model)

        # -- Circuit×Compound multiplicative gate --
        circ_v = self.circuit_emb(circuit_idx)                    # (B, 48)
        comp_v = self.compound_emb_gate(compound_idx_static)      # (B, 24)
        gate   = torch.sigmoid(
            self.cc_gate_ln(self.cc_gate_proj(torch.cat([circ_v, comp_v], dim=-1)))
        )                                                          # (B, d_model)
        x = x * gate.unsqueeze(1)                                 # (B, T, d_model)

        # -- LayerNorm + positional encoding --
        x = self.input_ln(x)
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_enc(pos_ids)

        # -- Transformer encoder --
        x = self.transformer(x)  # (B, T, d_model)
        x = x[:, -1, :]          # (B, d_model)  last-token rep

        return self.head_delta(x), self.head_abs(x), self.head_deg(x), self.head_value(x)


# ---------------------------------------------------------------------------
# Context-seed builders
# ---------------------------------------------------------------------------


def build_context_seed(
    practice_dist: PracticeDistribution,
    context_len: int,
    delta_std: float,
    track_temp_ref: float,
    air_temp_ref: float,
    compound_int: int,
    session_type_int: int,
    circuit_int: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    v2 context-seed builder.  Returns float32 array of shape (context_len, 14).

    Features 7-12 (telemetry deviations) are set to 0 — realistic at inference time
    when live telemetry is unavailable.  push_index (feature 12) is sampled from
    the neutral end of the range.  race_lap_norm (feature 13) defaults to 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = context_len
    deltas = rng.normal(practice_dist.delta_mu, practice_dist.delta_sigma, size=T)
    delta_norm = (deltas / (delta_std if delta_std > 0 else 1.0)).astype(np.float32)

    pace_intent = np.zeros(T, dtype=np.float32)
    gap_norm    = np.zeros(T, dtype=np.float32)

    track_temp = rng.normal(practice_dist.track_temp_mu, practice_dist.track_temp_sigma)
    air_temp   = rng.normal(practice_dist.air_temp_mu,   practice_dist.air_temp_sigma)
    track_temp_norm = float((track_temp - track_temp_ref) / 10.0)
    air_temp_norm   = float((air_temp   - air_temp_ref)   / 8.0)

    stint_age_norm = (np.arange(1, T + 1, dtype=np.float32) / 40.0)

    # Telemetry deviations (features 7-11, 14) — zero at inference
    zeros = np.zeros(T, dtype=np.float32)

    # push_index (feature 12): neutral (0)
    push_index = np.zeros(T, dtype=np.float32)

    # race_lap_norm (feature 13): 0 (start of stint assumption)
    race_lap_norm = np.zeros(T, dtype=np.float32)

    x_cont = np.column_stack([
        stint_age_norm,                                          # 1
        delta_norm,                                              # 2
        pace_intent,                                             # 3
        gap_norm,                                                # 4
        np.full(T, track_temp_norm, dtype=np.float32),          # 5
        np.full(T, air_temp_norm,   dtype=np.float32),          # 6
        zeros,   # throttle_dev                                  # 7
        zeros,   # brake_dev                                     # 8
        zeros,   # speed_dev                                     # 9
        zeros,   # rpm_norm                                      # 10
        zeros,   # drs_dev                                       # 11
        push_index,                                              # 12
        race_lap_norm,                                           # 13
        zeros,   # lap_speed_deviation                           # 14
    ])  # (T, 14)

    return x_cont.astype(np.float32)


def build_context_seed_v3(
    practice_dist: PracticeDistribution,
    context_len: int,
    delta_std: float,
    track_temp_ref: float,
    air_temp_ref: float,
    compound_int: int,
    session_type_int: int,
    circuit_int: int = 0,
    rng: Optional[np.random.Generator] = None,
    stint_number: int = 1,
) -> np.ndarray:
    """
    v3 context-seed builder.  Returns float32 array of shape (context_len, 15).

    Identical to build_context_seed (v2) except adds feature 14: stint_number_norm.
    The 15th feature is filled with the constant value clip((stint_number-1)/3, 0, 1.5).
    """
    # Build v2 seed first (14 features)
    seed_v2 = build_context_seed(
        practice_dist, context_len, delta_std, track_temp_ref, air_temp_ref,
        compound_int, session_type_int, circuit_int, rng,
    )  # (T, 14)

    T = context_len
    stint_number_norm = float(np.clip((stint_number - 1) / 3.0, 0.0, 1.5))
    stint_col = np.full((T, 1), stint_number_norm, dtype=np.float32)

    return np.concatenate([seed_v2, stint_col], axis=1)  # (T, 15)


def _build_context_seed_v1(
    practice_dist: PracticeDistribution,
    context_len: int,
    delta_std: float,
    track_temp_ref: float,
    air_temp_ref: float,
    compound_int: int,
    session_type_int: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    v1 context-seed builder (kept for backward compat).
    Returns (x_cont (T,6), x_cat (T,3)).
    """
    if rng is None:
        rng = np.random.default_rng()

    T = context_len
    deltas     = rng.normal(practice_dist.delta_mu, practice_dist.delta_sigma, size=T)
    delta_norm = (deltas / (delta_std if delta_std > 0 else 1.0)).astype(np.float32)

    track_temp = rng.normal(practice_dist.track_temp_mu, practice_dist.track_temp_sigma)
    air_temp   = rng.normal(practice_dist.air_temp_mu,   practice_dist.air_temp_sigma)
    track_temp_norm = float((track_temp - track_temp_ref) / 10.0)
    air_temp_norm   = float((air_temp   - air_temp_ref)   / 10.0)

    stint_ages = np.arange(1, T + 1, dtype=np.float32) / 40.0

    x_cont = np.column_stack([
        stint_ages,
        delta_norm,
        np.zeros(T, dtype=np.float32),  # pace_intent
        np.zeros(T, dtype=np.float32),  # gap_norm
        np.full(T, track_temp_norm, dtype=np.float32),
        np.full(T, air_temp_norm,   dtype=np.float32),
    ])  # (T, 6)

    lap_type_int = LAP_TYPE_VOCAB.get("normal", 1)
    x_cat = np.column_stack([
        np.full(T, compound_int,     dtype=np.int64),
        np.full(T, lap_type_int,     dtype=np.int64),
        np.full(T, session_type_int, dtype=np.int64),
    ])  # (T, 3)

    return x_cont, x_cat


# ---------------------------------------------------------------------------
# High-level model wrapper
# ---------------------------------------------------------------------------


class TransformerPaceModel:
    """
    Unified wrapper for all Transformer architectures.

    model_version="v3" uses TyreDegradationTransformerV3 (d_model=768, 12 layers, CC gate).
    model_version="v2" uses TyreDegradationTransformerV2 (d_model=256, 6 layers).
    model_version="v1" uses TyreTransformerNet with v1 constants (backward compat).
    """

    def __init__(
        self,
        context_len: int = TRANSFORMER_V2_CONTEXT_LAPS,
        d_model:     int = TRANSFORMER_V2_D_MODEL,
        n_heads:     int = TRANSFORMER_V2_N_HEADS,
        n_layers:    int = TRANSFORMER_V2_N_LAYERS,
        dim_ff:      int = TRANSFORMER_V2_DIM_FF,
        dropout:     float = TRANSFORMER_V2_DROPOUT,
        model_version: str = "v2",
    ):
        self.context_len   = context_len
        self.d_model       = d_model
        self.n_heads       = n_heads
        self.n_layers      = n_layers
        self.dim_ff        = dim_ff
        self.dropout       = dropout
        self.model_version = model_version

        self.model: Optional[Union[TyreTransformerNet, TyreDegradationTransformerV2, TyreDegradationTransformerV3]] = None
        self.encoders: Dict[str, Dict] = {
            "compound":     COMPOUND_VOCAB.copy(),
            "lap_type":     LAP_TYPE_VOCAB.copy(),
            "session_type": SESSION_TYPE_VOCAB.copy(),
        }
        self.stats: Dict = {}
        self.practice_dist: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode(series: pd.Series, encoder: Dict) -> np.ndarray:
        def _lookup(x):
            if isinstance(x, str):
                return encoder.get(x.upper(), encoder.get(x, 0))
            return encoder.get(x, 0)
        return series.map(_lookup).fillna(0).astype(int).values

    @staticmethod
    def _encode_lap_type(series: pd.Series) -> np.ndarray:
        return series.map(lambda x: LAP_TYPE_VOCAB.get(x, 0)).fillna(0).astype(int).values

    @staticmethod
    def _encode_session_type(series: pd.Series) -> np.ndarray:
        return series.map(lambda x: SESSION_TYPE_VOCAB.get(x, 0)).fillna(0).astype(int).values

    # ------------------------------------------------------------------
    # Statistics fitting
    # ------------------------------------------------------------------

    def _compute_stats(self, df: pd.DataFrame) -> None:
        """Compute base statistics (common to v1 and v2)."""
        self.stats["lap_mean"] = float(df["lap_time"].mean())
        self.stats["lap_std"]  = float(df["lap_time"].std()) or 1.0

        deltas: List[float] = []
        for _, grp in df.groupby(["session_key", "driver_id", "stint_number"]):
            d = grp.sort_values("lap_number")["lap_time"].diff().dropna().tolist()
            deltas.extend(d)
        self.stats["delta_std"]       = float(np.std(deltas)) if deltas else 1.0
        self.stats["track_temp_ref"]  = float(df["track_temp"].mean()) if "track_temp" in df.columns else 30.0
        self.stats["air_temp_ref"]    = float(df["air_temp"].mean())   if "air_temp"   in df.columns else 25.0

    def _compute_circuit_compound_stats(self, df: pd.DataFrame) -> Dict:
        """
        Compute per-(circuit_id, compound) telemetry stats for deviation features.
        Returns a dict keyed by (circuit_str, compound_upper_str).
        """
        cc_stats: Dict = {}
        if "circuit_id" not in df.columns or "compound" not in df.columns:
            return cc_stats

        telemetry_map = [
            ("mean_throttle", "throttle"),
            ("mean_brake",    "brake"),
            ("max_speed",     "speed"),
            ("drs_fraction",  "drs"),
            ("mean_rpm",      "rpm"),
        ]

        for (circuit_id, compound), grp in df.groupby(["circuit_id", "compound"]):
            entry: Dict = {}
            for col, alias in telemetry_map:
                if col in grp.columns and grp[col].notna().any():
                    vals = grp[col].dropna()
                    entry[f"mu_{alias}"]    = float(vals.mean())
                    entry[f"sigma_{alias}"] = float(max(vals.std(), 1e-3))
                else:
                    entry[f"mu_{alias}"]    = 0.5 if alias in ("throttle", "brake") else 0.0
                    entry[f"sigma_{alias}"] = 0.1
            cc_stats[(str(circuit_id), str(compound).upper())] = entry

        return cc_stats

    # ------------------------------------------------------------------
    # Sequence building — v1
    # ------------------------------------------------------------------

    def _build_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """v1 sequence builder — returns (X_cont N,T,6), (X_cat N,T,3), (y N)."""
        delta_std = self.stats["delta_std"]
        track_ref = self.stats["track_temp_ref"]
        air_ref   = self.stats["air_temp_ref"]
        lap_std   = self.stats["lap_std"]

        all_X_cont: List[np.ndarray] = []
        all_X_cat:  List[np.ndarray] = []
        all_y:      List[float]      = []

        for (session_key, driver_id), grp in df.groupby(["session_key", "driver_id"]):
            grp = grp[grp["lap_type"].isin(CONTEXT_LAP_TYPES)].sort_values("lap_number").copy()
            if len(grp) < self.context_len + 1:
                continue

            stint_med = grp.groupby("stint_number")["lap_time"].transform("median")
            grp["pace_intent_norm"] = (grp["lap_time"] - stint_med) / (lap_std or 1.0)
            grp["delta_norm"]       = grp["lap_time"].diff().fillna(0.0) / (delta_std or 1.0)
            grp["gap_norm"] = (
                grp["gap_to_car_ahead"].clip(0, 5).fillna(0.0) / 5.0
                if "gap_to_car_ahead" in grp.columns else 0.0
            )
            grp["track_temp_norm"] = (grp["track_temp"].fillna(track_ref) - track_ref) / 10.0
            grp["air_temp_norm"]   = (grp["air_temp"].fillna(air_ref)     - air_ref)   / 10.0
            grp["stint_age_norm"]  = grp["stint_age"] / 40.0

            comp_enc = self._encode(grp["compound"], self.encoders["compound"])
            lt_enc   = self._encode_lap_type(grp["lap_type"])
            st_enc   = self._encode_session_type(
                grp.get("session_type", pd.Series("RACE", index=grp.index))
            )

            cont = np.column_stack([
                grp["stint_age_norm"].values,
                grp["delta_norm"].values,
                grp["pace_intent_norm"].values,
                grp["gap_norm"].values,
                grp["track_temp_norm"].values,
                grp["air_temp_norm"].values,
            ]).astype(np.float32)

            cat       = np.column_stack([comp_enc, lt_enc, st_enc]).astype(np.int64)
            lap_types = grp["lap_type"].values
            lap_times = grp["lap_time"].values

            for i in range(self.context_len, len(grp)):
                if lap_types[i] not in TRAINING_LAP_TYPES:
                    continue
                delta_target = (lap_times[i] - lap_times[i - 1]) / (delta_std or 1.0)
                all_X_cont.append(cont[i - self.context_len: i])
                all_X_cat.append(cat[i - self.context_len: i])
                all_y.append(float(delta_target))

        if not all_X_cont:
            return (
                np.empty((0, self.context_len, 6), dtype=np.float32),
                np.empty((0, self.context_len, 3), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )
        return (
            np.array(all_X_cont, dtype=np.float32),
            np.array(all_X_cat,  dtype=np.int64),
            np.array(all_y,      dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Sequence building — v2
    # ------------------------------------------------------------------

    def _build_sequences_v2(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        v2 sequence builder.
        Returns (X_cont N,T,14), (X_cat N,T,3), (circuit_ids N),
                (y_delta N), (y_abs N), (y_deg N).
        """
        delta_std = self.stats["delta_std"]
        lap_std   = self.stats["lap_std"]
        lap_mean  = self.stats["lap_mean"]
        track_ref = self.stats["track_temp_ref"]
        air_ref   = self.stats["air_temp_ref"]
        cc_stats  = self.stats.get("circuit_compound_stats", {})

        all_X_cont: List[np.ndarray] = []
        all_X_cat:  List[np.ndarray] = []
        all_circuit: List[int]       = []
        all_y_delta: List[float]     = []
        all_y_abs:   List[float]     = []
        all_y_deg:   List[float]     = []

        for (session_key, driver_id), grp in df.groupby(["session_key", "driver_id"]):
            grp = grp[grp["lap_type"].isin(CONTEXT_LAP_TYPES)].sort_values("lap_number").copy()
            if len(grp) < self.context_len + 1:
                continue

            circuit_id  = str(grp["circuit_id"].iloc[0]) if "circuit_id" in grp.columns else "Unknown"
            circuit_int = CIRCUIT_VOCAB.get(circuit_id, 0)
            exp_laps    = CIRCUIT_EXPECTED_LAPS.get(circuit_id, 55)

            # -- Base features --
            stint_med = grp.groupby("stint_number")["lap_time"].transform("median")
            grp["pace_intent_norm"] = (grp["lap_time"] - stint_med) / (lap_std or 1.0)
            grp["delta_norm"]       = grp["lap_time"].diff().fillna(0.0) / (delta_std or 1.0)
            grp["gap_norm"] = (
                grp["gap_to_car_ahead"].clip(0, 5).fillna(0.0) / 5.0
                if "gap_to_car_ahead" in grp.columns else 0.0
            )
            grp["track_temp_norm"] = (grp["track_temp"].fillna(track_ref) - track_ref) / 10.0
            grp["air_temp_norm"]   = (grp["air_temp"].fillna(air_ref)     - air_ref)   / 8.0
            grp["stint_age_norm"]  = grp["stint_age"] / 40.0
            grp["race_lap_norm"]   = (grp["lap_number"] / max(exp_laps, 1)).clip(0, 1.2)

            # -- Telemetry deviation features --
            def _dev(col, alias, default_mu):
                key_c = (circuit_id, grp["compound"].fillna("UNKNOWN").str.upper().iloc[0])
                mu    = cc_stats.get(key_c, {}).get(f"mu_{alias}", default_mu)
                sigma = cc_stats.get(key_c, {}).get(f"sigma_{alias}", 0.1)
                if col in grp.columns and grp[col].notna().any():
                    return ((grp[col].fillna(mu) - mu) / max(sigma, 1e-3)).clip(-5, 5)
                return pd.Series(0.0, index=grp.index)

            throttle_dev = _dev("mean_throttle", "throttle", 0.5)
            brake_dev    = _dev("mean_brake",    "brake",    0.3)
            speed_dev    = _dev("max_speed",     "speed",    0.0)
            drs_dev      = _dev("drs_fraction",  "drs",      0.0)

            if "mean_rpm" in grp.columns and grp["mean_rpm"].notna().any():
                rpm_norm = ((grp["mean_rpm"].fillna(10000) - 10000) / 3000).clip(-3, 3)
            else:
                rpm_norm = pd.Series(0.0, index=grp.index)

            push_index = (
                0.4 * throttle_dev
                + 0.3 * speed_dev
                + 0.2 * brake_dev
                + 0.1 * drs_dev
            ).clip(-3, 3)

            # -- Deg rate label (per stint polyfit) --
            deg_rate = pd.Series(0.0, index=grp.index)
            for stint_num, sdf in grp.groupby("stint_number"):
                normal_sdf = sdf[sdf["lap_type"] == "normal"]
                if len(normal_sdf) >= 3:
                    slope = float(
                        np.polyfit(normal_sdf["stint_age"].values,
                                   normal_sdf["lap_time"].values, 1)[0]
                    )
                else:
                    slope = 0.05
                deg_rate[sdf.index] = slope / (lap_std or 1.0)

            # -- Assemble continuous feature matrix (N, 14) --
            cont = np.column_stack([
                grp["stint_age_norm"].values,    # 1
                grp["delta_norm"].values,         # 2
                grp["pace_intent_norm"].values,   # 3
                grp["gap_norm"].values,           # 4
                grp["track_temp_norm"].values,    # 5
                grp["air_temp_norm"].values,      # 6
                throttle_dev.values,              # 7
                brake_dev.values,                 # 8
                speed_dev.values,                 # 9
                rpm_norm.values,                  # 10
                drs_dev.values,                   # 11
                push_index.values,                # 12
                grp["race_lap_norm"].values,      # 13
                speed_dev.values,                 # 14  (lap_speed_deviation = speed_dev)
            ]).astype(np.float32)
            np.nan_to_num(cont, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)

            # -- Categorical encoding --
            comp_enc = self._encode(grp["compound"], self.encoders["compound"])
            lt_enc   = self._encode_lap_type(grp["lap_type"])
            st_enc   = self._encode_session_type(
                grp.get("session_type", pd.Series("RACE", index=grp.index))
            )
            cat = np.column_stack([comp_enc, lt_enc, st_enc]).astype(np.int64)

            lap_types = grp["lap_type"].values
            lap_times = grp["lap_time"].values
            deg_arr   = deg_rate.values

            for i in range(self.context_len, len(grp)):
                if lap_types[i] not in TRAINING_LAP_TYPES:
                    continue
                delta_target = (lap_times[i] - lap_times[i - 1]) / (delta_std or 1.0)
                abs_target   = (lap_times[i] - lap_mean)          / (lap_std  or 1.0)
                deg_target   = float(deg_arr[i])

                all_X_cont.append(cont[i - self.context_len: i])
                all_X_cat.append(cat[i - self.context_len: i])
                all_circuit.append(circuit_int)
                all_y_delta.append(float(delta_target))
                all_y_abs.append(float(abs_target))
                all_y_deg.append(float(deg_target))

        if not all_X_cont:
            T = self.context_len
            return (
                np.empty((0, T, 14), dtype=np.float32),
                np.empty((0, T, 3),  dtype=np.int64),
                np.empty((0,),       dtype=np.int64),
                np.empty((0,),       dtype=np.float32),
                np.empty((0,),       dtype=np.float32),
                np.empty((0,),       dtype=np.float32),
            )
        return (
            np.array(all_X_cont,  dtype=np.float32),
            np.array(all_X_cat,   dtype=np.int64),
            np.array(all_circuit, dtype=np.int64),
            np.array(all_y_delta, dtype=np.float32),
            np.array(all_y_abs,   dtype=np.float32),
            np.array(all_y_deg,   dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 15,
        batch_size: int = 64,
        learning_rate: float = 5e-4,
        val_df: Optional[pd.DataFrame] = None,
        patience: int = 5,
        aux_loss_w: float = TRANSFORMER_V2_AUX_LOSS_W,
        deg_loss_w: float = TRANSFORMER_V2_DEG_LOSS_W,
        log_csv_path: Optional[str] = None,
    ) -> ModelBundle:
        """Train the model on the provided feature DataFrame."""
        if df.empty:
            raise ValueError("Cannot train on an empty DataFrame.")

        torch.manual_seed(RANDOM_SEED)
        self._compute_stats(df)

        if self.model_version == "v3":
            # Use v3 loss weights and LR unless caller explicitly passed v2 defaults
            eff_aux = aux_loss_w if aux_loss_w != TRANSFORMER_V2_AUX_LOSS_W else TRANSFORMER_V3_AUX_LOSS_W
            eff_deg = deg_loss_w if deg_loss_w != TRANSFORMER_V2_DEG_LOSS_W else TRANSFORMER_V3_DEG_LOSS_W
            eff_lr  = learning_rate if learning_rate != 5e-4 else 2e-4
            return self._train_v3(
                df, epochs, batch_size, eff_lr, val_df, patience,
                eff_aux, eff_deg, log_csv_path,
            )
        if self.model_version == "v2":
            return self._train_v2(
                df, epochs, batch_size, learning_rate, val_df, patience,
                aux_loss_w, deg_loss_w, log_csv_path,
            )
        return self._train_v1(df, epochs, batch_size, learning_rate)

    def _train_v1(
        self,
        df: pd.DataFrame,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> ModelBundle:
        X_cont, X_cat, y = self._build_sequences(df)
        if len(y) == 0:
            raise ValueError("No valid training sequences found.")

        logger.info("TransformerPaceModel v1 train: n=%d d_model=%d", len(y), self.d_model)

        self.model = TyreTransformerNet(
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
            dim_ff=self.dim_ff, dropout=self.dropout, context_len=self.context_len,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        steps_per_epoch = max(1, math.ceil(len(y) / batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, steps_per_epoch=steps_per_epoch, epochs=epochs,
        )
        loss_fn = nn.HuberLoss(delta=1.0)
        loader  = DataLoader(SequenceDataset(X_cont, X_cat, y), batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total, n = 0.0, 0
            for bx_cont, bx_cat, by in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(bx_cont, bx_cat), by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                total += loss.item(); n += 1
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("  v1 epoch %d/%d  loss=%.4f", epoch + 1, epochs, total / max(1, n))

        return ModelBundle(model_state=self.model.state_dict(), encoders=self.encoders, stats=self.stats)

    def _train_v2(
        self,
        df: pd.DataFrame,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        val_df: Optional[pd.DataFrame],
        patience: int,
        aux_loss_w: float,
        deg_loss_w: float,
        log_csv_path: Optional[str],
    ) -> ModelBundle:
        # Compute extended stats (circuit×compound telemetry)
        self.stats["circuit_compound_stats"] = self._compute_circuit_compound_stats(df)

        X_cont, X_cat, circuit_ids, y_delta, y_abs, y_deg = self._build_sequences_v2(df)
        if len(y_delta) == 0:
            raise ValueError("No valid v2 training sequences found.")

        logger.info(
            "TransformerPaceModel v2 train: n=%d  context=%d  d_model=%d  layers=%d",
            len(y_delta), self.context_len, self.d_model, self.n_layers,
        )

        # Optional validation sequences
        has_val = val_df is not None and not val_df.empty
        if has_val:
            vX_cont, vX_cat, v_circuit, vy_delta, _, _ = self._build_sequences_v2(val_df)
            has_val = len(vy_delta) > 0

        self.model = TyreDegradationTransformerV2(
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
            dim_ff=self.dim_ff, dropout=self.dropout, context_len=self.context_len,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        steps_per_epoch = max(1, math.ceil(len(y_delta) / batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, steps_per_epoch=steps_per_epoch, epochs=epochs,
        )
        loss_fn = nn.HuberLoss(delta=1.0)
        mse_fn  = nn.MSELoss()

        dataset = SequenceDatasetV2(X_cont, X_cat, circuit_ids, y_delta, y_abs, y_deg)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_mae  = float("inf")
        best_state    = None
        patience_left = patience

        # CSV log
        csv_file = None
        csv_writer = None
        if log_csv_path:
            csv_file   = open(log_csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["epoch", "train_loss", "val_mae"])

        try:
            for epoch in range(epochs):
                self.model.train()
                total, n = 0.0, 0
                for bx_cont, bx_cat, b_circuit, by_delta, by_abs, by_deg in loader:
                    optimizer.zero_grad()
                    d_pred, abs_pred, deg_pred = self.model(bx_cont, bx_cat, b_circuit)
                    loss = (
                        loss_fn(d_pred, by_delta)
                        + aux_loss_w * mse_fn(abs_pred, by_abs)
                        + deg_loss_w * mse_fn(deg_pred, by_deg)
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step(); scheduler.step()
                    total += loss.item(); n += 1

                avg_loss = total / max(1, n)

                # Validation MAE on delta head
                val_mae = float("nan")
                if has_val:
                    self.model.eval()
                    with torch.no_grad():
                        vX_c = torch.tensor(vX_cont,    dtype=torch.float32)
                        vX_k = torch.tensor(vX_cat,     dtype=torch.long)
                        v_ci = torch.tensor(v_circuit,  dtype=torch.long)
                        vy   = torch.tensor(vy_delta,   dtype=torch.float32).reshape(-1, 1)
                        preds, _ , _ = self.model(vX_c, vX_k, v_ci)
                        val_mae = float(torch.mean(torch.abs(preds - vy)).item())

                if csv_writer:
                    csv_writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{val_mae:.6f}"])

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(
                        "  v2 epoch %d/%d  loss=%.4f  val_mae=%s",
                        epoch + 1, epochs, avg_loss,
                        f"{val_mae:.4f}" if not math.isnan(val_mae) else "N/A",
                    )

                # Early stopping
                if has_val and not math.isnan(val_mae):
                    if val_mae < best_val_mae - 1e-5:
                        best_val_mae  = val_mae
                        best_state    = {k: v.clone() for k, v in self.model.state_dict().items()}
                        patience_left = patience
                    else:
                        patience_left -= 1
                        if patience_left <= 0:
                            logger.info(
                                "  Early stopping at epoch %d (best val_mae=%.4f)",
                                epoch + 1, best_val_mae,
                            )
                            break
        finally:
            if csv_file:
                csv_file.close()

        # Restore best checkpoint if early stopping triggered
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        return ModelBundle(model_state=self.model.state_dict(), encoders=self.encoders, stats=self.stats)

    # ------------------------------------------------------------------
    # Sequence building — v3
    # ------------------------------------------------------------------

    def _build_sequences_v3(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        v3 sequence builder.
        Returns (X_cont N,T,15), (X_cat N,T,3), (circuit_ids N), (compound_static N),
                (y_delta N), (y_abs N), (y_deg N), (y_value N).

        Differences from v2:
          - X_cont has 15 features (adds stint_number_norm at index 14)
          - compound_static: compound int for the stint (for CC gate)
          - y_value: normalised cumulative remaining-stint lap time
        """
        delta_std = self.stats["delta_std"]
        lap_std   = self.stats["lap_std"]
        lap_mean  = self.stats["lap_mean"]
        track_ref = self.stats["track_temp_ref"]
        air_ref   = self.stats["air_temp_ref"]
        cc_stats  = self.stats.get("circuit_compound_stats", {})

        all_X_cont:        List[np.ndarray] = []
        all_X_cat:         List[np.ndarray] = []
        all_circuit:       List[int]        = []
        all_compound_stat: List[int]        = []
        all_y_delta:       List[float]      = []
        all_y_abs:         List[float]      = []
        all_y_deg:         List[float]      = []
        all_y_value:       List[float]      = []

        for (session_key, driver_id), grp in df.groupby(["session_key", "driver_id"]):
            grp = grp[grp["lap_type"].isin(CONTEXT_LAP_TYPES)].sort_values("lap_number").copy()
            if len(grp) < self.context_len + 1:
                continue

            circuit_id  = str(grp["circuit_id"].iloc[0]) if "circuit_id" in grp.columns else "Unknown"
            circuit_int = CIRCUIT_VOCAB.get(circuit_id, 0)
            exp_laps    = CIRCUIT_EXPECTED_LAPS.get(circuit_id, 55)

            # -- Base features (same as v2) --
            stint_med = grp.groupby("stint_number")["lap_time"].transform("median")
            grp["pace_intent_norm"] = (grp["lap_time"] - stint_med) / (lap_std or 1.0)
            grp["delta_norm"]       = grp["lap_time"].diff().fillna(0.0) / (delta_std or 1.0)
            grp["gap_norm"] = (
                grp["gap_to_car_ahead"].clip(0, 5).fillna(0.0) / 5.0
                if "gap_to_car_ahead" in grp.columns else 0.0
            )
            grp["track_temp_norm"] = (grp["track_temp"].fillna(track_ref) - track_ref) / 10.0
            grp["air_temp_norm"]   = (grp["air_temp"].fillna(air_ref)     - air_ref)   / 8.0
            grp["stint_age_norm"]  = grp["stint_age"] / 40.0
            grp["race_lap_norm"]   = (grp["lap_number"] / max(exp_laps, 1)).clip(0, 1.2)

            # -- stint_number_norm (NEW feature 14) --
            if "stint_number" in grp.columns:
                grp["stint_number_norm"] = (grp["stint_number"] / 3.0).clip(0, 1.5)
            else:
                grp["stint_number_norm"] = 0.0

            # -- Telemetry deviation features (same as v2) --
            def _dev(col, alias, default_mu):
                key_c = (circuit_id, grp["compound"].fillna("UNKNOWN").str.upper().iloc[0])
                mu    = cc_stats.get(key_c, {}).get(f"mu_{alias}", default_mu)
                sigma = cc_stats.get(key_c, {}).get(f"sigma_{alias}", 0.1)
                if col in grp.columns and grp[col].notna().any():
                    return ((grp[col].fillna(mu) - mu) / max(sigma, 1e-3)).clip(-5, 5)
                return pd.Series(0.0, index=grp.index)

            throttle_dev = _dev("mean_throttle", "throttle", 0.5)
            brake_dev    = _dev("mean_brake",    "brake",    0.3)
            speed_dev    = _dev("max_speed",     "speed",    0.0)
            drs_dev      = _dev("drs_fraction",  "drs",      0.0)

            if "mean_rpm" in grp.columns and grp["mean_rpm"].notna().any():
                rpm_norm = ((grp["mean_rpm"].fillna(10000) - 10000) / 3000).clip(-3, 3)
            else:
                rpm_norm = pd.Series(0.0, index=grp.index)

            push_index = (
                0.4 * throttle_dev + 0.3 * speed_dev + 0.2 * brake_dev + 0.1 * drs_dev
            ).clip(-3, 3)

            # -- Deg rate label (per stint polyfit) --
            deg_rate = pd.Series(0.0, index=grp.index)
            for stint_num, sdf in grp.groupby("stint_number"):
                normal_sdf = sdf[sdf["lap_type"] == "normal"]
                if len(normal_sdf) >= 3:
                    slope = float(
                        np.polyfit(normal_sdf["stint_age"].values,
                                   normal_sdf["lap_time"].values, 1)[0]
                    )
                else:
                    slope = 0.05
                deg_rate[sdf.index] = slope / (lap_std or 1.0)

            # -- Assemble continuous feature matrix (N, 15) --
            cont = np.column_stack([
                grp["stint_age_norm"].values,    # 0
                grp["delta_norm"].values,         # 1
                grp["pace_intent_norm"].values,   # 2
                grp["gap_norm"].values,           # 3
                grp["track_temp_norm"].values,    # 4
                grp["air_temp_norm"].values,      # 5
                throttle_dev.values,              # 6
                brake_dev.values,                 # 7
                speed_dev.values,                 # 8
                rpm_norm.values,                  # 9
                drs_dev.values,                   # 10
                push_index.values,                # 11
                grp["race_lap_norm"].values,      # 12
                speed_dev.values,                 # 13  (lap_speed_deviation = speed_dev)
                grp["stint_number_norm"].values,  # 14  NEW
            ]).astype(np.float32)
            np.nan_to_num(cont, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)

            # -- Categorical encoding --
            comp_enc = self._encode(grp["compound"], self.encoders["compound"])
            lt_enc   = self._encode_lap_type(grp["lap_type"])
            st_enc   = self._encode_session_type(
                grp.get("session_type", pd.Series("RACE", index=grp.index))
            )
            cat = np.column_stack([comp_enc, lt_enc, st_enc]).astype(np.int64)

            lap_types = grp["lap_type"].values
            lap_times = grp["lap_time"].values
            deg_arr   = deg_rate.values

            # Precompute suffix sums for y_value (cumulative remaining lap times per stint)
            # We compute per-stint suffix sums and store by row index
            stint_suffix: Dict[int, np.ndarray] = {}
            for stint_num, sdf in grp.groupby("stint_number"):
                sdf_sorted = sdf.sort_values("lap_number")
                times_arr  = sdf_sorted["lap_time"].values
                # suffix_sum[k] = sum(times_arr[k+1:]) / ((len - k - 1) * lap_std)
                n = len(times_arr)
                suffix      = np.zeros(n, dtype=np.float64)
                running_sum = 0.0
                for k in range(n - 1, -1, -1):
                    suffix[k]    = running_sum
                    running_sum += times_arr[k]
                # Normalise: divide by expected remaining laps * lap_std
                remaining_counts = np.arange(n - 1, -1, -1, dtype=np.float64)  # n-1, n-2, ..., 0
                remaining_counts = np.maximum(remaining_counts, 1.0)
                suffix_norm = suffix / (remaining_counts * max(lap_std, 1.0))
                for pos, idx in enumerate(sdf_sorted.index):
                    stint_suffix[grp.index.get_loc(idx)] = suffix_norm[pos]

            for i in range(self.context_len, len(grp)):
                if lap_types[i] not in TRAINING_LAP_TYPES:
                    continue
                delta_target = (lap_times[i] - lap_times[i - 1]) / (delta_std or 1.0)
                abs_target   = (lap_times[i] - lap_mean)          / (lap_std  or 1.0)
                deg_target   = float(deg_arr[i])
                value_target = float(stint_suffix.get(i, 0.0))

                # compound for the stint at position i (for CC gate)
                comp_int_i = int(comp_enc[i])

                all_X_cont.append(cont[i - self.context_len: i])
                all_X_cat.append(cat[i - self.context_len: i])
                all_circuit.append(circuit_int)
                all_compound_stat.append(comp_int_i)
                all_y_delta.append(float(delta_target))
                all_y_abs.append(float(abs_target))
                all_y_deg.append(float(deg_target))
                all_y_value.append(value_target)

        if not all_X_cont:
            T = self.context_len
            return (
                np.empty((0, T, 15), dtype=np.float32),
                np.empty((0, T, 3),  dtype=np.int64),
                np.empty((0,),       dtype=np.int64),
                np.empty((0,),       dtype=np.int64),
                np.empty((0,),       dtype=np.float32),
                np.empty((0,),       dtype=np.float32),
                np.empty((0,),       dtype=np.float32),
                np.empty((0,),       dtype=np.float32),
            )
        return (
            np.array(all_X_cont,        dtype=np.float32),
            np.array(all_X_cat,         dtype=np.int64),
            np.array(all_circuit,       dtype=np.int64),
            np.array(all_compound_stat, dtype=np.int64),
            np.array(all_y_delta,       dtype=np.float32),
            np.array(all_y_abs,         dtype=np.float32),
            np.array(all_y_deg,         dtype=np.float32),
            np.array(all_y_value,       dtype=np.float32),
        )

    def _train_v3(
        self,
        df: pd.DataFrame,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        val_df: Optional[pd.DataFrame],
        patience: int,
        aux_loss_w: float,
        deg_loss_w: float,
        log_csv_path: Optional[str],
    ) -> ModelBundle:
        value_loss_w = TRANSFORMER_V3_VALUE_LOSS_W
        self.stats["circuit_compound_stats"] = self._compute_circuit_compound_stats(df)

        X_cont, X_cat, circuit_ids, compound_static, y_delta, y_abs, y_deg, y_value = \
            self._build_sequences_v3(df)
        if len(y_delta) == 0:
            raise ValueError("No valid v3 training sequences found.")

        logger.info(
            "TransformerPaceModel v3 train: n=%d  context=%d  d_model=%d  layers=%d",
            len(y_delta), self.context_len, self.d_model, self.n_layers,
        )

        has_val = val_df is not None and not val_df.empty
        if has_val:
            vX_cont, vX_cat, v_circuit, v_comp_stat, vy_delta, _, _, _ = \
                self._build_sequences_v3(val_df)
            has_val = len(vy_delta) > 0

        self.model = TyreDegradationTransformerV3(
            d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
            dim_ff=self.dim_ff, dropout=self.dropout, context_len=self.context_len,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=0.0,
        )
        steps_per_epoch = max(1, math.ceil(len(y_delta) / batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, steps_per_epoch=steps_per_epoch, epochs=epochs,
        )
        loss_fn = nn.HuberLoss(delta=1.0)
        mse_fn  = nn.MSELoss()

        dataset = SequenceDatasetV3(
            X_cont, X_cat, circuit_ids, compound_static,
            y_delta, y_abs, y_deg, y_value,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_mae  = float("inf")
        best_state    = None
        patience_left = patience

        csv_file = None
        csv_writer = None
        if log_csv_path:
            csv_file   = open(log_csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["epoch", "train_loss", "val_mae"])

        try:
            for epoch in range(epochs):
                self.model.train()
                total, n = 0.0, 0
                for bx_cont, bx_cat, b_circuit, b_comp_stat, by_delta, by_abs, by_deg, by_value in loader:
                    optimizer.zero_grad()
                    d_pred, abs_pred, deg_pred, val_pred = self.model(
                        bx_cont, bx_cat, b_circuit, b_comp_stat,
                    )
                    loss = (
                        loss_fn(d_pred,   by_delta)
                        + aux_loss_w   * mse_fn(abs_pred, by_abs)
                        + deg_loss_w   * mse_fn(deg_pred, by_deg)
                        + value_loss_w * mse_fn(val_pred, by_value)
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step(); scheduler.step()
                    total += loss.item(); n += 1

                avg_loss = total / max(1, n)

                val_mae = float("nan")
                if has_val:
                    self.model.eval()
                    with torch.no_grad():
                        vX_c = torch.tensor(vX_cont,       dtype=torch.float32)
                        vX_k = torch.tensor(vX_cat,        dtype=torch.long)
                        v_ci = torch.tensor(v_circuit,     dtype=torch.long)
                        v_cs = torch.tensor(v_comp_stat,   dtype=torch.long)
                        vy   = torch.tensor(vy_delta,      dtype=torch.float32).reshape(-1, 1)
                        preds, _, _, _ = self.model(vX_c, vX_k, v_ci, v_cs)
                        val_mae = float(torch.mean(torch.abs(preds - vy)).item())

                if csv_writer:
                    csv_writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{val_mae:.6f}"])

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(
                        "  v3 epoch %d/%d  loss=%.4f  val_mae=%s",
                        epoch + 1, epochs, avg_loss,
                        f"{val_mae:.4f}" if not math.isnan(val_mae) else "N/A",
                    )

                if has_val and not math.isnan(val_mae):
                    if val_mae < best_val_mae - 1e-5:
                        best_val_mae  = val_mae
                        best_state    = {k: v.clone() for k, v in self.model.state_dict().items()}
                        patience_left = patience
                    else:
                        patience_left -= 1
                        if patience_left <= 0:
                            logger.info(
                                "  Early stopping at epoch %d (best val_mae=%.4f)",
                                epoch + 1, best_val_mae,
                            )
                            break
        finally:
            if csv_file:
                csv_file.close()

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        return ModelBundle(model_state=self.model.state_dict(), encoders=self.encoders, stats=self.stats)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, bundle: ModelBundle, input_dim: int) -> None:
        """Restore model from a saved ModelBundle."""
        if self.model_version == "v3":
            self.model = TyreDegradationTransformerV3(
                d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
                dim_ff=self.dim_ff, dropout=self.dropout, context_len=self.context_len,
            )
        elif self.model_version == "v2":
            self.model = TyreDegradationTransformerV2(
                d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
                dim_ff=self.dim_ff, dropout=self.dropout, context_len=self.context_len,
            )
        else:
            self.model = TyreTransformerNet(
                d_model=self.d_model, n_heads=self.n_heads, n_layers=self.n_layers,
                dim_ff=self.dim_ff, dropout=self.dropout, context_len=self.context_len,
            )
        self.model.load_state_dict(bundle.model_state)
        self.model.eval()
        self.encoders = bundle.encoders
        self.stats    = bundle.stats

    # ------------------------------------------------------------------
    # Inference — rollout_mc
    # ------------------------------------------------------------------

    def rollout_mc(
        self,
        context_seed: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        stint_len: int,
        pace_bounds_or_intent: Union[Tuple[float, float], float],
        track_temp: float,
        air_temp: float,
        compound_int: int,
        lap_type_int: int = 1,
        session_type_int: int = 4,
        base_lap_time: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        circuit_int: int = 0,
        race_lap_start: int = 0,
    ) -> np.ndarray:
        """
        Autoregressively generate `stint_len` lap times.

        Dispatches to _rollout_v2 when model is TyreDegradationTransformerV2,
        otherwise falls back to _rollout_v1 for backward compatibility.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        if rng is None:
            rng = np.random.default_rng()
        if base_lap_time is None:
            base_lap_time = float(self.stats.get("lap_mean", 90.0))

        if isinstance(self.model, TyreDegradationTransformerV3):
            # v3 path (single-sim fallback; batched path is called directly by strategy_engine)
            if not isinstance(context_seed, np.ndarray):
                raise ValueError("v3 rollout_mc expects context_seed as np.ndarray (T, 15)")
            if not isinstance(pace_bounds_or_intent, tuple):
                pace_bounds_or_intent = (pace_bounds_or_intent, pace_bounds_or_intent)
            return self._rollout_v3(
                context_seed, stint_len, pace_bounds_or_intent,
                track_temp, air_temp, compound_int, lap_type_int, session_type_int,
                base_lap_time, rng, circuit_int, race_lap_start,
            )
        elif isinstance(self.model, TyreDegradationTransformerV2):
            # v2 path
            if not isinstance(context_seed, np.ndarray):
                raise ValueError("v2 rollout_mc expects context_seed as np.ndarray (T, 14)")
            if not isinstance(pace_bounds_or_intent, tuple):
                # Single float → treat as symmetric around that value
                pace_bounds_or_intent = (pace_bounds_or_intent, pace_bounds_or_intent)
            return self._rollout_v2(
                context_seed, stint_len, pace_bounds_or_intent,
                track_temp, air_temp, compound_int, lap_type_int, session_type_int,
                base_lap_time, rng, circuit_int, race_lap_start,
            )
        else:
            # v1 path
            if isinstance(context_seed, np.ndarray):
                raise ValueError("v1 rollout_mc expects context_seed as Tuple (x_cont, x_cat)")
            pace_intent = (
                pace_bounds_or_intent[0]
                if isinstance(pace_bounds_or_intent, tuple)
                else pace_bounds_or_intent
            )
            return self._rollout_v1(
                context_seed, stint_len, pace_intent,
                track_temp, air_temp, compound_int, lap_type_int, session_type_int,
                base_lap_time, rng,
            )

    def _rollout_v1(
        self,
        context_seed: Tuple[np.ndarray, np.ndarray],
        stint_len: int,
        pace_intent: float,
        track_temp: float,
        air_temp: float,
        compound_int: int,
        lap_type_int: int,
        session_type_int: int,
        base_lap_time: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        delta_std       = self.stats.get("delta_std", 1.0)
        track_ref       = self.stats.get("track_temp_ref", 30.0)
        air_ref         = self.stats.get("air_temp_ref",   25.0)
        track_temp_norm = float((track_temp - track_ref) / 10.0)
        air_temp_norm   = float((air_temp   - air_ref)   / 10.0)

        x_cont_ctx, x_cat_ctx = context_seed
        ctx_cont = x_cont_ctx.copy().astype(np.float32)
        ctx_cat  = x_cat_ctx.copy().astype(np.int64)

        current = float(base_lap_time)
        generated: List[float] = []
        self.model.eval()

        with torch.no_grad():
            for step in range(stint_len):
                new_cont = np.array([[
                    float(step + 1) / 40.0, 0.0, float(pace_intent),
                    0.0, track_temp_norm, air_temp_norm,
                ]], dtype=np.float32)
                new_cat = np.array([[compound_int, lap_type_int, session_type_int]], dtype=np.int64)

                ctx_cont_full = np.concatenate([ctx_cont, new_cont], axis=0)
                ctx_cat_full  = np.concatenate([ctx_cat,  new_cat],  axis=0)

                in_cont = torch.tensor(ctx_cont_full[-self.context_len:][np.newaxis], dtype=torch.float32)
                in_cat  = torch.tensor(ctx_cat_full[-self.context_len:][np.newaxis],  dtype=torch.long)

                delta_norm_pred = float(self.model(in_cont, in_cat).item())
                delta_lap       = delta_norm_pred * delta_std
                new_lap         = float(np.clip(current + delta_lap, 60.0, 300.0))
                generated.append(new_lap)
                current = new_lap

                new_cont[0, 1] = delta_norm_pred
                ctx_cont = np.concatenate([ctx_cont[1:], new_cont], axis=0)
                ctx_cat  = np.concatenate([ctx_cat[1:],  new_cat],  axis=0)

        return np.array(generated, dtype=np.float32)

    def _rollout_v2(
        self,
        ctx_seed: np.ndarray,                         # (T, 14)
        stint_len: int,
        pace_bounds: Tuple[float, float],             # (pace_hi_norm, pace_lo_norm)
        track_temp: float,
        air_temp: float,
        compound_int: int,
        lap_type_int: int,
        session_type_int: int,
        base_lap_time: float,
        rng: np.random.Generator,
        circuit_int: int,
        race_lap_start: int,
    ) -> np.ndarray:
        delta_std = self.stats.get("delta_std", 1.0)
        track_ref = self.stats.get("track_temp_ref", 30.0)
        air_ref   = self.stats.get("air_temp_ref",   25.0)
        lap_mean  = self.stats.get("lap_mean", 90.0)
        lap_std   = self.stats.get("lap_std",  1.0)
        exp_laps  = CIRCUIT_EXPECTED_LAPS.get(
            {v: k for k, v in CIRCUIT_VOCAB.items()}.get(circuit_int, ""), 55
        )

        pace_hi_norm, pace_lo_norm = pace_bounds
        # Sample two phase values for lerp trajectory
        phase_a = float(rng.uniform(pace_hi_norm, pace_lo_norm))
        phase_b = float(rng.uniform(pace_hi_norm, pace_lo_norm))

        track_temp_norm = float((track_temp - track_ref) / 10.0)
        air_temp_norm   = float((air_temp   - air_ref)   / 8.0)

        ctx = ctx_seed.copy().astype(np.float32)  # (T, 14)
        # Initialise cat context buffer: all positions are the current compound/session
        ctx_cat_buf = np.tile(
            np.array([[compound_int, lap_type_int, session_type_int]], dtype=np.int64),
            (self.context_len, 1),
        )  # (T, 3)

        current = float(base_lap_time)
        generated: List[float] = []
        circuit_tensor = torch.tensor([circuit_int], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            for step in range(stint_len):
                t = step / max(1, stint_len - 1)
                pace_intent_t = phase_a * (1 - t) + phase_b * t
                # push_index synthesised from pace (more negative intent = faster = push harder)
                push_index_t  = float(np.clip(-pace_intent_t, -3.0, 3.0))
                race_lap_norm_t = float(np.clip((race_lap_start + step) / max(exp_laps, 1), 0, 1.2))

                new_row = np.array([[
                    float(step + 1) / 40.0,   # 1 stint_age_norm
                    0.0,                        # 2 delta_norm (placeholder)
                    pace_intent_t,              # 3 pace_intent_norm
                    0.0,                        # 4 gap_norm
                    track_temp_norm,            # 5
                    air_temp_norm,              # 6
                    0.0, 0.0, 0.0, 0.0, 0.0,   # 7-11 telemetry devs (zero at inference)
                    push_index_t,               # 12
                    race_lap_norm_t,            # 13
                    0.0,                        # 14 lap_speed_deviation
                ]], dtype=np.float32)  # (1, 14)

                new_cat = np.array(
                    [[compound_int, lap_type_int, session_type_int]], dtype=np.int64
                )  # (1, 3)

                # Sliding window: append new row and take last context_len
                ctx_full = np.concatenate([ctx,         new_row], axis=0)  # (T+1, 14)
                cat_full = np.concatenate([ctx_cat_buf, new_cat], axis=0)  # (T+1, 3)

                in_cont = torch.tensor(ctx_full[-self.context_len:][np.newaxis], dtype=torch.float32)
                in_cat  = torch.tensor(cat_full[-self.context_len:][np.newaxis], dtype=torch.long)

                delta_pred, _, _ = self.model(in_cont, in_cat, circuit_tensor)
                delta_norm_pred  = float(delta_pred.item())
                delta_lap        = delta_norm_pred * delta_std
                new_lap          = float(np.clip(current + delta_lap, 60.0, 300.0))
                generated.append(new_lap)
                current = new_lap

                # Update context: slide window forward, fill predicted delta
                new_row[0, 1] = delta_norm_pred
                ctx         = np.concatenate([ctx[1:],         new_row], axis=0)
                ctx_cat_buf = np.concatenate([ctx_cat_buf[1:], new_cat], axis=0)

        return np.array(generated, dtype=np.float32)

    def _rollout_v2_batched(
        self,
        ctx_seeds: np.ndarray,           # (n_sim, T, 14)
        phase_as: np.ndarray,            # (n_sim,) pre-sampled pace lerp start
        phase_bs: np.ndarray,            # (n_sim,) pre-sampled pace lerp end
        stint_len: int,
        track_temp_norms: np.ndarray,    # (n_sim,) pre-normalised
        air_temp_norms: np.ndarray,      # (n_sim,) pre-normalised
        compound_int: int,
        lap_type_int: int,
        session_type_int: int,
        base_lap_time: float,
        circuit_int: int,
        race_lap_start: int,
    ) -> np.ndarray:                     # (n_sim, stint_len)
        """
        Vectorised v2 rollout: processes all n_sim simulations in a single batched
        forward pass per lap step, instead of n_sim sequential forward passes.
        Speedup: ~n_sim× over sequential _rollout_v2 calls.
        """
        n_sim = ctx_seeds.shape[0]
        delta_std = self.stats.get("delta_std", 1.0)
        exp_laps  = CIRCUIT_EXPECTED_LAPS.get(
            {v: k for k, v in CIRCUIT_VOCAB.items()}.get(circuit_int, ""), 55
        )

        ctx = ctx_seeds.copy().astype(np.float32)               # (n_sim, T, 14)
        ctx_cat_buf = np.tile(
            np.array([[compound_int, lap_type_int, session_type_int]], dtype=np.int64),
            (n_sim, self.context_len, 1),
        )                                                        # (n_sim, T, 3)

        currents       = np.full(n_sim, base_lap_time, dtype=np.float32)
        generated      = np.zeros((n_sim, stint_len), dtype=np.float32)
        circuit_tensor = torch.full((n_sim,), circuit_int, dtype=torch.long)

        new_cat = np.tile(
            np.array([[compound_int, lap_type_int, session_type_int]], dtype=np.int64),
            (n_sim, 1, 1),
        )  # (n_sim, 1, 3) — constant across steps

        self.model.eval()
        with torch.no_grad():
            for step in range(stint_len):
                t = step / max(1, stint_len - 1)
                pace_intent_t  = phase_as * (1.0 - t) + phase_bs * t    # (n_sim,)
                push_index_t   = np.clip(-pace_intent_t, -3.0, 3.0)     # (n_sim,)
                race_lap_norm_t = float(
                    np.clip((race_lap_start + step) / max(exp_laps, 1), 0.0, 1.2)
                )

                # Build new context row for every sim: (n_sim, 1, 14)
                new_rows = np.zeros((n_sim, 1, 14), dtype=np.float32)
                new_rows[:, 0, 0]  = float(step + 1) / 40.0   # stint_age_norm
                # new_rows[:, 0, 1] = 0.0                      # delta_norm placeholder
                new_rows[:, 0, 2]  = pace_intent_t             # pace_intent_norm
                # new_rows[:, 0, 3] = 0.0                      # gap_norm
                new_rows[:, 0, 4]  = track_temp_norms          # track_temp_norm
                new_rows[:, 0, 5]  = air_temp_norms            # air_temp_norm
                # features 6-10: telemetry devs — already 0
                new_rows[:, 0, 11] = push_index_t              # push_index
                new_rows[:, 0, 12] = race_lap_norm_t           # race_lap_norm
                # feature 13: lap_speed_deviation — already 0

                ctx_full = np.concatenate([ctx,         new_rows], axis=1)  # (n_sim, T+1, 14)
                cat_full = np.concatenate([ctx_cat_buf, new_cat],  axis=1)  # (n_sim, T+1, 3)

                in_cont = torch.tensor(
                    ctx_full[:, -self.context_len:, :], dtype=torch.float32
                )  # (n_sim, T, 14)
                in_cat  = torch.tensor(
                    cat_full[:, -self.context_len:, :], dtype=torch.long
                )  # (n_sim, T, 3)

                delta_pred, _, _ = self.model(in_cont, in_cat, circuit_tensor)
                delta_norm_preds  = delta_pred.squeeze(-1).cpu().numpy()  # (n_sim,)
                delta_laps        = delta_norm_preds * delta_std
                new_laps          = np.clip(currents + delta_laps, 60.0, 300.0)
                generated[:, step] = new_laps
                currents           = new_laps

                # Slide context window forward, writing the predicted delta
                new_rows[:, 0, 1] = delta_norm_preds  # fill delta_norm with prediction
                ctx         = np.concatenate([ctx[:, 1:, :],         new_rows], axis=1)
                ctx_cat_buf = np.concatenate([ctx_cat_buf[:, 1:, :], new_cat],  axis=1)

        return generated  # (n_sim, stint_len)

    # ------------------------------------------------------------------
    # v3 rollout
    # ------------------------------------------------------------------

    def _rollout_v3(
        self,
        context_seed: np.ndarray,           # (T, 15)
        stint_len: int,
        pace_bounds: Tuple[float, float],   # (pace_hi_norm, pace_lo_norm)
        track_temp: float,
        air_temp: float,
        compound_int: int,
        lap_type_int: int,
        session_type_int: int,
        base_lap_time: float,
        rng: np.random.Generator,
        circuit_int: int,
        race_lap_start: int,
        stint_number: int = 1,
    ) -> np.ndarray:
        """Single-trajectory v3 rollout (single sim). Used as fallback."""
        delta_std       = self.stats.get("delta_std", 1.0)
        track_ref       = self.stats.get("track_temp_ref", 30.0)
        air_ref         = self.stats.get("air_temp_ref",   25.0)
        track_temp_norm = float((track_temp - track_ref) / 10.0)
        air_temp_norm   = float((air_temp   - air_ref)   / 8.0)
        exp_laps        = CIRCUIT_EXPECTED_LAPS.get(
            {v: k for k, v in CIRCUIT_VOCAB.items()}.get(circuit_int, ""), 55
        )

        pace_hi_norm, pace_lo_norm = pace_bounds
        phase_a = rng.uniform(pace_hi_norm, pace_lo_norm)
        phase_b = rng.uniform(pace_hi_norm, pace_lo_norm)

        stint_num_norm = float(np.clip((stint_number - 1) / 3.0, 0.0, 1.5))

        ctx            = context_seed.copy().astype(np.float32)  # (T, 15)
        ctx_cat_buf    = np.tile(
            np.array([[compound_int, lap_type_int, session_type_int]], dtype=np.int64),
            (self.context_len, 1),
        )  # (T, 3)

        current          = float(base_lap_time)
        generated: List[float] = []
        circuit_tensor   = torch.tensor([circuit_int],    dtype=torch.long)
        comp_stat_tensor = torch.tensor([compound_int],   dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            for step in range(stint_len):
                t = step / max(1, stint_len - 1)
                pace_intent_t  = phase_a * (1.0 - t) + phase_b * t
                push_index_t   = float(np.clip(-pace_intent_t, -3.0, 3.0))
                race_lap_norm_t = float(
                    np.clip((race_lap_start + step) / max(exp_laps, 1), 0.0, 1.2)
                )

                new_row = np.zeros((1, 15), dtype=np.float32)
                new_row[0, 0]  = float(step + 1) / 40.0
                new_row[0, 2]  = pace_intent_t
                new_row[0, 4]  = track_temp_norm
                new_row[0, 5]  = air_temp_norm
                new_row[0, 11] = push_index_t
                new_row[0, 12] = race_lap_norm_t
                new_row[0, 14] = stint_num_norm

                new_cat = np.array([[compound_int, lap_type_int, session_type_int]], dtype=np.int64)

                ctx_full     = np.concatenate([ctx,         new_row],  axis=0)
                cat_full     = np.concatenate([ctx_cat_buf, new_cat],  axis=0)

                in_cont = torch.tensor(
                    ctx_full[-self.context_len:][np.newaxis], dtype=torch.float32
                )
                in_cat  = torch.tensor(
                    cat_full[-self.context_len:][np.newaxis], dtype=torch.long
                )

                delta_pred, _, _, _ = self.model(in_cont, in_cat, circuit_tensor, comp_stat_tensor)
                delta_norm   = float(delta_pred.item())
                new_lap      = float(np.clip(current + delta_norm * delta_std, 60.0, 300.0))
                generated.append(new_lap)
                current      = new_lap

                new_row[0, 1] = delta_norm
                ctx            = np.concatenate([ctx[1:],         new_row],  axis=0)
                ctx_cat_buf    = np.concatenate([ctx_cat_buf[1:], new_cat],  axis=0)

        return np.array(generated, dtype=np.float32)

    def _rollout_v3_batched(
        self,
        ctx_seeds: np.ndarray,           # (n_sim, T, 15)
        phase_as: np.ndarray,            # (n_sim,) pre-sampled pace lerp start
        phase_bs: np.ndarray,            # (n_sim,) pre-sampled pace lerp end
        stint_len: int,
        track_temp_norms: np.ndarray,    # (n_sim,) pre-normalised
        air_temp_norms: np.ndarray,      # (n_sim,) pre-normalised
        compound_int: int,
        lap_type_int: int,
        session_type_int: int,
        base_lap_time: float,
        circuit_int: int,
        race_lap_start: int,
        stint_number: int = 1,
    ) -> np.ndarray:                     # (n_sim, stint_len)
        """
        Vectorised v3 rollout: same structure as _rollout_v2_batched but:
          - Tensors are (n_sim, T, 15) instead of (n_sim, T, 14)
          - Feature 14 is filled with stint_number_norm (constant per rollout)
          - compound_idx_static tensor passed to forward()
          - 4 outputs from forward(); only delta is used
        """
        n_sim     = ctx_seeds.shape[0]
        delta_std = self.stats.get("delta_std", 1.0)
        exp_laps  = CIRCUIT_EXPECTED_LAPS.get(
            {v: k for k, v in CIRCUIT_VOCAB.items()}.get(circuit_int, ""), 55
        )
        stint_num_norm = float(np.clip((stint_number - 1) / 3.0, 0.0, 1.5))

        ctx         = ctx_seeds.copy().astype(np.float32)          # (n_sim, T, 15)
        ctx_cat_buf = np.tile(
            np.array([[compound_int, lap_type_int, session_type_int]], dtype=np.int64),
            (n_sim, self.context_len, 1),
        )                                                           # (n_sim, T, 3)

        currents         = np.full(n_sim, base_lap_time, dtype=np.float32)
        generated        = np.zeros((n_sim, stint_len), dtype=np.float32)
        circuit_tensor   = torch.full((n_sim,), circuit_int,  dtype=torch.long)
        comp_stat_tensor = torch.full((n_sim,), compound_int, dtype=torch.long)

        new_cat = np.tile(
            np.array([[compound_int, lap_type_int, session_type_int]], dtype=np.int64),
            (n_sim, 1, 1),
        )  # (n_sim, 1, 3)

        self.model.eval()
        with torch.no_grad():
            for step in range(stint_len):
                t = step / max(1, stint_len - 1)
                pace_intent_t   = phase_as * (1.0 - t) + phase_bs * t   # (n_sim,)
                push_index_t    = np.clip(-pace_intent_t, -3.0, 3.0)    # (n_sim,)
                race_lap_norm_t = float(
                    np.clip((race_lap_start + step) / max(exp_laps, 1), 0.0, 1.2)
                )

                new_rows = np.zeros((n_sim, 1, 15), dtype=np.float32)
                new_rows[:, 0, 0]  = float(step + 1) / 40.0  # stint_age_norm
                # new_rows[:, 0, 1] = 0.0                     # delta_norm placeholder
                new_rows[:, 0, 2]  = pace_intent_t            # pace_intent_norm
                # new_rows[:, 0, 3] = 0.0                     # gap_norm
                new_rows[:, 0, 4]  = track_temp_norms         # track_temp_norm
                new_rows[:, 0, 5]  = air_temp_norms           # air_temp_norm
                # features 6-10: telemetry devs — already 0
                new_rows[:, 0, 11] = push_index_t             # push_index
                new_rows[:, 0, 12] = race_lap_norm_t          # race_lap_norm
                # feature 13: lap_speed_deviation — already 0
                new_rows[:, 0, 14] = stint_num_norm           # stint_number_norm (NEW)

                ctx_full = np.concatenate([ctx,         new_rows], axis=1)  # (n_sim, T+1, 15)
                cat_full = np.concatenate([ctx_cat_buf, new_cat],  axis=1)  # (n_sim, T+1, 3)

                in_cont = torch.tensor(
                    ctx_full[:, -self.context_len:, :], dtype=torch.float32
                )  # (n_sim, T, 15)
                in_cat  = torch.tensor(
                    cat_full[:, -self.context_len:, :], dtype=torch.long
                )  # (n_sim, T, 3)

                delta_pred, _, _, _ = self.model(in_cont, in_cat, circuit_tensor, comp_stat_tensor)
                delta_norm_preds     = delta_pred.squeeze(-1).cpu().numpy()  # (n_sim,)
                delta_laps           = delta_norm_preds * delta_std
                new_laps             = np.clip(currents + delta_laps, 60.0, 300.0)
                generated[:, step]   = new_laps
                currents             = new_laps

                new_rows[:, 0, 1] = delta_norm_preds
                ctx         = np.concatenate([ctx[:, 1:, :],         new_rows], axis=1)
                ctx_cat_buf = np.concatenate([ctx_cat_buf[:, 1:, :], new_cat],  axis=1)

        return generated  # (n_sim, stint_len)

    # ------------------------------------------------------------------
    # Deterministic prediction (kept for visualisation / LSTM-compat)
    # ------------------------------------------------------------------

    def predict_stint(self, stint_df: pd.DataFrame) -> np.ndarray:
        """
        Deterministic prediction on a real stint DataFrame (visualisation curves).
        Warm-up window returns actual lap times; remaining laps use the Transformer.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        df = stint_df.copy().sort_values("lap_number").reset_index(drop=True)
        if len(df) < self.context_len + 1:
            return df["lap_time"].values

        delta_std = self.stats.get("delta_std", 1.0)
        lap_std   = self.stats.get("lap_std",   1.0)
        track_ref = self.stats.get("track_temp_ref", 30.0)
        air_ref   = self.stats.get("air_temp_ref",   25.0)

        if "stint_number" in df.columns:
            stint_med = df.groupby("stint_number")["lap_time"].transform("median")
        else:
            stint_med = float(df["lap_time"].median())

        df["pace_intent_norm"] = (df["lap_time"] - stint_med) / (lap_std or 1.0)
        df["delta_norm"]       = df["lap_time"].diff().fillna(0.0) / (delta_std or 1.0)
        df["gap_norm"] = (
            df["gap_to_car_ahead"].clip(0, 5).fillna(0.0) / 5.0
            if "gap_to_car_ahead" in df.columns else 0.0
        )
        df["track_temp_norm"] = (df["track_temp"].fillna(track_ref) - track_ref) / 10.0
        df["air_temp_norm"]   = (df["air_temp"].fillna(air_ref)     - air_ref)   / 10.0
        df["stint_age_norm"]  = df["stint_age"] / 40.0

        comp_enc = self._encode(df["compound"], self.encoders["compound"])
        lt_enc   = self._encode_lap_type(df.get("lap_type", pd.Series("normal", index=df.index)))
        st_enc   = self._encode_session_type(df.get("session_type", pd.Series("RACE", index=df.index)))

        if isinstance(self.model, TyreDegradationTransformerV3):
            n_feat = 15
            cont = np.zeros((len(df), n_feat), dtype=np.float32)
            cont[:, 0] = df["stint_age_norm"].values
            cont[:, 1] = df["delta_norm"].values
            cont[:, 2] = df["pace_intent_norm"].values
            cont[:, 3] = df["gap_norm"].values
            cont[:, 4] = df["track_temp_norm"].values
            cont[:, 5] = df["air_temp_norm"].values
            if "stint_number" in df.columns:
                cont[:, 14] = (df["stint_number"].values / 3.0).clip(0, 1.5)
            cat = np.column_stack([comp_enc, lt_enc, st_enc]).astype(np.int64)
            circuit_id    = str(df["circuit_id"].iloc[0]) if "circuit_id" in df.columns else ""
            circuit_int   = CIRCUIT_VOCAB.get(circuit_id, 0)
            circ_t        = torch.tensor([circuit_int], dtype=torch.long)
            comp_int_val  = int(comp_enc[self.context_len]) if len(comp_enc) > self.context_len else 0
            comp_stat_t   = torch.tensor([comp_int_val], dtype=torch.long)
        elif isinstance(self.model, TyreDegradationTransformerV2):
            cont = np.zeros((len(df), 14), dtype=np.float32)
            cont[:, 0] = df["stint_age_norm"].values
            cont[:, 1] = df["delta_norm"].values
            cont[:, 2] = df["pace_intent_norm"].values
            cont[:, 3] = df["gap_norm"].values
            cont[:, 4] = df["track_temp_norm"].values
            cont[:, 5] = df["air_temp_norm"].values
            # features 6-13 remain 0
            cat = np.column_stack([comp_enc, lt_enc, st_enc]).astype(np.int64)
            circuit_id  = str(df["circuit_id"].iloc[0]) if "circuit_id" in df.columns else ""
            circuit_int = CIRCUIT_VOCAB.get(circuit_id, 0)
            circ_t      = torch.tensor([circuit_int], dtype=torch.long)
            comp_stat_t = None
        else:
            cont = np.column_stack([
                df["stint_age_norm"].values,
                df["delta_norm"].values,
                df["pace_intent_norm"].values,
                df["gap_norm"].values,
                df["track_temp_norm"].values,
                df["air_temp_norm"].values,
            ]).astype(np.float32)
            cat         = np.column_stack([comp_enc, lt_enc, st_enc]).astype(np.int64)
            circ_t      = None
            comp_stat_t = None

        preds = list(df["lap_time"].values[: self.context_len])

        self.model.eval()
        with torch.no_grad():
            for i in range(self.context_len, len(df)):
                x_cont = torch.tensor(cont[i - self.context_len: i][np.newaxis], dtype=torch.float32)
                x_cat  = torch.tensor(cat[i  - self.context_len: i][np.newaxis], dtype=torch.long)
                if isinstance(self.model, TyreDegradationTransformerV3):
                    # Update compound_static based on current position
                    comp_stat_t = torch.tensor([int(comp_enc[i])], dtype=torch.long)
                    delta_pred, _, _, _ = self.model(x_cont, x_cat, circ_t, comp_stat_t)
                    delta_norm = float(delta_pred.item())
                elif isinstance(self.model, TyreDegradationTransformerV2):
                    delta_pred, _, _ = self.model(x_cont, x_cat, circ_t)
                    delta_norm = float(delta_pred.item())
                else:
                    delta_norm = float(self.model(x_cont, x_cat).item())
                predicted = float(np.clip(preds[-1] + delta_norm * delta_std, 60.0, 300.0))
                preds.append(predicted)

        return np.array(preds, dtype=np.float32)
