"""
benchmark_models.py — Head-to-head performance comparison of Transformer v1 vs v2.

All measurements use synthetic tensors — no pipeline data required.

Metrics
-------
  architecture   : parameter count, state-dict size on disk (MB)
  forward_pass   : single-sample and batched latency (mean / p50 / p95 / p99, ms)
  throughput     : forward-pass samples per second at batch sizes 1, 32, 128
  rollout_mc     : wall-clock time per stint rollout (20 laps, repeated 100×)
  training       : time per epoch on synthetic dataset (500 samples, 1 epoch)

Usage
-----
  cd code/backend_fastapi
  source .venv_demo/bin/activate
  python -m scripts.benchmark_models [--output benchmark_models.json] [--reps 200]
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _state_dict_size_mb(model: nn.Module) -> float:
    """Serialize state dict to a buffer and measure bytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def _timing_stats(timings_ms: List[float]) -> Dict[str, float]:
    return {
        "mean_ms":  round(statistics.mean(timings_ms), 4),
        "median_ms": round(statistics.median(timings_ms), 4),
        "p95_ms":   round(_percentile(timings_ms, 95), 4),
        "p99_ms":   round(_percentile(timings_ms, 99), 4),
        "min_ms":   round(min(timings_ms), 4),
        "max_ms":   round(max(timings_ms), 4),
        "stdev_ms": round(statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Synthetic tensor generators
# ---------------------------------------------------------------------------

def _v1_batch(batch: int, context: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x_cont = torch.randn(batch, context, 6)
    x_cat  = torch.randint(0, 8, (batch, context, 3))
    return x_cont, x_cat


def _v2_batch(batch: int, context: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_cont    = torch.randn(batch, context, 14)
    x_cat     = torch.randint(0, 8, (batch, context, 3))
    circ      = torch.randint(0, 24, (batch,))
    return x_cont, x_cat, circ


# ---------------------------------------------------------------------------
# Architecture info
# ---------------------------------------------------------------------------

def _arch_info(model: nn.Module, label: str) -> Dict:
    return {
        "label":             label,
        "total_params":      _param_count(model),
        "trainable_params":  _trainable_param_count(model),
        "state_dict_mb":     round(_state_dict_size_mb(model), 4),
    }


# ---------------------------------------------------------------------------
# Forward pass benchmarks
# ---------------------------------------------------------------------------

def _bench_forward_v1(
    model: nn.Module,
    context: int,
    batch_sizes: List[int],
    reps: int,
) -> Dict:
    model.eval()
    results: Dict = {}
    with torch.no_grad():
        for bs in batch_sizes:
            x_cont, x_cat = _v1_batch(bs, context)
            # warm-up
            for _ in range(10):
                model(x_cont, x_cat)
            timings: List[float] = []
            for _ in range(reps):
                t0 = time.perf_counter()
                model(x_cont, x_cat)
                timings.append((time.perf_counter() - t0) * 1000)
            stats = _timing_stats(timings)
            stats["throughput_per_sec"] = round(bs / (stats["mean_ms"] / 1000), 1)
            results[f"batch_{bs}"] = stats
    return results


def _bench_forward_v2(
    model: nn.Module,
    context: int,
    batch_sizes: List[int],
    reps: int,
) -> Dict:
    model.eval()
    results: Dict = {}
    with torch.no_grad():
        for bs in batch_sizes:
            x_cont, x_cat, circ = _v2_batch(bs, context)
            # warm-up
            for _ in range(10):
                model(x_cont, x_cat, circ)
            timings: List[float] = []
            for _ in range(reps):
                t0 = time.perf_counter()
                model(x_cont, x_cat, circ)
                timings.append((time.perf_counter() - t0) * 1000)
            stats = _timing_stats(timings)
            stats["throughput_per_sec"] = round(bs / (stats["mean_ms"] / 1000), 1)
            results[f"batch_{bs}"] = stats
    return results


# ---------------------------------------------------------------------------
# Rollout MC benchmark (uses the TransformerPaceModel wrapper)
# ---------------------------------------------------------------------------

def _bench_rollout(
    version: str,
    context_len: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_ff: int,
    dropout: float,
    stint_len: int,
    reps: int,
) -> Dict:
    from app.models_transformer import (
        TransformerPaceModel,
        PracticeDistribution,
        build_context_seed,
        _build_context_seed_v1,
        COMPOUND_VOCAB,
        SESSION_TYPE_VOCAB,
    )

    wrapper = TransformerPaceModel(
        context_len=context_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        model_version=version,
    )

    # Initialise model and minimal stats so rollout works
    if version == "v2":
        from app.models_transformer import TyreDegradationTransformerV2
        wrapper.model = TyreDegradationTransformerV2(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dim_ff=dim_ff, dropout=dropout, context_len=context_len,
        )
    else:
        from app.models_transformer import TyreTransformerNet
        wrapper.model = TyreTransformerNet(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dim_ff=dim_ff, dropout=dropout, context_len=context_len,
        )
    wrapper.model.eval()
    wrapper.stats = {
        "lap_mean": 90.0, "lap_std": 1.5,
        "delta_std": 0.4, "track_temp_ref": 30.0, "air_temp_ref": 25.0,
    }

    pd_dist = PracticeDistribution(compound="SOFT")
    compound_int     = COMPOUND_VOCAB.get("SOFT", 1)
    session_type_int = SESSION_TYPE_VOCAB.get("RACE", 4)
    rng              = np.random.default_rng(42)

    if version == "v2":
        ctx = build_context_seed(
            pd_dist, context_len, 0.4, 30.0, 25.0,
            compound_int, session_type_int, circuit_int=1,
            rng=np.random.default_rng(42),
        )
        pace_bounds = (-0.5, 0.5)

        # warm-up
        for _ in range(5):
            wrapper.rollout_mc(
                ctx, stint_len, pace_bounds,
                track_temp=30.0, air_temp=22.0,
                compound_int=compound_int, circuit_int=1,
                rng=np.random.default_rng(0),
            )
        timings: List[float] = []
        for i in range(reps):
            t0 = time.perf_counter()
            wrapper.rollout_mc(
                ctx, stint_len, pace_bounds,
                track_temp=30.0, air_temp=22.0,
                compound_int=compound_int, circuit_int=1,
                rng=np.random.default_rng(i),
            )
            timings.append((time.perf_counter() - t0) * 1000)
    else:
        ctx_v1 = _build_context_seed_v1(
            pd_dist, context_len, 0.4, 30.0, 25.0,
            compound_int, session_type_int,
            rng=np.random.default_rng(42),
        )
        # warm-up
        for _ in range(5):
            wrapper.rollout_mc(
                ctx_v1, stint_len, 0.0,
                track_temp=30.0, air_temp=22.0,
                compound_int=compound_int,
                rng=np.random.default_rng(0),
            )
        timings = []
        for i in range(reps):
            t0 = time.perf_counter()
            wrapper.rollout_mc(
                ctx_v1, stint_len, 0.0,
                track_temp=30.0, air_temp=22.0,
                compound_int=compound_int,
                rng=np.random.default_rng(i),
            )
            timings.append((time.perf_counter() - t0) * 1000)

    stats = _timing_stats(timings)
    stats["stint_len_laps"] = stint_len
    stats["forward_passes_per_rollout"] = stint_len
    stats["rollouts_per_sec"] = round(1000.0 / stats["mean_ms"], 2)
    stats["laps_per_sec"]     = round(stint_len * 1000.0 / stats["mean_ms"], 1)
    return stats


# ---------------------------------------------------------------------------
# Training throughput benchmark
# ---------------------------------------------------------------------------

def _bench_training(
    version: str,
    context_len: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dim_ff: int,
    dropout: float,
    n_samples: int,
    batch_size: int,
) -> Dict:
    from app.models_transformer import (
        TyreTransformerNet,
        TyreDegradationTransformerV2,
        SequenceDataset,
        SequenceDatasetV2,
    )

    torch.manual_seed(42)

    if version == "v2":
        X_cont    = np.random.randn(n_samples, context_len, 14).astype(np.float32)
        X_cat     = np.random.randint(0, 8, (n_samples, context_len, 3)).astype(np.int64)
        circuit   = np.random.randint(0, 24, (n_samples,)).astype(np.int64)
        y_delta   = np.random.randn(n_samples).astype(np.float32)
        y_abs     = np.random.randn(n_samples).astype(np.float32)
        y_deg     = np.random.randn(n_samples).astype(np.float32)
        dataset   = SequenceDatasetV2(X_cont, X_cat, circuit, y_delta, y_abs, y_deg)
        model = TyreDegradationTransformerV2(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dim_ff=dim_ff, dropout=dropout, context_len=context_len,
        )
        loss_fn = nn.HuberLoss(delta=1.0)
        mse_fn  = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
        t0 = time.perf_counter()
        for bx_cont, bx_cat, b_circ, by_d, by_a, by_g in loader:
            optimizer.zero_grad()
            dp, ap, gp = model(bx_cont, bx_cat, b_circ)
            loss = loss_fn(dp, by_d) + 0.15 * mse_fn(ap, by_a) + 0.10 * mse_fn(gp, by_g)
            loss.backward()
            optimizer.step()
        elapsed_ms = (time.perf_counter() - t0) * 1000
    else:
        X_cont  = np.random.randn(n_samples, context_len, 6).astype(np.float32)
        X_cat   = np.random.randint(0, 8, (n_samples, context_len, 3)).astype(np.int64)
        y       = np.random.randn(n_samples).astype(np.float32)
        dataset = SequenceDataset(X_cont, X_cat, y)
        model   = TyreTransformerNet(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dim_ff=dim_ff, dropout=dropout, context_len=context_len,
        )
        loss_fn   = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
        t0 = time.perf_counter()
        for bx_cont, bx_cat, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(bx_cont, bx_cat), by)
            loss.backward()
            optimizer.step()
        elapsed_ms = (time.perf_counter() - t0) * 1000

    n_batches  = max(1, n_samples // batch_size)
    return {
        "n_samples":           n_samples,
        "batch_size":          batch_size,
        "epoch_ms":            round(elapsed_ms, 2),
        "ms_per_batch":        round(elapsed_ms / n_batches, 3),
        "samples_per_sec":     round(n_samples / (elapsed_ms / 1000), 1),
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_comparison(report: Dict) -> None:
    v1 = report["v1"]
    v2 = report["v2"]

    def _row(label, v1_val, v2_val, unit="", lower_is_better=True):
        if isinstance(v1_val, float) and isinstance(v2_val, float) and v1_val > 0 and v2_val > 0:
            ratio = v2_val / v1_val
            if lower_is_better:
                flag = "✓" if ratio <= 1.05 else ("~" if ratio <= 2.0 else "↑")
            else:
                flag = "✓" if ratio >= 0.95 else ("~" if ratio >= 0.5 else "↓")
        else:
            flag = ""
        v1_s = f"{v1_val:>12,.1f}" if isinstance(v1_val, float) else f"{v1_val:>12,}"
        v2_s = f"{v2_val:>12,.1f}" if isinstance(v2_val, float) else f"{v2_val:>12,}"
        print(f"  {label:<42} {v1_s} {unit:<4}   {v2_s} {unit:<4}  {flag}")

    print("\n" + "=" * 80)
    print(f"{'  BENCHMARK REPORT — Transformer v1 vs v2':^80}")
    print("=" * 80)

    # Architecture
    print("\n--- Architecture ---")
    print(f"  {'Metric':<42} {'v1':>12}      {'v2':>12}")
    print(f"  {'-'*70}")
    _row("Total parameters",
         float(v1["architecture"]["total_params"]),
         float(v2["architecture"]["total_params"]),
         "", lower_is_better=False)
    _row("Trainable parameters",
         float(v1["architecture"]["trainable_params"]),
         float(v2["architecture"]["trainable_params"]),
         "", lower_is_better=False)
    _row("State dict size (MB)",
         v1["architecture"]["state_dict_mb"],
         v2["architecture"]["state_dict_mb"],
         "MB", lower_is_better=True)

    # Forward pass - batch 1
    print("\n--- Forward Pass (batch=1) ---")
    print(f"  {'Metric':<42} {'v1':>12}      {'v2':>12}")
    print(f"  {'-'*70}")
    fp1_v1 = v1["forward_pass"]["batch_1"]
    fp1_v2 = v2["forward_pass"]["batch_1"]
    for key in ("mean_ms", "median_ms", "p95_ms", "p99_ms"):
        _row(key, fp1_v1[key], fp1_v2[key], "ms")
    _row("throughput (samples/s)",
         float(fp1_v1["throughput_per_sec"]),
         float(fp1_v2["throughput_per_sec"]),
         "/s", lower_is_better=False)

    # Forward pass - batch 32
    print("\n--- Forward Pass (batch=32) ---")
    print(f"  {'Metric':<42} {'v1':>12}      {'v2':>12}")
    print(f"  {'-'*70}")
    fp32_v1 = v1["forward_pass"].get("batch_32", {})
    fp32_v2 = v2["forward_pass"].get("batch_32", {})
    if fp32_v1 and fp32_v2:
        for key in ("mean_ms", "p95_ms"):
            _row(key, fp32_v1[key], fp32_v2[key], "ms")
        _row("throughput (samples/s)",
             float(fp32_v1["throughput_per_sec"]),
             float(fp32_v2["throughput_per_sec"]),
             "/s", lower_is_better=False)

    # Rollout MC
    print("\n--- Rollout MC (single stint) ---")
    print(f"  {'Metric':<42} {'v1':>12}      {'v2':>12}")
    print(f"  {'-'*70}")
    rm_v1 = v1["rollout_mc"]
    rm_v2 = v2["rollout_mc"]
    _row(f"stint length (laps)",
         float(rm_v1["stint_len_laps"]),
         float(rm_v2["stint_len_laps"]), "")
    for key in ("mean_ms", "p95_ms"):
        _row(key, rm_v1[key], rm_v2[key], "ms")
    _row("rollouts/sec",
         rm_v1["rollouts_per_sec"],
         rm_v2["rollouts_per_sec"],
         "/s", lower_is_better=False)
    _row("laps/sec",
         rm_v1["laps_per_sec"],
         rm_v2["laps_per_sec"],
         "/s", lower_is_better=False)

    # Training
    print("\n--- Training Throughput (1 epoch, 500 samples) ---")
    print(f"  {'Metric':<42} {'v1':>12}      {'v2':>12}")
    print(f"  {'-'*70}")
    tr_v1 = v1["training"]
    tr_v2 = v2["training"]
    _row("epoch time (ms)", tr_v1["epoch_ms"], tr_v2["epoch_ms"], "ms")
    _row("ms per batch",    tr_v1["ms_per_batch"], tr_v2["ms_per_batch"], "ms")
    _row("samples/sec",
         tr_v1["samples_per_sec"],
         tr_v2["samples_per_sec"],
         "/s", lower_is_better=False)

    print("\n" + "=" * 80)
    print("  Legend: ✓ within 5%  ~ up to 2×  ↑/↓ >2× difference")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    output_path: str = "benchmark_models.json",
    reps: int = 200,
    rollout_reps: int = 100,
    batch_sizes: List[int] | None = None,
    training_samples: int = 500,
    training_batch_size: int = 64,
    stint_len: int = 20,
) -> Dict:
    if batch_sizes is None:
        batch_sizes = [1, 32, 128]

    from app.models_transformer import (
        TyreTransformerNet,
        TyreDegradationTransformerV2,
    )
    from app.config import (
        TRANSFORMER_CONTEXT_LAPS, TRANSFORMER_D_MODEL, TRANSFORMER_N_HEADS,
        TRANSFORMER_N_LAYERS, TRANSFORMER_DIM_FF, TRANSFORMER_DROPOUT,
        TRANSFORMER_V2_CONTEXT_LAPS, TRANSFORMER_V2_D_MODEL, TRANSFORMER_V2_N_HEADS,
        TRANSFORMER_V2_N_LAYERS, TRANSFORMER_V2_DIM_FF, TRANSFORMER_V2_DROPOUT,
    )

    print("Building models...")
    m_v1 = TyreTransformerNet(
        d_model=TRANSFORMER_D_MODEL, n_heads=TRANSFORMER_N_HEADS,
        n_layers=TRANSFORMER_N_LAYERS, dim_ff=TRANSFORMER_DIM_FF,
        dropout=TRANSFORMER_DROPOUT, context_len=TRANSFORMER_CONTEXT_LAPS,
    )
    m_v2 = TyreDegradationTransformerV2(
        d_model=TRANSFORMER_V2_D_MODEL, n_heads=TRANSFORMER_V2_N_HEADS,
        n_layers=TRANSFORMER_V2_N_LAYERS, dim_ff=TRANSFORMER_V2_DIM_FF,
        dropout=TRANSFORMER_V2_DROPOUT, context_len=TRANSFORMER_V2_CONTEXT_LAPS,
    )

    report: Dict = {
        "config": {
            "reps": reps,
            "rollout_reps": rollout_reps,
            "batch_sizes": batch_sizes,
            "stint_len": stint_len,
            "training_samples": training_samples,
            "training_batch_size": training_batch_size,
        },
        "v1": {},
        "v2": {},
    }

    # Architecture
    print("Collecting architecture stats...")
    report["v1"]["architecture"] = _arch_info(m_v1, "TyreTransformerNet (v1)")
    report["v2"]["architecture"] = _arch_info(m_v2, "TyreDegradationTransformerV2 (v2)")

    # Forward pass
    print(f"Benchmarking v1 forward pass ({reps} reps per batch size)...")
    report["v1"]["forward_pass"] = _bench_forward_v1(
        m_v1, TRANSFORMER_CONTEXT_LAPS, batch_sizes, reps,
    )
    print(f"Benchmarking v2 forward pass ({reps} reps per batch size)...")
    report["v2"]["forward_pass"] = _bench_forward_v2(
        m_v2, TRANSFORMER_V2_CONTEXT_LAPS, batch_sizes, reps,
    )

    # Rollout MC
    print(f"Benchmarking v1 rollout_mc ({rollout_reps} reps, stint={stint_len} laps)...")
    report["v1"]["rollout_mc"] = _bench_rollout(
        version="v1",
        context_len=TRANSFORMER_CONTEXT_LAPS,
        d_model=TRANSFORMER_D_MODEL,
        n_heads=TRANSFORMER_N_HEADS,
        n_layers=TRANSFORMER_N_LAYERS,
        dim_ff=TRANSFORMER_DIM_FF,
        dropout=TRANSFORMER_DROPOUT,
        stint_len=stint_len,
        reps=rollout_reps,
    )
    print(f"Benchmarking v2 rollout_mc ({rollout_reps} reps, stint={stint_len} laps)...")
    report["v2"]["rollout_mc"] = _bench_rollout(
        version="v2",
        context_len=TRANSFORMER_V2_CONTEXT_LAPS,
        d_model=TRANSFORMER_V2_D_MODEL,
        n_heads=TRANSFORMER_V2_N_HEADS,
        n_layers=TRANSFORMER_V2_N_LAYERS,
        dim_ff=TRANSFORMER_V2_DIM_FF,
        dropout=TRANSFORMER_V2_DROPOUT,
        stint_len=stint_len,
        reps=rollout_reps,
    )

    # Training throughput
    print(f"Benchmarking v1 training throughput ({training_samples} samples, 1 epoch)...")
    report["v1"]["training"] = _bench_training(
        version="v1",
        context_len=TRANSFORMER_CONTEXT_LAPS,
        d_model=TRANSFORMER_D_MODEL,
        n_heads=TRANSFORMER_N_HEADS,
        n_layers=TRANSFORMER_N_LAYERS,
        dim_ff=TRANSFORMER_DIM_FF,
        dropout=TRANSFORMER_DROPOUT,
        n_samples=training_samples,
        batch_size=training_batch_size,
    )
    print(f"Benchmarking v2 training throughput ({training_samples} samples, 1 epoch)...")
    report["v2"]["training"] = _bench_training(
        version="v2",
        context_len=TRANSFORMER_V2_CONTEXT_LAPS,
        d_model=TRANSFORMER_V2_D_MODEL,
        n_heads=TRANSFORMER_V2_N_HEADS,
        n_layers=TRANSFORMER_V2_N_LAYERS,
        dim_ff=TRANSFORMER_V2_DIM_FF,
        dropout=TRANSFORMER_V2_DROPOUT,
        n_samples=training_samples,
        batch_size=training_batch_size,
    )

    # Write JSON
    out = Path(output_path)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out}")

    # Pretty print
    _print_comparison(report)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer v1 vs v2: forward pass, rollout_mc, training."
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_models.json",
        help="Output JSON path (default: benchmark_models.json).",
    )
    parser.add_argument(
        "--reps", type=int, default=200,
        help="Forward-pass repetitions per batch size (default: 200).",
    )
    parser.add_argument(
        "--rollout-reps", type=int, default=100,
        help="Rollout_mc repetitions (default: 100).",
    )
    parser.add_argument(
        "--stint-len", type=int, default=20,
        help="Stint length in laps for rollout benchmark (default: 20).",
    )
    parser.add_argument(
        "--training-samples", type=int, default=500,
        help="Synthetic training samples for throughput benchmark (default: 500).",
    )
    args = parser.parse_args()

    run_benchmark(
        output_path=args.output,
        reps=args.reps,
        rollout_reps=args.rollout_reps,
        stint_len=args.stint_len,
        training_samples=args.training_samples,
    )


if __name__ == "__main__":
    main()
