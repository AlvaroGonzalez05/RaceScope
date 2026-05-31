"""
scripts/benchmark_architectures.py — Comparativa de rendimiento v1 / v2 / v3.

Métricas:
  - Parámetros entrenables
  - Entrenamiento: tiempo/época y tiempo total hasta early-stop en datos sintéticos
  - Inferencia individual: tiempo de un rollout de 60 vueltas (1 trayectoria)
  - Inferencia batched: tiempo de n_sim=100 simulaciones MC de 60 vueltas
  - Pico de memoria RSS durante inferencia batched

Uso:
  cd code/backend_fastapi
  .venv_demo/bin/python -m scripts.benchmark_architectures [--n-train 512] [--n-sim 100] [--laps 60]
"""
from __future__ import annotations

import argparse
import time
import tracemalloc
import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

logging.basicConfig(level=logging.WARNING)  # suppress training logs during bench

# ---------------------------------------------------------------------------
# Config imports
# ---------------------------------------------------------------------------
from app.config import (
    TRANSFORMER_CONTEXT_LAPS, TRANSFORMER_D_MODEL, TRANSFORMER_N_HEADS,
    TRANSFORMER_N_LAYERS, TRANSFORMER_DIM_FF, TRANSFORMER_DROPOUT, TRANSFORMER_INPUT_DIM,
    TRANSFORMER_V2_CONTEXT_LAPS, TRANSFORMER_V2_D_MODEL, TRANSFORMER_V2_N_HEADS,
    TRANSFORMER_V2_N_LAYERS, TRANSFORMER_V2_DIM_FF, TRANSFORMER_V2_DROPOUT,
    TRANSFORMER_V3_CONTEXT_LAPS, TRANSFORMER_V3_D_MODEL, TRANSFORMER_V3_N_HEADS,
    TRANSFORMER_V3_N_LAYERS, TRANSFORMER_V3_DIM_FF, TRANSFORMER_V3_DROPOUT,
    RANDOM_SEED,
)
from app.models_transformer import (
    TyreTransformerNet,
    TyreDegradationTransformerV2,
    TyreDegradationTransformerV3,
    TransformerPaceModel,
    SequenceDatasetV3,
    PracticeDistribution,
    build_context_seed,
    build_context_seed_v3,
    COMPOUND_VOCAB,
    SESSION_TYPE_VOCAB,
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Architecture descriptors
# ---------------------------------------------------------------------------

ARCHS = {
    "v1": dict(
        context_len  = TRANSFORMER_CONTEXT_LAPS,
        d_model      = TRANSFORMER_D_MODEL,
        n_heads      = TRANSFORMER_N_HEADS,
        n_layers     = TRANSFORMER_N_LAYERS,
        dim_ff       = TRANSFORMER_DIM_FF,
        dropout      = TRANSFORMER_DROPOUT,
        input_dim    = TRANSFORMER_INPUT_DIM,     # 13
        n_cont       = 6,                          # v1 splits: 6 cont + 7 emb-proj
        n_cat        = 3,
        has_circuit  = False,
        has_compound_static = False,
    ),
    "v2": dict(
        context_len  = TRANSFORMER_V2_CONTEXT_LAPS,
        d_model      = TRANSFORMER_V2_D_MODEL,
        n_heads      = TRANSFORMER_V2_N_HEADS,
        n_layers     = TRANSFORMER_V2_N_LAYERS,
        dim_ff       = TRANSFORMER_V2_DIM_FF,
        dropout      = TRANSFORMER_V2_DROPOUT,
        input_dim    = 14,
        n_cont       = 14,
        n_cat        = 3,
        has_circuit  = True,
        has_compound_static = False,
    ),
    "v3": dict(
        context_len  = TRANSFORMER_V3_CONTEXT_LAPS,
        d_model      = TRANSFORMER_V3_D_MODEL,
        n_heads      = TRANSFORMER_V3_N_HEADS,
        n_layers     = TRANSFORMER_V3_N_LAYERS,
        dim_ff       = TRANSFORMER_V3_DIM_FF,
        dropout      = TRANSFORMER_V3_DROPOUT,
        input_dim    = 15,
        n_cont       = 15,
        n_cat        = 3,
        has_circuit  = True,
        has_compound_static = True,
    ),
}


def _make_model(version: str) -> torch.nn.Module:
    a = ARCHS[version]
    if version == "v1":
        return TyreTransformerNet(
            d_model=a["d_model"], n_heads=a["n_heads"], n_layers=a["n_layers"],
            dim_ff=a["dim_ff"], dropout=a["dropout"], context_len=a["context_len"],
        )
    if version == "v2":
        return TyreDegradationTransformerV2(
            d_model=a["d_model"], n_heads=a["n_heads"], n_layers=a["n_layers"],
            dim_ff=a["dim_ff"], dropout=a["dropout"], context_len=a["context_len"],
        )
    return TyreDegradationTransformerV3(
        d_model=a["d_model"], n_heads=a["n_heads"], n_layers=a["n_layers"],
        dim_ff=a["dim_ff"], dropout=a["dropout"], context_len=a["context_len"],
    )


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

def _make_synthetic_batch(version: str, n: int, batch_size: int):
    """Yield (inputs, targets) tensors for one synthetic epoch."""
    a = ARCHS[version]
    T = a["context_len"]
    rng = np.random.default_rng(RANDOM_SEED)

    x_cont_all = rng.normal(0, 1, (n, T, a["n_cont"])).astype(np.float32)
    x_cat_all  = rng.integers(0, 5, (n, T, 3)).astype(np.int64)
    circ_all   = rng.integers(0, 25, n).astype(np.int64)
    comp_all   = rng.integers(0, 8,  n).astype(np.int64)
    y_all      = rng.normal(0, 1, n).astype(np.float32)

    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        x_c  = torch.from_numpy(x_cont_all[start:end])
        x_k  = torch.from_numpy(x_cat_all[start:end])
        circ = torch.from_numpy(circ_all[start:end])
        comp = torch.from_numpy(comp_all[start:end])
        y    = torch.from_numpy(y_all[start:end])
        yield x_c, x_k, circ, comp, y


def _train_epoch(version: str, model: torch.nn.Module, n: int, batch_size: int,
                 optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for x_c, x_k, circ, comp, y in _make_synthetic_batch(version, n, batch_size):
        optimizer.zero_grad()
        if version == "v1":
            pred = model(x_c, x_k).squeeze(-1)
        elif version == "v2":
            delta, _, _ = model(x_c, x_k, circ)
            pred = delta.squeeze(-1)
        else:
            delta, _, _, _ = model(x_c, x_k, circ, comp)
            pred = delta.squeeze(-1)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _bench_inference_single(version: str, model: torch.nn.Module,
                             n_laps: int, n_reps: int = 30) -> float:
    """Mean time (s) for a single autoregressive rollout of n_laps."""
    a = ARCHS[version]
    T = a["context_len"]
    model.eval()

    x_c   = torch.randn(1, T, a["n_cont"])
    x_k   = torch.zeros(1, T, 3, dtype=torch.long)
    circ  = torch.zeros(1, dtype=torch.long)
    comp  = torch.zeros(1, dtype=torch.long)

    times = []
    with torch.no_grad():
        for _ in range(n_reps):
            ctx_c = x_c.clone()
            ctx_k = x_k.clone()
            t0 = time.perf_counter()
            for _ in range(n_laps):
                if version == "v1":
                    out = model(ctx_c, ctx_k)          # (1,1)
                elif version == "v2":
                    out, _, _ = model(ctx_c, ctx_k, circ)
                else:
                    out, _, _, _ = model(ctx_c, ctx_k, circ, comp)
                # slide window
                new_step = torch.cat([out.detach().unsqueeze(1),
                                      torch.zeros(1, 1, a["n_cont"] - 1)], dim=-1)
                ctx_c = torch.cat([ctx_c[:, 1:, :], new_step], dim=1)
                ctx_k = torch.cat([ctx_k[:, 1:, :], ctx_k[:, -1:, :]], dim=1)
            times.append(time.perf_counter() - t0)
    return float(np.mean(times[3:]))   # skip first 3 warm-up


def _bench_inference_batched(version: str, model: torch.nn.Module,
                              n_laps: int, n_sim: int, n_reps: int = 10) -> tuple[float, int]:
    """Mean time (s) and peak memory (MB) for a batched MC rollout."""
    a = ARCHS[version]
    T = a["context_len"]
    model.eval()

    x_c   = torch.randn(n_sim, T, a["n_cont"])
    x_k   = torch.zeros(n_sim, T, 3, dtype=torch.long)
    circ  = torch.zeros(n_sim, dtype=torch.long)
    comp  = torch.zeros(n_sim, dtype=torch.long)

    times = []
    peak_mb = 0
    with torch.no_grad():
        for rep in range(n_reps):
            ctx_c = x_c.clone()
            ctx_k = x_k.clone()
            if rep == n_reps - 1:
                tracemalloc.start()
            t0 = time.perf_counter()
            for _ in range(n_laps):
                if version == "v1":
                    out = model(ctx_c, ctx_k)
                elif version == "v2":
                    out, _, _ = model(ctx_c, ctx_k, circ)
                else:
                    out, _, _, _ = model(ctx_c, ctx_k, circ, comp)
                new_step = torch.cat([out.detach().unsqueeze(1),
                                      torch.zeros(n_sim, 1, a["n_cont"] - 1)], dim=-1)
                ctx_c = torch.cat([ctx_c[:, 1:, :], new_step], dim=1)
                ctx_k = torch.cat([ctx_k[:, 1:, :], ctx_k[:, -1:, :]], dim=1)
            times.append(time.perf_counter() - t0)
            if rep == n_reps - 1:
                _, peak = tracemalloc.get_traced_memory()
                peak_mb = peak // (1024 * 1024)
                tracemalloc.stop()
    return float(np.mean(times[2:])), peak_mb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark v1/v2/v3 architectures")
    parser.add_argument("--n-train",   type=int, default=512,  help="Synthetic training sequences")
    parser.add_argument("--batch-size",type=int, default=64,   help="Training batch size")
    parser.add_argument("--epochs",    type=int, default=5,    help="Training epochs to time")
    parser.add_argument("--n-sim",     type=int, default=100,  help="MC simulations for batched bench")
    parser.add_argument("--laps",      type=int, default=60,   help="Race laps for rollout bench")
    parser.add_argument("--inf-reps",  type=int, default=15,   help="Repetitions per inference bench")
    args = parser.parse_args()

    import sys
    sep  = "─" * 80
    col  = "{:<8} {:>12} {:>14} {:>14} {:>14} {:>12}"
    hdr  = col.format("Version", "Params", "Train/epoch(s)", f"Infer 1×{args.laps}(ms)",
                      f"Infer {args.n_sim}×{args.laps}(ms)", "Peak RAM(MB)")

    results: Dict[str, dict] = {}

    for version in ("v1", "v2", "v3"):
        print(f"\n{'='*80}")
        print(f"  Benchmarking {version.upper()}  —  "
              f"d_model={ARCHS[version]['d_model']}  "
              f"layers={ARCHS[version]['n_layers']}  "
              f"context={ARCHS[version]['context_len']}")
        print(f"{'='*80}")

        model = _make_model(version)
        model.eval()
        n_params = count_params(model)
        print(f"  Parámetros: {n_params:,}")

        # --- Training bench ---
        print(f"  Entrenamiento ({args.epochs} épocas, n={args.n_train}, bs={args.batch_size})...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        criterion = torch.nn.HuberLoss()
        epoch_times = []
        for ep in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            loss = _train_epoch(version, model, args.n_train, args.batch_size, optimizer, criterion)
            dt = time.perf_counter() - t0
            epoch_times.append(dt)
            print(f"    época {ep}/{args.epochs}  loss={loss:.4f}  {dt:.2f}s")
        mean_epoch = float(np.mean(epoch_times))
        total_train = float(np.sum(epoch_times))
        print(f"  → media/época: {mean_epoch:.2f}s  |  total {args.epochs} épocas: {total_train:.2f}s")

        # --- Single inference ---
        print(f"  Inferencia individual ({args.laps} vueltas, {args.inf_reps} repeticiones)...")
        model.eval()
        t_single = _bench_inference_single(version, model, args.laps, args.inf_reps)
        print(f"  → {t_single*1000:.1f} ms")

        # --- Batched inference ---
        print(f"  Inferencia batched ({args.n_sim} sims × {args.laps} vueltas, {args.inf_reps} reps)...")
        t_batch, peak_mb = _bench_inference_batched(version, model, args.laps, args.n_sim, args.inf_reps)
        print(f"  → {t_batch*1000:.1f} ms  |  pico RAM: {peak_mb} MB")

        results[version] = {
            "params":       n_params,
            "epoch_s":      mean_epoch,
            "single_ms":    t_single * 1000,
            "batched_ms":   t_batch  * 1000,
            "peak_mb":      peak_mb,
        }

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n\n{sep}")
    print("  RESUMEN COMPARATIVO")
    print(sep)
    print(hdr)
    print(sep)
    for version in ("v1", "v2", "v3"):
        r = results[version]
        print(col.format(
            version.upper(),
            f"{r['params']:,}",
            f"{r['epoch_s']:.2f}",
            f"{r['single_ms']:.1f}",
            f"{r['batched_ms']:.1f}",
            f"{r['peak_mb']}",
        ))
    print(sep)

    # Ratios vs v1
    print("\n  Ratios relativos a V1:")
    ref = results["v1"]
    ratio_col = "{:<8} {:>12} {:>14} {:>14} {:>14}"
    print(ratio_col.format("Version", "Params ×", "Train/época ×", "Infer 1× ×", f"Infer {args.n_sim}× ×"))
    print("─" * 54)
    for version in ("v1", "v2", "v3"):
        r = results[version]
        print(ratio_col.format(
            version.upper(),
            f"{r['params']/ref['params']:.1f}×",
            f"{r['epoch_s']/ref['epoch_s']:.1f}×",
            f"{r['single_ms']/ref['single_ms']:.1f}×",
            f"{r['batched_ms']/ref['batched_ms']:.1f}×",
        ))

    # ---------------------------------------------------------------------------
    # Save to CSV
    # ---------------------------------------------------------------------------
    import csv, json
    out_csv = "reports/benchmark_architectures.csv"
    import os; os.makedirs("reports", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["version","params","epoch_s","single_ms","batched_ms","peak_mb"])
        w.writeheader()
        for version, r in results.items():
            w.writerow({"version": version, **r})
    print(f"\n  Resultados guardados en {out_csv}")


if __name__ == "__main__":
    main()
