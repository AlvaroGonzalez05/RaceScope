from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.train import train_per_driver
from app.config import (
    TRANSFORMER_V2_CONTEXT_LAPS,
    TRANSFORMER_V2_D_MODEL,
    TRANSFORMER_V2_N_HEADS,
    TRANSFORMER_V2_N_LAYERS,
    TRANSFORMER_V2_DIM_FF,
    TRANSFORMER_V2_DROPOUT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Transformer pace models per driver (+ global fallback)."
    )
    parser.add_argument(
        "--driver-ids",
        type=str,
        default=None,
        help="Comma-separated driver IDs to train (e.g. '55,16'). Default: all drivers.",
    )
    parser.add_argument(
        "--min-laps",
        type=int,
        default=200,
        help="Minimum clean 'normal' laps required for a per-driver model (default: 200).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum training epochs; early stopping may terminate earlier (default: 15).",
    )
    parser.add_argument(
        "--max-train-year",
        type=int,
        default=2024,
        help="Hold out seasons after this year from training (default: 2024).",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=TRANSFORMER_V2_CONTEXT_LAPS,
        help=f"Transformer context window length (default: {TRANSFORMER_V2_CONTEXT_LAPS}).",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=TRANSFORMER_V2_D_MODEL,
        help=f"Transformer model dimension (default: {TRANSFORMER_V2_D_MODEL}).",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=TRANSFORMER_V2_N_HEADS,
        help=f"Number of attention heads (default: {TRANSFORMER_V2_N_HEADS}).",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=TRANSFORMER_V2_N_LAYERS,
        help=f"Number of encoder layers (default: {TRANSFORMER_V2_N_LAYERS}).",
    )
    parser.add_argument(
        "--dim-ff",
        type=int,
        default=TRANSFORMER_V2_DIM_FF,
        help=f"Feedforward dimension (default: {TRANSFORMER_V2_DIM_FF}).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=TRANSFORMER_V2_DROPOUT,
        help=f"Dropout rate (default: {TRANSFORMER_V2_DROPOUT}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (default: 64).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v2",
        choices=["v1", "v2", "v3"],
        help="Transformer architecture version (default: v2).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of sessions held out as validation for early stopping (default: 0.15).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs; 0 to disable (default: 5).",
    )
    parser.add_argument(
        "--hparam-json",
        type=str,
        default=None,
        help="Path to JSON file with hyperparameter overrides (output of hparam_search).",
    )
    args = parser.parse_args()

    driver_ids = None
    if args.driver_ids:
        driver_ids = [int(d.strip()) for d in args.driver_ids.split(",") if d.strip()]

    hparam_overrides = None
    if args.hparam_json:
        hparam_path = Path(args.hparam_json)
        if hparam_path.exists():
            with open(hparam_path) as f:
                hparam_overrides = json.load(f)
            print(f"Loaded hyperparameter overrides from {hparam_path}")
        else:
            print(f"Warning: --hparam-json path not found: {hparam_path}")

    trained = train_per_driver(
        max_train_year=args.max_train_year,
        driver_ids=driver_ids,
        min_laps=args.min_laps,
        epochs=args.epochs,
        context_len=args.context_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        model_version=args.model_version,
        val_frac=args.val_frac,
        patience=args.patience,
        hparam_overrides=hparam_overrides,
    )
    print(f"Trained {len(trained)} per-driver model(s): {sorted(trained.keys())}")


if __name__ == "__main__":
    main()
