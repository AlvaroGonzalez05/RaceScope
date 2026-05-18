"""
hparam_search.py — Hyperparameter search for TyreDegradationTransformerV2.

Uses Optuna to explore the architecture and training search space.
Trains a global model (all drivers) for a fixed number of epochs per trial
and evaluates val_mae on a chronological hold-out of 2024 sessions.
2025 data is never loaded.

Usage
-----
python -m scripts.hparam_search --n-trials 50 --timeout-hours 4 --output hparam_results/

Outputs
-------
  hparam_results/best_hparams.json   — best hyperparameters (load with --hparam-json)
  hparam_results/optuna_study.pkl    — full Optuna study (for analysis)
  hparam_results/trials.csv          — all trials summary
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _objective(trial, df_train, df_val, train_fn):
    """Optuna objective: train global model for N epochs, return val_mae."""
    import optuna  # noqa: PLC0415

    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    # n_heads must divide d_model
    valid_heads = [h for h in [4, 8, 16] if d_model % h == 0]
    n_heads  = trial.suggest_categorical("n_heads",  valid_heads)
    n_layers = trial.suggest_categorical("n_layers", [3, 4, 6, 8])
    dim_ff   = trial.suggest_categorical("dim_ff",   [512, 1024, 2048])
    context_len = trial.suggest_categorical("context_len", [15, 20, 25, 30])
    dropout  = trial.suggest_categorical("dropout",  [0.05, 0.10, 0.15, 0.20])
    lr       = trial.suggest_categorical("lr",       [5e-5, 1e-4, 5e-4])
    batch_size  = trial.suggest_categorical("batch_size",  [32, 64, 128])
    aux_loss_w  = trial.suggest_categorical("aux_loss_w",  [0.05, 0.10, 0.20])

    from app.models_transformer import TransformerPaceModel  # noqa: PLC0415

    model = TransformerPaceModel(
        context_len=context_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        model_version="v2",
    )
    try:
        bundle = model.train(
            df_train,
            epochs=8,
            batch_size=batch_size,
            learning_rate=lr,
            val_df=df_val,
            patience=0,           # no early stopping during HPO — measure at epoch 8
            aux_loss_w=aux_loss_w,
        )
    except Exception as exc:
        logger.warning("Trial %d failed: %s", trial.number, exc)
        raise optuna.exceptions.TrialPruned()

    # Compute val_mae from the model stats (logged during training)
    # We re-evaluate on val_df directly using the trained model
    try:
        from app.models_transformer import TyreDegradationTransformerV2  # noqa: PLC0415
        import torch  # noqa: PLC0415

        model.load(bundle, 14)
        vX_cont, vX_cat, v_circuit, vy_delta, _, _ = model._build_sequences_v2(df_val)
        if len(vy_delta) == 0:
            raise optuna.exceptions.TrialPruned()

        vX_c = torch.tensor(vX_cont,   dtype=torch.float32)
        vX_k = torch.tensor(vX_cat,    dtype=torch.long)
        v_ci = torch.tensor(v_circuit, dtype=torch.long)
        vy   = torch.tensor(vy_delta,  dtype=torch.float32).reshape(-1, 1)

        model.model.eval()
        with torch.no_grad():
            preds, _, _ = model.model(vX_c, vX_k, v_ci)
            val_mae = float(torch.mean(torch.abs(preds - vy)).item())
    except Exception as exc:
        logger.warning("Val eval failed for trial %d: %s", trial.number, exc)
        raise optuna.exceptions.TrialPruned()

    logger.info(
        "Trial %d  val_mae=%.4f  d_model=%d  n_layers=%d  context=%d",
        trial.number, val_mae, d_model, n_layers, context_len,
    )
    return val_mae


def run_search(
    n_trials: int = 50,
    timeout_hours: float = 4.0,
    output_dir: str = "hparam_results",
    max_train_year: int = 2024,
    val_frac: float = 0.15,
    study_name: str = "transformer_v2_hpo",
) -> dict:
    try:
        import optuna  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "optuna is required for hyperparameter search. "
            "Install with: pip install optuna"
        )

    from app.train import _load_features, filter_training_laps, _split_val  # noqa: PLC0415

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features (max_year=%d)...", max_train_year)
    df_all   = _load_features(max_year=max_train_year)
    if df_all.empty:
        raise ValueError("No feature data found. Run the data pipeline first.")

    if "lap_type" not in df_all.columns:
        df_all["lap_type"]       = "normal"
        df_all["is_valid_train"] = True
    df_clean = filter_training_laps(df_all)

    df_train, df_val = _split_val(df_clean, val_frac=val_frac)
    if df_train.empty or df_val.empty:
        raise ValueError("Empty train or val split.")

    logger.info("Train: %d rows | Val: %d rows", len(df_train), len(df_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    timeout_secs = timeout_hours * 3600 if timeout_hours > 0 else None
    study.optimize(
        lambda trial: _objective(trial, df_train, df_val, None),
        n_trials=n_trials,
        timeout=timeout_secs,
        catch=(Exception,),
    )

    best = study.best_params
    best["val_mae"] = study.best_value
    logger.info("Best hyperparameters: %s", best)

    # Save outputs
    best_path = out_path / "best_hparams.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    logger.info("Best hyperparameters saved to %s", best_path)

    study_path = out_path / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    logger.info("Optuna study saved to %s", study_path)

    # Trials CSV
    trials_path = out_path / "trials.csv"
    with open(trials_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "val_mae", "state"] + list(study.best_params.keys()))
        for t in study.trials:
            if t.values is None:
                continue
            row = [t.number, t.values[0] if t.values else None, t.state.name]
            row += [t.params.get(k, "") for k in study.best_params.keys()]
            writer.writerow(row)
    logger.info("All trials saved to %s", trials_path)

    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for TyreDegradationTransformerV2 using Optuna."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50).",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=4.0,
        help="Maximum search duration in hours; 0 = no timeout (default: 4.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hparam_results",
        help="Output directory for results (default: hparam_results/).",
    )
    parser.add_argument(
        "--max-train-year",
        type=int,
        default=2024,
        help="Hold out seasons after this year (default: 2024).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of sessions held out as validation (default: 0.15).",
    )
    args = parser.parse_args()

    best = run_search(
        n_trials=args.n_trials,
        timeout_hours=args.timeout_hours,
        output_dir=args.output,
        max_train_year=args.max_train_year,
        val_frac=args.val_frac,
    )
    print(f"\nBest val_mae: {best.get('val_mae', 'N/A'):.4f}")
    print(f"Best params:  {json.dumps({k: v for k, v in best.items() if k != 'val_mae'}, indent=2)}")
    print(f"\nTo train with best params:")
    print(f"  python -m scripts.train_models --hparam-json {args.output}/best_hparams.json")


if __name__ == "__main__":
    main()
