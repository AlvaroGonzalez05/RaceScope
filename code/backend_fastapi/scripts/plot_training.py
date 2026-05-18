"""
plot_training.py — Visual evaluation of Transformer v2 training runs.

Reads the per-epoch CSV logs produced by train_models.py and generates:

  1. global_curves.png       — Global model: train_loss + val_mae vs epoch
  2. driver_grid.png         — Small-multiples val_mae per epoch for all 25 drivers
  3. final_val_mae_bar.png   — Ranked bar chart of final val_mae across drivers
  4. convergence_epochs.png  — Epochs trained per driver (early stopping visible)
  5. loss_vs_valmae.png      — Scatter: final train_loss vs final val_mae per driver

Usage
-----
  cd code/backend_fastapi
  python3.11 -m scripts.plot_training [--logs-dir models/logs] [--out-dir plots]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

DARK_BG   = "#0f1117"
PANEL_BG  = "#16181b"
TEXT      = "#e8eaf0"
MUTED     = "#9ea6b1"
ACCENT    = "#e8002d"   # F1 red
ACCENT2   = "#00c4b4"   # teal
GRID_COL  = "#2a2d35"

def _apply_dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   DARK_BG,
        "axes.facecolor":     PANEL_BG,
        "axes.edgecolor":     GRID_COL,
        "axes.labelcolor":    TEXT,
        "axes.titlecolor":    TEXT,
        "xtick.color":        MUTED,
        "ytick.color":        MUTED,
        "text.color":         TEXT,
        "grid.color":         GRID_COL,
        "grid.linewidth":     0.6,
        "legend.facecolor":   PANEL_BG,
        "legend.edgecolor":   GRID_COL,
        "legend.labelcolor":  TEXT,
        "font.family":        "monospace",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
    })


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def _latest_log(logs_dir: Path, prefix: str) -> Optional[pd.DataFrame]:
    """Return the most recent CSV matching logs_dir/<prefix>_*.csv."""
    matches = sorted(logs_dir.glob(f"{prefix}_*.csv"))
    if not matches:
        return None
    df = pd.read_csv(matches[-1])
    df["epoch"] = df["epoch"].astype(int)
    return df


def _load_all_latest(logs_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load latest run for every prefix found in logs_dir."""
    seen: Dict[str, str] = {}
    for p in sorted(logs_dir.glob("*.csv"), reverse=True):
        # filename pattern: <prefix>_YYYYMMDD_HHMMSS.csv
        m = re.match(r"^(.+)_\d{8}_\d{6}\.csv$", p.name)
        if not m:
            continue
        prefix = m.group(1)
        if prefix not in seen:
            seen[prefix] = str(p)

    result: Dict[str, pd.DataFrame] = {}
    for prefix, path in seen.items():
        df = pd.read_csv(path)
        df["epoch"] = df["epoch"].astype(int)
        result[prefix] = df
    return result


# ---------------------------------------------------------------------------
# Plot 1 — Global model curves
# ---------------------------------------------------------------------------

def plot_global_curves(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(DARK_BG)

    epochs = df["epoch"].values
    ax1.plot(epochs, df["train_loss"].values, color=ACCENT,  lw=2,   label="Train loss (Huber)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color=ACCENT)
    ax1.tick_params(axis="y", labelcolor=ACCENT)
    ax1.grid(True, axis="both")
    ax1.set_xlim(1, epochs.max())

    ax2 = ax1.twinx()
    ax2.plot(epochs, df["val_mae"].values,   color=ACCENT2, lw=2,   label="Val MAE (Δlap, norm.)")
    ax2.set_ylabel("Val MAE (norm.)", color=ACCENT2)
    ax2.tick_params(axis="y", labelcolor=ACCENT2)
    ax2.set_facecolor(PANEL_BG)

    # Best val_mae marker
    best_idx = df["val_mae"].idxmin()
    best_ep  = int(df.loc[best_idx, "epoch"])
    best_mae = float(df.loc[best_idx, "val_mae"])
    ax2.axvline(best_ep, color=ACCENT2, lw=1, ls="--", alpha=0.6)
    ax2.annotate(
        f"best={best_mae:.4f}\n@ epoch {best_ep}",
        xy=(best_ep, best_mae),
        xytext=(best_ep + 0.5, best_mae + 0.01),
        color=ACCENT2, fontsize=8,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax1.set_title("Global model — Training curves (v2)", fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Driver small-multiples grid
# ---------------------------------------------------------------------------

def plot_driver_grid(
    logs: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    driver_keys = sorted(
        [k for k in logs if k.startswith("driver_")],
        key=lambda k: int(k.split("_")[1]),
    )
    n = len(driver_keys)
    ncols = 5
    nrows = -(-n // ncols)  # ceil division

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.6))
    fig.patch.set_facecolor(DARK_BG)
    axes_flat = axes.flatten()

    for i, key in enumerate(driver_keys):
        ax  = axes_flat[i]
        df  = logs[key]
        drv = key.split("_")[1]

        epochs = df["epoch"].values
        ax.plot(epochs, df["train_loss"].values, color=ACCENT,  lw=1.4, label="loss")
        ax2 = ax.twinx()
        ax2.plot(epochs, df["val_mae"].values,   color=ACCENT2, lw=1.4, label="val_mae")
        ax2.set_facecolor(PANEL_BG)

        # Mark best val_mae
        valid = df["val_mae"].dropna()
        if not valid.empty:
            best_ep = int(df.loc[valid.idxmin(), "epoch"])
            ax2.axvline(best_ep, color=MUTED, lw=0.8, ls="--", alpha=0.7)

        final_mae = float(df["val_mae"].iloc[-1])
        ax.set_title(f"Driver {drv}  (final mae={final_mae:.3f})", fontsize=8, pad=3)
        ax.set_xlim(1, max(epochs))
        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)
        ax.grid(True, alpha=0.4)

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Legend in last visible slot if there's room
    if n < len(axes_flat):
        leg_ax = axes_flat[n]
        leg_ax.set_visible(True)
        leg_ax.set_facecolor(PANEL_BG)
        leg_ax.axis("off")
        leg_ax.plot([], [], color=ACCENT,  lw=2, label="Train loss")
        leg_ax.plot([], [], color=ACCENT2, lw=2, label="Val MAE")
        leg_ax.plot([], [], color=MUTED,   lw=1, ls="--", label="Best epoch")
        leg_ax.legend(loc="center", fontsize=9)

    fig.suptitle("Per-driver training curves — Transformer v2", fontweight="bold",
                 fontsize=12, color=TEXT, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Final val_mae bar chart (ranked)
# ---------------------------------------------------------------------------

def plot_final_val_mae_bar(
    logs: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    records: List[Tuple[str, float, float]] = []
    for key, df in logs.items():
        if not key.startswith("driver_") and key != "global":
            continue
        label    = key.replace("driver_", "D") if key != "global" else "GLOBAL"
        valid    = df["val_mae"].dropna()
        if valid.empty:
            continue
        best_mae = float(valid.min())
        final_mae= float(valid.iloc[-1])
        records.append((label, best_mae, final_mae))

    records.sort(key=lambda x: x[1])
    labels    = [r[0] for r in records]
    best_maes = [r[1] for r in records]

    fig, ax = plt.subplots(figsize=(max(10, len(records) * 0.55), 5))
    fig.patch.set_facecolor(DARK_BG)

    colors = [ACCENT if l == "GLOBAL" else ACCENT2 for l in labels]
    bars   = ax.bar(labels, best_maes, color=colors, width=0.7, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, best_maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7, color=TEXT,
        )

    # Median line
    median = float(np.median(best_maes))
    ax.axhline(median, color=MUTED, lw=1.2, ls="--", zorder=4,
               label=f"Median {median:.3f}")

    ax.set_xlabel("Model")
    ax.set_ylabel("Best val MAE (normalised Δlap_time)")
    ax.set_title("Best val MAE per driver — Transformer v2 (lower is better)",
                 fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.5, zorder=0)
    ax.set_ylim(0, max(best_maes) * 1.18)
    plt.xticks(rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Convergence (epochs trained)
# ---------------------------------------------------------------------------

def plot_convergence_epochs(
    logs: Dict[str, pd.DataFrame],
    out_path: Path,
    max_epochs: int = 15,
) -> None:
    records: List[Tuple[str, int, bool]] = []
    for key, df in logs.items():
        if not key.startswith("driver_"):
            continue
        label  = key.replace("driver_", "D")
        n_ep   = int(df["epoch"].max())
        stopped_early = n_ep < max_epochs
        records.append((label, n_ep, stopped_early))

    records.sort(key=lambda x: x[1])
    labels   = [r[0] for r in records]
    n_epochs = [r[1] for r in records]
    colors   = [ACCENT if r[2] else ACCENT2 for r in records]

    fig, ax = plt.subplots(figsize=(max(10, len(records) * 0.55), 4.5))
    fig.patch.set_facecolor(DARK_BG)

    bars = ax.bar(labels, n_epochs, color=colors, width=0.7, zorder=3)
    ax.axhline(max_epochs, color=MUTED, lw=1.2, ls="--", zorder=4,
               label=f"Max epochs ({max_epochs})")

    for bar, val in zip(bars, n_epochs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val),
            ha="center", va="bottom", fontsize=7.5, color=TEXT,
        )

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=ACCENT,  label="Early stopped"),
        Patch(facecolor=ACCENT2, label="Full training"),
        plt.Line2D([0], [0], color=MUTED, ls="--", label=f"Max ({max_epochs})"),
    ]
    ax.legend(handles=legend_els, fontsize=9)
    ax.set_xlabel("Driver model")
    ax.set_ylabel("Epochs trained")
    ax.set_title("Convergence — epochs trained per driver (red = early stopped)",
                 fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.5, zorder=0)
    ax.set_ylim(0, max_epochs + 2)
    plt.xticks(rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5 — Scatter: final train_loss vs final val_mae
# ---------------------------------------------------------------------------

def plot_loss_vs_valmae(
    logs: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    xs, ys, labels = [], [], []
    for key, df in logs.items():
        if not key.startswith("driver_"):
            continue
        valid = df["val_mae"].dropna()
        if valid.empty:
            continue
        xs.append(float(df["train_loss"].iloc[-1]))
        ys.append(float(valid.iloc[-1]))
        labels.append(key.replace("driver_", "D"))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(DARK_BG)

    ax.scatter(xs, ys, color=ACCENT2, s=60, zorder=3, alpha=0.85)

    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(4, 3), fontsize=7, color=MUTED)

    # Identity line for reference
    lim = max(max(xs), max(ys)) * 1.1
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", alpha=0.5, label="loss = val_mae")

    # Diagonal bands
    ax.fill_between([0, lim], [0, lim * 0.5], [0, lim * 2.0],
                    alpha=0.05, color=ACCENT2, label="2× band")

    ax.set_xlabel("Final train loss (Huber)")
    ax.set_ylabel("Final val MAE (norm.)")
    ax.set_title("Overfitting check — train loss vs val MAE",
                 fontweight="bold", pad=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4, zorder=0)
    ax.set_xlim(0)
    ax.set_ylim(0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Transformer v2 training evaluation charts."
    )
    parser.add_argument(
        "--logs-dir", type=str, default="models/logs",
        help="Directory containing training CSV logs (default: models/logs).",
    )
    parser.add_argument(
        "--out-dir", type=str, default="plots",
        help="Output directory for PNG files (default: plots/).",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=15,
        help="Max epochs used during training, for convergence plot (default: 15).",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not logs_dir.exists():
        print(f"ERROR: logs directory not found: {logs_dir}")
        return

    _apply_dark_style()

    print("Loading training logs...")
    logs = _load_all_latest(logs_dir)
    if not logs:
        print("No CSV logs found.")
        return
    print(f"  Found logs for: {sorted(logs.keys())}")

    print("\nGenerating plots...")

    if "global" in logs:
        plot_global_curves(logs["global"], out_dir / "global_curves.png")

    driver_logs = {k: v for k, v in logs.items() if k.startswith("driver_")}
    if driver_logs:
        plot_driver_grid(driver_logs,                out_dir / "driver_grid.png")
        plot_final_val_mae_bar(logs,                 out_dir / "final_val_mae_bar.png")
        plot_convergence_epochs(driver_logs,         out_dir / "convergence_epochs.png",
                                max_epochs=args.max_epochs)
        plot_loss_vs_valmae(driver_logs,             out_dir / "loss_vs_valmae.png")

    print(f"\nAll plots saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
