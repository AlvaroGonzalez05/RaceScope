"""
plot_eval_2025.py — Visualise predicted vs actual race time for 2025 for
four drivers (Norris, Piastri, Verstappen, Leclerc).

Reads the CSV produced by scripts/evaluate_2025.py and emits twelve
publication-grade PDFs into a target directory:

  Per driver (one PDF each):
    dumbbell_<CODE>.pdf
        Dumbbell of predicted vs actual race time, race by race.
    error_signed_<CODE>.pdf
        Signed prediction error across the season.

  Global:
    parity_global.pdf
        Scatter of actual vs predicted across all drivers and races, with
        identity line and ±1 % / ±2 % bands.
    error_signed_grid.pdf
        2×2 small multiples of the signed-error series.
    error_abs_distribution.pdf
        Boxplot + stripplot of |error| per driver.
    mape_bias_summary.pdf
        Horizontal bar summary of MAPE (%) and signed bias (s) per driver.

Usage
-----
python -m scripts.plot_eval_2025 \
    --input reports/eval_2025_4drivers.csv \
    --outdir reports/eval_2025_4drivers/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

DRIVER_ORDER = ["NOR", "PIA", "VER", "LEC"]
DRIVER_ID_TO_CODE = {4: "NOR", 81: "PIA", 1: "VER", 16: "LEC"}
DRIVER_LABEL = {
    "NOR": "Norris",
    "PIA": "Piastri",
    "VER": "Verstappen",
    "LEC": "Leclerc",
}
DRIVER_PALETTE = {
    "NOR": "#E07A2A",
    "PIA": "#8C5A2A",
    "VER": "#2A4D7A",
    "LEC": "#B33A3A",
}
DNF_REL_THRESHOLD = 0.60  # actual_race_time < 0.60 × median(circuit) → DNF


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "DejaVu Serif", "Times New Roman"],
        "font.size": 9.5,
        "axes.titlesize": 11.0,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#EAEAEA",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 9.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _calendar_order(year: int = 2025) -> dict[str, int]:
    """Map circuit_id → chronological index using silver session dates.

    For sprint weekends (multiple Race sessions on the same circuit) we keep
    the latest, which is the main Grand Prix.
    """
    gold = pd.read_parquet(f"data/gold/year={year}/features.parquet")
    race_g = gold[gold.session_type == "RACE"][["session_key", "circuit_id"]]
    race_g = race_g.drop_duplicates()
    silver = pd.read_parquet(f"data/silver/year={year}/sessions.parquet")
    race_s = silver[silver.session_type == "Race"][["session_key", "date_start"]]
    merged = race_g.merge(race_s, on="session_key", how="left").dropna()
    merged = (
        merged.sort_values("date_start")
        .groupby("circuit_id", as_index=False)
        .tail(1)
        .sort_values("date_start")
        .reset_index(drop=True)
    )
    return {row.circuit_id: i + 1 for i, row in merged.iterrows()}


def _short_circuit_label(circuit: str) -> str:
    """Compact label for tight axes."""
    aliases = {
        "Abu Dhabi": "Abu Dhabi",
        "Austin": "Austin",
        "Baku": "Baku",
        "Barcelona": "Barcelona",
        "Budapest": "Budapest",
        "Imola": "Imola",
        "Jeddah": "Jeddah",
        "Las Vegas": "Las Vegas",
        "Lusail": "Lusail",
        "Melbourne": "Melbourne",
        "Mexico City": "Mexico",
        "Miami": "Miami",
        "Monaco": "Monaco",
        "Montreal": "Montreal",
        "Monza": "Monza",
        "Sakhir": "Sakhir",
        "Shanghai": "Shanghai",
        "Silverstone": "Silverstone",
        "Singapore": "Singapore",
        "Spa": "Spa",
        "Spielberg": "Spielberg",
        "Suzuka": "Suzuka",
        "São Paulo": "São Paulo",
        "Zandvoort": "Zandvoort",
    }
    return aliases.get(circuit, circuit)


def load_evaluation(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"empty eval csv: {csv_path}")

    df = df[df.driver_id.isin(DRIVER_ID_TO_CODE)].copy()
    df["driver_code"] = df["driver_id"].map(DRIVER_ID_TO_CODE)

    median_per_circuit = df.groupby("circuit_id")["actual_race_time"].transform("median")
    dnf_mask = df["actual_race_time"] < DNF_REL_THRESHOLD * median_per_circuit
    n_dropped = int(dnf_mask.sum())
    if n_dropped:
        dropped = df.loc[dnf_mask, ["circuit_id", "driver_code", "actual_race_time"]]
        print(f"[info] dropping {n_dropped} suspected DNF rows:")
        print(dropped.to_string(index=False))
    df = df.loc[~dnf_mask].copy()

    try:
        cal = _calendar_order(2025)
    except Exception as exc:
        print(f"[warn] calendar order unavailable ({exc}); falling back to alphabetical")
        cal = {c: i + 1 for i, c in enumerate(sorted(df["circuit_id"].unique()))}

    df["race_index"] = df["circuit_id"].map(cal)
    df = df.dropna(subset=["race_index"]).copy()
    df["race_index"] = df["race_index"].astype(int)
    df["error"] = df["best_predicted_time"] - df["actual_race_time"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["error"] / df["actual_race_time"] * 100.0
    df = df.sort_values(["driver_code", "race_index"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plots — per driver
# ---------------------------------------------------------------------------


def plot_dumbbell(df: pd.DataFrame, code: str, out_path: Path) -> None:
    sub = df[df.driver_code == code].sort_values("race_index", ascending=False)
    if sub.empty:
        print(f"[warn] no data for {code} in dumbbell plot")
        return
    color = DRIVER_PALETTE[code]
    y = np.arange(len(sub))
    labels = [_short_circuit_label(c) for c in sub.circuit_id]

    fig, ax = plt.subplots(figsize=(6.8, max(3.5, 0.32 * len(sub) + 1.0)))
    for yi, (_, row) in zip(y, sub.iterrows()):
        ax.plot(
            [row.actual_race_time, row.best_predicted_time],
            [yi, yi],
            color=color, linewidth=1.0, alpha=0.55, zorder=1,
        )
    ax.scatter(sub.actual_race_time, y, s=34, facecolor=color,
               edgecolor=color, linewidth=0.8, zorder=3, label="Real")
    ax.scatter(sub.best_predicted_time, y, s=34, facecolor="white",
               edgecolor=color, linewidth=1.1, zorder=3, label="Predicho")

    err_max_x = np.maximum(sub.actual_race_time.values,
                           sub.best_predicted_time.values)
    pad = (err_max_x.max() - sub[["actual_race_time", "best_predicted_time"]].min().min()) * 0.04
    for yi, (_, row) in zip(y, sub.iterrows()):
        ax.text(
            err_max_x[yi] + pad, yi,
            f"{row.error:+.0f} s",
            color="#555555", fontsize=8.0, va="center", ha="left",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Tiempo total de carrera (s)")
    ax.set_title(
        f"{DRIVER_LABEL[code]} — predicho vs real por carrera (2025)",
        loc="left",
    )
    ax.legend(loc="lower right", ncol=2)
    ax.margins(y=0.02)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_error_signed(df: pd.DataFrame, code: str, out_path: Path) -> None:
    sub = df[df.driver_code == code].sort_values("race_index")
    if sub.empty:
        print(f"[warn] no data for {code} in error series")
        return
    color = DRIVER_PALETTE[code]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    ax.axhspan(-60, 60, color="#F2F2F2", zorder=0, label="Margen ±60 s")
    ax.axhline(0, color="#444444", linewidth=0.7, zorder=1)
    ax.plot(sub.race_index, sub.error, color=color, linewidth=1.1, zorder=2)
    ax.scatter(sub.race_index, sub.error, color=color, s=22, zorder=3)

    ax.set_xticks(sub.race_index)
    ax.set_xticklabels([_short_circuit_label(c) for c in sub.circuit_id],
                       rotation=45, ha="right")
    ax.set_xlabel("Carrera (orden de calendario 2025)")
    ax.set_ylabel("Error firmado: predicho − real (s)")
    ax.set_title(
        f"{DRIVER_LABEL[code]} — sesgo de predicción a lo largo de la temporada",
        loc="left",
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots — global
# ---------------------------------------------------------------------------


def plot_parity(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    lo = min(df.actual_race_time.min(), df.best_predicted_time.min())
    hi = max(df.actual_race_time.max(), df.best_predicted_time.max())
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad

    x = np.linspace(lo, hi, 100)
    ax.fill_between(x, x * 0.98, x * 1.02, color="#F4F4F4", label="±2 %")
    ax.fill_between(x, x * 0.99, x * 1.01, color="#E8E8E8", label="±1 %")
    ax.plot(x, x, color="#444444", linewidth=0.8, label="Identidad")

    for code in DRIVER_ORDER:
        sub = df[df.driver_code == code]
        ax.scatter(
            sub.actual_race_time, sub.best_predicted_time,
            s=34, color=DRIVER_PALETTE[code],
            edgecolor="white", linewidth=0.6, alpha=0.95,
            label=DRIVER_LABEL[code],
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Tiempo real (s)")
    ax.set_ylabel("Tiempo predicho (s)")
    ax.set_title("Calibración global: predicho vs real (2025)", loc="left")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_error_signed_grid(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharex=False, sharey=True)
    axes = axes.ravel()

    y_max = float(df.error.abs().max()) * 1.08
    y_max = max(y_max, 80.0)
    for ax, code in zip(axes, DRIVER_ORDER):
        sub = df[df.driver_code == code].sort_values("race_index")
        color = DRIVER_PALETTE[code]
        ax.axhspan(-60, 60, color="#F4F4F4", zorder=0)
        ax.axhline(0, color="#444444", linewidth=0.7, zorder=1)
        if not sub.empty:
            ax.plot(sub.race_index, sub.error, color=color, linewidth=1.0, zorder=2)
            ax.scatter(sub.race_index, sub.error, color=color, s=18, zorder=3)
            ax.set_xticks(sub.race_index)
            ax.set_xticklabels(
                [_short_circuit_label(c) for c in sub.circuit_id],
                rotation=60, ha="right", fontsize=7.5,
            )
        ax.set_ylim(-y_max, y_max)
        ax.set_title(DRIVER_LABEL[code], loc="left", fontsize=10.5)
        ax.set_ylabel("Error (s)")
    fig.suptitle(
        "Error firmado (predicho − real) por carrera — temporada 2025",
        x=0.06, ha="left", fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_abs_error_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    data = [df.loc[df.driver_code == code, "abs_error"].values for code in DRIVER_ORDER]

    bp = ax.boxplot(
        data, vert=False, widths=0.55, patch_artist=True,
        medianprops=dict(color="#222222", linewidth=1.0),
        whiskerprops=dict(color="#666666", linewidth=0.8),
        capprops=dict(color="#666666", linewidth=0.8),
        flierprops=dict(marker="", linestyle=""),
    )
    for patch, code in zip(bp["boxes"], DRIVER_ORDER):
        patch.set_facecolor("white")
        patch.set_edgecolor(DRIVER_PALETTE[code])
        patch.set_linewidth(1.0)

    rng = np.random.default_rng(42)
    for i, (code, values) in enumerate(zip(DRIVER_ORDER, data)):
        if len(values) == 0:
            continue
        y_jit = (i + 1) + rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(values, y_jit, s=22, color=DRIVER_PALETTE[code],
                   alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_yticks(range(1, len(DRIVER_ORDER) + 1))
    ax.set_yticklabels([DRIVER_LABEL[c] for c in DRIVER_ORDER])
    ax.set_xlabel("Error absoluto |predicho − real| (s)")
    ax.set_title("Distribución del error absoluto por piloto (2025)", loc="left")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_mape_bias_summary(df: pd.DataFrame, out_path: Path) -> tuple[pd.DataFrame, None]:
    rows = []
    for code in DRIVER_ORDER:
        sub = df[df.driver_code == code]
        if sub.empty:
            continue
        rows.append({
            "driver_code": code,
            "driver_label": DRIVER_LABEL[code],
            "mape_pct": float(sub.abs_error.mean() / sub.actual_race_time.mean() * 100.0),
            "bias_s": float(sub.error.mean()),
            "median_error_s": float(sub.error.median()),
            "n_races": int(len(sub)),
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        print("[warn] empty summary — skipping MAPE/bias plot")
        return summary, None

    fig, (ax_mape, ax_bias) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    y = np.arange(len(summary))
    colors = [DRIVER_PALETTE[c] for c in summary.driver_code]
    labels = list(summary.driver_label)

    ax_mape.barh(y, summary.mape_pct, color=colors, edgecolor="white", linewidth=0.8)
    for yi, v in zip(y, summary.mape_pct):
        ax_mape.text(v + 0.05, yi, f"{v:.2f} %", va="center", fontsize=8.5,
                     color="#333333")
    ax_mape.set_yticks(y)
    ax_mape.set_yticklabels(labels)
    ax_mape.invert_yaxis()
    ax_mape.set_xlabel("MAPE (%)")
    ax_mape.set_title("Precisión (MAPE)", loc="left")
    ax_mape.grid(axis="x")
    ax_mape.grid(axis="y", visible=False)
    ax_mape.margins(x=0.18)

    ax_bias.axvline(0, color="#444444", linewidth=0.7)
    ax_bias.barh(y, summary.bias_s, color=colors, edgecolor="white", linewidth=0.8)
    for yi, v in zip(y, summary.bias_s):
        offset = max(abs(v) * 0.02, 4.0)
        ax_bias.text(
            v + (offset if v >= 0 else -offset), yi, f"{v:+.0f} s",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8.5, color="#333333",
        )
    ax_bias.set_yticks(y)
    ax_bias.set_yticklabels([])
    ax_bias.invert_yaxis()
    ax_bias.set_xlabel("Sesgo: media(predicho − real) (s)")
    ax_bias.set_title("Sesgo medio", loc="left")
    ax_bias.grid(axis="x")
    ax_bias.grid(axis="y", visible=False)
    ax_bias.margins(x=0.22)

    fig.suptitle("Resumen de calibración por piloto (2025)",
                 x=0.06, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return summary, None


# ---------------------------------------------------------------------------
# Driver console summary
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("\n=== Resumen por piloto (2025) ===")
    if summary.empty:
        print("(sin datos)")
        return
    print(summary.to_string(index=False,
                            float_format=lambda x: f"{x:.2f}"))
    print(f"\nCarreras evaluadas (tras filtro DNF): {len(df)}")
    print(f"Circuitos únicos cubiertos: {df['circuit_id'].nunique()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to evaluate_2025.py CSV output")
    parser.add_argument("--outdir", required=True, type=Path,
                        help="Directory where PDFs are written")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    df = load_evaluation(args.input)

    for code in DRIVER_ORDER:
        plot_dumbbell(df, code, args.outdir / f"dumbbell_{code}.pdf")
        plot_error_signed(df, code, args.outdir / f"error_signed_{code}.pdf")

    plot_parity(df, args.outdir / "parity_global.pdf")
    plot_error_signed_grid(df, args.outdir / "error_signed_grid.pdf")
    plot_abs_error_distribution(df, args.outdir / "error_abs_distribution.pdf")
    summary, _ = plot_mape_bias_summary(df, args.outdir / "mape_bias_summary.pdf")

    summary_csv = args.outdir / "summary_by_driver.csv"
    if not summary.empty:
        summary.to_csv(summary_csv, index=False)
        print(f"[ok] summary table → {summary_csv}")

    print_summary(df, summary)
    print(f"\n[ok] PDFs written to {args.outdir}/")


if __name__ == "__main__":
    main()
