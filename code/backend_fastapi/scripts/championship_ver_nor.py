"""
championship_ver_nor.py — Build a head-to-head 2025 championship between
Verstappen (VER) and Norris (NOR) using the strategies predicted by RaceScope
versus the official 2025 race times from the gold parquet.

For each race:
  - The driver with the lower time gets 25 points (P1).
  - The other gets 18 points (P2).
  - This mirrors the top-two F1 points allocation and keeps the predicted vs.
    real comparison strictly apples-to-apples.

Outputs:
  - reports/championship_ver_nor.csv  : per-race table (cumulative points).
  - reports/championship_ver_nor_pred.pdf : predicted cumulative chart.
  - reports/championship_ver_nor_real.pdf : real cumulative chart.

Usage:
  python -m scripts.championship_ver_nor \
      --eval-csv evaluation_VER_NOR_2025.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


POINTS_WINNER = 25
POINTS_LOSER = 18


# Paleta y estilo compartidos con plot_eval_2025.py para coherencia visual.
DRIVER_PALETTE = {
    "VER": "#2A4D7A",  # azul navy muted (Red Bull)
    "NOR": "#E07A2A",  # papaya muted (McLaren)
}


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


def _calendar_order(year: int = 2025) -> dict[str, int]:
    """Return a {circuit_id: chronological_index} dict using the main GP date.

    The gold features parquet binds (session_key, circuit_id) and the silver
    sessions parquet adds the date. For circuits with multiple Race sessions
    (sprint weekends) we pick the latest, which is the main Grand Prix.
    """
    gold = pd.read_parquet(f"data/gold/year={year}/features.parquet")
    race_g = gold[gold.session_type == "RACE"][["session_key", "circuit_id"]]
    race_g = race_g.drop_duplicates()
    ses = pd.read_parquet(f"data/silver/year={year}/sessions.parquet")
    race_s = ses[ses.session_type == "Race"][["session_key", "date_start"]]
    merged = race_g.merge(race_s, on="session_key", how="left").dropna()
    merged = (
        merged.sort_values("date_start")
        .groupby("circuit_id", as_index=False)
        .tail(1)
        .sort_values("date_start")
        .reset_index(drop=True)
    )
    return {row.circuit_id: i + 1 for i, row in merged.iterrows()}


def build_table(eval_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(eval_csv)
    if df.empty:
        raise SystemExit(f"empty eval csv: {eval_csv}")

    # Keep only VER and NOR; abandon races with a DNF (race_time < 2000s).
    sub = df[df.driver_id.isin([1, 4]) & (df.actual_race_time > 2000.0)].copy()
    grouped = (
        sub.pivot_table(
            index="circuit_id",
            columns="driver_id",
            values=[
                "actual_race_time",
                "best_predicted_time",
                "predicted_top5_includes_actual",
            ],
            aggfunc="first",
        )
    )

    # Drop circuits where either driver is missing.
    grouped = grouped.dropna(
        subset=[
            ("actual_race_time", 1),
            ("actual_race_time", 4),
            ("best_predicted_time", 1),
            ("best_predicted_time", 4),
        ]
    )

    # Real 2025 calendar order, derived from silver session dates.
    cal = _calendar_order(2025)
    valid = [c for c in grouped.index if c in cal]
    grouped = grouped.loc[valid]
    grouped = grouped.reindex(sorted(valid, key=lambda c: cal[c]))
    grouped[("race_index", "")] = [cal[c] for c in grouped.index]

    rows = []
    cum_pred = {1: 0, 4: 0}
    cum_real = {1: 0, 4: 0}
    for circuit, row in grouped.iterrows():
        ver_pred = row[("best_predicted_time", 1)]
        nor_pred = row[("best_predicted_time", 4)]
        ver_real = row[("actual_race_time", 1)]
        nor_real = row[("actual_race_time", 4)]

        # Predicted H2H.
        if ver_pred <= nor_pred:
            pts_pred = {1: POINTS_WINNER, 4: POINTS_LOSER}
            pred_winner = "VER"
        else:
            pts_pred = {1: POINTS_LOSER, 4: POINTS_WINNER}
            pred_winner = "NOR"
        # Real H2H.
        if ver_real <= nor_real:
            pts_real = {1: POINTS_WINNER, 4: POINTS_LOSER}
            real_winner = "VER"
        else:
            pts_real = {1: POINTS_LOSER, 4: POINTS_WINNER}
            real_winner = "NOR"

        for did in (1, 4):
            cum_pred[did] += pts_pred[did]
            cum_real[did] += pts_real[did]

        rows.append({
            "race_index": int(row[("race_index", "")]),
            "circuit": circuit,
            "ver_predicted_time": round(float(ver_pred), 2),
            "nor_predicted_time": round(float(nor_pred), 2),
            "predicted_winner": pred_winner,
            "ver_real_time": round(float(ver_real), 2),
            "nor_real_time": round(float(nor_real), 2),
            "real_winner": real_winner,
            "ver_pred_cum": cum_pred[1],
            "nor_pred_cum": cum_pred[4],
            "ver_real_cum": cum_real[1],
            "nor_real_cum": cum_real[4],
            "agreement": pred_winner == real_winner,
        })

    return pd.DataFrame(rows)


def chart_cumulative(df: pd.DataFrame, kind: str, out_path: Path) -> None:
    """kind in {'pred','real'} → renders a 1-panel line chart."""
    if kind == "pred":
        ver_col, nor_col = "ver_pred_cum", "nor_pred_cum"
        title = "RaceScope: simulación predicha"
    else:
        ver_col, nor_col = "ver_real_cum", "nor_real_cum"
        title = "Resultados reales 2025"

    color_ver = DRIVER_PALETTE["VER"]
    color_nor = DRIVER_PALETTE["NOR"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = df["race_index"].tolist()
    ax.plot(x, df[ver_col], color=color_ver, linewidth=1.1, zorder=2,
            label="Verstappen (VER)")
    ax.scatter(x, df[ver_col], color=color_ver, s=22, zorder=3)
    ax.plot(x, df[nor_col], color=color_nor, linewidth=1.1, zorder=2,
            label="Norris (NOR)")
    ax.scatter(x, df[nor_col], color=color_nor, s=22, zorder=3)

    ax.set_xlabel("Carrera (orden cronológico del calendario 2025)")
    ax.set_ylabel("Puntos acumulados")
    ax.set_title(title, loc="left")
    ax.set_xticks(x)
    ax.legend(loc="upper left")
    # Annotate the final value of each line at the right edge.
    for col, color in [(ver_col, color_ver), (nor_col, color_nor)]:
        y_end = df[col].iloc[-1]
        ax.annotate(f"{int(y_end)}", xy=(x[-1], y_end),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9, va="center")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("reports"), type=Path)
    parser.add_argument("--charts-dir", default=None, type=Path,
                        help="Optional second destination for the PDF charts.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    table = build_table(args.eval_csv)
    table_csv = args.out_dir / "championship_ver_nor.csv"
    table.to_csv(table_csv, index=False)
    print(f"[ok] table written → {table_csv}  ({len(table)} races)")

    pred_pdf = args.out_dir / "championship_ver_nor_pred.pdf"
    real_pdf = args.out_dir / "championship_ver_nor_real.pdf"
    chart_cumulative(table, "pred", pred_pdf)
    chart_cumulative(table, "real", real_pdf)
    print(f"[ok] charts written → {pred_pdf}, {real_pdf}")

    if args.charts_dir is not None:
        args.charts_dir.mkdir(parents=True, exist_ok=True)
        for pdf in (pred_pdf, real_pdf):
            dest = args.charts_dir / pdf.name
            dest.write_bytes(pdf.read_bytes())
            print(f"[ok] charts copied → {dest}")

    # Summary.
    final = table.iloc[-1]
    print("\n=== FINAL STANDINGS ===")
    print(f"Predicted: VER {final['ver_pred_cum']} - {final['nor_pred_cum']} NOR  "
          f"=> champion: {'VER' if final['ver_pred_cum'] > final['nor_pred_cum'] else 'NOR'}")
    print(f"Real:      VER {final['ver_real_cum']} - {final['nor_real_cum']} NOR  "
          f"=> champion: {'VER' if final['ver_real_cum'] > final['nor_real_cum'] else 'NOR'}")
    print(f"H2H agreement: {table['agreement'].sum()}/{len(table)} races "
          f"({table['agreement'].mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
