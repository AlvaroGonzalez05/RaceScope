from __future__ import annotations

import argparse

from app.data_store import load_features
from app.driver_profile import train_driver_profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-laps", type=int, default=120)
    args = parser.parse_args()

    df = load_features()
    if df.empty:
        raise SystemExit("No features available. Run preprocess first.")

    train_driver_profiles(df, min_laps=args.min_laps)


if __name__ == "__main__":
    main()
