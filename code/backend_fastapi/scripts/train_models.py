from __future__ import annotations

import argparse

from app.train import train_per_driver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-laps", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    train_per_driver(min_laps=args.min_laps, epochs=args.epochs)


if __name__ == "__main__":
    main()
