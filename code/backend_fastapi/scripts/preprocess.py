from __future__ import annotations

import argparse

from app.preprocess import build_features_for_year, build_features_range


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    if args.year:
        build_features_for_year(args.year)
    else:
        if args.start is None or args.end is None:
            raise SystemExit("Provide --year or --start/--end")
        build_features_range(args.start, args.end)


if __name__ == "__main__":
    main()
