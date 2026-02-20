from __future__ import annotations

import argparse

from app.ingest import ingest_season, ingest_range


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Single season to ingest")
    parser.add_argument("--start", type=int, help="Start year")
    parser.add_argument("--end", type=int, help="End year")
    args = parser.parse_args()

    if args.year:
        ingest_season(args.year)
    else:
        if args.start is None or args.end is None:
            raise SystemExit("Provide --year or --start/--end")
        ingest_range(args.start, args.end)


if __name__ == "__main__":
    main()
