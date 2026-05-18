from __future__ import annotations

import argparse

from app.ingest import ingest_season, ingest_range
from app.data_store import refresh_snapshot_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OpenF1 data into bronze/silver layers.")
    parser.add_argument("--year", type=int, help="Single season to ingest")
    parser.add_argument("--start", type=int, help="Start year (range mode)")
    parser.add_argument("--end",   type=int, help="End year (range mode)")
    parser.add_argument("--sleep-s",      type=float, default=1.5,  help="Sleep between sessions (s)")
    parser.add_argument("--min-interval", type=float, default=1.2,  help="Min interval between requests (s)")
    parser.add_argument(
        "--force-optional",
        action="store_true",
        help=(
            "Re-fetch optional endpoints (car_data, location, race_control, pit, intervals) "
            "for sessions where silver+bronze mandatory files are complete but optional "
            "bronze files are missing. Does NOT re-download mandatory endpoints."
        ),
    )
    args = parser.parse_args()

    try:
        kwargs = dict(
            sleep_s=args.sleep_s,
            min_interval=args.min_interval,
            force_optional=args.force_optional,
        )
        if args.year:
            ingest_season(args.year, **kwargs)
        else:
            if args.start is None or args.end is None:
                raise SystemExit("Provide --year or --start/--end")
            ingest_range(args.start, args.end, **kwargs)
    except Exception as exc:
        refresh_snapshot_state(stale_mode=True, last_error=str(exc))
        raise


if __name__ == "__main__":
    main()
