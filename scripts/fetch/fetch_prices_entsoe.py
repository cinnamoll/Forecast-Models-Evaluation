#!/usr/bin/env python3
"""Fetch hourly day-ahead prices from ENTSO-E Transparency (L3).

Requires ``ENTSOE_API_KEY`` in the environment.  Without the token the script
stops with instructions instead of writing a placeholder price series.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.exogenous.config import ENTSOE_ZONES  # noqa: E402
from utils.exogenous.sources import entsoe  # noqa: E402

# Belgium (PV/BESS task, Jan-Jun 2024) and Germany DE-LU (WPuQ 2024 window).
DEFAULT_JOBS = [("BE", 2024), ("DE-LU", 2024)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", nargs="+", choices=sorted(ENTSOE_ZONES), default=None)
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--jobs", nargs="+", default=None,
                        help="Explicit zone:year pairs, e.g. BE:2024 DE-LU:2024")
    args = parser.parse_args(argv)

    if not os.environ.get("ENTSOE_API_KEY"):
        print("ENTSOE_API_KEY is not set.\n"
              "1) register at https://transparency.entsoe.eu/usrm/user/myAccountSettings\n"
              "2) export ENTSOE_API_KEY='<token>'\n"
              "3) re-run this script", file=sys.stderr)
        return 2

    if args.jobs:
        jobs = [tuple(job.split(":")) for job in args.jobs]
        jobs = [(zone, int(year)) for zone, year in jobs]
    elif args.zone:
        years = args.years or [2024]
        jobs = [(zone, year) for zone in args.zone for year in years]
    else:
        jobs = DEFAULT_JOBS

    failures = 0
    for zone, year in jobs:
        try:
            path = entsoe.fetch_year(zone, year)
            print(f"[ok]   {zone:6s} {year}  -> {path.relative_to(PROJECT_ROOT)}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[fail] {zone:6s} {year}  {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
