#!/usr/bin/env python3
"""Fetch hourly ERA5 weather (temperature, humidity, cloud, GHI, DNI, wind, rain).

Source: Open-Meteo archive API (ERA5), free and keyless.
Caches one CSV per site-year under ``data/exogenous/weather``.
"""

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.exogenous.config import SITES  # noqa: E402
from utils.exogenous.sources import openmeteo  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=[*SITES, "all"], default=["all"])
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Explicit years; defaults to the years each task needs.")
    args = parser.parse_args(argv)

    # Years each task actually needs, derived from the benchmark series spans.
    required_years = {
        "belgium": [2024],
        "germany": [2018, 2019, 2023, 2024],
        "london": [2013],
        "zonnedael": [2013],
    }

    datasets = list(SITES) if "all" in args.dataset else args.dataset
    failures = 0

    for dataset in datasets:
        years = args.years or required_years.get(dataset, [])
        for year in years:
            try:
                path = openmeteo.fetch_year(dataset, year)
                rows = sum(1 for _ in open(path)) - 1
                print(f"[ok]   {dataset:11s} {year}  {rows:6d} hourly rows  -> "
                      f"{path.relative_to(Path(__file__).resolve().parents[2])}")
            except Exception as exc:  # noqa: BLE001 - report and continue with other years
                failures += 1
                print(f"[fail] {dataset:11s} {year}  {type(exc).__name__}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
