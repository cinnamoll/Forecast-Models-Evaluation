#!/usr/bin/env python3
"""Fetch school-holiday calendars from the OpenHolidays API.

OpenHolidays coverage is uneven: NL is populated for recent years, GB is absent
and 2013 is generally missing.  The script therefore reports a coverage table and
only writes what the API actually returns.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.exogenous.config import SITES  # noqa: E402


def fetch(country, subdivision, year):
    languages = {"GB": "EN", "DE": "DE", "NL": "NL", "BE": "NL"}
    params = {
        "countryIsoCode": country,
        "validFrom": f"{year}-01-01",
        "validTo": f"{year}-12-31",
        "languageIsoCode": languages.get(country, "EN"),
    }
    if subdivision:
        params["subdivisionCode"] = subdivision
    response = requests.get(
        "https://openholidaysapi.org/SchoolHolidays", params=params, timeout=60
    )
    response.raise_for_status()
    payload = response.json()

    rows = []
    for entry in payload:
        name = entry.get("name") or []
        label = name[0]["text"] if name else ""
        rows.append(
            {
                "country": country,
                "subdivision": subdivision or "",
                "start": entry["startDate"],
                "end": entry["endDate"],
                "name": label,
                "source": "openholidaysapi.org",
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=[*SITES, "all"], default=["all"])
    parser.add_argument("--years", nargs="+", type=int, default=[2013, 2018, 2019, 2023, 2024, 2026])
    args = parser.parse_args(argv)

    # data/exogenous/holidays, from scripts/fetch/.
    holiday_dir = Path(__file__).resolve().parents[2] / "data" / "exogenous" / "holidays"

    datasets = list(SITES) if "all" in args.dataset else args.dataset
    countries = {}
    for dataset in datasets:
        site = SITES[dataset]
        countries.setdefault(site["country"], site.get("subdivision"))

    holiday_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'country':8s} {'subdiv':8s} {'year':6s} rows")
    for country, subdivision in countries.items():
        frames = []
        for year in args.years:
            rows = fetch(country, subdivision, year)
            print(f"{country:8s} {str(subdivision):8s} {year:<6d} {len(rows)}")
            if rows:
                frames.append(pd.DataFrame(rows))
        out = holiday_dir / f"school_holidays_{country.upper()}.csv"
        if frames:
            pd.concat(frames).drop_duplicates().to_csv(out, index=False)
            print(f"  -> {out.relative_to(Path(__file__).resolve().parents[2])}")
        else:
            print(f"  no data returned for {country}; school holidays stay n/a there")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
