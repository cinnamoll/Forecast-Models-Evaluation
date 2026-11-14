#!/usr/bin/env python3
"""Fetch school-holiday calendars from the OpenHolidays API (L1).

OpenHolidays coverage is uneven: NL is populated for recent years, GB is absent
and 2013 is generally missing.  The script therefore reports a coverage table and
only writes what the API actually returns, so the feature store can mark the
school-holiday column n/a instead of pretending it exists.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.exogenous.config import HOLIDAY_DIR, SITES  # noqa: E402

API_URL = "https://openholidaysapi.org/SchoolHolidays"
LANGUAGES = {"GB": "EN", "DE": "DE", "NL": "NL", "BE": "NL"}
REQUEST_TIMEOUT = 60


def fetch(country, subdivision, year):
    params = {
        "countryIsoCode": country,
        "validFrom": f"{year}-01-01",
        "validTo": f"{year}-12-31",
        "languageIsoCode": LANGUAGES.get(country, "EN"),
    }
    if subdivision:
        params["subdivisionCode"] = subdivision
    response = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
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

    datasets = list(SITES) if "all" in args.dataset else args.dataset
    countries = {}
    for dataset in datasets:
        site = SITES[dataset]
        countries.setdefault(site["country"], site.get("subdivision"))

    HOLIDAY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'country':8s} {'subdiv':8s} {'year':6s} rows")
    for country, subdivision in countries.items():
        frames = []
        for year in args.years:
            rows = fetch(country, subdivision, year)
            print(f"{country:8s} {str(subdivision):8s} {year:<6d} {len(rows)}")
            if rows:
                frames.append(pd.DataFrame(rows))
        out = HOLIDAY_DIR / f"school_holidays_{country.upper()}.csv"
        if frames:
            pd.concat(frames).drop_duplicates().to_csv(out, index=False)
            print(f"  -> {out.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  no data returned for {country}; school holidays stay n/a there")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
