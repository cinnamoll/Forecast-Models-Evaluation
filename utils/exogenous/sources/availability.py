"""Per-level availability of the exogenous ladder.

The paper needs one table that says, for every (dataset, level), whether the
level could be built and -- when it could not -- why.  "n/a" is a result; a
substituted proxy is not.
"""

from pathlib import Path

import pandas as pd

from utils.exogenous.config import PAPER_PRICE_FILES, PRICE_ZONES, site_for
from utils.exogenous.market_static_features import (
    get_static_household_metadata,
    load_controller_schedule,
)
from utils.exogenous.sources import entsoe, openmeteo


def weather_available(dataset_name, start, end):
    site = site_for(dataset_name)
    missing_years = [
        year
        for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
        if not openmeteo.cache_path(site["site_id"], year).exists()
    ]
    if missing_years:
        return False, f"ERA5 not cached for {missing_years}"
    return True, "ok"


def price_available(dataset_name, start, end):
    """ok only when a real series covers the whole window, from either source."""
    key = str(dataset_name).lower()
    zone = PRICE_ZONES.get(key)
    if zone is None:
        return False, f"no day-ahead market for this task in that period (zones: {sorted(PRICE_ZONES)})"
    zone_code = zone[0]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    # A dataset that ships its own auction series (Belgium) needs no token; it
    # only helps when it actually spans the window being reported.
    release_detail = ""
    local = PAPER_PRICE_FILES.get(key)
    if local is not None and Path(local).exists():
        times = pd.read_csv(local, usecols=["time"], parse_dates=["time"])["time"]
        first, last = times.min(), times.max()
        if first <= start_ts and last >= end_ts - pd.Timedelta(hours=1):
            return True, f"ok, release file {Path(local).name} ({first} .. {last})"
        release_detail = f"; release file {Path(local).name} stops at {last}"

    missing_years = [
        year
        for year in range(start_ts.year, end_ts.year + 1)
        if not entsoe.cache_path(zone_code, year).exists()
    ]
    if missing_years:
        return False, f"day-ahead prices not cached for {zone_code} {missing_years}{release_detail}"
    return True, f"ok{release_detail}"


def static_available(dataset_name, entity_id):
    meta = get_static_household_metadata(dataset_name, entity_id)
    if not meta.available:
        return False, meta.missing.get("all", "no static metadata")
    return True, "ok" if not meta.missing else f"partial: missing {sorted(meta.missing)}"


def report(dataset_name, entity_id, start, end):
    """dict block -> (available: bool, detail: str) for the blocks behind ``ALL``.

    There is no ladder any more: a model runs ``L0`` (paper baseline, no
    covariates) or ``ALL``.  So the question this answers is "what does ALL
    actually contain for this dataset", which is what the appendix has to state
    beside the run -- the blocks genuinely differ, and that is why the experiment
    is compared on the *eliminated* sets rather than on the raw ALL columns.
    """
    return {
        "calendar": (True, "always: holidays, bridge days, DST, cyclic encodings"),
        "weather": weather_available(dataset_name, start, end),
        "price": price_available(dataset_name, start, end),
        "static": static_available(dataset_name, entity_id),
    }


def missing_blocks(dataset_name, entity_id, start, end):
    return {name: detail for name, (ok, detail)
            in report(dataset_name, entity_id, start, end).items() if not ok}
