"""ENTSO-E Transparency day-ahead prices (L3 source).

Requires a free security token in the ``ENTSOE_API_KEY`` environment variable.
The token is never written to disk or to the command line by these helpers.
"""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

from utils.exogenous.config import ENTSOE_ZONES, PRICE_DIR, MissingSourceError

API_URL = "https://web-api.tp.entsoe.eu/api"
REQUEST_TIMEOUT = 120
NAMESPACE = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"}


def _token():
    token = os.environ.get("ENTSOE_API_KEY")
    if not token:
        raise MissingSourceError(
            "ENTSOE_API_KEY is not set. Create a free token at "
            "https://transparency.entsoe.eu/usrm/user/myAccountSettings and export it "
            "before running the price fetch (never commit it)."
        )
    return token


def cache_path(zone_code, year):
    return Path(PRICE_DIR) / f"day_ahead_{zone_code}_{int(year)}.csv"


def fetch_year(zone_code, year, session=None):
    """Download one bidding-zone-year of hourly day-ahead prices (EUR/MWh)."""
    if zone_code not in ENTSOE_ZONES:
        raise ValueError(f"Unknown zone {zone_code!r}. Known: {sorted(ENTSOE_ZONES)}")
    domain = ENTSOE_ZONES[zone_code]
    path = cache_path(zone_code, year)
    if path.exists():
        return path

    params = {
        "securityToken": _token(),
        "documentType": "A44",
        "businessType": "A01",
        "in_Domain": domain,
        "out_Domain": domain,
        "periodStart": f"{int(year)}01010000",
        "periodEnd": f"{int(year) + 1}01010000",
    }
    # Imported here so that reading a cached price file (and every model run)
    # does not require the HTTP client to be installed.
    import requests

    response = (session or requests).get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    frame = parse_prices(response.content)
    if frame.empty:
        raise RuntimeError(f"ENTSO-E returned no price points for {zone_code} {year}")

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.5f")
    return path


def _naive_utc(timestamp):
    """ENTSO-E returns tz-aware UTC; the benchmark series are naive UTC."""
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tz is not None:
        return timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _parse_resolution(text):
    """ISO-8601 duration as returned by ENTSO-E: PT60M, PT15M, PT1H."""
    raw = str(text).strip().upper().replace("PT", "")
    if raw.endswith("M"):
        return pd.Timedelta(minutes=int(raw[:-1]))
    if raw.endswith("H"):
        return pd.Timedelta(hours=int(raw[:-1]))
    raise ValueError(f"Unsupported ENTSO-E resolution {text!r}")


def parse_prices(xml_bytes):
    """Parse an ENTSO-E A44 document into (time, price_eur_mwh)."""
    root = ET.fromstring(xml_bytes)
    rows = []

    for series in root.findall("ns:TimeSeries", NAMESPACE):
        for period in series.findall("ns:Period", NAMESPACE):
            start_el = period.find("ns:timeInterval/ns:start", NAMESPACE)
            resolution_el = period.find("ns:resolution", NAMESPACE)
            if start_el is None or resolution_el is None:
                continue
            start = _naive_utc(pd.Timestamp(start_el.text))
            resolution = _parse_resolution(resolution_el.text)

            for point in period.findall("ns:Point", NAMESPACE):
                position_el = point.find("ns:position", NAMESPACE)
                price_el = point.find("ns:price.amount", NAMESPACE)
                if position_el is None or price_el is None:
                    continue
                timestamp = start + (int(position_el.text) - 1) * resolution
                rows.append({"time": timestamp, "price_eur_mwh": float(price_el.text)})

    if not rows:
        return pd.DataFrame(columns=["time", "price_eur_mwh"])

    frame = pd.DataFrame(rows).sort_values("time")
    return frame.drop_duplicates(subset="time", keep="last").reset_index(drop=True)


def load_hourly(zone_code, start, end):
    """Cached hourly prices covering [start, end], or raise MissingSourceError."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    frames, absent = [], []

    for year in range(start_ts.year, end_ts.year + 1):
        path = cache_path(zone_code, year)
        if not path.exists():
            absent.append(year)
            continue
        frames.append(pd.read_csv(path, parse_dates=["time"]))

    if absent:
        raise MissingSourceError(
            f"Day-ahead prices for {zone_code} are missing years {absent}. "
            f"Set ENTSOE_API_KEY and run: python scripts/fetch/fetch_prices_entsoe.py --zone {zone_code}"
        )

    frame = pd.concat(frames).sort_values("time")
    frame = frame.drop_duplicates(subset="time", keep="last")
    if frame["time"].min() > start_ts:
        raise MissingSourceError(
            f"Day-ahead prices for {zone_code} start at {frame['time'].min()}, after {start_ts}"
        )
    return frame.set_index("time")["price_eur_mwh"]
