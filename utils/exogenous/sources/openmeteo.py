"""Open-Meteo ERA5 archive access (L2 source).

Free, no API key, hourly, global, back to 1940 -- which is what makes the 2013
datasets (London LCL, Zonnedael) and the 2024 Belgium/WPuQ windows usable with
the same code path.

The fetcher writes one CSV per site-year under ``data/exogenous/weather``.  The
loader refuses to run when a year is missing: a silently synthetic temperature
series is worse than a stopped experiment.
"""

from pathlib import Path

import pandas as pd

from utils.exogenous.config import MissingSourceError, site_for

HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "wind_speed_10m",
    "precipitation",
)


def cache_path(site_id, year):
    # data/exogenous/weather/<site>_<year>.csv, from utils/exogenous/sources/.
    weather_dir = Path(__file__).resolve().parents[3] / "data" / "exogenous" / "weather"
    return weather_dir / f"{site_id}_{int(year)}.csv"


def fetch_year(dataset_name, year, session=None):
    """Download one site-year of ERA5 hourly weather and cache it."""
    site = site_for(dataset_name)
    path = cache_path(site["site_id"], year)
    if path.exists():
        return path

    params = {
        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "start_date": f"{int(year)}-01-01",
        "end_date": f"{int(year)}-12-31",
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "models": "era5",
    }
    # Imported here so reading cached weather (and every model run) does not
    # require the HTTP client to be installed.
    import requests

    response = (session or requests).get(
        "https://archive-api.open-meteo.com/v1/archive", params=params, timeout=120
    )
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload:
        raise RuntimeError(f"Open-Meteo returned no hourly block for {dataset_name} {year}: {payload}")

    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.set_index("time").sort_index()
    missing = [c for c in HOURLY_VARIABLES if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Open-Meteo response is missing {missing}")

    path.parent.mkdir(parents=True, exist_ok=True)
    frame[list(HOURLY_VARIABLES)].to_csv(path, float_format="%.4f")
    return path


def load_hourly(dataset_name, start, end, pad=pd.Timedelta(hours=6)):
    """Concatenate cached years covering [start, end], or raise MissingSourceError.

    Cache files are required only for the years the span itself touches; adjacent
    years are used for the ``pad`` when they happen to be cached, so a window that
    starts on 1 January does not force a download of the previous year.
    """
    site = site_for(dataset_name)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    core_years = list(range(start_ts.year, end_ts.year + 1))
    frames, absent_years = [], []

    for year in core_years:
        path = cache_path(site["site_id"], year)
        if not path.exists():
            absent_years.append(year)
            continue
        frames.append(pd.read_csv(path, parse_dates=["time"], index_col="time"))

    if absent_years:
        raise MissingSourceError(
            f"ERA5 weather for {dataset_name} is missing years {absent_years}. "
            f"Run: python scripts/fetch/fetch_weather_openmeteo.py --dataset {dataset_name}"
        )

    for neighbour in (start_ts.year - 1, end_ts.year + 1):
        path = cache_path(site["site_id"], neighbour)
        if path.exists():
            frames.append(pd.read_csv(path, parse_dates=["time"], index_col="time"))

    hourly = pd.concat(frames).sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="first")]
    hourly = hourly.loc[start_ts - pad: end_ts + pad]

    # A sub-hourly tail (e.g. 23:30 when the archive ends at 23:00) cannot be
    # interpolated without the next hour, so it stays NaN and shows up in the
    # reported missing fraction instead of aborting the build.
    if hourly.index.min() > start_ts or hourly.index.max() < end_ts - pd.Timedelta(hours=1):
        raise MissingSourceError(
            f"ERA5 weather for {dataset_name} covers {hourly.index.min()}..{hourly.index.max()}, "
            f"which does not contain {start_ts}..{end_ts}"
        )
    return hourly


def available_years(dataset_name):
    site = site_for(dataset_name)
    weather_dir = Path(__file__).resolve().parents[3] / "data" / "exogenous" / "weather"
    return sorted(
        int(p.stem.split("_")[-1])
        for p in weather_dir.glob(f"{site['site_id']}_*.csv")
        if p.stem.split("_")[-1].isdigit()
    )
