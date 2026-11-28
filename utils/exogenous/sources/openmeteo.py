"""Open-Meteo ERA5 archive access (L2 source).

Free, no API key, hourly, global, back to 1940 -- which is what makes the 2013
datasets (London LCL, Zonnedael) and the 2024 Belgium/WPuQ windows usable with
the same code path.

The fetcher writes one CSV per site-year under ``data/exogenous/weather``;
``utils/covariates.weather_frame`` reads those files back, so nothing that only
trains a model imports this module.
"""

from pathlib import Path

import pandas as pd

from utils.exogenous.config import site_for

HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
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
