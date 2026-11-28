"""Per-dataset ERA5 sampling site.

Only the site coordinates are here.  Data directories are spelled out where the
fetch script writes them, and the covariate set itself (calendar + weather
everywhere, soc / day_ahead_price for Belgium) is built in
``utils/covariates.py``.
"""

# Coordinates are the point ERA5 is sampled at.  They are deliberately coarse:
# for household load a 50 km error in temperature is well below the noise the
# benchmark measures.  Belgium is exact (VUB Etterbeek campus), the other three
# are regional approximations because the source datasets do not publish the
# exact district.
SITES = {
    "belgium": {
        "site_id": "brussels",
        "latitude": 50.8225,
        "longitude": 4.3922,
        "country": "BE",
        "subdivision": None,
        "timezone": "Europe/Brussels",
        "approx": False,
    },
    "germany": {
        "site_id": "lower_saxony_approx",
        "latitude": 52.3759,
        "longitude": 9.7320,
        "country": "DE",
        "subdivision": "DE-NI",
        "timezone": "Europe/Berlin",
        "approx": True,
    },
    "london": {
        "site_id": "london",
        "latitude": 51.5072,
        "longitude": -0.1276,
        "country": "GB",
        "subdivision": "GB-ENG",
        "timezone": "Europe/London",
        "approx": True,
    },
    "zonnedael": {
        "site_id": "zonnedael_approx",
        "latitude": 52.1300,
        "longitude": 5.2900,
        "country": "NL",
        "subdivision": "NL-NH",
        "timezone": "Europe/Amsterdam",
        "approx": True,
    },
}


def site_for(dataset_name):
    key = str(dataset_name).strip().lower()
    if key not in SITES:
        raise ValueError(f"Unknown dataset {dataset_name!r}. Known: {sorted(SITES)}")
    return SITES[key]
