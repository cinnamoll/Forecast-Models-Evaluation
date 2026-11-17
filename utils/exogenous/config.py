"""Per-dataset exogenous source configuration.

Only the ERA5 sampling site per dataset is left here.  Data directories are
spelled out where the fetch scripts write them, and the covariate set itself
(calendar + weather everywhere, soc / day_ahead_price / hardware for Belgium) is
built in ``utils/covariates.py``.
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


class MissingSourceError(RuntimeError):
    """Raised when a real data source is absent instead of silently faking it."""


def minutes_per_step(freq):
    """Return the number of minutes in one step for a pandas-style freq string."""
    if freq is None:
        raise ValueError("freq is required; pass the series frequency, e.g. '15min'")
    aliases = {
        "15min": 15,
        "15t": 15,
        "30min": 30,
        "30t": 30,
        "1h": 60,
        "h": 60,
        "60min": 60,
        "hourly": 60,
    }
    key = str(freq).strip().lower()
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unsupported freq {freq!r}. Known: {sorted(aliases)}")


def steps_per_hour(freq):
    minutes = minutes_per_step(freq)
    if 60 % minutes:
        raise ValueError(f"freq {freq!r} does not divide an hour evenly")
    return 60 // minutes


def steps_per_day(freq):
    return 24 * steps_per_hour(freq)


def site_for(dataset_name):
    key = str(dataset_name).strip().lower()
    if key not in SITES:
        raise ValueError(f"Unknown dataset {dataset_name!r}. Known: {sorted(SITES)}")
    return SITES[key]
