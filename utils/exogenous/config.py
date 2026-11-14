"""Central configuration for exogenous feature construction.

Everything that is dataset-specific (coordinates, timezone, calendar rules,
where the real source files live) is declared here so that the feature code
never has to guess and never has to invent a fallback.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXOGENOUS_DATA_DIR = PROJECT_ROOT / "data" / "exogenous"
WEATHER_DIR = EXOGENOUS_DATA_DIR / "weather"
PRICE_DIR = EXOGENOUS_DATA_DIR / "prices"
NEIGHBOR_DIR = EXOGENOUS_DATA_DIR / "neighbors"
METADATA_DIR = EXOGENOUS_DATA_DIR / "metadata"
HOLIDAY_DIR = EXOGENOUS_DATA_DIR / "holidays"

# Two-decimal grid the benchmark series live on.  All exogenous data is
# resampled onto the series' own index, never the other way round.
DEFAULT_MAX_INTERPOLATION_HOURS = 3.0
DEFAULT_BASE_TEMPERATURE_C = 15.5

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

# Bidding zone used for the day-ahead price per dataset.  Only the two tasks that
# actually have a traded day-ahead market are listed; everything else is n/a by
# rule (plan_updated.md section 6: "bậc không áp dụng được thì ghi n/a").
# Belgium: PV/BESS task, 2024.  Germany: WPuQ task, 2024 (DE-LU bidding zone).
PRICE_ZONES = {
    "belgium": ("BE", "10YBE----------2"),
    "germany": ("DE-LU", "10Y1001A1001A82H"),
}

# Days of {country: (zone name used by ENTSO-E)} kept explicit so a typo fails
# loudly instead of silently returning an empty price series.
ENTSOE_ZONES = {
    "BE": "10YBE----------2",
    "DE-LU": "10Y1001A1001A82H",
    "NL": "10YNL----------L",
    "GB": "10YGB----------A",
}

# Which datasets the neighbour (L5) block can be built for.  London needs the
# full LCL extract, Zonnedael ships all 56 customers in the repo file, WPuQ
# ships a single house and therefore cannot have neighbours.  Belgium has no
# entry: every house replays the same SFH19 load (so a peer would be the target
# itself), the BESS output is whatever EMS ran that day, and its PV only differs
# across houses in the release's experiment window (Apr-Jun), which the benchmark
# does not use.  L5 is therefore n/a for Belgium by construction.
NEIGHBOR_SOURCES = {
    "london": "lcl_households_2013.csv.gz",
    "zonnedael": "zonnedael_customers_2013.csv.gz",
}

# Measured plant state (L6) is retired with L5: only Belgium ships a state
# channel, so the block could never sit in a cross-dataset table.  The source is
# still in the release (``metadata/Belgium/*/soc.csv``) and
# ``scripts/fetch/build_belgium_covariates.py`` still writes
# ``metadata/belgium_state.csv`` from it, so the decision is reversible; nothing
# in the pipeline reads either file any more.

# Target series per dataset, used to exclude the target itself from its own
# neighbour aggregate.
TARGET_ENTITIES = {
    "london": "MAC000033",
    "zonnedael": "Klant 8",
}

# Local day-ahead series shipped with a dataset release, in the ENTSO-E cache
# schema (``time,price_eur_mwh``).  Belgium's release quotes the exact market
# series the source experiment was billed against (Ruddick et al. 2024, section
# 3.2.3), so L3 builds from a file instead of a token and the network.  ENTSO-E
# A44 stays the source wherever no such file exists; neither is a proxy for the
# other, both are the same auction and one just arrives with the dataset.
PAPER_PRICE_FILES = {
    "belgium": PRICE_DIR / "day_ahead_BE_2024_paper.csv",
}

FREQ_ALIASES = {
    "15min": 15,
    "15t": 15,
    "30min": 30,
    "30t": 30,
    "1h": 60,
    "h": 60,
    "60min": 60,
    "hourly": 60,
}


class MissingSourceError(RuntimeError):
    """Raised when a real data source is absent instead of silently faking it."""


class ExogenousLevelUnavailable(RuntimeError):
    """Raised when an exog level is n/a for a dataset (never replaced by a proxy)."""

    def __init__(self, level, reason, dataset=None):
        self.level = level
        self.reason = reason
        self.dataset = dataset
        super().__init__(
            f"Exogenous level {level} is not available"
            f"{f' for {dataset}' if dataset else ''}: {reason}"
        )


def minutes_per_step(freq):
    """Return the number of minutes in one step for a pandas-style freq string."""
    if freq is None:
        raise ValueError("freq is required; pass the series frequency, e.g. '15min'")
    key = str(freq).strip().lower()
    if key in FREQ_ALIASES:
        return FREQ_ALIASES[key]
    raise ValueError(f"Unsupported freq {freq!r}. Known: {sorted(FREQ_ALIASES)}")


def steps_per_hour(freq):
    minutes = minutes_per_step(freq)
    if 60 % minutes:
        raise ValueError(f"freq {freq!r} does not divide an hour evenly")
    return 60 // minutes


def steps_per_day(freq):
    return 24 * steps_per_hour(freq)


def horizon_steps(horizon_hours, freq):
    """Convert a wall-clock horizon to a number of steps at the given freq."""
    return int(round(horizon_hours * steps_per_hour(freq)))


def site_for(dataset_name):
    key = str(dataset_name).strip().lower()
    if key not in SITES:
        raise ValueError(f"Unknown dataset {dataset_name!r}. Known: {sorted(SITES)}")
    return SITES[key]
