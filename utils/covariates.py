"""Exogenous covariates for the six improved models, and their roles.

The improvement this module implements is one block of **calendar** features that
every model receives, plus — where the dataset has them — measured weather, the
day-ahead price, static hardware metadata and the BESS state of charge:

    calendar   quarter_hour, hour, minute, dayofweek, is_weekend
    weather    temperature, humidity, cloud_cover, ghi, dni, wind_speed, precipitation
    price      day_ahead_price            (Belgium only)
    static     static_<column>            (Belgium only, one value per house)
    state      soc                        (Belgium BESS only)

The calendar columns use exactly the definitions of the paper's loaders
(``utils/dataset_config.py``, ``quarter_hour = hour * 4 + minute // 15``), so the
baseline of the original table is reconstructed when ``level="L0"``.

Role split (the part that decides whether a model leaks):

* ``known_future`` — calendar and day-ahead price.  The clock is known in advance
  and the auction clears a day ahead, so passing the horizon values is correct.
* ``past_only`` — weather and state of charge.  These are *measured*, so no value
  after the forecast origin may enter the model; they are offered over the
  context only.  The horizon therefore behaves like persistence even though no
  model is ever handed a persisted number: the channel simply stops at the
  origin.
* ``static`` — hardware metadata, constant per entity.

Data sources, all under ``data/exogenous``:

* ``weather/<site>.csv`` — hourly reanalysis (Open-Meteo ERA5 archive), one file
  per dataset.  Hourly values are linearly interpolated onto the series grid;
  interpolation only ever brackets two *observations*, and nothing after the
  forecast origin is used, so it does not create look-ahead.
* ``Belgium/<house dir>/day_ahead_prices.csv`` — 15-minute, EUR/kWh.
* ``Belgium/<house dir>/soc.csv`` — 15-minute state of charge.  Readings outside
  ``[0, 1]`` are sensor glitches and are treated as missing.
* ``Belgium/belgium_hardware.csv`` — one row per house.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXOGENOUS_DIR = PROJECT_ROOT / "data" / "exogenous"

#: dataset -> weather file under ``data/exogenous/weather``.
WEATHER_FILES = {
    "belgium": "brussels_2024.csv",
    "germany": "lower_saxony_approx_2024.csv",
    "london": "london_2013.csv",
    "zonnedael": "zonnedael_approx_2013.csv",
}

#: Raw weather columns -> feature names used by every model.
WEATHER_COLUMNS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "cloud_cover": "cloud_cover",
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "wind_speed_10m": "wind_speed",
    "precipitation": "precipitation",
}

PRICE_COLUMN = "day_ahead_price"
SOC_COLUMN = "soc"

#: Static hardware columns of the Belgium release.  ``bess_model`` and
#: ``ac_phase`` are strings and are integer-coded like the calendar columns.
STATIC_NUMERIC = (
    "pv_peak_kwp",
    "bess_energy_kwh",
    "bess_charge_kw",
    "bess_discharge_kw",
    "inverter_max_ac_kw",
    "house_breaker_kva",
)
STATIC_CATEGORICAL = ("bess_model", "ac_phase")

CALENDAR_COLUMNS = ("quarter_hour", "hour", "minute", "dayofweek", "is_weekend")

#: Columns that may be handed to a model as observed-in-advance values.
KNOWN_FUTURE_COLUMNS = CALENDAR_COLUMNS + (PRICE_COLUMN,)
#: Codes, embedded or target-encoded by the model rather than scaled.  Static
#: columns are *not* categorical as a class: the hardware table is mostly
#: physical ratings (kWp, kWh), and embedding a rating would throw away its order.
CATEGORICAL_COLUMNS = CALENDAR_COLUMNS + ("static_ac_phase", "static_bess_model")

#: Blocks used by the TimeSHAP player set.  Grouping is physical, not syntactic:
#: ``quarter_hour`` is ``hour * 4 + minute // 15``, so the three time-of-day
#: columns carry the same information and must not be scored as three
#: independent players (Shapley would split the credit arbitrarily between them).
FEATURE_GROUPS = {
    "time_of_day": ("quarter_hour", "hour", "minute"),
    "day_of_week": ("dayofweek", "is_weekend"),
    "temperature": ("temperature",),
    "radiation": ("ghi", "dni", "cloud_cover"),
    "air": ("humidity", "wind_speed", "precipitation"),
    "price": (PRICE_COLUMN,),
    "state": (SOC_COLUMN,),
}

GROUP_ORDER = ("time_of_day", "day_of_week", "temperature", "radiation", "air", "price", "state", "static")


# --------------------------------------------------------------------- calendar

def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """The paper's day/hour block, one column per feature.

    Definitions are copied from ``utils/dataset_config.py`` so the ``L0`` run
    reproduces the original design: ``quarter_hour`` is the index of the quarter
    hour inside the day, ``is_weekend`` flags Saturday and Sunday.
    """
    index = pd.DatetimeIndex(index)
    hour = index.hour.to_numpy()
    minute = index.minute.to_numpy()
    dayofweek = index.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "quarter_hour": hour * 4 + minute // 15,
            "hour": hour,
            "minute": minute,
            "dayofweek": dayofweek,
            "is_weekend": (dayofweek >= 5).astype(int),
        },
        index=index,
    )


# ---------------------------------------------------------------------- weather

def _naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Drop the timezone: the exogenous archives are stored on wall clock.

    The paper's loaders treat every dataset on its own wall clock (London and
    Zonnedael are local 2013 timestamps), so the exogenous files are joined the
    same way instead of being shifted by a timezone rule they never had.
    """
    index = pd.DatetimeIndex(index)
    return index.tz_localize(None) if index.tz is not None else index


def weather_frame(dataset: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly reanalysis interpolated onto ``index``, one column per feature."""
    path = EXOGENOUS_DIR / "weather" / WEATHER_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"weather archive missing for {dataset}: {path}")
    raw = pd.read_csv(path, parse_dates=["time"])
    raw = raw.set_index("time").sort_index()
    present = [c for c in WEATHER_COLUMNS if c in raw.columns]
    if not present:
        raise ValueError(f"{path.name} carries none of {sorted(WEATHER_COLUMNS)}")

    target = _naive(index)
    grid = raw[present].reindex(raw.index.union(target)).interpolate(method="time")
    values = grid.reindex(target)[present].rename(columns=WEATHER_COLUMNS)
    # Edges only: a linear gap in the middle of the archive stays NaN so a
    # missing measurement can never be disguised as data.
    values = values.ffill().bfill()
    if values.isna().any().any():
        raise ValueError(f"{path.name} leaves NaN inside {target[0]} .. {target[-1]}")
    values.index = pd.DatetimeIndex(index)
    return values


# ------------------------------------------------------------------------ price

def _belgium_house_dir(house: int) -> Path:
    root = EXOGENOUS_DIR / "Belgium"
    for candidate in sorted(root.glob(f"house{house}_*")):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no Belgium exogenous directory for house {house} under {root}")


def price_frame(dataset: str, house: int | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Day-ahead price aligned to ``index``.

    Only Belgium has a traded day-ahead series in this release; the 2013 datasets
    do not, and returning an empty frame is the honest answer rather than a
    silent zero column.
    """
    if dataset != "belgium" or house is None:
        return pd.DataFrame(index=index)
    path = _belgium_house_dir(house) / "day_ahead_prices.csv"
    if not path.exists():
        return pd.DataFrame(index=index)
    raw = pd.read_csv(path, parse_dates=["datetime"])
    column = [c for c in raw.columns if c != "datetime"][0]
    series = raw.set_index("datetime")[column].sort_index().astype(float)
    target = _naive(index)
    # The auction price is constant per hour: hold the last cleared value rather
    # than interpolate, which would invent prices between two auctions.
    aligned = series.reindex(series.index.union(target)).ffill().bfill().reindex(target)
    frame = pd.DataFrame({PRICE_COLUMN: aligned.to_numpy()}, index=pd.DatetimeIndex(index))
    return frame


# ------------------------------------------------------------------ state of charge

def soc_frame(dataset: str, house: int | None, task: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """BESS state of charge, BESS tasks only.

    Readings outside [0, 1] are sensor glitches (the released file reaches 6.5);
    they are treated as missing and carried forward, which is what a controller
    would see.  The column is past-only: the state at the origin is known, the
    state inside the horizon is not.
    """
    if dataset != "belgium" or house is None or task != "battery":
        return pd.DataFrame(index=index)
    path = _belgium_house_dir(house) / "soc.csv"
    if not path.exists():
        return pd.DataFrame(index=index)
    raw = pd.read_csv(path, parse_dates=["datetime"])
    column = [c for c in raw.columns if c != "datetime"][0]
    series = raw.set_index("datetime")[column].astype(float).sort_index()
    series = series.where(series.between(0.0, 1.0))
    target = _naive(index)
    aligned = series.reindex(series.index.union(target)).ffill().reindex(target)
    if aligned.isna().all():
        return pd.DataFrame(index=index)
    aligned = aligned.bfill()
    return pd.DataFrame({SOC_COLUMN: aligned.to_numpy()}, index=pd.DatetimeIndex(index))


# ----------------------------------------------------------------------- static

def static_frame(dataset: str, entity: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hardware metadata of one entity, broadcast over the series."""
    if dataset != "belgium":
        return pd.DataFrame(index=index)
    path = EXOGENOUS_DIR / "Belgium" / "belgium_hardware.csv"
    if not path.exists():
        return pd.DataFrame(index=index)
    table = pd.read_csv(path).set_index("entity_id")
    if entity not in table.index:
        return pd.DataFrame(index=index)
    row = table.loc[entity]
    values = {}
    for column in STATIC_NUMERIC:
        if column in row.index and pd.notna(row[column]):
            values[f"static_{column}"] = np.full(len(index), float(row[column]))
    for column in STATIC_CATEGORICAL:
        if column in row.index and pd.notna(row[column]):
            values[f"static_{column}"] = np.full(len(index), float(_code(str(row[column]))))
    if not values:
        return pd.DataFrame(index=index)
    return pd.DataFrame(values, index=pd.DatetimeIndex(index))


def _code(value: str, vocabulary: tuple[str, ...] | None = None) -> int:
    """Stable integer code of one categorical value (sorted vocabulary)."""
    vocabulary = vocabulary or ("mono", "three")
    ordered = tuple(sorted(set(vocabulary) | {value}))
    return ordered.index(value)


# ------------------------------------------------------------------- assembly

def roles_for(frame: pd.DataFrame) -> dict:
    """Split the covariate columns of ``frame`` by role."""
    known_future, past_only, static, categorical = [], [], [], []
    for column in frame.columns:
        if column.startswith("static_"):
            static.append(column)
        elif column in KNOWN_FUTURE_COLUMNS:
            known_future.append(column)
        else:
            past_only.append(column)
        if column in CATEGORICAL_COLUMNS:
            categorical.append(column)
    return {
        "known_future": known_future,
        "past_only": past_only,
        "static": static,
        "categorical": [c for c in categorical if c in frame.columns],
    }


def build_covariates(dataset: str, entity: str, index: pd.DatetimeIndex, task: str,
                     level: str = "COV", house: int | None = None):
    """Covariate frame aligned to ``index`` plus its role split.

    ``level="L0"`` returns the paper's baseline: the calendar block only, and for
    a direct multi-step regressor the two columns the original loaders used
    (``quarter_hour``, plus ``is_weekend`` for Zonnedael).  ``level="COV"``
    returns everything the dataset has.
    """
    index = pd.DatetimeIndex(index)
    if level not in ("L0", "COV"):
        raise ValueError(f"unknown covariate level {level!r}; known: ('L0', 'COV')")

    blocks = [calendar_frame(index)]
    if level == "COV":
        blocks.append(weather_frame(dataset, index))
        blocks.append(price_frame(dataset, house, index))
        blocks.append(soc_frame(dataset, house, task, index))
        blocks.append(static_frame(dataset, entity, index))

    frame = pd.concat(blocks, axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.index = index
    frame.index.name = "ds"
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"covariates {missing} carry NaN for {dataset}/{entity}")
    return frame, roles_for(frame)


def scale_covariates(frame: pd.DataFrame, roles: dict, train_end: int) -> pd.DataFrame:
    """Min-max scale continuous covariates to [0, 1] on the training rows.

    Neural models need bounded inputs; a raw ``quarter_hour`` in 0..95 next to a
    temperature of 5 C would otherwise dominate the first layer.  The bounds are
    taken from rows ``< train_end`` only, so the scaling cannot import the test
    range.  Categorical codes are left as integers because the models embed them.
    """
    scaled = frame.copy()
    for column in frame.columns:
        if column in roles["categorical"]:
            continue
        train = frame[column].iloc[:train_end]
        low, high = float(train.min()), float(train.max())
        span = high - low
        if span <= 0:
            scaled[column] = 0.0
        else:
            scaled[column] = ((frame[column] - low) / span).clip(0.0, 1.0)
    return scaled


def shift_past_only(frame: pd.DataFrame, roles: dict, points: int) -> pd.DataFrame:
    """Lag every past-only column by ``points`` steps.

    A direct multi-step regressor predicts row ``t`` from row ``t``'s covariates,
    so a measured covariate must be the value a forecaster had at the origin:
    ``t - horizon``.  Without the shift the model would read tomorrow's weather
    while predicting tomorrow.
    """
    if points <= 0 or not roles["past_only"]:
        return frame
    shifted = frame.copy()
    for column in roles["past_only"]:
        shifted[column] = frame[column].shift(int(points)).bfill()
    return shifted


def column_groups(columns) -> dict:
    """``column -> block name`` for the TimeSHAP player set."""
    groups = {}
    for column in columns:
        if column.startswith("static_"):
            groups[column] = "static"
            continue
        for group, members in FEATURE_GROUPS.items():
            if column in members:
                groups[column] = group
                break
        else:
            groups[column] = "other"
    return groups


def group_columns(columns) -> dict:
    """``block name -> [columns]``, in ``GROUP_ORDER``."""
    per_column = column_groups(columns)
    grouped = {}
    for column, group in per_column.items():
        grouped.setdefault(group, []).append(column)
    ordered = {name: grouped[name] for name in GROUP_ORDER if name in grouped}
    for name in sorted(set(grouped) - set(ordered)):
        ordered[name] = grouped[name]
    return ordered
