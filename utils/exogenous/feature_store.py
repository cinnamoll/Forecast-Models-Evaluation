"""Unified exogenous feature store: two modes, ``L0`` and ``ALL``.

Design rules, in priority order:

1. A block is either built from a real source or reported as absent.  There is no
   synthetic fallback: a fabricated temperature column produces results that look
   like a weather ablation but measure a sine wave.
2. Time-based parameters are declared in hours and converted with the series' own
   frequency.  A "24 h mean" is 96 steps at 15 min but 48 steps at 30 min.
3. Everything is knowable at the forecast origin: calendar from the clock,
   weather from a reanalysis archive, price from the day-ahead auction.

There is no ladder of levels.  ``L0`` is the paper baseline (the original table's
calendar, byte-identical, kept for parity) and ``ALL`` is every covariate the
dataset has.  Which of those covariates earns its place is not answered by fixed
rungs but by the TimeSHAP elimination in ``scripts/xai_elimination.py``, which
drops one group at a time and records what each drop costs.  The blocks of ``ALL``
are always attempted, and one that cannot be built for this dataset/window is
recorded under ``report["unavailable"]`` instead of aborting the run.
"""

import inspect

import numpy as np
import pandas as pd
from pathlib import Path

from utils.exogenous.calendar_features import (
    build_l1_calendar_matrix,
    to_local_calendar,
)
from utils.exogenous.config import (
    DEFAULT_BASE_TEMPERATURE_C,
    DEFAULT_MAX_INTERPOLATION_HOURS,
    PAPER_PRICE_FILES,
    PRICE_ZONES,
    ExogenousLevelUnavailable,
    MissingSourceError,
    site_for,
    steps_per_hour,
)
from utils.exogenous.market_static_features import (
    controller_codes,
    get_static_household_metadata,
    load_controller_schedule,
    load_price_file,
    resample_price_forward_fill,
)
from utils.exogenous.sources import availability, entsoe, openmeteo
from utils.exogenous.weather_features import (
    calculate_solar_elevation,
    clear_sky_ghi,
    compute_apparent_temperature,
    compute_degree_days,
    resample_to_grid,
)

#: The only level that is a *level* any more: the paper baseline.  ``ALL`` is the
#: other mode (see ``ALL_COVARIATES_LEVEL``) and walks the whole cascade below.
SUPPORTED_EXOG_LEVELS = ("L0",)
LEVEL_ORDER = {level: i for i, level in enumerate(SUPPORTED_EXOG_LEVELS)}

#: "Give me every covariate this dataset has", not a rung of the ladder.  Every
#: stage below returns early only for its own level, so an unknown-but-allowed
#: value falls through the whole ladder and each block that can be built is
#: added; blocks that cannot are recorded as unavailable instead of raising,
#: because the point of this mode is one training run with everything.
ALL_COVARIATES_LEVEL = "ALL"

# Columns that carry a category, not a magnitude.  They are integer-coded so
# min-max normalisation and NeuralForecast/GluonTS categorical inputs both work.
CATEGORICAL_COLUMNS = (
    "static_controller_id",
    "static_tariff_type",
    "static_building_type",
    "static_heat_pump_model",
    "static_klanttypering",
)

# Grouping used by the XAI comparison (plan_updated.md section 9).
FEATURE_GROUPS = {
    "hour": {"hour", "quarter_hour", "hour_sin", "hour_cos", "is_weekend"},
    "day_of_week": {"dayofweek", "dayofweek_sin", "dayofweek_cos"},
    "month": {"month", "month_sin", "month_cos"},
    "calendar": {"is_holiday", "is_bridge_day", "is_dst"},
    "weather": {
        "temperature",
        "apparent_temperature",
        "hdd",
        "cdd",
        "ghi",
        "dni",
        "cloud_cover",
        "humidity",
        "wind_speed",
        "precipitation",
        "solar_elevation",
        "clear_sky_ghi",
        "temp_rolling_mean_24h",
        "temp_rolling_std_24h",
        "temp_rolling_mean_72h",
        "ghi_rolling_mean_24h",
    },
    "price": {"day_ahead_price"},
}

_METADATA_STRING_COLUMNS = {
    "controller_id": "static_controller_id",
    "tariff_type": "static_tariff_type",
    "building_type": "static_building_type",
    "heat_pump_model": "static_heat_pump_model",
    "klanttypering": "static_klanttypering",
}

# Metadata columns that describe the source rather than the household.  A citation
# string or a product name is not a feature and must not become a constant column.
# ``acorn_group`` sits here by decision, not by accident: it is a postcode-level
# geodemographic label (proxy for income and dwelling stock) and is not used as a
# covariate anywhere in the ladder, so a stray column in a metadata CSV is dropped
# and reported instead of silently becoming a feature.
NON_FEATURE_METADATA_KEYS = {"source", "bess_model", "inverter_model", "notes", "acorn_group"}


def build_exogenous_feature_dataframe(
    dataset_name,
    entity_id="default",
    datetime_sequence=None,
    exog_level="L0",
    forecast_horizon=None,
    freq=None,
    strict=True,
    return_report=False,
    task=None,
):
    """Build the exogenous block for one entity at one level.

    Parameters
    ----------
    forecast_horizon : int, optional
        Horizon in *steps* at ``freq``.  Required for L5 so the neighbour lag is
        at least one horizon long.
    freq : str, optional
        Series frequency, e.g. ``"15min"``.  Inferred from the index when omitted.
    strict : bool
        When True (default) an ``ALL`` run that could not build **any** covariate
        block raises ``ExogenousLevelUnavailable``, so a typo cannot quietly
        produce a baseline run labelled ``ALL``.  When False the run comes back
        with the calendar only and every reason in the report.  A proxy is never
        substituted either way.
    task : {"pv", "battery", "load"}, optional
        Which asset the target describes.  Accepted for the dataset loaders'
        signature; the only task-scoped block (measured plant state) was removed
        with L6.
    exog_level : str
        ``L0`` (paper baseline) or ``"ALL"`` (every covariate the dataset has).
        There is no ladder in between: which covariates earn their place is
        answered by the TimeSHAP elimination, not by fixed rungs.
    """
    if exog_level not in SUPPORTED_EXOG_LEVELS and exog_level != ALL_COVARIATES_LEVEL:
        from utils.protocol import BLOCKS_ONLY_IN_ALL

        block = BLOCKS_ONLY_IN_ALL.get(exog_level)
        detail = (f" It is a block inside ALL: {block}." if block else "") 
        raise ValueError(
            f"Unsupported exog_level {exog_level!r}.{detail}"
            f" Known: {SUPPORTED_EXOG_LEVELS + (ALL_COVARIATES_LEVEL,)}"
        )
    if datetime_sequence is None:
        raise ValueError("datetime_sequence is required")

    index = pd.DatetimeIndex(datetime_sequence)
    if index.empty:
        raise ValueError("datetime_sequence is empty")

    dataset_key = str(dataset_name).lower()
    site = site_for(dataset_key)
    if freq is None:
        freq = infer_freq(index)

    report = {"dataset": dataset_key, "entity_id": str(entity_id), "level": exog_level, "freq": freq}
    features = {}

    # ---------------------------------------------------------------- L0
    # Kept byte-identical to the paper's baseline calendar so that the L0 run
    # reproduces the original table (plan_updated.md section 0).
    features["quarter_hour"] = index.hour * 4 + index.minute // 15
    features["is_weekend"] = (index.dayofweek >= 5).astype(int)
    if exog_level == "L0":
        return _finish(features, index, report, return_report)

    # ------------------------------------------------------------ ALL: L1 block
    local_index = to_local_calendar(index, site["timezone"])
    calendar_block = build_l1_calendar_matrix(
        local_index, site["country"], site.get("subdivision"), timezone_name=site["timezone"]
    )
    calendar_block.pop("is_weekend", None)  # already emitted above
    calendar_block["hour"] = local_index.hour
    calendar_block["dayofweek"] = local_index.dayofweek
    calendar_block["month"] = local_index.month
    features.update(calendar_block)

    # School holidays are *not* a feature by decision.  The tables cover 2024
    # (Belgium, WPuQ) but not 2013 (LCL, Zonnedael), so the column would exist for
    # two datasets and be absent for the other two; a covariate set that differs
    # per dataset cannot be compared across datasets.  The table, its loader and
    # its availability check stay in ``calendar_features`` for the day all four
    # windows can be covered.
    report["school_holidays"] = ("excluded by design: the table covers 2024 only, "
                                 "not the 2013 datasets")

    # ------------------------------------------------------------ ALL: L2 block
    # A block that is genuinely unavailable is recorded and skipped, never filled
    # with a proxy: LCL and Zonnedael are 2013 and have no day-ahead market, and
    # that must not stop the weather block from being built.
    try:
        features.update(_build_weather(dataset_key, site, index, freq, report))
    except MissingSourceError as exc:
        report.setdefault("unavailable", {})["L2"] = str(exc)

    # ------------------------------------------------------------ ALL: L3 block
    try:
        features.update(_build_price(dataset_key, index, report))
    except MissingSourceError as exc:
        report.setdefault("unavailable", {})["L3"] = str(exc)

    # ------------------------------------------------------------ ALL: L4 block
    static_block = _build_static(dataset_key, entity_id, index, report)
    if static_block is None:
        report.setdefault("unavailable", {})["L4"] = report.get(
            "static_reason", "no static metadata")
    else:
        features.update(static_block)

    # L5 (neighbour aggregate) and L6 (measured plant state) were removed, not
    # omitted: L5 exists for two datasets and L6 for one, so neither could ever be
    # part of a comparison across datasets.  ``utils/exogenous/sources/neighbors.py``
    # and the L6 reader were taken out of the pipeline with them.

    # An ALL run that produced nothing but the calendar is a mistake, not a
    # result: name the reasons instead of shipping a baseline run under an ALL
    # label.  "Nothing but the calendar" means every covariate block failed, not
    # that few columns came back -- the calendar block itself is many columns.
    unavailable = report.get("unavailable", {})
    if strict and all(block in unavailable for block in ("L2", "L3", "L4")):
        reasons = "; ".join(f"{block}: {unavailable[block]}" for block in ("L2", "L3", "L4"))
        raise ExogenousLevelUnavailable("ALL", reasons)
    return _finish(features, index, report, return_report)


def infer_freq(index):
    """Infer a pandas freq string from the modal step of an index."""
    index = pd.DatetimeIndex(index)
    if len(index) < 2:
        raise ValueError("cannot infer freq from fewer than two timestamps")
    deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
    modal_seconds = float(pd.Series(deltas).mode().iloc[0])
    minutes = int(round(modal_seconds / 60.0))
    if minutes < 1:
        raise ValueError("sub-minute resolution is not supported")
    return f"{minutes}min"


def _build_weather(dataset_key, site, index, freq, report):
    hourly = openmeteo.load_hourly(dataset_key, index[0], index[-1])
    grid, imputation = resample_to_grid(hourly, index, freq, max_gap_hours=DEFAULT_MAX_INTERPOLATION_HOURS)

    report["weather_source"] = (
        f"Open-Meteo ERA5 archive, site '{site['site_id']}'"
        + (" (approximate location)" if site.get("approx") else "")
    )
    report["weather_imputed_fraction"] = {k: round(float(v), 5) for k, v in imputation.items()}
    report["weather_missing_fraction"] = {
        column: round(float(grid[column].isna().mean()), 5) for column in grid.columns
    }
    # Linear resampling brackets each grid point with the nearest hourly
    # observations, so a grid point up to one hour before the forecast origin can
    # lean on the next hourly value.  Reported explicitly instead of being hidden.
    report["weather_interpolation_bracket_hours"] = 1.0

    temperature = grid["temperature_2m"]
    humidity = grid["relative_humidity_2m"]
    wind = grid["wind_speed_10m"]

    hdd, cdd = compute_degree_days(temperature.tolist(), DEFAULT_BASE_TEMPERATURE_C)
    apparent = compute_apparent_temperature(temperature.tolist(), humidity.tolist(), wind.tolist())
    elevation = calculate_solar_elevation(index, latitude=site["latitude"], longitude=site["longitude"])

    per_hour = steps_per_hour(freq)
    mean_24, std_24 = _rolling_stats(temperature, 24 * per_hour)
    mean_72, _ = _rolling_stats(temperature, 72 * per_hour)
    ghi_24, _ = _rolling_stats(grid["shortwave_radiation"], 24 * per_hour)

    return {
        "temperature": temperature,
        "apparent_temperature": apparent,
        "hdd": hdd,
        "cdd": cdd,
        "ghi": grid["shortwave_radiation"],
        "dni": grid["direct_normal_irradiance"],
        "cloud_cover": grid["cloud_cover"],
        "humidity": humidity,
        "wind_speed": wind,
        "precipitation": grid["precipitation"],
        "solar_elevation": elevation,
        "clear_sky_ghi": clear_sky_ghi(elevation),
        "temp_rolling_mean_24h": mean_24,
        "temp_rolling_std_24h": std_24,
        "temp_rolling_mean_72h": mean_72,
        "ghi_rolling_mean_24h": ghi_24,
    }


def _rolling_stats(series, window_steps):
    """Causal rolling mean/std; pandas rolling is right-aligned, so no look-ahead."""
    window_steps = max(int(window_steps), 1)
    mean = series.rolling(window_steps, min_periods=1).mean()
    std = series.rolling(window_steps, min_periods=1).std(ddof=1)
    return mean.to_numpy(), std.to_numpy()


def _build_price(dataset_key, index, report):
    zone = PRICE_ZONES.get(dataset_key)
    if zone is None:
        raise MissingSourceError(
            f"no day-ahead market is defined for {dataset_key}: L3 is n/a for this task "
            "(the 2013 datasets have no traded day-ahead series)"
        )
    zone_code = zone[0]
    hourly, source = _price_series(dataset_key, zone_code, index)
    # Pad inside the year only: a window starting on 1 January must not require
    # the previous year's price file just to forward-fill the first hour.
    hourly = hourly.loc[: index[-1]]
    values = resample_price_forward_fill(hourly.index.to_numpy(), hourly.to_numpy(), index.to_numpy())
    series = pd.Series(values, index=index, dtype="float64")

    missing = float(series.isna().mean())
    if missing > 0.01:
        raise MissingSourceError(f"day-ahead prices for {zone_code} leave {missing:.1%} of the grid uncovered")

    report["price_source"] = source
    report["price_missing_fraction"] = round(missing, 6)
    return {"day_ahead_price": series}


def _price_series(dataset_key, zone_code, index):
    """The dataset's own price file when its release ships one, else ENTSO-E A44.

    Belgium's release quotes the market series the source experiment was billed
    against (Ruddick et al. 2024, section 3.2.3), which removes the API token and
    the network from the L3 build.  Both routes yield the same auction result in
    the same unit, so the reported source says which one was read.
    """
    local = PAPER_PRICE_FILES.get(dataset_key)
    if local is not None and Path(local).exists():
        return (
            load_price_file(local, index[0], index[-1]),
            f"dataset release {Path(local).name} (Ruddick et al. 2024, section 3.2.3)",
        )
    return (
        entsoe.load_hourly(zone_code, index[0], index[-1]),
        f"ENTSO-E A44 bidding zone {zone_code}",
    )


def _build_static(dataset_key, entity_id, index, report):
    meta = get_static_household_metadata(dataset_key, entity_id)
    report["static_source"] = meta.source
    report["static_missing"] = meta.missing
    if not meta.available:
        report["static_reason"] = meta.missing.get(
            "all",
            f"no usable static column for {dataset_key}/{entity_id} (missing: {sorted(meta.missing) or 'none'})",
        )
        return None

    block = {}
    columns = []
    skipped = []
    for key, value in meta.values.items():
        if key in NON_FEATURE_METADATA_KEYS:
            skipped.append(key)
            continue
        column = _METADATA_STRING_COLUMNS.get(key, f"static_{key}")
        if isinstance(value, str):
            block[column] = _encode_categorical(column, value)
        else:
            block[column] = np.full(len(index), float(value))
        columns.append(column)
    if skipped:
        report["static_not_features"] = skipped

    schedule = load_controller_schedule(dataset_key)
    if schedule is not None:
        codes = controller_codes(schedule, index, entity_id)
        if codes is None:
            # The schedule exists but covers none of this window (the Belgium
            # release only ships the Apr-Jun experiment, not the Jan-Mar window
            # the benchmark forecasts).  n/a, never a constant column.
            report["static_missing"] = {
                **meta.missing,
                "controller_id": f"schedule covers none of {index[0]} .. {index[-1]}",
            }
        else:
            block["static_controller_id"] = _encode_categorical("static_controller_id", codes)
            columns.append("static_controller_id")
            coverage = sum(code is not None for code in codes) / len(codes)
            report["controller_coverage"] = round(float(coverage), 4)
            if coverage < 1.0:
                report.setdefault("static_missing", {})["controller_id_uncovered"] = (
                    f"{(1 - coverage):.1%} of the window has no controller in the schedule"
                )
    else:
        report["static_missing"] = {
            **meta.missing,
            "controller_id": "EMS switches daily at 15:00; schedule file not built",
        }

    report["static_columns"] = columns
    return block


def _encode_categorical(column, values):
    """Deterministic integer coding (sorted vocabulary) so codes are reproducible."""
    if isinstance(values, str):
        values = [values]
    unique = sorted({str(v) for v in values if v is not None})
    mapping = {name: code for code, name in enumerate(unique)}
    return np.array([mapping.get(str(v), -1) for v in values], dtype="float64")


def _finish(features, index, report, return_report):
    frame = pd.DataFrame(features, index=index)
    return (frame, report) if return_report else frame


def xai_feature_groups(columns):
    """Map every column to an XAI group; unknown columns fall back to their prefix."""
    groups = {}
    for column in columns:
        for group, members in FEATURE_GROUPS.items():
            if column in members:
                groups[column] = group
                break
        else:
            if column.startswith("neighbor_"):
                groups[column] = "neighbor"
            elif column.startswith("static_"):
                groups[column] = "metadata"
            else:
                groups[column] = "other"
    return groups


# Columns that are known over the whole horizon (calendar, prices, clear-sky
# geometry) versus columns that only exist up to the forecast origin (measured
# weather, lagged neighbours).  Getting this split wrong is how future target
# information leaks into a covariate model.
FUTR_EXOG_COLUMNS = {
    "quarter_hour",
    "is_weekend",
    "hour",
    "dayofweek",
    "month",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "is_holiday",
    "is_bridge_day",
    "is_dst",
    "is_school_holiday",
    "day_ahead_price",
    "solar_elevation",
    "clear_sky_ghi",
}


def partition_for_neuralforecast(columns):
    """Split feature columns into NeuralForecast's futr / hist / stat lists."""
    futr, hist, stat = [], [], []
    for column in columns:
        if column in FUTR_EXOG_COLUMNS:
            futr.append(column)
        elif column.startswith("static_"):
            stat.append(column)
        else:
            hist.append(column)
    return {"futr_exog_list": futr, "hist_exog_list": hist, "stat_exog_list": stat}


#: NeuralForecast 3.x declares per-class Boolean support flags; the base model
#: raises when a list is passed for a kind the class does not implement ("PatchTST
#: does not support future exogenous variables").  The lists must therefore be
#: filtered by capability, not by the constructor signature: PatchTST *inherits*
#: ``futr_exog_list`` in its signature and still refuses it.
NEURALFORECAST_EXOG_FLAGS = {
    "futr_exog_list": "EXOGENOUS_FUTR",
    "hist_exog_list": "EXOGENOUS_HIST",
    "stat_exog_list": "EXOGENOUS_STAT",
}


def neuralforecast_exog_capabilities(model_class):
    """``{list name: bool}`` support flags of one NeuralForecast class, or None.

    ``None`` means the class does not declare the flags (a different build), in
    which case the caller falls back to the constructor signature.
    """
    flags = {}
    for list_name, attribute in NEURALFORECAST_EXOG_FLAGS.items():
        value = getattr(model_class, attribute, None)
        if value is None:
            return None
        flags[list_name] = bool(value)
    return flags


def partition_for_model(model_class, columns):
    """Covariate kwargs for one NeuralForecast class, and what it had to drop.

    Three outcomes, all recorded rather than silent:

    * a kind the class supports is passed as that kind;
    * known-future columns become ``hist_exog_list`` when the class takes
      historical but not future covariates — the input window is entirely before
      the forecast origin, so they stay causal while still being usable;
    * a kind the class cannot take at all is dropped and listed in the report,
      because a model with no covariate channel would otherwise look like a model
      that used covariates and gained nothing.
    """
    lists = partition_for_neuralforecast(columns)
    capabilities = neuralforecast_exog_capabilities(model_class)
    if capabilities is None:  # unknown build: keep the old signature check
        try:
            accepted = set(inspect.signature(model_class.__init__).parameters)
        except (TypeError, ValueError):
            accepted = set()
        kwargs = {key: value for key, value in lists.items() if value and key in accepted}
        dropped = {key: value for key, value in lists.items() if value and key not in accepted}
        return kwargs, {"mode": "signature", "accepted": sorted(kwargs), "dropped": dropped}

    kwargs, dropped = {}, {}
    hist = list(lists["hist_exog_list"])
    futr = list(lists["futr_exog_list"])
    if capabilities["hist_exog_list"]:
        if not capabilities["futr_exog_list"] and futr:
            hist = hist + futr  # usable, just only over the input window
            futr = []
    elif futr and capabilities["futr_exog_list"]:
        dropped["hist_exog_list"] = hist
        hist = []

    if hist and capabilities["hist_exog_list"]:
        kwargs["hist_exog_list"] = hist
    elif hist:
        dropped["hist_exog_list"] = hist
    if futr and capabilities["futr_exog_list"]:
        kwargs["futr_exog_list"] = futr
    elif futr:
        dropped["futr_exog_list"] = futr
    if lists["stat_exog_list"] and capabilities["stat_exog_list"]:
        kwargs["stat_exog_list"] = lists["stat_exog_list"]
    elif lists["stat_exog_list"]:
        dropped["stat_exog_list"] = lists["stat_exog_list"]

    return kwargs, {
        "mode": "capability",
        "capabilities": capabilities,
        "accepted": sorted(kwargs),
        "dropped": dropped,
        "n_dropped": sum(len(v) for v in dropped.values()),
    }


def static_summary(frame, columns, unique_id="unique_id"):
    """One modal value per series and static column, plus the non-constant ones.

    NeuralForecast and GluonTS both want one row per series for static inputs, and
    taking ``.first()`` would silently turn a daily-changing controller id into a
    constant.  The mode is the value the series spends most of its time in; a
    column that is not constant is returned in the second element so the caller
    can log that the "static" assumption does not hold for it.
    """
    if not columns:
        return None, []
    grouped = frame.groupby(unique_id)
    summary = pd.DataFrame(index=grouped.size().index)
    varying = []
    for column in columns:
        modes, spreads = [], []
        for _, block in grouped[column]:
            values = pd.Series(block).dropna()
            modes.append(values.mode().iloc[0] if not values.empty else np.nan)
            spreads.append(values.nunique())
        summary[column] = modes
        if any(spread > 1 for spread in spreads):
            varying.append(column)
    summary.index.name = unique_id
    return summary.reset_index(), varying


# Columns whose value is a category, not a magnitude.  Chronos-2 and Timestral
# both target-encode categorical rows for a univariate target, which is how the
# holiday/weekend information survives (plan_updated.md section 3, point 3).
CATEGORICAL_EXOG_COLUMNS = {
    "quarter_hour",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    "is_bridge_day",
    "is_dst",
    "is_school_holiday",
}


def is_categorical_column(column):
    return column in CATEGORICAL_EXOG_COLUMNS or column in CATEGORICAL_COLUMNS


def covariate_roles(columns):
    """Role split used by the TSFM wrappers (Chronos-2, Timestral).

    ``known_future`` covariates are known over the horizon (calendar, day-ahead
    price, clear-sky geometry).  ``past_only`` covariates stop at the forecast
    origin (measured weather, lagged neighbours, static metadata).  ``categorical``
    lists the subset that must be passed as category codes rather than numbers.
    """
    known_future, past_only, categorical = [], [], []
    for column in columns:
        if column in FUTR_EXOG_COLUMNS:
            known_future.append(column)
        else:
            past_only.append(column)
        if is_categorical_column(column):
            categorical.append(column)
    return {
        "known_future": known_future,
        "past_only": past_only,
        "categorical": categorical,
    }



def availability_report(dataset_name, entity_id, start, end):
    """dict level -> (available, detail) for the level table in the paper."""
    return availability.report(str(dataset_name).lower(), entity_id, start, end)
