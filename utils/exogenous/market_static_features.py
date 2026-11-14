"""Market price (L3), static metadata (L4) and lagged neighbours (L5).

Everything here reads a real file.  When a file is absent the function returns
``None`` plus a reason and the feature store reports the level as n/a; it never
invents a stand-in value (an earlier version hard-coded PV/BESS sizes that did
not match the source paper, and zero-filled neighbours).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.exogenous.config import METADATA_DIR, MissingSourceError

DEFAULT_OFFTAKE_SURCHARGE = 0.114  # EUR/kWh, Ruddick et al. (2024) Eq. (4)
VAT_RATE = 0.06

# Columns that must never become features, even when a metadata file carries
# them.  ACORN is a postcode-level geodemographic label: it proxies income and
# dwelling stock rather than describing the connection itself, and the LCL
# download does not ship it.  Decision: ACORN is not a covariate (plan_updated.md
# section 6, L4), so it is dropped here instead of being reported as missing.
EXCLUDED_STATIC_KEYS = {"acorn_group"}

METADATA_FILES = {
    "belgium": "belgium_hardware.csv",
    "london": "lcl_households.csv",
    "zonnedael": "zonnedael_customers.csv",
    "belgium_controllers": "belgium_controllers.csv",
}


@dataclass
class StaticMetadata:
    """Result of an L4 lookup: what exists, and what is genuinely missing."""

    values: dict = field(default_factory=dict)
    source: str = ""
    missing: dict = field(default_factory=dict)

    @property
    def available(self):
        return bool(self.values)


def resample_price_forward_fill(hourly_timestamps, hourly_prices, target_timestamps):
    """Forward-fill hourly prices onto the series grid.

    Strictly step-wise: a price that is published per hour applies to the whole
    hour, so linear interpolation between two hourly prices would invent a price
    the market never quoted.
    """
    if hourly_timestamps is None or hourly_prices is None:
        return [None] * len(target_timestamps)

    price_frame = pd.DataFrame(
        {
            "time": pd.to_datetime(pd.Series(hourly_timestamps)),
            "price": pd.Series(hourly_prices, dtype="float64"),
        }
    ).sort_values("time")
    price_frame = price_frame[~price_frame["time"].duplicated(keep="last")]

    target = pd.DataFrame({"time": pd.to_datetime(pd.Series(target_timestamps)).values})
    merged = pd.merge_asof(target, price_frame, on="time", direction="backward")
    return merged["price"].tolist()


def price_is_available(hourly_prices, target_timestamps):
    """True when the price series covers the first target timestamp."""
    if hourly_prices is None or len(hourly_prices) == 0:
        return False
    first_valid = pd.to_datetime(pd.Series(hourly_prices).dropna().index) if isinstance(hourly_prices, pd.Series) else None
    if len(target_timestamps) == 0:
        return True
    return first_valid is not None and len(first_valid) > 0


def load_price_file(path, start, end):
    """Hourly day-ahead prices (EUR/MWh) from a local CSV built by the fetch scripts.

    The schema is the one the ENTSO-E cache writes (``time,price_eur_mwh``), so a
    dataset that ships its own series plugs into the same resampling step as the
    API download.  Raises ``MissingSourceError`` when the file starts after
    ``start``: forward-filling the first hours from nothing would shift the whole
    price curve relative to the load it is meant to explain.
    """
    frame = pd.read_csv(path, parse_dates=["time"])
    if frame.empty:
        raise MissingSourceError(f"{path} holds no price rows")
    frame = frame.sort_values("time").drop_duplicates(subset="time", keep="last")

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if frame["time"].min() > start_ts:
        raise MissingSourceError(
            f"prices in {path} start at {frame['time'].min()}, after {start_ts}. "
            "Rebuild the file with scripts/fetch/build_belgium_covariates.py"
        )
    return frame.set_index("time")["price_eur_mwh"].loc[:end_ts]


def load_metadata_table(key):
    """Read one metadata CSV, or ``None`` when the file has not been built yet."""
    filename = METADATA_FILES.get(key, f"{key}.csv")
    path = METADATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def _row_for(table, entity_id, id_column):
    if table is None or id_column not in table.columns:
        return None
    match = table[table[id_column].astype(str) == str(entity_id)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def get_static_household_metadata(dataset_name, entity_id="default"):
    """L4 static description of the target entity, from the real tables only.

    * Belgium: PV peak power, BESS energy and power limits, inverter limits.
      Numbers come from Table A.6/A.7 of Ruddick et al., Energy and AI 18 (2024)
      100448 (``papers/1-s2.0-S2666546824001149-main.pdf``), written to
      ``data/exogenous/metadata/belgium_hardware.csv``.
    * London: tariff code (``stdorToU``) per household, extracted from the LCL
      extract.  ACORN is deliberately NOT a covariate (design decision): it is a
      postcode-level geodemographic label, i.e. a proxy for income and dwelling
      stock rather than a property of the connection, and it is not part of the
      LCL download.  When a metadata file does carry an ``acorn_group`` column it
      is dropped, never emitted as a feature.
    * Zonnedael: ``Klanttypering`` per customer, if the Liander table is present.
    * Germany: no published per-house metadata in the repo -> n/a.
    """
    dataset_key = str(dataset_name).lower()

    if "belgium" in dataset_key:
        table = load_metadata_table("belgium")
        row = _row_for(table, entity_id, "entity_id")
        if row is None:
            return StaticMetadata(missing={"all": "data/exogenous/metadata/belgium_hardware.csv has no row"})
        values = {
            k: v
            for k, v in row.items()
            if k not in ("entity_id", *EXCLUDED_STATIC_KEYS) and not pd.isna(v)
        }
        return StaticMetadata(values=values, source="Ruddick et al. 2024, Tables A.6-A.7")

    if "london" in dataset_key:
        table = load_metadata_table("london")
        row = _row_for(table, entity_id, "entity_id")
        if row is None:
            values = {}
            missing = {"tariff_type": "no row in lcl_households.csv"}
        else:
            values = {
                k: v
                for k, v in row.items()
                if k not in ("entity_id", *EXCLUDED_STATIC_KEYS) and not pd.isna(v)
            }
            missing = {}
        return StaticMetadata(
            values=values,
            source="LCL smart-meter extract (stdorToU)",
            missing=missing,
        )

    if "zonnedael" in dataset_key:
        table = load_metadata_table("zonnedael")
        row = _row_for(table, entity_id, "entity_id")
        if row is None:
            return StaticMetadata(missing={"all": "no row in zonnedael_customers.csv (Klanttypering not in the repo file)"})
        values = {
            k: v
            for k, v in row.items()
            if k not in ("entity_id", *EXCLUDED_STATIC_KEYS) and not pd.isna(v)
        }
        return StaticMetadata(values=values, source="Liander Zonnedael metadata table")

    if "germany" in dataset_key:
        return StaticMetadata(missing={"all": "WPuQ ships a single house and no per-house metadata in this repo"})

    return StaticMetadata(missing={"all": f"unknown dataset {dataset_name}"})


def load_controller_schedule(dataset_name):
    """Belgium EMS controller over time, or ``None`` when no schedule is used.

    The paper switches EMS every 24 h at 15:00 (Ruddick et al. 2024, section
    3.2), so the controller is *not* a static attribute of a house.  It is also
    not a covariate of the benchmark: the schedule only exists for the release's
    experiment window (2024-04-08 .. 2024-06-17) while the benchmark forecasts
    2024-01-01 .. 2024-04-01, so no file is built and the feature is n/a rather
    than a wrong per-house constant.  The reader is kept so that a schedule
    covering the benchmark window would be used unchanged if one ever exists.
    """
    if "belgium" not in str(dataset_name).lower():
        return None
    table = load_metadata_table("belgium_controllers")
    if table is None:
        return None
    for column in ("start", "end", "controller"):
        if column not in table.columns:
            raise ValueError(f"belgium_controllers.csv is missing column {column!r}")
    table["start"] = pd.to_datetime(table["start"])
    table["end"] = pd.to_datetime(table["end"])
    if table.empty:
        return None
    return table


def controller_codes(schedule, timestamps, entity_id):
    """Map each timestamp to the controller active then, or ``None`` when uncovered.

    Returns ``None`` when the schedule covers none of the window: an all-``-1``
    column would enter the regression matrix as a constant "unknown" covariate,
    which is worse than reporting the level as n/a.  A partially covered window
    keeps the codes and leaves the holes at ``-1``; the caller reports the
    coverage so the table in the paper states it.
    """
    if schedule is None:
        return None
    subset = schedule
    if "entity_id" in subset.columns:
        subset = subset[subset["entity_id"].astype(str) == str(entity_id)]
    if subset.empty:
        return None
    codes = []
    for ts in pd.DatetimeIndex(timestamps):
        row = subset[(subset["start"] <= ts) & (ts < subset["end"])]
        codes.append(row.iloc[0]["controller"] if len(row) else None)
    if not any(code is not None for code in codes):
        return None
    return codes


def build_lagged_neighbor_features(neighbor_series_dict, lag_steps):
    """Lag every neighbour series by ``lag_steps`` rows (lag >= horizon).

    At row ``t`` the feature is ``neighbour[t - lag_steps]``.  With
    ``lag_steps`` equal to the forecast horizon, every value used for a forecast
    issued at ``t0`` is dated at or before ``t0``, so the feature cannot see any
    part of the window being predicted.
    """
    if lag_steps < 1:
        raise ValueError("lag_steps must be >= 1")
    lagged_features = {}
    for name, series in neighbor_series_dict.items():
        values = np.asarray(series, dtype="float64")
        lagged = np.full(values.shape, np.nan)
        if len(values) > lag_steps:
            lagged[lag_steps:] = values[:-lag_steps]
        lagged_features[f"{name}_lag{lag_steps}"] = lagged
    return lagged_features
