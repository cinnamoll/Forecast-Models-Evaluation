import pandas as pd
import numpy as np
from pathlib import Path
from utils.metrics import min_max_normalize
from scipy import stats
from utils.exogenous.feature_store import (
    FUTR_EXOG_COLUMNS,
    build_exogenous_feature_dataframe,
)


EXOG_HORIZON_POINTS = 192  # horizon of the paper's protocol, in points

# Direct multi-step models (LightGBM, KNN) get one row per target step, so a
# column whose value at t is only observed at t is a leak when t lies in the
# test window.  Columns outside FUTR_EXOG_COLUMNS (measured weather, lagged
# neighbours) are therefore shifted back by one full horizon before they reach
# the regression matrix: the row for t sees the value from t - 192 points, which
# is the freshest value a forecaster has at the origin of a two-day horizon.
EXOG_CAUSAL_LAG_POINTS = EXOG_HORIZON_POINTS


def _naive_bound(timestamp):
    """Drop the timezone so bounds compare against the tz-naive series index."""
    if timestamp is None:
        return None
    bound = pd.Timestamp(timestamp)
    return bound.tz_localize(None) if bound.tzinfo is not None else bound


def _exog_frame(dataset_name, entity_id, timestamps, exog_level, freq, task=None):
    """Build the exogenous block for one entity, aligned to ``timestamps``."""
    frame = build_exogenous_feature_dataframe(
        dataset_name=dataset_name,
        entity_id=entity_id,
        datetime_sequence=timestamps,
        exog_level=exog_level,
        forecast_horizon=EXOG_HORIZON_POINTS,
        freq=freq,
        task=task,
    )
    expected = pd.DatetimeIndex(timestamps)
    if not pd.DatetimeIndex(frame.index).equals(expected):
        raise ValueError(
            f"exogenous frame for {dataset_name}/{entity_id} is not aligned with the series index"
        )
    return frame


def _attach_exog(df, dataset_name, entity_id, exog_level, freq, timestamp_column="ds", task=None):
    """Attach exogenous columns to a long-format frame, verifying the alignment."""
    frame = _exog_frame(dataset_name, entity_id, df[timestamp_column], exog_level, freq, task=task)
    if len(frame) != len(df):
        raise ValueError(
            f"exogenous frame has {len(frame)} rows but {dataset_name}/{entity_id} has {len(df)}"
        )
    for column in frame.columns:
        values = frame[column]
        # A continuous source can start one step after the series (the first
        # reanalysis stamp).  Fill only the two edge rows; interior gaps stay
        # NaN so a missing measurement is never disguised as data.
        if column not in ("is_holiday", "is_bridge_day", "is_dst", "is_school_holiday") and values.isna().any():
            values = values.bfill().ffill()
        df[column] = values.to_numpy()
    return df


def _causal_exog_frame(
    dataset_name, entity_id, timestamps, exog_level, freq,
    lag_steps=EXOG_CAUSAL_LAG_POINTS, task=None,
):
    """Exogenous frame for direct multi-step regressors.

    Values only observed at their own timestamp (measured weather, lagged
    neighbours) are shifted back by one horizon, so the row that predicts t only
    carries information available at the forecast origin.  Known-future columns
    (calendar, prices, clear-sky geometry) are left untouched.
    """
    frame = _exog_frame(dataset_name, entity_id, timestamps, exog_level, freq, task=task)
    if lag_steps <= 0:
        return frame
    shifted = frame.copy()
    for column in frame.columns:
        if column in FUTR_EXOG_COLUMNS:
            continue
        # The first lag_steps rows have no history yet; backfill them from the
        # oldest shifted value so the regression matrix has no leading hole.
        shifted[column] = frame[column].shift(int(lag_steps)).bfill()
    return shifted

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize_datetime_series(series):
    dt_series = pd.to_datetime(series)
    if hasattr(dt_series.dt, "tz") and dt_series.dt.tz is not None:
        dt_series = dt_series.dt.tz_localize(None)
    return dt_series


def _find_belgium_file(house, filename, start_dt, end_dt):
    start_str = start_dt.strftime("%Y-%m-%d_%H%M")
    end_str = end_dt.strftime("%Y-%m-%d_%H%M")
    exact_file = PROJECT_ROOT / f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/{filename}"
    if exact_file.exists():
        return exact_file
    house_dir = PROJECT_ROOT / f"data/belgium_dataset/house_{house}"
    if house_dir.exists():
        for candidate_dir in sorted(house_dir.glob("*_*")):
            cand_file = candidate_dir / filename
            if cand_file.exists():
                return cand_file
    return exact_file


class DatasetBelgiumNF:

    def get_inputs_for_pv(self, house, start_dt, end_dt, exog_level="L0"):
        start_dt, end_dt = _naive_bound(start_dt), _naive_bound(end_dt)
        pv_file = _find_belgium_file(house, "solar.csv", start_dt, end_dt)
        df = pd.read_csv(pv_file)
        df["datetime"] = _normalize_datetime_series(df["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.rename(columns={"datetime": "ds", "SolarPv_0 (kW)": "y"})
        df["y"] = min_max_normalize(df[["y"]])
        df["unique_id"] = f"series_pv_house_{house}"

        if exog_level != "L0":
            return _attach_exog(
                df, dataset_name="belgium", entity_id=f"house_{house}",
                exog_level=exog_level, freq="15min", timestamp_column="ds", task="pv",
            )

        return df[["unique_id", "ds", "y"]]

    def get_inputs_for_battery(self, house, start_dt, end_dt, freq="15min", exog_level="L0"):
        start_dt, end_dt = _naive_bound(start_dt), _naive_bound(end_dt)
        battery_file = _find_belgium_file(house, "battery.csv", start_dt, end_dt)
        df = pd.read_csv(battery_file)
        df["datetime"] = _normalize_datetime_series(df["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.rename(columns={"datetime": "ds", "Battery_0 (kW)": "y"})
        full_range = pd.date_range(start=start_dt, end=end_dt - pd.Timedelta(freq), freq=freq)
        df = df.set_index("ds").reindex(full_range)
        df.index.name = "ds"
        if df["y"].notna().sum() > 0:
            z_scores = np.abs(stats.zscore(df["y"], nan_policy="omit"))
            df.loc[z_scores >= 3, "y"] = np.nan
        df["y"] = df["y"].interpolate(method="linear", limit_direction="both")
        df["y"] = min_max_normalize(df[["y"]])
        df["unique_id"] = f"series_battery_house_{house}"
        df = df.reset_index()

        if exog_level != "L0":
            return _attach_exog(
                df, dataset_name="belgium", entity_id=f"house_{house}",
                exog_level=exog_level, freq="15min", timestamp_column="ds", task="battery",
            )

        return df[["unique_id", "ds", "y"]]


class DatasetBelgium1D:

    def get_pv_data(self, house, start_dt, end_dt, exog_level="L0"):
        start_dt, end_dt = _naive_bound(start_dt), _naive_bound(end_dt)
        pv_file = _find_belgium_file(house, "solar.csv", start_dt, end_dt)
        df = pd.read_csv(pv_file)
        df["datetime"] = _normalize_datetime_series(df["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.sort_values("datetime").set_index("datetime")
        df["y"] = min_max_normalize(df[["SolarPv_0 (kW)"]])

        if exog_level == "L0":
            df["hour"] = df.index.hour
            df["minute"] = df.index.minute
            df["dayofweek"] = df.index.dayofweek
            df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
            df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
            X = df[["quarter_hour"]]
        else:
            X = _causal_exog_frame(
                "belgium", f"house_{house}", df.index, exog_level, "15min", task="pv"
            )
        y = df["y"]
        return X, y

    def get_battery_data(self, house, start_dt, end_dt, exog_level="L0"):
        start_dt, end_dt = _naive_bound(start_dt), _naive_bound(end_dt)
        battery_file = _find_belgium_file(house, "battery.csv", start_dt, end_dt)
        df = pd.read_csv(battery_file)
        df["datetime"] = _normalize_datetime_series(df["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.sort_values("datetime").set_index("datetime")
        df["y"] = df["Battery_0 (kW)"]
        z_scores = np.abs(stats.zscore(df["y"], nan_policy="omit"))
        df.loc[z_scores >= 3, "y"] = np.nan
        df["y"] = df["y"].interpolate(method="linear", limit_direction="both")
        df["y"] = min_max_normalize(df[["y"]])

        if exog_level == "L0":
            df["hour"] = df.index.hour
            df["minute"] = df.index.minute
            df["dayofweek"] = df.index.dayofweek
            df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
            df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
            X = df[["quarter_hour"]]
        else:
            X = _causal_exog_frame(
                "belgium", f"house_{house}", df.index, exog_level, "15min", task="battery"
            )
        y = df["y"]
        return X, y


class DatasetGermanyNF:

    def get_inputs_for_load(self, start_dt, end_dt, exog_level="L0"):
        start_dt, end_dt = _naive_bound(start_dt), _naive_bound(end_dt)
        load_file = PROJECT_ROOT / "data/germany_wpuq_dataset/SFH19_2023_2024_15min_3_month.csv"
        load_df = pd.read_csv(load_file)
        load_df["datetime"] = _normalize_datetime_series(load_df["datetime"])

        df = load_df[
            (load_df["datetime"] >= start_dt) & (load_df["datetime"] < end_dt)
        ][["datetime", "Consumer_0_electric (kW)"]]
        df = df.rename(columns={"datetime": "ds", "Consumer_0_electric (kW)": "y"})
        df["y"] = min_max_normalize(df[["y"]])
        df["unique_id"] = "series_load"

        if exog_level != "L0":
            return _attach_exog(
                df, dataset_name="germany", entity_id="SFH19",
                exog_level=exog_level, freq="15min", timestamp_column="ds"
            )

        return df[["unique_id", "ds", "y"]]


class DatasetGermany1D:

    def get_load_data(self, start_dt, end_dt, exog_level="L0"):
        start_dt, end_dt = _naive_bound(start_dt), _naive_bound(end_dt)
        load_file = PROJECT_ROOT / "data/germany_wpuq_dataset/SFH19_2023_2024_15min_3_month.csv"
        df = pd.read_csv(load_file)
        df["datetime"] = _normalize_datetime_series(df["datetime"])

        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.sort_values("datetime").set_index("datetime")
        df["y"] = min_max_normalize(df[["Consumer_0_electric (kW)"]])

        if exog_level == "L0":
            df["hour"] = df.index.hour
            df["minute"] = df.index.minute
            df["dayofweek"] = df.index.dayofweek
            df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
            df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
            X = df[["quarter_hour"]]
        else:
            X = _causal_exog_frame(
                "germany", "SFH19", df.index, exog_level, "15min"
            )
        y = df["y"]
        return X, y


class DatasetLondonNF:

    def get_inputs_for_load(self, exog_level="L0"):
        london_consumption_file = PROJECT_ROOT / "data/london_dataset/LCL_london_consumption_2013.csv"
        df = pd.read_csv(london_consumption_file)
        df["datetime"] = _normalize_datetime_series(df["DateTime"])
        df = df.rename(columns={"datetime": "ds", "KWH/hh (per half hour)": "y"})
        # Keep only the target and its id: the source file carries household
        # metadata (LCLid, stdorToU, DateTime) that must not become covariates.
        df = df[["ds", "y"]]
        df["y"] = min_max_normalize(df[["y"]])
        df["unique_id"] = "series_load"

        if exog_level != "L0":
            return _attach_exog(
                df, dataset_name="london", entity_id="MAC000033",
                exog_level=exog_level, freq="30min", timestamp_column="ds"
            )

        return df[["unique_id", "ds", "y"]]


class DatasetLondon1D:

    def get_load_data(self, exog_level="L0"):
        london_consumption_file = PROJECT_ROOT / "data/london_dataset/LCL_london_consumption_2013.csv"
        df = pd.read_csv(london_consumption_file)
        df["datetime"] = _normalize_datetime_series(df["DateTime"])
        df = df.sort_values("datetime").set_index("datetime")
        df["y"] = min_max_normalize(df[["KWH/hh (per half hour)"]])

        if exog_level == "L0":
            df["hour"] = df.index.hour
            df["minute"] = df.index.minute
            df["dayofweek"] = df.index.dayofweek
            df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
            df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
            X = df[["quarter_hour"]]
        else:
            X = _causal_exog_frame(
                "london", "MAC000033", df.index, exog_level, "30min"
            )
        y = df["y"]
        return X, y


class DatasetZonnedaelNF:

    def get_inputs_for_zonnedael_consumption(self, customer_number: int, exog_level="L0"):
        zonnedael_consumption_file = PROJECT_ROOT / "data/zonnedael_dataset/liander_zonnedael_2013_original.csv"
        df = pd.read_csv(zonnedael_consumption_file)
        df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
        df = df.dropna(subset=["datetime"])
        df["datetime"] = _normalize_datetime_series(df["datetime"])
        client_no = f"Klant {customer_number}"
        df = df.rename(columns={"datetime": "ds", client_no: "y"})
        # Keep only the target series: the source file carries all 56 customers,
        # and leaving them in would hand NeuralForecast 55 undeclared covariates.
        df = df[["ds", "y"]]
        df["y"] = min_max_normalize(df["y"])
        df["unique_id"] = f"series_load_{customer_number}"

        if exog_level != "L0":
            return _attach_exog(
                df, dataset_name="zonnedael", entity_id=f"Klant {customer_number}",
                exog_level=exog_level, freq="15min", timestamp_column="ds"
            )

        return df[["unique_id", "ds", "y"]]


class DatasetZonnedael1D:

    def get_inputs_for_zonnedael_consumption(self, customer_number: int, exog_level="L0"):
        zonnedael_consumption_file = PROJECT_ROOT / "data/zonnedael_dataset/liander_zonnedael_2013_original.csv"
        df = pd.read_csv(zonnedael_consumption_file)
        df["datetime"] = pd.to_datetime(
            df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce"
        )
        df = df.dropna(subset=["datetime"])
        df["datetime"] = _normalize_datetime_series(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
        customer_column = f"Klant {customer_number}"
        df["y"] = min_max_normalize(df[[customer_column]])

        if exog_level == "L0":
            df["hour"] = df.index.hour
            df["minute"] = df.index.minute
            df["dayofweek"] = df.index.dayofweek
            df["quarter_hour"] = df["hour"] * 4 + df["minute"] // 15
            df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
            X = df[["quarter_hour", "is_weekend"]]
        else:
            X = _causal_exog_frame(
                "zonnedael", f"Klant {customer_number}", df.index, exog_level, "15min"
            )
        y = df["y"]
        return X, y
