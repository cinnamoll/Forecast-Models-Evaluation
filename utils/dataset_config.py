import pandas as pd
import numpy as np
from pathlib import Path
from utils.metrics import min_max_normalize
from scipy import stats
from utils.covariates import build_covariates
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _attach_covariates(df, dataset, task, level="L0", house=None):
    """Append the covariate block to a long frame, aligned on ``ds``.
    """
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    if level == "L0":
        return df
    frame, _roles = build_covariates(
        dataset, pd.DatetimeIndex(df["ds"]), task, level=level, house=house
    )
    covariates = frame.reset_index(drop=True)
    covariates.index = df.index
    return pd.concat([df, covariates], axis=1)


def _covariate_columns(df, dataset, task, level, house=None):
    """Feature matrix of the direct multi-step models (LightGBM, KNN)."""
    frame, _roles = build_covariates(
        dataset, pd.DatetimeIndex(df.index), task, level=level, house=house
    )
    return frame


def _with_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """The paper's day/hour columns, exactly as the original 1D loaders built them."""
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    return df

class DatasetBelgiumNF:

    # Load PV data for specified range and reformat for NeuralForecast
    def get_inputs_for_pv(self, house, start_dt, end_dt, level="L0"):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        pv_file = PROJECT_ROOT / f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/solar.csv"
        df = pd.read_csv(pv_file, parse_dates=["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.rename(columns={"datetime": "ds", "SolarPv_0 (kW)": "y"})
        df["y"] = min_max_normalize(df[["y"]])
        df["unique_id"] = f"series_pv_house_{house}"
        df = df[["unique_id", "ds", "y"]].reset_index(drop=True)
        return _attach_covariates(df, "belgium", "pv", level, house=house)

    def get_inputs_for_battery(self, house, start_dt, end_dt, freq="15min", level="L0"):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        battery_file = PROJECT_ROOT / f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/battery.csv"
        df = pd.read_csv(battery_file, parse_dates=["datetime"])
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
        df = df[["unique_id", "ds", "y"]].reset_index(drop=True)
        return _attach_covariates(df, "belgium", "battery", level, house=house)

class DatasetBelgium1D:

    # Load PV data for specified range and reformat for LightGBM
    def get_pv_data(self, house, start_dt, end_dt, level="L0"):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        pv_file = PROJECT_ROOT / f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/solar.csv"
        df = pd.read_csv(pv_file, parse_dates=["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.sort_values("datetime").set_index("datetime")
        df["y"] = min_max_normalize(df[["SolarPv_0 (kW)"]])
        if level == "L0":
            df = _with_calendar(df)
            X = df[["hour"]]
        else:
            X = _covariate_columns(df, "belgium", "pv", level, house=house)
        y = df["y"]
        return X, y

    # Load battery data for specified range and reformat for LightGBM
    def get_battery_data(self, house, start_dt, end_dt, level="L0"):
        start_str = start_dt.strftime("%Y-%m-%d_%H%M")
        end_str = end_dt.strftime("%Y-%m-%d_%H%M")
        battery_file = PROJECT_ROOT / f"data/belgium_dataset/house_{house}/{start_str}_{end_str}/battery.csv"
        df = pd.read_csv(battery_file, parse_dates=["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.sort_values("datetime").set_index("datetime")
        df["y"] = df["Battery_0 (kW)"]
        z_scores = np.abs(stats.zscore(df["y"], nan_policy="omit"))
        df.loc[z_scores >= 3, "y"] = np.nan
        df["y"] = df["y"].interpolate(method="linear", limit_direction="both")
        df["y"] = min_max_normalize(df[["y"]])
        if level == "L0":
            df = _with_calendar(df)
            X = df[["hour"]]
        else:
            X = _covariate_columns(df, "belgium", "battery", level, house=house)
        y = df["y"]
        return X, y

class DatasetGermanyNF:

    # Load electric consumption data for specified range and reformat for NeuralForecast
    def get_inputs_for_load(self, start_dt, end_dt, level="L0"):
        load_file = PROJECT_ROOT / "data/germany_wpuq_dataset/SFH19_2023_2024_15min_3_month.csv"
        load_df = pd.read_csv(load_file)
        load_df["datetime"] = pd.to_datetime(load_df["datetime"])
        df = load_df[
            (load_df["datetime"] >= start_dt) & (load_df["datetime"] < end_dt)
        ][["datetime", "Consumer_0_electric (kW)"]]
        df = df.rename(columns={"datetime": "ds", "Consumer_0_electric (kW)": "y"})
        df["y"] = min_max_normalize(df[["y"]])
        df["unique_id"] = "series_load"
        df = df[["unique_id", "ds", "y"]].reset_index(drop=True)
        return _attach_covariates(df, "germany", "load", level)

class DatasetGermany1D:

    # Load electric consumption data for specified range and reformat for LightGBM
    def get_load_data(self, start_dt, end_dt, level="L0"):
        load_file = PROJECT_ROOT / "data/germany_wpuq_dataset/SFH19_2023_2024_15min_3_month.csv"
        df = pd.read_csv(load_file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]
        df = df.sort_values("datetime").set_index("datetime")
        df["y"] = min_max_normalize(df[["Consumer_0_electric (kW)"]])
        if level == "L0":
            df = _with_calendar(df)
            X = df[["hour"]]
        else:
            X = _covariate_columns(df, "germany", "load", level)
        y = df["y"]
        return X, y

class DatasetLondonNF:

    # Load electric consumption data for specified range and reformat for NeuralForecast
    def get_inputs_for_load(self, level="L0"):
        london_consumption_file = PROJECT_ROOT / "data/london_dataset/LCL_london_consumption_2013.csv"
        df = pd.read_csv(london_consumption_file)
        df["datetime"] = pd.to_datetime(df["DateTime"])
        df = df.rename(columns={"datetime": "ds", "KWH/hh (per half hour)": "y"})
        df["y"] = min_max_normalize(df[["y"]])
        df["unique_id"] = "series_load"
        df = df[["unique_id", "ds", "y"]].reset_index(drop=True)
        return _attach_covariates(df, "london", "load", level)

class DatasetLondon1D:

    # Load electric consumption data for specified range and reformat for LightGBM
    def get_load_data(self, level="L0"):
        london_consumption_file = PROJECT_ROOT / "data/london_dataset/LCL_london_consumption_2013.csv"
        df = pd.read_csv(london_consumption_file)
        df["datetime"] = pd.to_datetime(df["DateTime"])
        df = df.sort_values("datetime").set_index("datetime")
        df["y"] = min_max_normalize(df[["KWH/hh (per half hour)"]])
        if level == "L0":
            df = _with_calendar(df)
            X = df[["hour"]]
        else:
            X = _covariate_columns(df, "london", "load", level)
        y = df["y"]
        return X, y

class DatasetZonnedaelNF:

    # Load electric consumption data for specified range and reformat for NeuralForecast
    def get_inputs_for_zonnedael_consumption(self, customer_number: int, level="L0"):
        zonnedael_consumption_file = PROJECT_ROOT / "data/zonnedael_dataset/liander_zonnedael_2013_original.csv"
        df = pd.read_csv(zonnedael_consumption_file)
        df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce")
        df = df.dropna(subset=["datetime"])
        client_no = f"Klant {customer_number}"
        df = df.rename(columns={"datetime": "ds", client_no: "y"})
        df["y"] = min_max_normalize(df["y"])
        df["unique_id"] = f"series_load_{customer_number}"
        df = df[["unique_id", "ds", "y"]].reset_index(drop=True)
        return _attach_covariates(df, "zonnedael", "load", level)

class DatasetZonnedael1D:
    def get_inputs_for_zonnedael_consumption(self, customer_number: int, level="L0"):
        zonnedael_consumption_file = PROJECT_ROOT / "data/zonnedael_dataset/liander_zonnedael_2013_original.csv"
        df = pd.read_csv(zonnedael_consumption_file)
        df["datetime"] = pd.to_datetime(
            df["datetime"], format="%d-%m-%Y %H:%M", errors="coerce"
        )
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
        customer_column = f"Klant {customer_number}"
        df["y"] = min_max_normalize(df[[customer_column]])
        if level == "L0":
            df = _with_calendar(df)
            X = df[["hour", "dayofweek"]]
        else:
            X = _covariate_columns(df, "zonnedael", "load", level)
        y = df["y"]
        return X, y
