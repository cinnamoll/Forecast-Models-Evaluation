import argparse
import json
import os
import sys
import time
import logging
import warnings
import gc
from pathlib import Path
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import pmdarima as pm
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.metrics import split_train_test, calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from utils.dataset_config import DatasetBelgium1D, DatasetGermany1D, DatasetLondon1D, DatasetZonnedael1D
from utils.covariates import attach_target_features, roles_for, shift_past_only
from utils.device import GPU_HELP, uses_gpu
from utils.protocol import horizon_for, run_dir

def setup_model_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log.txt")
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
    logging.info(f"Logger initialized at {log_file}")

def load_tuned_params(path):
    """Best parameters written by ``scripts/tune_optuna.py`` ({} when absent)."""
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        logging.warning("tuned parameters not found at %s; using defaults", path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("best_params", payload))

# Generalized model training function for both univariate and multivariate models
def generic_model(X, y, model_name, save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, params_override=None):
    current_seed = int(time.time() * 1000) % (2**32 - 1)  # Ensure higher variability
    is_univariate = X is None
    logging.info(f"Run {run_num} - {model_name} - {model_type}")
    logging.info(f"Using random seed: {current_seed}")
    if is_univariate:
        # Univariate setup
        y = y.iloc[::int(100/sampling_rate)]
        y_train = y.iloc[:-forecast_horizon]
        y_test = y.iloc[-forecast_horizon:]
        logging.info(f"Training {model_type} model for {model_name} on {len(y_train)} samples (univariate)...")
        if model_type == "ARIMA":
                y_arr = y_train.to_numpy(dtype=float)[-(48 * 60):]
                d_order = pm.arima.ndiffs(y_arr, alpha=0.05, test='kpss', max_d=2)
                # If you MUST use seasonality, pre-calculate D as well.
                # Note: With m=48, this can still be slow.
                D_order = pm.arima.nsdiffs(y_arr, m=48, max_D=1, test='ch')
                model = pm.auto_arima(
                    y_arr,
                    start_p=0, start_q=0,
                    max_p=1, max_q=1,
                    start_P=0, start_Q=0,
                    max_P=1, max_Q=1,
                    d=d_order,                # Use pre-calculated d
                    D=D_order,                # Use pre-calculated D
                    m=48,                     
                    seasonal=True,
                    stepwise=True,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    information_criterion='aic',
                    max_order=4,
                    maxiter=10,               # Allow the model to converge reasonably
                    # Use n_jobs=-1 for parallel processing when stepwise=False,
                    # but with stepwise=True, this has limited effect.
                )
                preds = model.predict(n_periods=forecast_horizon)
        elif model_type == "NaiveMovingAverage":
            k = int(forecast_horizon / 1)
            hist = y_train.tolist().copy()
            for i in range(forecast_horizon):
                hist.append(sum(hist[-k:]) / k)
            preds = hist[-forecast_horizon:]
            model = "NaiveMovingAverage"
        else:
            raise NotImplementedError(f"{model_type} not supported for univariate.")
        y_true = y_test.values
        mae, rmse = calculate_metrics(preds, y_true)
        logging.info(f"{model_name} ({model_type}) - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        df_plot = pd.DataFrame({"datetime": y_test.index, "Actual": y_true, "Forecast": preds}).set_index("datetime")
        forecast_plot_and_csv(df_plot, model_name, save_dir)
        return model, mae, rmse
    else:
        # Multivariate setup
        X = X.iloc[::int(100/sampling_rate)]
        y = y.iloc[::int(100/sampling_rate)]
        # A direct multi-step regressor reads the covariates of the row it is
        # predicting, so every measured covariate (weather, state of charge) is
        # shifted back by one horizon: the row for t then only carries what a
        # forecaster had at the origin.  Known-future columns (calendar, price)
        # are left alone because they are announced in advance.
        roles = roles_for(X)
        if roles["past_only"]:
            X = shift_past_only(X, roles, forecast_horizon)
            logging.info(
                "lagged %d past-only covariate(s) by %d steps: %s",
                len(roles["past_only"]), forecast_horizon, ", ".join(roles["past_only"]),
            )
        if model_type == "LightGBM":
            # The target's own history.  A sequence model reads it from its context
            # window; a direct regressor has to be handed it as columns, and every
            # column is shifted by at least one horizon so the row for t still only
            # carries what was known at the origin.
            X = attach_target_features(X, y, forecast_horizon)
            logging.info(
                "target-history features: %d column(s) (lags >= %d steps): %s",
                len(X.columns), forecast_horizon,
                ", ".join(c for c in X.columns if c.startswith("y_")),
            )
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=forecast_horizon)
        logging.info(f"Training {model_type} model for {model_name} on {X_train.shape[0]} samples ({sampling_rate}% of original data)...")
        params = {
            'random_state': current_seed,
            'bagging_fraction': 0.5,
            'feature_fraction': 0.5,
            'bagging_freq': 1,
            'verbosity': -1,
            'n_jobs': -1,
            'deterministic': False
        }
        if model_type == "LightGBM":
            if uses_gpu(gpu):
                params.update(device_type="gpu", gpu_device_id=gpu)
            else:
                params["device_type"] = "cpu"
            # The tables report MAE, so train the loss that matches it.  The paper's
            # refit left LightGBM on its default L2; L1 is better aligned with the
            # reported metric and less sensitive to the spikes of a load/PV series.
            # A tuned JSON may still override either key.
            params.setdefault("objective", "regression_l1")
            params.setdefault("metric", "mae")
            if params_override:
                applied = {k: v for k, v in params_override.items()
                           if k not in {"random_state", "n_jobs", "verbosity", "device_type"}}
                if applied:
                    logging.info("LightGBM tuned parameters: %s", applied)
                    params.update(applied)
            logging.info("LightGBM objective=%s metric=%s (n_estimators=%s, learning_rate=%s)",
                         params.get("objective"), params.get("metric"),
                         params.get("n_estimators", 100), params.get("learning_rate", 0.1))
            model = lgb.LGBMRegressor(**params).fit(X_train, y_train)
            preds = model.predict(X_test)
        elif model_type == "KNNRegression":
            model = KNeighborsRegressor().fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            raise NotImplementedError(f"{model_type} not supported for multivariate models.")
        mae, rmse = calculate_metrics(preds, y_test)
        logging.info(f"{model_name} ({model_type}) - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        df_plot = pd.DataFrame({"datetime": y_test.index, "Actual": y_test, "Forecast": preds}).set_index("datetime")
        forecast_plot_and_csv(df_plot, model_name, save_dir)
        return model, mae, rmse

def train_pv_model(dataset, start_dt, end_dt, save_dir, house, model_type, run_num, sampling_rate, forecast_horizon, gpu, level="L0", params_override=None):
    if model_type in ["NaiveMovingAverage", "ARIMA"]:
        y = dataset.get_pv_data(house, start_dt, end_dt, level=level)[1]
        return generic_model(None, y, f"pv_house_{house}", save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu)
    else:
        X, y = dataset.get_pv_data(house, start_dt, end_dt, level=level)
        return generic_model(X, y, f"pv_house_{house}", save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, params_override)

def train_battery_model(dataset, start_dt, end_dt, save_dir, house, model_type, run_num, sampling_rate, forecast_horizon, gpu, level="L0", params_override=None):
    if model_type in ["NaiveMovingAverage", "ARIMA"]:
        y = dataset.get_battery_data(house, start_dt, end_dt, level=level)[1]
        return generic_model(None, y, f"bess_house_{house}", save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu)
    else:
        X, y = dataset.get_battery_data(house, start_dt, end_dt, level=level)
        return generic_model(X, y, f"bess_house_{house}", save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, params_override)

def train_london_consumption_model(dataset, save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, level="L0", params_override=None):
    if model_type in ["NaiveMovingAverage", "ARIMA"]:
        y = dataset.get_load_data(level=level)[1]
        return generic_model(None, y, "london_load", save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu)
    else:
        X, y = dataset.get_load_data(level=level)
        return generic_model(X, y, "london_load", save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, params_override)

def train_germany_consumption_model(dataset, start_dt, end_dt, save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, level="L0", params_override=None):
    if model_type in ["NaiveMovingAverage", "ARIMA"]:
        y = dataset.get_load_data(start_dt, end_dt, level=level)[1]
        return generic_model(None, y, "germany_load", save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu)
    else:
        X, y = dataset.get_load_data(start_dt, end_dt, level=level)
        return generic_model(X, y, "germany_load", save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, params_override)

def train_zonnedael_consumption_model(dataset, save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, level="L0", params_override=None):
    metrics = []
    customer_ids = [8, 9, 43]
    for customer_number in customer_ids:
        if model_type in ["NaiveMovingAverage", "ARIMA"]:
            _, y = dataset.get_inputs_for_zonnedael_consumption(customer_number, level=level)
            _, mae, rmse = generic_model(
                None, y, f"zonnedael_customer_{customer_number}", save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu,
            )
        else:
            X, y = dataset.get_inputs_for_zonnedael_consumption(customer_number, level=level)
            _, mae, rmse = generic_model(
                X, y, f"zonnedael_customer_{customer_number}", save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu,
                params_override,
            )
        metrics.append({"model": f"zonnedael_load_{customer_number}", "MAE": mae, "RMSE": rmse})
    return metrics

# Unified training pipeline
def train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, level="L0", params_override=None):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info(f"Run {run_num}: Start training {model_type} models (covariate level {level})...")
    if dataset_name == "belgium":
        for house in [1, 2, 3, 4]:
            _, pv_mae, pv_rmse = train_pv_model(dataset, start_dt, end_dt, save_dir, house, model_type, run_num, sampling_rate, forecast_horizon, gpu, level, params_override)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        for house in [1, 2, 3, 4]:
            _, battery_mae, battery_rmse = train_battery_model(dataset, start_dt, end_dt, save_dir, house, model_type, run_num, sampling_rate, forecast_horizon, gpu, level, params_override)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        _, mae, rmse = train_germany_consumption_model(dataset, start_dt, end_dt, save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, level, params_override)
        metrics.append({"model": "germany_load", "MAE": mae, "RMSE": rmse})
    elif dataset_name == "london":
        _, mae, rmse = train_london_consumption_model(dataset, save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, level, params_override)
        metrics.append({"model": "london_load", "MAE": mae, "RMSE": rmse})
    elif dataset_name == "zonnedael":
        zonnedael_metrics = train_zonnedael_consumption_model(dataset, save_dir, model_type, run_num, sampling_rate, forecast_horizon, gpu, level, params_override)
        metrics.extend(zonnedael_metrics)
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    end_time = time.time()
    logging.info(f"Run {run_num}: End training {model_type} models.")
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

def paper_forecasting_train(dataset_name, dataset, run_num, model_type, sampling_rate, gpu, results_dir=PROJECT_ROOT / "results",
                            level="L0", params_override=None):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    forecast_horizon = horizon_for(sampling_rate)
    save_dir = run_dir(results_dir, dataset_name, model_type, sampling_rate, run_num, exog_level=level)
    train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, model_type, run_num,
                     sampling_rate, forecast_horizon, gpu, level, params_override)

if __name__ == "__main__":
    dataset_classes = dict(belgium=DatasetBelgium1D, germany=DatasetGermany1D, london=DatasetLondon1D, zonnedael=DatasetZonnedael1D)
    deterministic_models = {"ARIMA",
                            "NaiveMovingAverage"}
    parser = argparse.ArgumentParser(description="Run statistical and machine-learning forecasts.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="london", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--models", nargs="+", choices=["KNNRegression", "LightGBM", "ARIMA", "NaiveMovingAverage"],
        default=["NaiveMovingAverage"], help="Models to evaluate.")
    parser.add_argument("--runs", type=int, help="Runs per model; otherwise 1 for deterministic models and 10 for others.")
    parser.add_argument("--exog_level", "--exog-level", choices=list(("L0", "COV")), default="COV",
                        help="Covariate block: COV (day/hour + weather + price + state) or "
                             "L0 (the paper's baseline).")
    parser.add_argument("--params_json", type=Path, default=None,
                        help="Tuned parameters from scripts/tune_optuna.py.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    params_override = load_tuned_params(args.params_json)
    for sampling_rate in args.sampling_rates:
        for model_type in args.models:
            n_runs = args.runs if args.runs is not None else (1 if model_type in deterministic_models else 10)
            for run_num in range(1, n_runs + 1):
                try:
                    gc.collect()
                    paper_forecasting_train(args.dataset, dataset, run_num, model_type, sampling_rate, args.gpu,
                                            args.results_dir, level=args.exog_level,
                                            params_override=params_override if model_type == "LightGBM" else None)
                    gc.collect()
                except Exception as e:
                    logging.error(f"{model_type} Run {run_num} (Sampling {sampling_rate}) failed: {str(e)}", exc_info=True)
