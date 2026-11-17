import argparse
import os
import sys
import warnings
import logging
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.metrics import calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from utils.dataset_config import DatasetBelgiumNF, DatasetGermanyNF, DatasetLondonNF, DatasetZonnedaelNF
from utils.covariates import roles_for
from utils.device import GPU_HELP, torch_device
from utils.protocol import horizon_for, run_dir
from utils.tsfm import prepare_inputs, timestral_arguments
import time
import gc
import torch
from timestral import TimestralForecaster

DEFAULT_CHECKPOINT = "mistralai/Timestral-123M"
PAPER_WINDOW = (pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-04-01 00:00:00", tz="UTC"))


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


# Core function for Timestral forecasting
def timestral_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, sampling_rate,
                             forecaster, device, level="L0"):
    y_df = y_df.iloc[::int(100 / sampling_rate)].reset_index(drop=True)
    test_df = y_df.iloc[-forecast_horizon:]
    feature_columns = [c for c in y_df.columns if c not in ("unique_id", "ds", "y")]
    roles = roles_for(y_df[feature_columns]) if feature_columns else {
        "known_future": [], "past_only": [], "static": [], "categorical": []}

    payload = prepare_inputs(y_df, roles, level, forecast_horizon)
    logging.info(
        "%s: context=%d steps, horizon=%d, covariates: %d known-future, %d past-only, "
        "%d static dropped (no per-series channel in this checkpoint)",
        model_name, payload["context_end"], forecast_horizon,
        payload["report"].get("n_known_future", 0), payload["report"].get("n_past_only", 0),
        len(payload["report"]["dropped_static"]),
    )
    if payload["report"]["dropped_static"]:
        logging.warning("%s: static metadata is not representable and was dropped: %s",
                        model_name, ", ".join(payload["report"]["dropped_static"]))

    forecast = forecaster.forecast(prediction_length=forecast_horizon,
                                   **timestral_arguments(payload))
    y_pred = np.asarray(forecast.median, dtype=float).reshape(-1)[:forecast_horizon]
    y_true = test_df["y"].values
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name,
        save_dir
    )
    return mae, rmse


# Full pipeline for Timestral
def train_all_models(dataset_name, dataset, save_dir, freq, forecast_horizon, sampling_rate,
                     forecaster, device, level="L0"):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info("...Start Timestral forecasting...")
    if dataset_name == "belgium":
        logging.info("Forecasting PV")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, *PAPER_WINDOW, level=level)
            pv_mae, pv_rmse = timestral_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq,
                                                       forecast_horizon, sampling_rate, forecaster, device, level)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        logging.info("Forecasting BESS")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, *PAPER_WINDOW, level=level)
            battery_mae, battery_rmse = timestral_forecast_model(battery_data, f"BESS_house_{house}", save_dir,
                                                                  freq, forecast_horizon, sampling_rate,
                                                                  forecaster, device, level)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        logging.info("Forecasting Germany load")
        load_data = dataset.get_inputs_for_load(*PAPER_WINDOW, level=level)
        load_mae, load_rmse = timestral_forecast_model(load_data, "germany_load", save_dir, freq,
                                                       forecast_horizon, sampling_rate, forecaster, device, level)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        logging.info("Forecasting London load")
        load_data = dataset.get_inputs_for_load(level=level)
        load_mae, load_rmse = timestral_forecast_model(load_data, "london_load", save_dir, freq,
                                                       forecast_horizon, sampling_rate, forecaster, device, level)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        logging.info("Forecasting Zonnedael customers")
        for customer_id in [8, 9, 43]:
            data_df = dataset.get_inputs_for_zonnedael_consumption(customer_id, level=level)
            cust_mae, cust_rmse = timestral_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir,
                                                           freq, forecast_horizon, sampling_rate, forecaster, device, level)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End Timestral forecasting...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")


def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, forecaster, device,
                            results_dir=PROJECT_ROOT / "results", level="L0"):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    freq_str = f"{int(15 * (100 / sampling_rate))}T"
    forecast_horizon = horizon_for(sampling_rate)
    try:
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        save_dir = run_dir(results_dir, dataset_name, "Timestral", sampling_rate, run_num, exog_level=level)
        train_all_models(dataset_name, dataset, save_dir, freq_str, forecast_horizon, sampling_rate,
                         forecaster, device, level)
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"Skipping Timestral due to error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    dataset_classes = dict(belgium=DatasetBelgiumNF,
                           germany=DatasetGermanyNF,
                           london=DatasetLondonNF,
                           zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run the Timestral-123M zero-shot benchmark.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (the model is deterministic).")
    parser.add_argument("--model_id", default=DEFAULT_CHECKPOINT, help="Timestral checkpoint ID.")
    parser.add_argument("--exog_level", "--exog-level", choices=list(("L0", "COV")), default="COV",
                        help="Covariate block: COV (day/hour + weather + price) or L0 (paper baseline).")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    device = torch_device(args.gpu)
    forecaster = TimestralForecaster.from_pretrained(args.model_id, device=device)
    for sampling_rate in args.sampling_rates:
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(args.dataset, dataset, run_num, sampling_rate, forecaster, device,
                                    args.results_dir, level=args.exog_level)