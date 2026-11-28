import argparse
import os
import sys
import warnings
import logging
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.metrics import calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from utils.dataset_config import DatasetBelgiumNF, DatasetGermanyNF, DatasetLondonNF, DatasetZonnedaelNF
from utils.covariates import restrict_columns, roles_for
from utils.device import GPU_HELP, torch_device
from utils.protocol import freq_for, horizon_for, run_dir, window
from utils.tsfm import prepare_inputs, timesfm3_covariates, to_numpy
import time
import gc
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator

#: Context the PyTorch checkpoint reads at most (``TimesFM3Forecaster.global_context``).
MAX_CONTEXT_POINTS = 15360

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


# Core function for TimesFM 3
def timesfm3_forecast_model(y_df, model_name, save_dir, freq, forecast_horizon, sampling_rate,
                            forecaster, level="L0", symmetric_averaging=True, covariates=None):
    y_df = y_df.iloc[::int(100 / sampling_rate)].reset_index(drop=True)
    test_df = y_df.iloc[-forecast_horizon:]
    feature_columns = [c for c in y_df.columns if c not in ("unique_id", "ds", "y")]
    roles = roles_for(y_df[feature_columns]) if feature_columns else {
        "known_future": [], "past_only": [], "categorical": []}

    if covariates:
        available = [c for c in covariates if c in feature_columns]
        missing = [c for c in covariates if c not in feature_columns]
        if not available:
            raise KeyError(
                f"{model_name}: none of the requested covariates is in the frame "
                f"(available: {feature_columns})")
        y_df, roles = restrict_columns(y_df, roles, available)
        feature_columns = [c for c in y_df.columns if c not in ("unique_id", "ds", "y")]
        logging.info("%s: covariate set restricted to %s", model_name, ", ".join(feature_columns))
        if missing:
            logging.warning("%s: requested covariates absent from this series, skipped: %s",
                            model_name, ", ".join(missing))

    payload = prepare_inputs(y_df, roles, level, forecast_horizon, scale_categorical=True)
    arguments = timesfm3_covariates(payload)
    logging.info(
        "%s: context=%d steps (model reads %d), horizon=%d, covariates: %d known-future, "
        "%d past-only",
        model_name, payload["context_end"], min(payload["context_end"], MAX_CONTEXT_POINTS),
        forecast_horizon, payload["report"].get("n_known_future", 0),
        payload["report"].get("n_past_only", 0),
    )

    output = forecaster.predict(
        context=payload["context"],
        horizon=forecast_horizon,
        return_quantiles=True,
        use_symmetric_averaging=symmetric_averaging,
        make_positive=True,
        **arguments,
    )
    # ``forecast`` is the median quantile, which is the point forecast of the
    # model card; ``quantiles`` stays unused here but the call above asks for it
    # so the evaluator does not fall back to a mean of the quantiles.
    y_pred = to_numpy(output.forecast).reshape(-1)[:forecast_horizon]
    y_true = test_df["y"].to_numpy(dtype=float)
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name,
        save_dir
    )
    return mae, rmse


# Full pipeline for TimesFM 3
def train_all_models(dataset_name, dataset, save_dir, freq, forecast_horizon, sampling_rate,
                     forecaster, device, level="L0", symmetric_averaging=True, covariates=None):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info("...Start TimesFM 3 forecasting...")
    if dataset_name == "belgium":
        start_dt, end_dt = window(dataset_name)
        logging.info("Forecasting PV")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt, level=level)
            pv_mae, pv_rmse = timesfm3_forecast_model(pv_data, f"PV_house_{house}", save_dir, freq,
                                                     forecast_horizon, sampling_rate, forecaster, level,
                                                     symmetric_averaging, covariates)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        logging.info("Forecasting BESS")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt, level=level)
            battery_mae, battery_rmse = timesfm3_forecast_model(battery_data, f"BESS_house_{house}", save_dir,
                                                               freq, forecast_horizon, sampling_rate, forecaster,
                                                               level, symmetric_averaging, covariates)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        start_dt, end_dt = window(dataset_name)
        logging.info("Forecasting Germany load")
        load_data = dataset.get_inputs_for_load(start_dt, end_dt, level=level)
        load_mae, load_rmse = timesfm3_forecast_model(load_data, "germany_load", save_dir, freq,
                                                     forecast_horizon, sampling_rate, forecaster, level,
                                                     symmetric_averaging, covariates)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        logging.info("Forecasting London load")
        load_data = dataset.get_inputs_for_load(level=level)
        load_mae, load_rmse = timesfm3_forecast_model(load_data, "london_load", save_dir, freq,
                                                     forecast_horizon, sampling_rate, forecaster, level,
                                                     symmetric_averaging, covariates)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        logging.info("Forecasting Zonnedael customers")
        for customer_id in [8, 9, 43]:
            data_df = dataset.get_inputs_for_zonnedael_consumption(customer_id, level=level)
            cust_mae, cust_rmse = timesfm3_forecast_model(data_df, f"zonnedael_customer_{customer_id}", save_dir,
                                                         freq, forecast_horizon, sampling_rate, forecaster,
                                                         level, symmetric_averaging, covariates)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End TimesFM 3 forecasting...")
    logging.info(f"Forecasting completed in {elapsed_time:.2f} seconds.")


def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, forecaster, device,
                            results_dir=PROJECT_ROOT / "results", level="L0", symmetric_averaging=True, covariates=None):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    freq_str = freq_for(sampling_rate, style="T")
    forecast_horizon = horizon_for(sampling_rate)
    try:
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        save_dir = run_dir(results_dir, dataset_name, "TimesFM3", sampling_rate, run_num, exog_level=level)
        train_all_models(dataset_name, dataset, save_dir, freq_str, forecast_horizon, sampling_rate,
                         forecaster, device, level, symmetric_averaging, covariates)
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"Skipping TimesFM 3 due to error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    dataset_classes = dict(belgium=DatasetBelgiumNF,
                           germany=DatasetGermanyNF,
                           london=DatasetLondonNF,
                           zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run the TimesFM 3.0 zero-shot benchmark.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (the model is deterministic).")
    parser.add_argument("--model_id", default="google/timesfm-3.0-pytorch", help="TimesFM 3.0 checkpoint ID.")
    parser.add_argument("--exog_level", "--exog-level", choices=list(("L0", "COV")), default="COV",
                        help="Covariate block: COV (day/hour + weather + price) or L0 (paper baseline).")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-core batch size of the checkpoint.")
    parser.add_argument("--symmetric_averaging", "--symmetric-averaging", action=argparse.BooleanOptionalAction,
                        default=True, help="Benchmark protocol: average the forecast of x and -x (two forward passes).")
    parser.add_argument("--covariates", nargs="+", default=None,
                        help="Restrict the COV block to these columns (e.g. the TimeSHAP set). "
                             "Omit to keep every block the dataset has.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    device = torch_device(args.gpu)
    forecaster = TimesFM3Evaluator(ModelConfig(
        checkpoint_path=args.model_id,
        per_core_batch_size=args.batch_size,
        device=device,
    ))
    for sampling_rate in args.sampling_rates:
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(args.dataset, dataset, run_num, sampling_rate, forecaster, device,
                                    args.results_dir, level=args.exog_level,
                                    symmetric_averaging=args.symmetric_averaging,
                                    covariates=args.covariates)
