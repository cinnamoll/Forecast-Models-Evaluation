"""Chronos-2 zero-shot benchmark (plan_updated.md sections 1 and 4).

Chronos-2 is evaluated *zero-shot*: the checkpoint is used as published, on the
same 192-point test window as the original paper.  Nothing in this script fits
the model to the benchmark data, which is what makes the column comparable to the
original Chronos (Bolt) column.

Covariates follow the role split of plan_updated.md section 3: every exogenous
column is passed over the context, and the columns that are knowable during the
horizon (calendar, day-ahead price, clear-sky geometry) are also passed as
``future_covariates``.  Measured weather and lagged neighbours are past-only, so
the model cannot read the test window through them.
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.benchmark_series import display_name, iter_series
from utils.dataset_config import (
    DatasetBelgiumNF,
    DatasetGermanyNF,
    DatasetLondonNF,
    DatasetZonnedaelNF,
)
from utils.device import GPU_HELP, torch_device
from utils.exogenous.feature_store import covariate_roles
from utils.metrics import calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from utils.protocol import (
    COVARIATE_LEVEL,
    EXOG_LEVEL_CHOICES,
    SAMPLING_RATES,
    freq_for,
    horizon_for,
    paper_bounds,
    run_dir,
    sampling_tag,
    step_minutes,
)

MODEL_DIR = "Chronos2"
DEFAULT_CHECKPOINT = "amazon/chronos-2"
IDENTIFIER_COLUMNS = ("unique_id", "ds", "y")


def setup_model_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log.txt")
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info("Logger initialized at %s", log_file)


def build_inputs(frame, horizon, context_length=None):
    """Chronos-2 input dict for one series, plus the test frame it refers to.

    ``frame`` is the already-subsampled series; the last ``horizon`` rows are the
    test window and are never part of the context.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(frame) <= horizon:
        raise ValueError(f"series of {len(frame)} points cannot hold horizon {horizon}")

    exog_columns = [c for c in frame.columns if c not in IDENTIFIER_COLUMNS]
    roles = covariate_roles(exog_columns)
    train = frame.iloc[:-horizon]
    test = frame.iloc[-horizon:]
    if context_length:
        train = train.iloc[-int(context_length):]

    past_covariates, future_covariates = {}, {}
    for column in exog_columns:
        values = train[column]
        categorical = column in roles["categorical"]
        past_covariates[column] = (
            values.astype(str).to_numpy() if categorical
            else values.to_numpy(dtype=np.float32)
        )
        if column not in roles["known_future"]:
            continue
        future = test[column]
        future_covariates[column] = (
            future.astype(str).to_numpy() if categorical
            else future.to_numpy(dtype=np.float32)
        )

    payload = {"target": train["y"].to_numpy(dtype=np.float32)}
    if past_covariates:
        payload["past_covariates"] = past_covariates
    if future_covariates:
        payload["future_covariates"] = future_covariates
    return payload, test, roles


def forecast_series(pipeline, frame, horizon, point="median", context_length=None):
    payload, test, roles = build_inputs(frame, horizon, context_length=context_length)
    logging.info(
        "context=%d steps, horizon=%d, covariates: %d past-only, %d known-future, %d categorical",
        len(payload["target"]), horizon,
        len(roles["past_only"]), len(roles["known_future"]), len(roles["categorical"]),
    )
    quantile_levels = [0.1, 0.5, 0.9]
    quantiles, mean = pipeline.predict_quantiles(
        inputs=[payload],
        prediction_length=horizon,
        quantile_levels=quantile_levels,
    )
    if point == "mean":
        y_pred = np.asarray(mean[0])[0, :horizon]
    else:
        y_pred = np.asarray(quantiles[0])[0, :horizon, quantile_levels.index(0.5)]
    y_true = test["y"].to_numpy()
    mae, rmse = calculate_metrics(y_pred, y_true)
    return y_pred, y_true, test, mae, rmse


def train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq,
                     forecast_horizon, sampling_rate, pipeline, point, context_length,
                     exog_level):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info("...Start Chronos-2 zero-shot forecasting (exogenous level %s)...", exog_level)
    step = step_minutes(sampling_rate)
    for series_id, frame in iter_series(
        dataset_name, dataset, start_dt, end_dt, exog_level=exog_level
    ):
        frame = frame.iloc[::step].reset_index(drop=True)
        y_pred, y_true, test, mae, rmse = forecast_series(
            pipeline, frame, forecast_horizon, point=point, context_length=context_length
        )
        logging.info("%s - MAE: %.4f, RMSE: %.4f", series_id, mae, rmse)
        forecast_plot_and_csv(
            pd.DataFrame(
                {"datetime": test["ds"].to_numpy(), "Actual": y_true, "Forecast": y_pred}
            ).set_index("datetime"),
            display_name(series_id),
            save_dir,
            # plan_updated.md section 4: the two TSFM columns carry the zero-shot
            # label in every figure; the file name stays stable for the tables.
            title=f"Forecast vs Actual - {display_name(series_id)} (zero-shot)",
        )
        metrics.append({"model": series_id, "MAE": mae, "RMSE": rmse,
                        "protocol": "zero-shot"})
    pd.DataFrame(metrics).to_csv(
        os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f"
    )
    plot_model_metrics(metrics, save_dir)
    logging.info("...End Chronos-2 forecasting...")
    logging.info("Forecasting completed in %.2f seconds.", time.time() - start_time)
    return metrics


def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, pipeline,
                            point, context_length, exog_level,
                            results_dir=PROJECT_ROOT / "results"):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt, end_dt = paper_bounds()
    horizon = horizon_for(sampling_rate)
    freq = freq_for(sampling_rate)
    save_dir = run_dir(results_dir, dataset_name, MODEL_DIR, sampling_rate, run_num,
                       exog_level=exog_level)
    return train_all_models(
        dataset_name, dataset, start_dt, end_dt, save_dir, freq, horizon,
        sampling_rate, pipeline, point, context_length, exog_level,
    )


def main(argv=None):
    dataset_classes = dict(
        belgium=DatasetBelgiumNF,
        germany=DatasetGermanyNF,
        london=DatasetLondonNF,
        zonnedael=DatasetZonnedaelNF,
    )
    parser = argparse.ArgumentParser(description="Run the Chronos-2 zero-shot benchmark.")
    parser.add_argument("--dataset", choices=list(dataset_classes), default="belgium")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float,
                        default=list(SAMPLING_RATES), help="Sampling percentages to evaluate.")
    parser.add_argument("--runs", type=int, default=1, help="Runs per sampling rate.")
    parser.add_argument("--model_id", default=DEFAULT_CHECKPOINT,
                        help="Chronos-2 checkpoint.  Chronos-Bolt checkpoints do not accept covariates.")
    parser.add_argument("--point", choices=["median", "mean"], default="median",
                        help="Point forecast used for MAE/RMSE (median matches the original Chronos column).")
    parser.add_argument("--context_length", type=int, default=None,
                        help="Cap the context in points; default keeps the whole training history, "
                             "which is what the original zero-shot runs used.")
    parser.add_argument("--exog_level", "--exog-level", choices=list(EXOG_LEVEL_CHOICES), default=None,
                        help="Ladder level (default L0, which reproduces the paper baseline). "
                             "Mutually exclusive with --use_covariates.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--use_covariates", "--use-covariates", action="store_true",
                        help="Use every exogenous variable the dataset has, in one block, "
                             "instead of one rung of the ladder (same as --exog_level ALL).")
    args = parser.parse_args(argv)
    # --use_covariates is a shorthand for --exog_level ALL, so the two must not
    # disagree.  The level defaults to None so "not given" can be told apart
    # from an explicit L0; a contradictory pair fails loudly instead of one
    # silently winning and the results landing in the wrong tree.
    if args.exog_level is None:
        args.exog_level = COVARIATE_LEVEL if args.use_covariates else "L0"
    elif args.use_covariates and args.exog_level != COVARIATE_LEVEL:
        parser.error(
            f"--use_covariates and --exog_level {args.exog_level} disagree; drop one "
            "(--use_covariates is --exog_level ALL)"
        )

    from chronos import Chronos2Pipeline

    dataset = dataset_classes[args.dataset]()
    device = torch_device(args.gpu)
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device)
    for sampling_rate in args.sampling_rates:
        for run_num in range(1, args.runs + 1):
            try:
                paper_forecasting_train(
                    args.dataset, dataset, run_num, sampling_rate, pipeline,
                    args.point, args.context_length, args.exog_level, args.results_dir,
                )
            except Exception as error:  # keep the other runs alive
                logging.error(
                    "Chronos2 run %d at sampling %s failed: %s",
                    run_num, sampling_tag(sampling_rate), error, exc_info=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
