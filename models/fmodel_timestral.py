"""Timestral-123M zero-shot benchmark (plan_updated.md sections 1-4).

The checkpoint is fixed to ``mistralai/Timestral-123M``: section 2 of the plan
justifies that size (within 0.15 % of 209M on GIFT-Eval MASE at three quarters of
the latency), so the provider argument defaults to it and changing it is an
explicit choice, not an accident.

Like Chronos-2 this is a *zero-shot* column: no fine-tuning, no fitting, and the
same 192-point test window as the rest of the table.

Covariate roles follow section 3 exactly:

* calendar, day-ahead price and clear-sky geometry are known-future covariates
  (they carry the horizon, ``T + H``);
* measured weather, lagged neighbours and static metadata are past-only
  covariates (``T`` only), so the target of the test window cannot be reached
  through them.

Categorical rows are passed as integer codes because Timestral target-encodes a
categorical row against the context for a univariate target; that is what keeps
the holiday information alive instead of collapsing it into an ordinal.
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
    DEFAULT_RUNS,
    EXOG_LEVEL_CHOICES,
    SAMPLING_RATES,
    freq_for,
    horizon_for,
    paper_bounds,
    run_dir,
    sampling_tag,
    step_minutes,
)

MODEL_DIR = "Timestral"
DEFAULT_CHECKPOINT = "mistralai/Timestral-123M"
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


def _codes(values):
    """Integer category codes for a categorical covariate row.

    ``-1`` (pandas' missing code) becomes NaN: Timestral treats a missing
    category as its own bucket, but an out-of-range integer would be treated as a
    real category.
    """
    codes, _ = pd.factorize(pd.Series(values), sort=True)
    codes = codes.astype(np.float32)
    codes[codes < 0] = np.nan
    return codes


def build_covariates(frame, horizon, context_length=None):
    """Split the exogenous block into Timestral's four covariate tensors."""
    exog_columns = [c for c in frame.columns if c not in IDENTIFIER_COLUMNS]
    roles = covariate_roles(exog_columns)
    train = frame.iloc[:-horizon]
    test = frame.iloc[-horizon:]
    if context_length:
        train = train.iloc[-int(context_length):]

    categorical = set(roles["categorical"])
    known_future = set(roles["known_future"])
    context_block = pd.concat([train, test], axis=0)

    numeric_past, numeric_known = [], []
    codes_past, codes_known = [], []
    for column in exog_columns:
        if column in categorical:
            # Codes are built on the context+horizon block so that the same
            # category keeps the same code in both halves.
            codes = _codes(context_block[column].to_numpy())
            codes = codes.reshape(1, -1)
            if column in known_future:
                codes_known.append(codes)
            else:
                codes_past.append(codes[:, : len(train)])
            continue
        values = context_block[column].to_numpy(dtype=np.float32).reshape(1, -1)
        if column in known_future:
            numeric_known.append(values)
        else:
            numeric_past.append(values[:, : len(train)])

    def stack(blocks):
        return np.concatenate(blocks, axis=0)[None, ...] if blocks else None

    return {
        "context": train["y"].to_numpy(dtype=np.float32)[None, None, :],
        "past_covariates": stack(numeric_past),
        "known_covariates": stack(numeric_known),
        "categorical_past_covariates": stack(codes_past),
        "categorical_known_covariates": stack(codes_known),
        "roles": roles,
    }


def forecast_series(forecaster, frame, horizon, context_length=None):
    payload = build_covariates(frame, horizon, context_length=context_length)
    logging.info(
        "context=%d steps, horizon=%d, covariates: %d past-only, %d known-future, %d categorical",
        payload["context"].shape[-1], horizon,
        len(payload["roles"]["past_only"]), len(payload["roles"]["known_future"]),
        len(payload["roles"]["categorical"]),
    )
    forecast = forecaster.forecast(
        payload["context"],
        prediction_length=horizon,
        past_covariates=payload["past_covariates"],
        known_covariates=payload["known_covariates"],
        categorical_past_covariates=payload["categorical_past_covariates"],
        categorical_known_covariates=payload["categorical_known_covariates"],
    )
    y_pred = np.asarray(forecast.median)[0, 0, :horizon]
    test = frame.iloc[-horizon:]
    y_true = test["y"].to_numpy()
    mae, rmse = calculate_metrics(y_pred, y_true)
    return y_pred, y_true, test, mae, rmse


def train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq,
                     forecast_horizon, sampling_rate, forecaster, context_length,
                     exog_level):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info("...Start Timestral zero-shot forecasting (exogenous level %s)...", exog_level)
    step = step_minutes(sampling_rate)
    for series_id, frame in iter_series(
        dataset_name, dataset, start_dt, end_dt, exog_level=exog_level
    ):
        frame = frame.iloc[::step].reset_index(drop=True)
        y_pred, y_true, test, mae, rmse = forecast_series(
            forecaster, frame, forecast_horizon, context_length=context_length
        )
        logging.info("%s - MAE: %.4f, RMSE: %.4f", series_id, mae, rmse)
        forecast_plot_and_csv(
            pd.DataFrame(
                {"datetime": test["ds"].to_numpy(), "Actual": y_true, "Forecast": y_pred}
            ).set_index("datetime"),
            display_name(series_id),
            save_dir,
            # plan_updated.md section 4: the zero-shot label lives in the figure.
            title=f"Forecast vs Actual - {display_name(series_id)} (zero-shot)",
        )
        metrics.append({"model": series_id, "MAE": mae, "RMSE": rmse,
                        "protocol": "zero-shot"})
    pd.DataFrame(metrics).to_csv(
        os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f"
    )
    plot_model_metrics(metrics, save_dir)
    logging.info("...End Timestral forecasting...")
    logging.info("Forecasting completed in %.2f seconds.", time.time() - start_time)
    return metrics


def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, forecaster,
                            context_length, exog_level, results_dir=PROJECT_ROOT / "results"):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt, end_dt = paper_bounds()
    horizon = horizon_for(sampling_rate)
    freq = freq_for(sampling_rate)
    save_dir = run_dir(results_dir, dataset_name, MODEL_DIR, sampling_rate, run_num,
                       exog_level=exog_level)
    return train_all_models(
        dataset_name, dataset, start_dt, end_dt, save_dir, freq, horizon,
        sampling_rate, forecaster, context_length, exog_level,
    )


def main(argv=None):
    dataset_classes = dict(
        belgium=DatasetBelgiumNF,
        germany=DatasetGermanyNF,
        london=DatasetLondonNF,
        zonnedael=DatasetZonnedaelNF,
    )
    parser = argparse.ArgumentParser(description="Run the Timestral zero-shot benchmark.")
    parser.add_argument("--dataset", choices=list(dataset_classes), default="belgium")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float,
                        default=list(SAMPLING_RATES), help="Sampling percentages to evaluate.")
    parser.add_argument("--runs", type=int, default=1, help="Runs per sampling rate.")
    parser.add_argument("--model_id", default=DEFAULT_CHECKPOINT,
                        help="Timestral checkpoint; the plan fixes 123M.")
    parser.add_argument("--context_length", type=int, default=None,
                        help="Cap the context in points (model limit is 32768). Default keeps the "
                             "whole training history, as the zero-shot benchmarks do.")
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

    from timestral import TimestralForecaster

    dataset = dataset_classes[args.dataset]()
    device = torch_device(args.gpu)
    forecaster = TimestralForecaster.from_pretrained(args.model_id, device=device)
    for sampling_rate in args.sampling_rates:
        for run_num in range(1, args.runs + 1):
            try:
                paper_forecasting_train(
                    args.dataset, dataset, run_num, sampling_rate, forecaster,
                    args.context_length, args.exog_level, args.results_dir,
                )
            except Exception as error:  # keep the other runs alive
                logging.error(
                    "Timestral run %d at sampling %s failed: %s",
                    run_num, sampling_tag(sampling_rate), error, exc_info=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
