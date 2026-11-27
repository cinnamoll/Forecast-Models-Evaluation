#!/usr/bin/env python
"""TimeXer on the covariate flow, as a single-model entry point.

TimeXer is a NeuralForecast model, so the training path is the one of
``models/models_neuralforecast.py`` — same loaders, same ``scale_covariates``
bounds, same ``Epochs_<steps>_<run>`` tree — and this file imports that path
instead of copying it; a second implementation of the window split is how the
benchmark grew two incomparable numbers before.

It exists as its own script for two reasons:

* TimeXer needs ``n_series=1``: its exogenous attention is per series, so one
  fit covers one series (Belgium's eight series are eight fits, not one).
* The covariate table has to be explainable. The default block is the study's
  set -- hour, dayofweek, humidity, temperature, cloud_cover, plus
  ``day_ahead_price`` and ``soc`` on Belgium -- and ``--covariates`` overrides
  it with a named subset, exactly like the zero-shot checkpoints
  (``models/fmodel_*.py``).  So the TimeSHAP game in ``scripts/xai_timeshap.py``
  and the benchmark cell describe one input mapping rather than two.  Hardware
  ratings (``static_*``) are never used: the variant has no static channel.

Covariate roles come from ``utils.covariates.roles_for``: measured columns ->
``hist_exog_list`` (``TimeXerHist`` reads them over the look-back window),
announced columns (calendar, price) -> ``futr_exog_list``.

Usage
-----
    conda run -n neuralforecast python models/model_timexer.py \
        --dataset germany --sampling_rates 100 --runs 1 --exog_level COV

    # a narrower cell: only the calendar and two weather channels
    conda run -n neuralforecast python models/model_timexer.py \
        --dataset belgium --sampling_rates 100 --runs 1 --exog_level COV \
        --covariates hour dayofweek temperature cloud_cover
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.models_neuralforecast import TimeXerHist, neural_forecast_model, setup_model_logger
from utils.covariates import TIMESHAP_COLUMNS, roles_for
from utils.dataset_config import (DatasetBelgiumNF, DatasetGermanyNF, DatasetLondonNF,
                                  DatasetZonnedaelNF)
from utils.device import GPU_HELP, uses_gpu
from utils.metrics import plot_model_metrics
from utils.protocol import DATASET_STEP_MINUTES, horizon_for, run_dir

PAPER_WINDOW = (pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-04-01 00:00:00", tz="UTC"))

LABELS = ("unique_id", "ds", "y")

#: Model name used everywhere in the tree and in ``scripts/run_benchmark.py``.
MODEL_NAME = "TimeXer"

#: The covariates this study asks of TimeXer: the five variables of the TimeSHAP
#: set.  Belgium adds the auction price and the battery state of charge; hardware
#: ratings (``static_*``) are never used -- the variant has no static channel.
DEFAULT_COVARIATES = tuple(TIMESHAP_COLUMNS)
BELGIUM_COVARIATES = DEFAULT_COVARIATES + ("day_ahead_price", "soc")


def covariate_columns(frame) -> list[str]:
    """Everything a series carries besides the labels."""
    return [c for c in frame.columns if c not in LABELS]


def restrict_covariates(frame, model_name: str, covariates, level: str):
    """Trim a series to ``covariates``; a no-op without the flag.

    The labels are always kept, so the trimmed frame is still a NeuralForecast
    input.  A column the series does not carry is reported and skipped rather
    than raised, because the same ``--covariates`` list is asked of every series
    of a dataset and ``soc`` only exists for the BESS ones.  A list that matches
    *nothing* is an error: that is a typo, not a dataset difference.

    ``static_*`` hardware ratings are refused even when asked for: this variant
    has no static covariate channel, and passing one raises inside nf.
    """
    if not covariates:
        return frame
    if level == "L0":
        logging.warning("%s: --covariates ignored at L0 (the paper's baseline has no "
                        "covariates); the level decides the block", model_name)
        return frame
    blocked = [c for c in covariates if c.startswith("static_")]
    if blocked:
        logging.warning("%s: static_* hardware ratings are not used, skipped: %s",
                        model_name, ", ".join(blocked))
    covariates = [c for c in covariates if not c.startswith("static_")]
    available = covariate_columns(frame)
    keep = [c for c in covariates if c in available]
    missing = [c for c in covariates if c not in available]
    if missing:
        logging.warning("%s: requested covariates absent from this series, skipped: %s",
                        model_name, ", ".join(missing))
    if not keep:
        raise KeyError(f"{model_name}: none of the requested covariates is in the frame "
                       f"(available: {available})")
    return frame[[c for c in LABELS if c in frame.columns] + keep]


def train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq,
                     forecast_horizon, sampling_rate, epochs, gpu, level="L0",
                     covariates=None):
    """Run TimeXer on every series of ``dataset_name`` at one sampling rate."""
    setup_model_logger(save_dir)
    if covariates is None:
        covariates = BELGIUM_COVARIATES if dataset_name == "belgium" else DEFAULT_COVARIATES
    metrics = []
    logging.info("...Start training using %s with %s%% sampling rate, covariate level %s...",
                 MODEL_NAME, sampling_rate, level)

    def run(frame, series_name):
        frame = restrict_covariates(frame, series_name, covariates, level)
        columns = covariate_columns(frame)
        roles = roles_for(frame[columns]) if columns else {
            "known_future": [], "past_only": [], "static": [], "categorical": []}
        logging.info(
            "%s: %d covariate column(s) -> %d announced, %d measured, %d static",
            series_name, len(columns), len(roles["known_future"]),
            len(roles["past_only"]), len(roles["static"]))
        _, mae, rmse = neural_forecast_model(
            frame, series_name, save_dir, freq, forecast_horizon, sampling_rate,
            epochs, gpu, nf_model_name=MODEL_NAME, NFmodel=TimeXerHist, level=level)
        metrics.append({"model": series_name.lower(), "MAE": mae, "RMSE": rmse})

    if dataset_name == "belgium":
        logging.info("Training PV models")
        for house in [1, 2, 3, 4]:
            run(dataset.get_inputs_for_pv(house, start_dt, end_dt, level=level),
                f"PV_house_{house}")
        logging.info("Training BESS models")
        for house in [1, 2, 3, 4]:
            run(dataset.get_inputs_for_battery(house, start_dt, end_dt, level=level),
                f"BESS_house_{house}")
    elif dataset_name == "germany":
        logging.info("Training Germany load model")
        run(dataset.get_inputs_for_load(start_dt, end_dt, level=level), "germany_load")
    elif dataset_name == "london":
        logging.info("Training London load model")
        run(dataset.get_inputs_for_load(level=level), "london_load")
    elif dataset_name == "zonnedael":
        logging.info("Training Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            run(dataset.get_inputs_for_zonnedael_consumption(customer_id, level=level),
                f"zonnedael_customer_{customer_id}")
    else:
        raise ValueError(f"unknown dataset {dataset_name!r}")

    pd.DataFrame(metrics).to_csv(
        str(Path(save_dir) / "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    logging.info("...End training...")


def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, epochs, gpu,
                            results_dir=PROJECT_ROOT / "results", level="L0",
                            covariates=None):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt, end_dt = PAPER_WINDOW
    base_step = DATASET_STEP_MINUTES[dataset_name]
    freq = f"{int(base_step * (100 / sampling_rate))}min"
    forecast_horizon = horizon_for(sampling_rate)
    save_dir = run_dir(results_dir, dataset_name, MODEL_NAME, sampling_rate, run_num,
                       epochs=epochs, style="epochs", exog_level=level)
    try:
        train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, freq,
                         forecast_horizon, sampling_rate, epochs, gpu, level, covariates)
    except Exception as error:  # one failed series must not lose the others' numbers
        logging.error("TimeXer run %s at %s%% failed: %s", run_num, sampling_rate,
                      error, exc_info=True)
        raise
    finally:
        gc.collect()
        if uses_gpu(gpu):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    dataset_classes = dict(belgium=DatasetBelgiumNF, germany=DatasetGermanyNF,
                           london=DatasetLondonNF, zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run TimeXer on the covariate flow.")
    parser.add_argument("--dataset", choices=list(dataset_classes), default="germany",
                        help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float,
                        default=[25, 100 / 3, 50, 100],
                        help="Sampling percentages to evaluate.")
    parser.add_argument("--runs", type=int, default=10,
                        help="Runs per cell (seeds are averaged).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training steps per model (NeuralForecast max_steps).")
    parser.add_argument("--exog_level", "--exog-level", choices=list(("L0", "COV")),
                        default="COV",
                        help="Covariate block: COV (day/hour + weather + price + static + state) "
                             "or L0 (paper baseline).")
    parser.add_argument("--covariates", nargs="+", default=None,
                        help="Covariate columns to use. Omit for the study's set: "
                             "hour dayofweek humidity temperature cloud_cover "
                             "(Belgium adds day_ahead_price soc). static_* is refused.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results",
                        help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    for sampling_rate in args.sampling_rates:
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(args.dataset, dataset, run_num, sampling_rate,
                                    args.epochs, args.gpu, args.results_dir,
                                    level=args.exog_level, covariates=args.covariates)
