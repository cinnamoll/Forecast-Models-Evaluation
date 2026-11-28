import argparse
import inspect
import json
import os
import sys
import warnings
import logging
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import mxnet as mx
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.metrics import calculate_metrics, forecast_plot_and_csv, plot_model_metrics
from utils.dataset_config import DatasetBelgiumNF, DatasetGermanyNF, DatasetLondonNF, DatasetZonnedaelNF
from utils.covariates import roles_for, scale_covariates
from utils.device import GPU_HELP, uses_gpu
from utils.protocol import freq_for, horizon_for, run_dir
from gluonts.mx.model.seq2seq import MQRNNEstimator
from gluonts.mx.model.seq2seq import MQCNNEstimator
from gluonts.mx.model.deep_factor import DeepFactorEstimator
from gluonts.mx.model.wavenet import WaveNetEstimator
from gluonts.mx.model.tft import TemporalFusionTransformerEstimator
from gluonts.mx.model.deepar import DeepAREstimator


PAPER_WINDOW = (pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-04-01 00:00:00", tz="UTC"))

#: Estimator arguments the benchmark controls.  A search may change the
#: architecture and the optimiser, never the window or the covariate wiring.
PROTECTED = ("freq", "prediction_length", "trainer", "context_length",
             "dynamic_feature_dims", "dynamic_cardinalities",
             "past_dynamic_features")


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


def covariate_spec(frame, roles, level, train_end):
    """Per-column dataset fields and the estimator kwargs describing them.

    GluonTS's temporal fusion transformer does not take one covariate matrix: it
    takes **one named field per covariate** and stacks them itself, and it needs
    to be told, per name, whether the field is categorical (codes plus a
    cardinality) or real (one dimension).  Measured covariates are additionally
    named in ``past_dynamic_features``, which makes the instance splitter cut
    their future part — that is the only way the model can *see* weather without
    being handed weather inside the horizon.

    ``frame`` decides the length of every field: the estimator stacks these
    arrays with its own time-feature block, whose length is the one of the target
    the predictor is handed.  So pass the rows of that same entry — ``train_df``
    for the training entry, ``y_df`` for the prediction entry — never a frame of
    a different length.  (``make_evaluation_predictions`` truncates the target by
    ``horizon`` before predicting and ``AddTimeFeatures`` adds it back, so the
    prediction arrays are ``len(y_df)``, not ``len(y_df) + horizon``.)
    ``train_end`` is the number of leading rows the scaling may look at, so both
    entries are scaled with the training bounds.
    """
    feature_columns = [c for c in frame.columns if c not in ("unique_id", "ds", "y")]
    if level == "L0" or not feature_columns:
        return {}, {}

    scaled = scale_covariates(frame[feature_columns], roles, train_end=train_end)
    entry, kwargs = {}, {}
    dynamic_dims, dynamic_cards = {}, {}

    for column in roles["known_future"]:
        values = scaled[column].to_numpy(dtype="float32")
        if column in roles["categorical"]:
            codes = np.rint(values).astype("float32")
            entry[column] = codes
            dynamic_cards[column] = int(codes.max()) + 1
        else:
            entry[column] = values.reshape(1, -1)
            dynamic_dims[column] = 1

    for column in roles["past_only"]:
        values = scaled[column].to_numpy(dtype="float32").reshape(1, -1)
        entry[column] = values
        dynamic_dims[column] = 1

    if dynamic_dims:
        kwargs["dynamic_feature_dims"] = dynamic_dims
    if dynamic_cards:
        kwargs["dynamic_cardinalities"] = dynamic_cards
    if roles["past_only"]:
        # The names must also appear in ``dynamic_feature_dims``: the estimator
        # pops them out of that dict to build the past-only channel.
        kwargs["past_dynamic_features"] = list(roles["past_only"])
    return entry, kwargs


def to_gluonts_dataset(df, freq, start, entry_extra=None):
    entry = {"start": start, "target": df["y"].to_numpy(dtype="float32")}
    if entry_extra:
        entry.update(entry_extra)
    return ListDataset([entry], freq=freq)


def split_params(model_class, params_override):
    """Tuned values -> (estimator kwargs, Trainer kwargs)."""
    params_override = dict(params_override or {})
    try:
        estimator_accepted = set(inspect.signature(model_class.__init__).parameters)
    except (TypeError, ValueError):
        estimator_accepted = set()
    try:
        trainer_accepted = set(inspect.signature(Trainer.__init__).parameters)
    except (TypeError, ValueError):
        trainer_accepted = set()
    estimator_kwargs, trainer_kwargs, ignored = {}, {}, {}
    for key, value in params_override.items():
        if key in PROTECTED:
            continue
        if key in estimator_accepted:
            estimator_kwargs[key] = value
        elif key in trainer_accepted:
            trainer_kwargs[key] = value
        else:
            ignored[key] = "not an estimator or Trainer argument"
    return estimator_kwargs, trainer_kwargs, ignored


def train_gluonts_model(y_df, model_name, save_dir, model_class, sampling_rate, forecast_horizon,
                        freq, epochs, ctx, level="L0", params_override=None):
    current_seed = int(time.time()) % (2**32 - 1)
    logging.info(f"Using random seed: {current_seed}")
    y_df = y_df.iloc[::int(100 / sampling_rate)].reset_index(drop=True)
    train_df = y_df.iloc[:-forecast_horizon]
    start_idx = pd.Timestamp(train_df["ds"].iloc[0])

    feature_columns = [c for c in y_df.columns if c not in ("unique_id", "ds", "y")]
    roles = roles_for(y_df[feature_columns]) if feature_columns else {
        "known_future": [], "past_only": [], "categorical": []}
    # The training entry ends at the forecast origin, so its covariate arrays must
    # end there too: the estimator stacks them with a time-feature block of
    # ``len(past_target)`` rows and refuses to stack arrays of two lengths.
    train_entry_extra, covariate_kwargs = covariate_spec(
        train_df, roles, level, train_end=len(train_df))
    # The prediction entry keeps the whole series and needs NO horizon block.
    # ``make_evaluation_predictions`` splits the series first: the target the
    # predictor sees is the first ``len - horizon`` points, and ``AddTimeFeatures``
    # (``is_train=False``) then builds ``len(target) + horizon`` rows — which is
    # exactly ``len(y_df)``, the length of these arrays.  Padding them any further
    # is what fails with "size N and size N+H" at predict time.
    test_entry_extra, _ = covariate_spec(y_df, roles, level, train_end=len(train_df))
    if level == "L0":
        logging.info("%s receives no covariates at L0 (paper baseline)", model_name)
    else:
        logging.info(
            "%s covariates: %d known-future fields, %d past-only dynamic, %d categorical",
            model_name, len(roles["known_future"]), len(roles["past_only"]),
            len(roles["categorical"]),
        )

    train_ds = to_gluonts_dataset(train_df, freq, start_idx, train_entry_extra)
    test_ds = to_gluonts_dataset(y_df, freq, start_idx, test_entry_extra)

    estimator_kwargs, trainer_kwargs, ignored = split_params(model_class, params_override)
    if ignored:
        logging.warning("%s: tuned parameter(s) ignored: %s", model_name, sorted(ignored))
    if estimator_kwargs:
        logging.info("%s tuned estimator parameters: %s", model_name, sorted(estimator_kwargs))
    trainer_kwargs.setdefault("num_batches_per_epoch", 1)
    estimator = model_class(
        freq=freq,
        prediction_length=forecast_horizon,
        **covariate_kwargs,
        **estimator_kwargs,
        trainer=Trainer(
            epochs=epochs,
            ctx=ctx,
            hybridize=model_class not in {MQCNNEstimator, MQRNNEstimator},
            **trainer_kwargs,
        ),
    )
    logging.info(f"Training {model_name} model on {len(train_df)} samples...")
    predictor = estimator.train(training_data=train_ds)
    forecast_it, _ = make_evaluation_predictions(dataset=test_ds, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    # ``forecast.mean`` on a quantile forecast has no mean to return: GluonTS logs
    # "the median is being returned instead" and hands back the p50 quantile.  That
    # is what the published GluonTS numbers are (``QuantileForecastGenerator``), so
    # it is kept so the new cells join the old table -- but it is a median, not a
    # mean of the quantiles, and the log says so on every run.
    y_pred = forecasts[0].mean[:forecast_horizon]
    y_true = y_df["y"].values[-forecast_horizon:]
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": y_df["ds"].values[-forecast_horizon:], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name,
        save_dir
    )
    return predictor, mae, rmse


def train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, model_name, model_class,
                     sampling_rate, epochs, ctx, level="L0", params_override=None):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    forecast_horizon = horizon_for(sampling_rate)
    # The label follows the dataset's native step: LCL is half-hourly, and the
    # published runs labelled it 15 minutes, which made the estimator's time
    # features (built from ``start + i * freq``) run at half speed.
    freq = freq_for(sampling_rate, dataset_name)
    logging.info(f"...Start training using {model_name} with {sampling_rate}% sampling rate, covariate level {level}...")
    if dataset_name == "belgium":
        logging.info("Training PV models")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt, level=level)
            _, pv_mae, pv_rmse = train_gluonts_model(pv_data, f"PV_house_{house}", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, level, params_override)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        logging.info("Training BESS models")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt, level=level)
            _, b_mae, b_rmse = train_gluonts_model(battery_data, f"BESS_house_{house}", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, level, params_override)
            metrics.append({"model": f"bess_house_{house}", "MAE": b_mae, "RMSE": b_rmse})
    elif dataset_name == "germany":
        logging.info("Training Germany load model")
        germany_data = dataset.get_inputs_for_load(start_dt, end_dt, level=level)
        _, load_mae, load_rmse = train_gluonts_model(germany_data, "germany_load", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, level, params_override)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        logging.info("Training London load model")
        london_data = dataset.get_inputs_for_load(level=level)
        _, load_mae, load_rmse = train_gluonts_model(london_data, "london_load", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, level, params_override)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        logging.info("Training Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            cust_data = dataset.get_inputs_for_zonnedael_consumption(customer_id, level=level)
            _, c_mae, c_rmse = train_gluonts_model(
                cust_data, f"zonnedael_customer_{customer_id}", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, level, params_override
            )
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": c_mae, "RMSE": c_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End training...")
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")


def paper_forecasting_train(dataset_name, dataset, sampling_rates, models, runs, epochs, gpu, ctx,
                            results_dir=PROJECT_ROOT / "results", level="L0", params_by_model=None):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt, end_dt = PAPER_WINDOW
    params_by_model = params_by_model or {}
    for sampling_rate in sampling_rates:
        for model_name, model_class in models.items():
            for run_num in range(1, runs + 1):
                try:
                    gc.collect()
                    if uses_gpu(gpu):
                        mx.nd.waitall()
                    save_dir = run_dir(results_dir, dataset_name, model_name, sampling_rate,
                                       run_num, epochs=epochs, style="epochs", exog_level=level)
                    train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, model_name,
                                     model_class, sampling_rate, epochs, ctx, level,
                                     params_by_model.get(model_name))
                    gc.collect()
                    if uses_gpu(gpu):
                        mx.nd.waitall()
                except Exception as e:
                    logging.error(f"Skipping model {model_name} run {run_num} at sampling {sampling_rate}% due to error: {str(e)}", exc_info=True)
                    continue

if __name__ == "__main__":
    model_classes = {
        "TemporalFusionTransformer": TemporalFusionTransformerEstimator,  # TFT
        "DeepAR": DeepAREstimator,
        "DeepFactor": DeepFactorEstimator,
        "MQCNN": MQCNNEstimator,
        "MQRNN": MQRNNEstimator,
        "WaveNet": WaveNetEstimator,
    }
    dataset_classes = dict(belgium=DatasetBelgiumNF, germany=DatasetGermanyNF, london=DatasetLondonNF, zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run GluonTS model benchmarks.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--models", nargs="+", choices=list(model_classes),
                        default=["TemporalFusionTransformer"], help="GluonTS models to evaluate.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--exog_level", "--exog-level", choices=list(("L0", "COV")), default="COV",
                        help="Covariate block: COV (day/hour + weather + price + state) or L0 (paper baseline).")
    parser.add_argument("--params_json", type=Path, default=None, help="Tuned parameters (JSON) from scripts/tune_optuna.py.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    selected_models = {name: model_classes[name] for name in args.models}
    ctx = mx.gpu(args.gpu) if uses_gpu(args.gpu) else mx.cpu()
    tuned = {}
    if args.params_json and args.params_json.exists():
        payload = json.loads(args.params_json.read_text(encoding="utf-8"))
        tuned = dict(payload.get("best_params", payload))
    params_by_model = {name: tuned for name in selected_models} if tuned else {}
    paper_forecasting_train(args.dataset, dataset, args.sampling_rates, selected_models, args.runs,
                            args.epochs, args.gpu, ctx, args.results_dir, level=args.exog_level,
                            params_by_model=params_by_model)
