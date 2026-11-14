import argparse
import inspect
import os
import sys
import warnings
import logging
import time
import gc
from pathlib import Path
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
from utils.device import GPU_HELP, uses_gpu
from utils.exogenous.feature_store import static_summary
from utils.protocol import COVARIATE_LEVEL, EXOG_LEVEL_CHOICES, run_dir
from utils.tuning import load_best_params
from gluonts.mx.model.seq2seq import MQRNNEstimator
from gluonts.mx.model.seq2seq import MQCNNEstimator
from gluonts.mx.model.deep_factor import DeepFactorEstimator
from gluonts.mx.model.wavenet import WaveNetEstimator
from gluonts.mx.model.tft import TemporalFusionTransformerEstimator
from gluonts.mx.model.deepar import DeepAREstimator


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

def to_gluonts_dataset(df, freq, start, dynamic_columns=None, static_row=None):
    """Build a ListDataset entry, with GluonTS covariates when they are requested.

    GluonTS splits covariates by *dataset field*, not by model role:
    ``feat_dynamic_real`` may extend past the target (so it is the only channel
    that can carry a known-future value into the horizon), and ``feat_static_cat``
    carries per-series categories.  Columns that are only known up to the origin
    cannot be expressed here without inventing horizon values, so they are passed
    for the context and left out of the horizon by the caller.
    """
    entry = {"start": start, "target": df["y"].to_numpy(dtype="float32")}
    if dynamic_columns:
        entry["feat_dynamic_real"] = df[dynamic_columns].to_numpy(dtype="float32").T
    if static_row is not None:
        entry["feat_static_cat"] = static_row
    return ListDataset([entry], freq=freq)


def _covariate_kwargs(model_class, dynamic_count, static_cardinalities):
    """Estimator arguments for covariates, limited to what the estimator accepts."""
    import inspect

    try:
        accepted = inspect.signature(model_class.__init__).parameters
    except (TypeError, ValueError):
        accepted = {}
    kwargs = {}
    if dynamic_count and "dynamic_dims" in accepted:
        kwargs["dynamic_dims"] = [dynamic_count]
    if static_cardinalities and "static_cardinalities" in accepted:
        kwargs["static_cardinalities"] = list(static_cardinalities)
    return kwargs


def _covariate_columns(df, exog_level):
    """(known-future, static) column lists for the GluonTS estimators."""
    from utils.exogenous.feature_store import covariate_roles

    feature_columns = [c for c in df.columns if c not in ("unique_id", "ds", "y")]
    roles = covariate_roles(feature_columns)
    static = [c for c in roles["past_only"] if c.startswith("static_")]
    # TFT has no past-only dynamic channel in GluonTS, so measured weather and
    # lagged neighbours stay out of this model instead of being carried forward.
    known_future = roles["known_future"]
    logging.info(
        "%s: %d known-future dynamic covariates, %d static covariates, "
        "%d past-only dynamic covariates not representable in GluonTS",
        exog_level, len(known_future), len(static),
        len(roles["past_only"]) - len(static),
    )
    return known_future, static


def train_gluonts_model(y_df, model_name, save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, exog_level="L0", params_override=None):
    current_seed = int(time.time()) % (2**32 - 1)
    logging.info(f"Using random seed: {current_seed}")
    y_df = y_df.iloc[::int(100 / sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df
    start_idx = pd.Timestamp(train_df["ds"].iloc[0])

    known_future, static = ([], [])
    if exog_level != "L0":
        known_future, static = _covariate_columns(y_df, exog_level)

    static_row = None
    if static:
        # One modal value per static column: ``.iloc[0]`` would freeze a
        # daily-changing controller id into whatever the first row happened to be.
        static_table, varying = static_summary(y_df, static)
        if varying:
            logging.warning(
                "static covariates that are not constant per series (mode is used): %s",
                ", ".join(varying),
            )
        static_row = static_table.iloc[0][static].to_numpy(dtype="int32")

    train_ds = to_gluonts_dataset(
        train_df, freq, start_idx,
        dynamic_columns=known_future or None,
        static_row=static_row,
    )
    test_ds = to_gluonts_dataset(
        test_df, freq, start_idx,
        dynamic_columns=known_future or None,
        static_row=static_row,
    )
    logging.info(f"Training {model_name} model on {len(train_df)} samples...")

    covariate_kwargs = _covariate_kwargs(
        model_class,
        dynamic_count=len(known_future),
        static_cardinalities=[int(y_df[c].max()) + 1 for c in static],
    )
    if (known_future or static) and not covariate_kwargs:
        logging.warning(
            "%s accepts no covariates in this GluonTS version; the %d covariate columns are "
            "dropped and the run is logged as covariate-free", model_name,
            len(known_future) + len(static),
        )
    if params_override:
        accepted = set(inspect.signature(model_class.__init__).parameters)
        protected = {"freq", "prediction_length", "trainer", "dynamic_dims",
                     "static_cardinalities", "past_length"}
        applied = {k: v for k, v in params_override.items()
                   if k in accepted and k not in protected}
        if applied:
            logging.info("%s tuned parameters: %s", model_name, applied)
            covariate_kwargs.update(applied)
    estimator = model_class(
        freq=freq,
        prediction_length=forecast_horizon,
        **covariate_kwargs,
        trainer=Trainer(
            epochs=epochs,
            num_batches_per_epoch=1,
            ctx=ctx,
            hybridize=model_class not in {MQCNNEstimator, MQRNNEstimator}
        )
    )
    predictor = estimator.train(training_data=train_ds)
    forecast_it, _ = make_evaluation_predictions(dataset=test_ds, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
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

def train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, model_name, model_class, sampling_rate, epochs, ctx, exog_level="L0", params_override=None):
    setup_model_logger(save_dir)
    metrics = []
    start_time = time.time()
    logging.info(f"...Start training using {model_name} with {sampling_rate}% sampling rate, exog level {exog_level}...")
    forecast_horizon = int(192 / (100 / sampling_rate))
    freq = f"{int(15 * (100 / sampling_rate))}min"
    if dataset_name == "belgium":
        logging.info("Training PV models")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt, exog_level=exog_level)
            _, pv_mae, pv_rmse = train_gluonts_model(pv_data, f"PV_house_{house}", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, exog_level, params_override)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        logging.info("Training BESS models")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt, exog_level=exog_level)
            _, b_mae, b_rmse = train_gluonts_model(battery_data, f"BESS_house_{house}", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, exog_level, params_override)
            metrics.append({"model": f"bess_house_{house}", "MAE": b_mae, "RMSE": b_rmse})
    elif dataset_name == "germany":
        logging.info("Training Germany load model")
        germany_data = dataset.get_inputs_for_load(start_dt, end_dt, exog_level=exog_level)
        _, load_mae, load_rmse = train_gluonts_model(germany_data, "germany_load", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, exog_level, params_override)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        logging.info("Training London load model")
        london_data = dataset.get_inputs_for_load(exog_level=exog_level)
        _, load_mae, load_rmse = train_gluonts_model(london_data, "london_load", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, exog_level, params_override)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        logging.info("Training Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            cust_data = dataset.get_inputs_for_zonnedael_consumption(customer_id, exog_level=exog_level)
            _, c_mae, c_rmse = train_gluonts_model(
                cust_data, f"zonnedael_customer_{customer_id}", save_dir, model_class, sampling_rate, forecast_horizon, freq, epochs, ctx, exog_level, params_override
            )
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": c_mae, "RMSE": c_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End training...")
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

def paper_forecasting_train(dataset_name, dataset, sampling_rates, models, runs, epochs, gpu, ctx, results_dir=PROJECT_ROOT / "results", exog_level="L0", params_by_model=None):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    params_by_model = params_by_model or {}
    for sampling_rate in sampling_rates:
        for model_name, model_class in models.items():
            # Run 10 times
            for run_num in range(1, runs + 1):
                try:
                    gc.collect()
                    if uses_gpu(gpu):
                        mx.nd.waitall()
                    save_dir = run_dir(results_dir, dataset_name, model_name, sampling_rate,
                                       run_num, epochs=epochs, style="epochs", exog_level=exog_level)
                    train_all_models(dataset_name, dataset, start_dt, end_dt, save_dir, model_name, model_class, sampling_rate, epochs, ctx, exog_level, params_by_model.get(model_name))
                    gc.collect()
                    if uses_gpu(gpu):
                        mx.nd.waitall()
                except Exception as e:
                    logging.error(f"Skipping model {model_name} run {run_num} at sampling {sampling_rate}% due to error: {str(e)}", exc_info=True)
                    continue

if __name__ == "__main__":
    model_classes = {
        "DeepAR": DeepAREstimator,
        "DeepFactor": DeepFactorEstimator,
        "MQCNN": MQCNNEstimator,
        "MQRNN": MQRNNEstimator,
        "TemporalFusionTransformer": TemporalFusionTransformerEstimator,  # TFT
        "WaveNet": WaveNetEstimator
    }
    dataset_classes = dict(belgium=DatasetBelgiumNF, germany=DatasetGermanyNF, london=DatasetLondonNF, zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run GluonTS model benchmarks.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--models", nargs="+", choices=list(model_classes), default=list(model_classes), help="GluonTS models to evaluate.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--exog_level", "--exog-level",
                        choices=list(EXOG_LEVEL_CHOICES), default=None,
                        help="Ladder level (default L0, which reproduces the paper baseline). "
                             "Mutually exclusive with --use_covariates.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--params_json", type=Path, default=None,
                        help="Tuned parameters from scripts/tune_models.py (JSON).")
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--use_covariates", "--use-covariates", action="store_true",
                        help="Use every exogenous variable the dataset has, in one block, "
                             "instead of one rung of the ladder (same as --exog_level ALL).")
    args = parser.parse_args()
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
    dataset = dataset_classes[args.dataset]()
    selected_models = {name: model_classes[name] for name in args.models}
    ctx = mx.gpu(args.gpu) if uses_gpu(args.gpu) else mx.cpu()
    tuned = load_best_params(args.params_json) if args.params_json else {}
    params_by_model = {name: tuned for name in selected_models} if tuned else {}
    paper_forecasting_train(args.dataset, dataset, args.sampling_rates, selected_models, args.runs, args.epochs, args.gpu, ctx, args.results_dir, exog_level=args.exog_level, params_by_model=params_by_model)
