import argparse
import inspect
import json
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
from utils.exogenous.feature_store import partition_for_model, static_summary
from utils.device import GPU_HELP, lightning_device_config, uses_gpu
from utils.protocol import COVARIATE_LEVEL, EXOG_LEVEL_CHOICES, run_dir
from utils.tuning import load_best_params
import torch
import time
import gc
from neuralforecast import NeuralForecast
from neuralforecast.models import (BiTCN, DeepNPTS, Informer, NBEATS, NHITS, NLinear, PatchTST,
    TCN, TiDE, TimesNet, TimeXer, VanillaTransformer, iTransformer)


# Custom logger setup per model
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

# Core training function for each model type
def neural_forecast_model(
    y_df, model_name, save_dir, freq, forecast_horizon, sampling_rate, epochs,
    gpu, nf_model_name="TimesNet", NFmodel=None, params_override=None
):
    current_seed = int(time.time()) % (2**32 - 1)  # Use current time for true randomness
    logging.info(f"Using random seed: {current_seed}")
    # Split data
    y_df = y_df.iloc[::int(100/sampling_rate)]
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]
    # train_df = full_train_df.iloc[::int(100/sampling_rate)]
    # plt.plot(train_df['ds'], train_df['y'])
    # plt.show()
    logging.info(f"Training {nf_model_name} {model_name} model on {len(train_df)} samples ({sampling_rate}% of original training set)...")
    device_config = lightning_device_config(gpu)

    feature_columns = [c for c in y_df.columns if c not in ("unique_id", "ds", "y")]
    exog_kwargs, exog_report = partition_for_model(NFmodel, feature_columns)
    if feature_columns and not exog_kwargs:
        logging.warning(
            "%s supports no exogenous inputs at all (NeuralForecast flags futr/hist/stat = "
            "%s); %d covariate columns are present and will be ignored, so this run is the "
            "L0 model for this architecture",
            nf_model_name, exog_report.get("capabilities"), len(feature_columns),
        )
    else:
        if exog_kwargs:
            logging.info(
                "%s exogenous inputs: %s", nf_model_name,
                ", ".join(f"{key}={len(value)}" for key, value in exog_kwargs.items()),
            )
        if exog_report.get("n_dropped"):
            logging.warning(
                "%s cannot use %d of %d covariate columns (%s); they are dropped for this "
                "model only",
                nf_model_name, exog_report["n_dropped"], len(feature_columns),
                ", ".join(f"{kind}={len(cols)}" for kind, cols in exog_report["dropped"].items()),
            )
    # Written next to the metrics so the paper can say which model actually
    # consumed covariates instead of inferring it from a shared ablation column.
    try:
        with open(os.path.join(save_dir, "exogenous_support.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {"model": nf_model_name,
                 "covariate_columns": feature_columns,
                 "accepted": {k: v for k, v in exog_kwargs.items()},
                 **exog_report},
                handle, indent=2, default=str,
            )
    except OSError:
        logging.warning("could not write exogenous_support.json in %s", save_dir)

    model_kwargs = dict(
        h=forecast_horizon,
        input_size=forecast_horizon * 2,
        max_steps=epochs,
        random_seed=current_seed,
        **device_config,
        **exog_kwargs,
    )
    if nf_model_name in ["TimeXer", "iTransformer"]:
        model_kwargs["n_series"] = 1

    # Tuned hyper-parameters (plan_updated.md section 5).  Protocol keys are
    # protected: a search must not change the window, the horizon or the
    # covariate lists, only the architecture and the optimiser.
    if params_override:
        accepted = set(inspect.signature(NFmodel.__init__).parameters)
        protected = {"h", "input_size", "stat_exog_list", "hist_exog_list", "futr_exog_list",
                     "n_series", "max_steps", "random_seed", "accelerator", "devices"}
        applied = {k: v for k, v in params_override.items()
                   if k in accepted and k not in protected}
        if applied:
            logging.info("%s tuned parameters: %s", nf_model_name, applied)
            model_kwargs.update(applied)

    static_df = None
    if exog_kwargs.get("stat_exog_list"):
        static_df, varying = static_summary(train_df, exog_kwargs["stat_exog_list"])
        if varying:
            logging.warning(
                "static covariates that are not constant per series (mode is used): %s",
                ", ".join(varying),
            )

    nf = NeuralForecast(models=[NFmodel(**model_kwargs)], freq=freq)
    try:
        nf.fit(df=train_df, val_size=forecast_horizon, static_df=static_df)
    except Exception as error:
        # Some models declare futr_exog_list in their signature and then reject it
        # at build time (PatchTST in NeuralForecast 3.0 raises a bare Exception).
        # Falling back to the past window keeps the covariates available without
        # pretending the model consumed the horizon values; the warning lands in
        # training_log.txt and in the run's metrics row.
        if "future exogenous" not in str(error).lower() or not exog_kwargs.get("futr_exog_list"):
            raise
        moved = exog_kwargs.pop("futr_exog_list")
        model_kwargs.pop("futr_exog_list", None)
        logging.warning(
            "%s %s does not accept future exogenous variables; moving %d columns to hist_exog_list",
            nf_model_name, model_name, len(moved),
        )
        if "hist_exog_list" in model_kwargs:
            exog_kwargs["hist_exog_list"] = list(exog_kwargs.get("hist_exog_list", [])) + moved
            model_kwargs["hist_exog_list"] = exog_kwargs["hist_exog_list"]
        nf = NeuralForecast(models=[NFmodel(**model_kwargs)], freq=freq)
        nf.fit(df=train_df, val_size=forecast_horizon, static_df=static_df)

    if exog_kwargs.get("futr_exog_list"):
        futr_df = test_df[["unique_id", "ds", *exog_kwargs["futr_exog_list"]]]
        forecast_df = nf.predict(futr_df=futr_df)
    else:
        forecast_df = nf.predict()
    y_pred = forecast_df[nf_model_name].values
    y_true = test_df["y"].values
    mae, rmse = calculate_metrics(y_pred, y_true)
    logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    forecast_plot_and_csv(
        pd.DataFrame({"datetime": test_df["ds"], "Actual": y_true, "Forecast": y_pred}).set_index("datetime"),
        model_name,
        save_dir
    )
    return nf, mae, rmse

# Main pipeline for training all models
def train_all_models(
    dataset_name, dataset, start_dt, end_dt, save_dir, freq, forecast_horizon,
    sampling_rate, epochs, gpu, nf_model_name, NFmodel, exog_level="L0",
    params_override=None
):
    setup_model_logger(save_dir)  # Ensure logging is set up first
    metrics = []
    start_time = time.time()
    logging.info(f"...Start training for {epochs} epochs using {nf_model_name} (exog level {exog_level})...")
    if dataset_name == "belgium":
        # Train PV model
        logging.info("Training PV models")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt, exog_level=exog_level)
            _, pv_mae, pv_rmse = neural_forecast_model(
                pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, sampling_rate,
                epochs, gpu, nf_model_name, NFmodel, params_override
            )
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        # Train battery model
        logging.info("Training BESS models")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt, exog_level=exog_level)
            _, battery_mae, battery_rmse = neural_forecast_model(
                battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon,
                sampling_rate, epochs, gpu, nf_model_name, NFmodel, params_override
            )
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        logging.info("Training Germany load model")
        germany_data = dataset.get_inputs_for_load(start_dt, end_dt, exog_level=exog_level)
        _, load_mae, load_rmse = neural_forecast_model(
            germany_data, "germany_load", save_dir, freq, forecast_horizon, sampling_rate,
            epochs, gpu, nf_model_name, NFmodel, params_override
        )
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        # Train london load model
        logging.info("Training London load model")
        london_data = dataset.get_inputs_for_load(exog_level=exog_level)
        _, load_mae, load_rmse = neural_forecast_model(
            london_data, "london_load", save_dir, freq, forecast_horizon, sampling_rate,
            epochs, gpu, nf_model_name, NFmodel, params_override
        )
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        # Train zonnedael customers
        logging.info("Training Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            customer_data = dataset.get_inputs_for_zonnedael_consumption(customer_id, exog_level=exog_level)
            _, cust_mae, cust_rmse = neural_forecast_model(
                customer_data, f"zonnedael_customer_{customer_id}", save_dir, freq,
                forecast_horizon, sampling_rate, epochs, gpu, nf_model_name, NFmodel,
                params_override
            )
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info("...End training...")
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, epochs, models, gpu, results_dir=PROJECT_ROOT / "results", exog_level="L0", params_by_model=None):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end_dt = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    # Adjust frequency based on sampling rate
    freq_str = f"{int(15 * (100 / sampling_rate))}min"  # Convert to string format for frequency
    forecast_horizon = int(192 / (100 / sampling_rate))  # 2 days of 15-minute intervals
    params_by_model = params_by_model or {}
    for model_name, model_class in models.items():
        try:
            # Clear memory before training a new model
            gc.collect()
            if uses_gpu(gpu):
                torch.cuda.empty_cache()
            save_dir = run_dir(results_dir, dataset_name, model_name, sampling_rate,
                               run_num, epochs=epochs, style="epochs", exog_level=exog_level)
            train_all_models(
                dataset_name, dataset, start_dt, end_dt, save_dir, freq_str, forecast_horizon,
                sampling_rate, epochs, gpu, nf_model_name=model_name, NFmodel=model_class,
                exog_level=exog_level, params_override=params_by_model.get(model_name)
            )
            # Clear memory after training a model
            gc.collect()
            if uses_gpu(gpu):
                torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Skipping model {model_name} due to error: {str(e)}", exc_info=True)
            continue

if __name__ == "__main__":
    model_classes = {
        "BiTCN": BiTCN,
        "DeepNPTS": DeepNPTS,
        "Informer": Informer,
        "NBEATS": NBEATS,
        "NHITS": NHITS,
        "NLinear": NLinear,
        "PatchTST": PatchTST,
        "TCN": TCN,
        "TiDE": TiDE,
        "TimesNet": TimesNet,
        "TimeXer": TimeXer,
        "iTransformer": iTransformer,
        "VanillaTransformer": VanillaTransformer
    }
    dataset_classes = dict(belgium=DatasetBelgiumNF, germany=DatasetGermanyNF, london=DatasetLondonNF, zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run NeuralForecast model benchmarks.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--models", nargs="+", choices=list(model_classes), default=list(model_classes), help="NeuralForecast models to evaluate.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs.")
    parser.add_argument("--epochs", type=int, default=100, help="Training steps per model.")
    parser.add_argument("--exog_level", "--exog-level", choices=list(EXOG_LEVEL_CHOICES), default=None,
                        help="Ladder level (default L0, which reproduces the paper baseline). "
                             "Mutually exclusive with --use_covariates.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--params_json", type=Path, default=None,
                        help="Tuned parameters from scripts/tune_models.py (JSON).")
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
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
    tuned = load_best_params(args.params_json) if args.params_json else {}
    params_by_model = {name: tuned for name in selected_models} if tuned else {}
    for sampling_rate in args.sampling_rates:
        # Loop from run_num = 1 to 10
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(args.dataset, dataset, run_num, sampling_rate, args.epochs, selected_models, args.gpu, args.results_dir, exog_level=args.exog_level, params_by_model=params_by_model)
