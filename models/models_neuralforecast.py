import argparse
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
from utils.covariates import roles_for, scale_covariates
from utils.device import GPU_HELP, lightning_device_config, uses_gpu
from utils.protocol import horizon_for, run_dir, DATASET_STEP_MINUTES
import torch
import time
import gc
from neuralforecast import NeuralForecast
from neuralforecast.models import (DeepNPTS, PatchTST, TCN, NBEATS, NHITS, NLinear,
    BiTCN, TiDE, TimesNet, TimeXer, VanillaTransformer, iTransformer)

#: Evaluation window of the paper (Belgium and WPuQ ship a Jan-Mar file).
PAPER_WINDOW = (pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2024-04-01 00:00:00", tz="UTC"))

#: Covariate channels a NeuralForecast model can accept, by role.
COVARIATE_KINDS = ("hist_exog_list", "futr_exog_list", "stat_exog_list")


class TimeXerHist(TimeXer):
    """TimeXer that also takes measured (historical) exogenous variables.

    Stock nf 3.0.2 TimeXer declares ``EXOGENOUS_HIST = False`` and reads only the
    look-back slice of ``futr_exog``, while the paper (arXiv 2402.19072, Eq.1)
    feeds every exogenous series over the look-back window, z_{1:Tex}.  A
    measured channel -- weather, state of charge -- is therefore impossible to
    pass in stock nf.  Here the ``hist_exog`` window becomes extra variate
    tokens beside the announced ones: same variate-level embedding, one token
    per series, no timestamp alignment needed.
    """

    EXOGENOUS_HIST = True

    def forward(self, windows_batch):
        insample_y = windows_batch["insample_y"]
        marks = []
        if self.futr_exog_size > 0:
            # nf hands the announced columns over history + horizon; only the
            # look-back part becomes a token, which is the paper's setting.
            marks.append(windows_batch["futr_exog"][:, :, : self.input_size, :])
        if self.hist_exog_size > 0:
            marks.append(windows_batch["hist_exog"])
        x_mark = None
        if marks:
            x_mark = torch.cat(marks, dim=1)  # [B, C, T, 1]
            batch, n_vars, steps, dim = x_mark.shape
            x_mark = x_mark.reshape(batch, steps, n_vars * dim)
        y_pred = self.forecast(insample_y, x_mark)
        return y_pred.reshape(insample_y.shape[0], self.h, -1)


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


def exog_lists(roles, level):
    """``{list_name: columns}`` for one model at one covariate level.

    ``hist_exog_list`` carries the measured covariates (weather, state of charge)
    and therefore never reaches the horizon; ``futr_exog_list`` carries calendar
    and price, which are announced in advance.
    """
    if level == "L0":
        return {}
    lists = {
        "hist_exog_list": list(roles["past_only"]),
        "futr_exog_list": list(roles["known_future"]),
    }
    return {name: cols for name, cols in lists.items() if cols}


def receptive_field(kernel_size=2, depth=5):
    """Steps of context a causal dilated stack can see: ``1 + (k-1) * sum(2^i)``.

    At the NeuralForecast defaults (``kernel_size=2``, dilations 1..16) that is
    32 steps — about 8 hours at 15 minutes, far shorter than the 192-point
    horizon.  The tuning space therefore searches the depth, not just the widths.
    """
    return 1 + (kernel_size - 1) * sum(2 ** level for level in range(depth))


# Core training function for each model type
def neural_forecast_model(
    y_df, model_name, save_dir, freq, forecast_horizon, sampling_rate, epochs,
    gpu, nf_model_name="DeepNPTS", NFmodel=None, level="L0", params_override=None
):
    current_seed = int(time.time()) % (2**32 - 1)  # Use current time for true randomness
    logging.info(f"Using random seed: {current_seed}")
    # Split data
    y_df = y_df.iloc[::int(100/sampling_rate)].reset_index(drop=True)
    feature_columns = [c for c in y_df.columns if c not in ("unique_id", "ds", "y")]
    roles = roles_for(y_df[feature_columns]) if feature_columns else {
        "known_future": [], "past_only": [], "categorical": []}
    exog = exog_lists(roles, level)
    if exog.get("hist_exog_list") and not getattr(NFmodel, "EXOGENOUS_HIST", True):
        # nf refuses measured covariates for models that only take announced ones
        # (EXOGENOUS_HIST = False).  Dropping is the only honest option: handing a
        # measured column over as ``futr_exog_list`` claims to know tomorrow's
        # weather.  TimeXerHist lifts the restriction in its own forward.
        dropped = exog.pop("hist_exog_list")
        logging.warning("%s %s accepts announced covariates only; dropping %d measured "
                        "covariate(s): %s", nf_model_name, model_name, len(dropped),
                        ", ".join(dropped))
    if level == "L0":
        logging.info("%s receives no covariates at L0 (paper baseline)", nf_model_name)
    elif exog:
        logging.info("%s covariate lists: %s", nf_model_name,
                     {name: len(cols) for name, cols in exog.items()})

    # Neural models need bounded inputs; categorical codes stay as integers
    # because the embedding tables index them.
    if feature_columns:
        scaled = scale_covariates(y_df[feature_columns], roles, train_end=len(y_df) - forecast_horizon)
        y_df = pd.concat([y_df.drop(columns=feature_columns), scaled], axis=1)
    train_df = y_df.iloc[:-forecast_horizon]
    test_df = y_df.iloc[-forecast_horizon:]

    logging.info(f"Training {nf_model_name} {model_name} model on {len(train_df)} samples ({sampling_rate}% of original training set)...")
    device_config = lightning_device_config(gpu)
    model_kwargs = dict(
        h=forecast_horizon,
        input_size=forecast_horizon * 2,
        max_steps=epochs,
        random_seed=current_seed,
        # nf names the forecast column after the class; the callers slice it by
        # nf_model_name (e.g. TimeXerHist arrives as "TimeXer").
        alias=nf_model_name,
        **device_config,
        **exog,
    )
    if nf_model_name in ["TimeXer", "iTransformer"]:
        model_kwargs["n_series"] = 1
    if nf_model_name == "TCN":
        # The receptive field decides whether a TCN can see the horizon at all:
        # 32 steps at the defaults is ~8 h at 15 minutes, far shorter than the
        # 192-point horizon.  Logged so a weak TCN is not read as a weak idea.
        logging.info("TCN receptive field: %d steps (horizon %d)", receptive_field(), forecast_horizon)

    # Tuned hyper-parameters.  Protocol keys stay off limits: a search may change
    # the architecture and the optimiser, never the window or the covariate lists.
    if params_override:
        protected = set(COVARIATE_KINDS) | {"h", "input_size", "max_steps", "random_seed",
                                            "accelerator", "devices", "n_series", "alias"}
        applied = {k: v for k, v in params_override.items() if k not in protected}
        if applied:
            logging.info("%s tuned parameters: %s", nf_model_name, applied)
            model_kwargs.update(applied)

    def fit_and_predict(kwargs):
        nf = NeuralForecast(models=[NFmodel(**kwargs)], freq=freq)
        nf.fit(df=train_df, val_size=forecast_horizon)
        if kwargs.get("futr_exog_list"):
            futr_df = nf.make_future_dataframe()
            # Assigning the values of that frame keeps its tz; assigning
            # ``.values`` into test_df's tz-aware column made it tz-naive and
            # nf.predict then refused to merge it back (datetime64[us] vs UTC).
            futr_df[list(kwargs["futr_exog_list"])] = test_df[
                list(kwargs["futr_exog_list"])].to_numpy()
            return nf, nf.predict(futr_df=futr_df)
        return nf, nf.predict()

    try:
        nf, forecast_df = fit_and_predict(model_kwargs)
    except Exception as error:
        if "future exogenous" not in str(error).lower() or not exog.get("futr_exog_list"):
            raise
        logging.warning("%s %s rejects future exogenous variables; moving %d columns to hist_exog_list",
                        nf_model_name, model_name, len(exog["futr_exog_list"]))
        moved = exog.pop("futr_exog_list")
        exog["hist_exog_list"] = list(exog.get("hist_exog_list", [])) + moved
        model_kwargs.pop("futr_exog_list", None)
        model_kwargs["hist_exog_list"] = exog["hist_exog_list"]
        nf, forecast_df = fit_and_predict(model_kwargs)

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
    sampling_rate, epochs, gpu, nf_model_name, NFmodel, level="L0", params_override=None
):
    setup_model_logger(save_dir)  # Ensure logging is set up first
    metrics = []
    start_time = time.time()
    logging.info(f"...Start training for {epochs} epochs using {nf_model_name} (covariate level {level})...")
    if dataset_name == "belgium":
        logging.info("Training PV models")
        for house in [1, 2, 3, 4]:
            pv_data = dataset.get_inputs_for_pv(house, start_dt, end_dt, level=level)
            _, pv_mae, pv_rmse = neural_forecast_model(
                pv_data, f"PV_house_{house}", save_dir, freq, forecast_horizon, sampling_rate,
                epochs, gpu, nf_model_name, NFmodel, level, params_override)
            metrics.append({"model": f"pv_house_{house}", "MAE": pv_mae, "RMSE": pv_rmse})
        logging.info("Training BESS models")
        for house in [1, 2, 3, 4]:
            battery_data = dataset.get_inputs_for_battery(house, start_dt, end_dt, level=level)
            _, battery_mae, battery_rmse = neural_forecast_model(
                battery_data, f"BESS_house_{house}", save_dir, freq, forecast_horizon,
                sampling_rate, epochs, gpu, nf_model_name, NFmodel, level, params_override)
            metrics.append({"model": f"bess_house_{house}", "MAE": battery_mae, "RMSE": battery_rmse})
    elif dataset_name == "germany":
        logging.info("Training Germany load model")
        germany_data = dataset.get_inputs_for_load(start_dt, end_dt, level=level)
        _, load_mae, load_rmse = neural_forecast_model(
            germany_data, "germany_load", save_dir, freq, forecast_horizon, sampling_rate,
            epochs, gpu, nf_model_name, NFmodel, level, params_override)
        metrics.append({"model": "germany_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "london":
        logging.info("Training London load model")
        london_data = dataset.get_inputs_for_load(level=level)
        _, load_mae, load_rmse = neural_forecast_model(
            london_data, "london_load", save_dir, freq, forecast_horizon, sampling_rate,
            epochs, gpu, nf_model_name, NFmodel, level, params_override)
        metrics.append({"model": "london_load", "MAE": load_mae, "RMSE": load_rmse})
    elif dataset_name == "zonnedael":
        logging.info("Training Zonnedael customer models")
        for customer_id in [8, 9, 43]:
            customer_data = dataset.get_inputs_for_zonnedael_consumption(customer_id, level=level)
            _, cust_mae, cust_rmse = neural_forecast_model(
                customer_data, f"zonnedael_customer_{customer_id}", save_dir, freq,
                forecast_horizon, sampling_rate, epochs, gpu, nf_model_name, NFmodel,
                level, params_override)
            metrics.append({"model": f"zonnedael_customer_{customer_id}", "MAE": cust_mae, "RMSE": cust_rmse})
    pd.DataFrame(metrics).to_csv(os.path.join(save_dir, "model_metrics_summary.csv"), index=False, float_format="%.6f")
    plot_model_metrics(metrics, save_dir)
    elapsed_time = time.time() - start_time
    logging.info(f"...End training...")
    logging.info(f"Training completed in {elapsed_time:.2f} seconds.")


def paper_forecasting_train(dataset_name, dataset, run_num, sampling_rate, epochs, models, gpu,
                            results_dir=PROJECT_ROOT / "results", level="L0", params_by_model=None):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt, end_dt = PAPER_WINDOW
    base_step = DATASET_STEP_MINUTES[dataset_name]
    freq_str = f"{int(base_step * (100 / sampling_rate))}min"  # the paper's label, see utils/protocol.py
    forecast_horizon = horizon_for(sampling_rate)
    params_by_model = params_by_model or {}
    for model_name, model_class in models.items():
        try:
            gc.collect()
            if uses_gpu(gpu):
                torch.cuda.empty_cache()
            save_dir = run_dir(results_dir, dataset_name, model_name, sampling_rate,
                               run_num, epochs=epochs, style="epochs", exog_level=level)
            train_all_models(
                dataset_name, dataset, start_dt, end_dt, save_dir, freq_str, forecast_horizon,
                sampling_rate, epochs, gpu, nf_model_name=model_name, NFmodel=model_class,
                level=level, params_override=params_by_model.get(model_name)
            )
            gc.collect()
            if uses_gpu(gpu):
                torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Skipping model {model_name} due to error: {str(e)}", exc_info=True)
            continue

if __name__ == "__main__":
    model_classes = {
        "DeepNPTS": DeepNPTS,
        "PatchTST": PatchTST,
        "TCN": TCN,
        "NBEATS": NBEATS,
        "NHITS": NHITS,
        "NLinear": NLinear,
        "BiTCN": BiTCN,
        "TiDE": TiDE,
        "TimesNet": TimesNet,
        "TimeXer": TimeXerHist,
        "VanillaTransformer": VanillaTransformer,
        "iTransformer": iTransformer,
    }
    dataset_classes = dict(belgium=DatasetBelgiumNF, germany=DatasetGermanyNF, london=DatasetLondonNF, zonnedael=DatasetZonnedaelNF)
    parser = argparse.ArgumentParser(description="Run NeuralForecast model benchmarks.")
    parser.add_argument("--dataset", choices=["belgium", "germany", "london", "zonnedael"], default="belgium", help="Dataset to forecast.")
    parser.add_argument("--sampling_rates", "--sampling_rate", nargs="+", type=float, default=[25, 100/3, 50, 100], help="Sampling percentages to evaluate.")
    parser.add_argument("--models", nargs="+", choices=list(model_classes),
                        default=["DeepNPTS", "PatchTST", "TCN"], help="NeuralForecast models to evaluate.")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs.")
    parser.add_argument("--epochs", type=int, default=100, help="Training steps per model.")
    parser.add_argument("--exog_level", "--exog-level", choices=list(("L0", "COV")), default="COV",
                        help="Covariate block: COV (day/hour + weather + price + state) or L0 (paper baseline).")
    parser.add_argument("--params_json", type=Path, default=None, help="Tuned parameters (JSON) from scripts/tune_optuna.py.")
    parser.add_argument("--gpu", type=int, default=-1, help=GPU_HELP)
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    args = parser.parse_args()
    dataset = dataset_classes[args.dataset]()
    selected_models = {name: model_classes[name] for name in args.models}
    tuned = {}
    if args.params_json and args.params_json.exists():
        payload = json.loads(args.params_json.read_text(encoding="utf-8"))
        tuned = dict(payload.get("best_params", payload))
    params_by_model = {name: tuned for name in selected_models} if tuned else {}
    for sampling_rate in args.sampling_rates:
        # Loop from run_num = 1 to 10
        for run_num in range(1, args.runs + 1):
            paper_forecasting_train(args.dataset, dataset, run_num, sampling_rate, args.epochs,
                                    selected_models, args.gpu, args.results_dir,
                                    level=args.exog_level, params_by_model=params_by_model)
