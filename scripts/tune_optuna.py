#!/usr/bin/env python
"""Optuna hyper-parameter search for the five trainable models.

Spaces are the ones of ``plan/plan_updated.md`` section 5.  The contract:

* the metric is MAE, because MAE is the table's metric;
* the search runs on a validation block that ends exactly where the test window
  starts, so nothing is selected on the window that is reported;
* at most 30 trials and 3 hours per model (whichever comes first), with Optuna's
  median pruner and an SQLite study so a session that dies can be resumed;
* parameters tuned on LCL (``--dataset london``) and WPuQ (``--dataset germany``)
  are transferred to Belgium and Zonnedael, so a parameter resting on the edge of
  its space is reported instead of being silently transferred.

Each model family is imported lazily, so the script runs in the environment that
holds the model (``statsml`` for LightGBM, ``neuralforecast`` for DeepNPTS / TCN, 
``gluonts`` for TFT), the same way ``models/`` does.

Example
-------
    conda run -n neuralforecast python scripts/tune_optuna.py \
        --model TCN --dataset germany --level COV --sampling 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
import mxnet as mx

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.covariates import roles_for, shift_past_only
from utils.protocol import horizon_for, sampling_tag, steps_per_day, validation_windows
from utils.covariates import scale_covariates
from utils.device import lightning_device_config
from models.models_gluonts import to_gluonts_dataset, covariate_spec, split_params
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx.trainer import Trainer
from utils.device import uses_gpu

def suggest_lightgbm(trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 5, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "objective": "l1",
    }


def suggest_deepnpts(trial) -> dict:
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


def suggest_tcn(trial) -> dict:
    """TCN: the depth of the dilation stack *is* the receptive field.

    ``dilations_depth`` is searched instead of a list, because a TPE sampler
    cannot move through a list of ints, and because the depth decides whether the
    network can see the horizon at all: with ``kernel_size=2`` the context is
    ``1 + sum(2^i)`` steps, i.e. 32 / 64 / 128 / 256 for depth 5..8, so at
    horizon 192 the deepest level of the plan's original space (7) still falls
    short.  Level 8 is included for exactly that reason, and a parameter that
    ends on the deepest level is reported as "widen the space" rather than as a
    verdict on the architecture (plan sections 1.1 and 5).

    ``scaler_type`` is not searched: the plan keeps each model's default so the
    columns stay comparable with the published table.
    """
    return {
        "kernel_size": trial.suggest_categorical("kernel_size", [2, 3]),
        "dilations_depth": trial.suggest_categorical("dilations_depth", [5, 6, 7, 8]),
        "encoder_hidden_size": trial.suggest_int("encoder_hidden_size", 64, 256, step=64),
        "context_size": trial.suggest_int("context_size", 5, 20),
        "decoder_layers": trial.suggest_int("decoder_layers", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def suggest_tft(trial) -> dict:
    """GluonTS MX ``TemporalFusionTransformerEstimator`` names.

    ``context_length`` is deliberately absent: it stays ``prediction_length``,
    which is what keeps the horizon definition of the paper.  ``learning_rate``
    and ``num_batches_per_epoch`` belong to the ``Trainer`` and are split out by
    ``models/models_gluonts.py``.
    """
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 64, step=8),
        "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "num_batches_per_epoch": trial.suggest_categorical("num_batches_per_epoch", [1, 5, 10, 25, 50]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


SUGGESTERS = {
    "LightGBM": suggest_lightgbm,
    "DeepNPTS": suggest_deepnpts,
    "TCN": suggest_tcn,
    "TFT": suggest_tft,
}

NUMERIC_BOUNDS = {
    "LightGBM": {"num_leaves": (5, 50), "learning_rate": (0.01, 0.1), "n_estimators": (100, 2000),
                 "min_data_in_leaf": (20, 200), "feature_fraction": (0.6, 1.0),
                 "bagging_fraction": (0.6, 1.0), "lambda_l1": (0.0, 10.0), "lambda_l2": (0.0, 10.0)},
    "DeepNPTS": {"n_layers": (1, 3), "dropout": (0.0, 0.5), "learning_rate": (1e-4, 1e-2)},
    "TCN": {"encoder_hidden_size": (64, 256), "context_size": (5, 20), "decoder_layers": (1, 3),
            "learning_rate": (1e-4, 1e-2)},
    "TFT": {"hidden_dim": (16, 64), "dropout_rate": (0.0, 0.3), "learning_rate": (1e-4, 1e-2)},
}

CATEGORICAL_CHOICES = {
    "DeepNPTS": {"hidden_size": (32, 64, 128, 256), "batch_size": (32, 64, 128)},
    "TCN": {"kernel_size": (2, 3), "dilations_depth": (5, 6, 7, 8)},
    "TFT": {"num_heads": (2, 4, 8), "num_batches_per_epoch": (1, 5, 10, 25, 50), "batch_size": (32, 64, 128)},
}

def expand_params(model: str, params: dict) -> dict:
    """Sampled names -> the names the model accepts (TCN's dilation list)."""
    expanded = dict(params)
    if model == "TCN" and "dilations_depth" in expanded:
        raw = expanded.pop("dilations_depth")
        depth = raw if isinstance(raw, int) else int(float(str(raw)))
        expanded["dilations"] = [2 ** level for level in range(depth)]
    return expanded


def boundary_hits(model: str, params: dict) -> list:
    hits = []
    for name, value in params.items():
        if name in NUMERIC_BOUNDS.get(model, {}):
            low, high = NUMERIC_BOUNDS[model][name]
            number = float(value)
            if np.isclose(number, low) or np.isclose(number, high):
                hits.append(name)
        elif name in CATEGORICAL_CHOICES.get(model, {}):
            choices = list(CATEGORICAL_CHOICES[model][name])
            if value in (choices[0], choices[-1]):
                hits.append(name)
    return sorted(hits)

def load_frame(dataset: str, level: str):
    """Long frame of the tuned series at the covariate level being searched.

    LCL and WPuQ each hold one series, which is what makes them the tuning sets:
    a search over four Belgian houses at once would spend the budget on the house
    the parameter set fits best.
    """
    from utils.dataset_config import DatasetGermanyNF, DatasetLondonNF, DatasetBelgiumNF

    if dataset == "london":
        frame = DatasetLondonNF().get_inputs_for_load(level=level)
    elif dataset == "germany":
        start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        end = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
        frame = DatasetGermanyNF().get_inputs_for_load(start, end, level=level)
    elif dataset == "belgium":
        start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        end = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
        frame = DatasetBelgiumNF().get_inputs_for_pv(house=1, start_dt=start, end_dt=end, level=level)
    else:
        raise ValueError(f"{dataset!r} is not a tuning dataset; use london (LCL) or germany (WPuQ)")
    return frame.reset_index(drop=True)

def lightgbm_objective(params, X, y, windows, seed, trial=None):
    """Mean MAE over the validation windows, refitting before each window."""
    import lightgbm as lgb

    maes = []
    for step, (train_end, eval_end) in enumerate(windows):
        model = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbosity=-1,
                                  deterministic=False, **params)
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
        prediction = model.predict(X.iloc[train_end:eval_end])
        mae = float(np.mean(np.abs(prediction - y.iloc[train_end:eval_end].to_numpy())))
        maes.append(mae)
        if trial is not None:
            trial.report(mae, step)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
    return float(np.mean(maes))


def neuralforecast_objective(model_class, params, frame, horizon, freq, windows, epochs,
                             seed, gpu, window=None):
    """MAE of the first validation window for a NeuralForecast model."""
    train_end, eval_end = windows[0] if window is None else window
    window_frame = frame.iloc[:eval_end].reset_index(drop=True)
    feature_columns = [c for c in window_frame.columns if c not in ("unique_id", "ds", "y")]
    roles = roles_for(window_frame[feature_columns]) if feature_columns else {
        "known_future": [], "past_only": [], "static": [], "categorical": []}
    if feature_columns:
        scaled = scale_covariates(window_frame[feature_columns], roles, train_end=train_end)
        window_frame = pd.concat([window_frame.drop(columns=feature_columns), scaled], axis=1)

    kwargs = dict(h=eval_end - train_end, input_size=(eval_end - train_end) * 2,
                  max_steps=int(epochs), random_seed=seed,
                  **lightning_device_config(gpu), **params)
    exog = {}
    if roles["past_only"]:
        exog["hist_exog_list"] = roles["past_only"]
    if roles["known_future"]:
        exog["futr_exog_list"] = roles["known_future"]
    if roles["static"]:
        exog["stat_exog_list"] = roles["static"]
    kwargs.update(exog)

    train = window_frame.iloc[:train_end]
    static_df = None
    if exog.get("stat_exog_list"):
        static_df = window_frame.groupby("unique_id")[exog["stat_exog_list"]].first().reset_index()
    forecast = NeuralForecast(models=[model_class(**kwargs)], freq=freq)
    forecast.fit(df=train, val_size=eval_end - train_end, static_df=static_df)
    if exog.get("futr_exog_list"):
        futr_df = window_frame.iloc[train_end:eval_end][["unique_id", "ds", *exog["futr_exog_list"]]]
        predicted = forecast.predict(futr_df=futr_df)
    else:
        predicted = forecast.predict()
    truth = window_frame.iloc[train_end:eval_end]["y"].to_numpy(dtype=float)
    return float(np.mean(np.abs(predicted.iloc[-len(truth):][model_class.__name__].to_numpy(dtype=float) - truth)))


def gluonts_objective(model_class, params, frame, horizon, freq, windows, epochs, seed, gpu):
    """MAE of the first validation window for the GluonTS TFT estimator."""
    train_end, eval_end = windows[0]
    window_frame = frame.iloc[:eval_end].reset_index(drop=True)
    feature_columns = [c for c in window_frame.columns if c not in ("unique_id", "ds", "y")]
    roles = roles_for(window_frame[feature_columns]) if feature_columns else {
        "known_future": [], "past_only": [], "static": [], "categorical": []}
    entry_extra, covariate_kwargs = covariate_spec(window_frame, roles, "COV", train_end=train_end)
    start = pd.Timestamp(window_frame["ds"].iloc[0])
    train_ds = to_gluonts_dataset(window_frame.iloc[:train_end], freq, start, entry_extra)
    test_ds = to_gluonts_dataset(window_frame, freq, start, entry_extra)

    estimator_kwargs, trainer_kwargs, _ignored = split_params(model_class, params)
    trainer_kwargs.setdefault("num_batches_per_epoch", 1)
    ctx = mx.gpu(gpu) if uses_gpu(gpu) else mx.cpu()
    estimator = model_class(freq=freq, prediction_length=eval_end - train_end,
                            **covariate_kwargs, **estimator_kwargs,
                            trainer=Trainer(epochs=epochs, ctx=ctx, **trainer_kwargs))
    mx.random.seed(seed)
    np.random.seed(seed)
    predictor = estimator.train(training_data=train_ds)
    forecast_it, _ = make_evaluation_predictions(dataset=test_ds, predictor=predictor, num_samples=100)
    prediction = np.asarray(list(forecast_it)[0].mean[: eval_end - train_end], dtype=float)
    truth = window_frame.iloc[train_end:eval_end]["y"].to_numpy(dtype=float)
    return float(np.mean(np.abs(prediction - truth)))


def build_study(storage: str, name: str, seed: int):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, n_min_trials=3),
        load_if_exists=True,
    )


def best_value_curve(study) -> list:
    curve, best = [], float("inf")
    for trial in study.trials:
        if trial.value is None:
            continue
        best = min(best, float(trial.value))
        curve.append([int(trial.number), float(best)])
    return curve


def is_saturated(curve, tail_fraction=0.3, tolerance=0.01) -> bool:
    if len(curve) < 5:
        return False
    values = np.array([point[1] for point in curve], dtype=float)
    tail_start = max(0, int(np.floor(len(values) * (1.0 - tail_fraction))))
    head_best = float(np.min(values[: max(tail_start, 1)]))
    final_best = float(np.min(values))
    if head_best <= 0:
        return final_best <= head_best
    return (head_best - final_best) / head_best < tolerance


def save_result(model: str, dataset: str, level: str, sampling: float, study,
                hours_used: float, windows, max_trials: int, max_hours: float, out_dir: Path) -> Path:
    search_params = dict(study.best_params)
    payload = {
        "model": model,
        "dataset": dataset,
        "level": level,
        "sampling": float(sampling),
        "horizon": horizon_for(sampling),
        "validation_windows": [list(map(int, w)) for w in windows],
        "budget": {"max_trials": int(max_trials), "max_hours": float(max_hours)},
        "used": {"trials": len(study.trials), "hours": round(float(hours_used), 4)},
        "best_value_mae": float(study.best_value),
        "best_params": expand_params(model, search_params),
        "search_params": search_params,
        "boundary_hits": boundary_hits(model, search_params),
        "best_value_curve": best_value_curve(study),
        "saturated": is_saturated(best_value_curve(study)),
        "selection_metric": f"MAE on the {14}-day validation block before the test window",
        "strategy": "B: the five trainable models are tuned with one shared budget; the two TSFM stay zero-shot",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}_{dataset}_{level}_{sampling_tag(sampling)}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def time_limit_callback(max_hours: float):
    start = time.time()

    def callback(study, trial):  # noqa: ARG001 - Optuna signature
        if time.time() - start >= float(max_hours) * 3600.0:
            study.stop()

    return callback

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Optuna search for one model (plan section 5).")
    parser.add_argument("--model", required=True,
                        choices=list(("LightGBM", "DeepNPTS", "TCN", "TFT")))
    parser.add_argument("--dataset", required=True, choices=["london", "germany", "belgium"])
    parser.add_argument("--level", default="COV", choices=list(("L0", "COV")))
    parser.add_argument("--sampling", type=float, default=100.0)
    parser.add_argument("--epochs", type=int, default=100, help="Training steps for the neural models.")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--seed", type=int, default=0, help="Sampler seed and model seed.")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "tuning")
    parser.add_argument("--print-space", action="store_true",
                        help="Sample one configuration and exit (no training).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # LCL = London, WPuQ = the German house.
    dataset = {"LCL": "london", "WPuQ": "germany"}.get(args.dataset) or args.dataset
    horizon = horizon_for(args.sampling)
    per_day = steps_per_day(args.sampling, dataset)
    frame = load_frame(dataset, args.level)
    windows = validation_windows(len(frame), horizon, per_day, 14)
    logging.getLogger("tune").info("%s/%s: %d points, horizon %d, %d validation window(s) of %d steps",
                                   args.model, dataset, len(frame), horizon, len(windows), horizon)

    if args.print_space:
        import optuna

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        print(SUGGESTERS[args.model](trial))
        return 0

    storage = f"sqlite:///{args.out_dir / 'optuna.db'}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    study = build_study(storage, f"{args.model}_{dataset}_{args.level}_{sampling_tag(args.sampling)}", args.seed)

    if args.model == "LightGBM":
        X, y = frame.drop(columns=["unique_id", "ds", "y"]), frame["y"]
        roles = roles_for(X)
        if roles["past_only"]:
            X = shift_past_only(X, roles, horizon)
        X = X.iloc[::int(round(100.0 / args.sampling))].reset_index(drop=True)
        y = y.iloc[::int(round(100.0 / args.sampling))].reset_index(drop=True)
        windows = validation_windows(len(y), horizon, per_day, 14)

        def objective(trial):
            params = SUGGESTERS["LightGBM"](trial)
            params.setdefault("bagging_freq", 1)
            return lightgbm_objective(params, X, y, windows, args.seed, trial)

    elif args.model == "TFT":
        from gluonts.mx.model.tft import TemporalFusionTransformerEstimator

        def objective(trial):
            params = expand_params("TFT", SUGGESTERS["TFT"](trial))
            return gluonts_objective(TemporalFusionTransformerEstimator, params, frame, horizon,
                                     f"{int(15 * (100 / args.sampling))}min", windows,
                                     args.epochs, args.seed, args.gpu)
    else:
        import neuralforecast.models as nf_models

        model_class = getattr(nf_models, args.model)

        def objective(trial):
            params = expand_params(args.model, SUGGESTERS[args.model](trial))
            return neuralforecast_objective(model_class, params, frame, horizon,
                                            f"{int(15 * (100 / args.sampling))}min", windows,
                                            args.epochs, args.seed, args.gpu)

    remaining = max(0, int(args.trials) - len(study.trials))
    if remaining == 0:
        logging.getLogger("tune").warning(
            "study already used its %d trials; writing the existing result", len(study.trials))
        hours_used = 0.0
    else:
        started = time.time()
        study.optimize(objective, n_trials=remaining,
                       callbacks=[time_limit_callback(args.hours)], gc_after_trial=True)
        hours_used = (time.time() - started) / 3600.0

    path = save_result(args.model, dataset, args.level, args.sampling, study, hours_used,
                       windows, args.trials, args.hours, args.out_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"saved {path}")
    print(f"best MAE={payload['best_value_mae']:.6f} after {payload['used']['trials']} trials / "
          f"{payload['used']['hours']:.2f} h")
    if payload["boundary_hits"]:
        print("WARNING: parameter(s) on the edge of the space - widen it or tune this dataset "
              "separately: " + ", ".join(payload["boundary_hits"]))
    if not payload["saturated"]:
        print("WARNING: the best value was still improving at the end - the budget was not enough; "
              "say so in the appendix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
