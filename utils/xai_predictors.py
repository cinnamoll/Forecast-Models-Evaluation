"""Predictors for the TimeSHAP game that never refit per coalition.

The zero-shot checkpoints in ``scripts/xai_timeshap.py`` accept a new input
mapping, so a coalition is just another array.  The five trainable models of the
benchmark need the same ability, and a refit per coalition is not an option
(2^N fits is days, not minutes):

* ``LightGBM`` -- one fitted booster, so a coalition is a new design matrix;
* ``TFT`` -- a trained GluonTS predictor predicts on a new ``ListDataset``;
* ``TCN`` / ``DeepNPTS`` / ``TimeXer`` -- ``NeuralForecast`` freezes the context
  and ``hist_exog`` into the fitted dataset, but the fitted LightningModule can be
  re-run on a window built here.  The window, the per-window normalization and the
  inverse transform are copied from ``_base_model``; on a synthetic cell this path
  reproduces ``nf.predict()`` bit for bit (max |diff| = 0.0), which is the parity
  test ``verify_predictor`` runs before a game is allowed to start.

Every predictor has the same contract as the checkpoint ones:
``predict(perturbed_frame) -> np.ndarray`` of ``horizon`` values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ("unique_id", "ds", "y")


# --------------------------------------------------------------------- helpers
def _exog_lists(roles: dict, model_class) -> tuple[list, list, list]:
    """The three lists the model can carry, gated by its own support flags."""
    hist = list(roles.get("past_only", [])) if getattr(model_class, "EXOGENOUS_HIST", True) else []
    futr = list(roles.get("known_future", [])) if getattr(model_class, "EXOGENOUS_FUTR", True) else []
    stat = list(roles.get("static", [])) if getattr(model_class, "EXOGENOUS_STAT", True) else []
    return hist, futr, stat


def _nf_window(rows_context: pd.DataFrame, rows_horizon: pd.DataFrame, hist: list, futr: list,
               horizon: int, input_size: int, dtype=np.float32):
    """Window tensor + mask, laid out exactly as ``_base_model._create_windows``.

    Context rows carry y, hist and futr; the horizon block carries announced
    columns only, and its mask is 0 so the normalization statistics ignore it.
    """
    columns = ["y", *hist, *futr]
    window = np.zeros((input_size + horizon, len(columns)), dtype=dtype)
    window[:input_size, 0] = rows_context["y"].to_numpy(dtype=dtype)[-input_size:]
    if hist:
        window[:input_size, 1:1 + len(hist)] = rows_context[hist].to_numpy(dtype=dtype)[-input_size:]
    if futr:
        window[:input_size, 1 + len(hist):] = rows_context[futr].to_numpy(dtype=dtype)[-input_size:]
        window[input_size:, 1 + len(hist):] = rows_horizon[futr].to_numpy(dtype=dtype)
    mask = np.ones((input_size + horizon, 1), dtype=dtype)
    mask[input_size:] = 0.0
    return window, mask


def _nf_forward(model, window: np.ndarray, mask: np.ndarray, horizon: int, input_size: int,
                n_hist: int, n_futr: int):
    """One forward pass through the fitted module, normalized like nf does."""
    import torch

    multivariate = bool(getattr(model, "MULTIVARIATE", False))
    tensor = torch.tensor(window).unsqueeze(0)                 # [1, L+h, C]
    mask_tensor = torch.tensor(mask).unsqueeze(0)              # [1, L+h, 1]
    with torch.no_grad():
        scaled = model.scaler.transform(x=tensor, mask=mask_tensor)
        insample_y = scaled[:, :input_size, 0:1]
        hist_exog = scaled[:, :input_size, 1:1 + n_hist] if n_hist else None
        futr_exog = scaled[:, :, 1 + n_hist:] if n_futr else None
        if hist_exog is not None:
            hist_exog = (hist_exog.unsqueeze(-1).swapaxes(1, 2) if multivariate
                         else hist_exog.squeeze(-1))
        if futr_exog is not None:
            futr_exog = (futr_exog.unsqueeze(-1).swapaxes(1, 2) if multivariate
                         else futr_exog.squeeze(-1))
        batch = {"insample_y": insample_y, "insample_mask": mask_tensor[:, :input_size],
                 "hist_exog": hist_exog, "futr_exog": futr_exog, "stat_exog": None}
        out = model(batch)
        out = model.scaler.inverse_transform(z=out, x_scale=model.scaler.x_scale[:, :, 0],
                                             x_shift=model.scaler.x_shift[:, :, 0])
    return out[0, :, 0].detach().cpu().numpy()


# ------------------------------------------------------------------ predictors
def neuralforecast_predictor(model_class, frame: pd.DataFrame, roles: dict, horizon: int,
                             freq: str, epochs: int, gpu: int, params: dict | None,
                             seed: int = 0, log=None):
    """Fit once on the context, then score a coalition by re-running the module."""
    from neuralforecast import NeuralForecast

    from utils.device import lightning_device_config

    input_size = horizon * 2
    hist, futr, stat = _exog_lists(roles, model_class)
    context = frame.iloc[:-horizon]
    kwargs = dict(h=horizon, input_size=input_size, max_steps=int(epochs),
                  random_seed=seed, alias=model_class.__name__,
                  **lightning_device_config(gpu))
    if getattr(model_class, "MULTIVARIATE", False):
        kwargs["n_series"] = 1
    if hist:
        kwargs["hist_exog_list"] = hist
    if futr:
        kwargs["futr_exog_list"] = futr
    if stat:
        kwargs["stat_exog_list"] = stat
    protected = {"h", "input_size", "max_steps", "random_seed", "n_series", "alias",
                 "hist_exog_list", "futr_exog_list", "stat_exog_list",
                 "accelerator", "devices"}
    applied = {k: v for k, v in dict(params or {}).items() if k not in protected}
    kwargs.update(applied)
    if applied and log:
        log.info("%s: tuned parameters %s", model_class.__name__, applied)

    static_df = None
    if stat:
        static_df = context.groupby("unique_id")[stat].first().reset_index()
    fitted = NeuralForecast(models=[model_class(**kwargs)], freq=freq)
    fitted.fit(df=context, val_size=horizon, static_df=static_df)
    module = fitted.models[0]
    module.eval()

    def predict(perturbed: pd.DataFrame) -> np.ndarray:
        window, mask = _nf_window(perturbed.iloc[:-horizon], perturbed.iloc[-horizon:],
                                  hist, futr, horizon, input_size)
        return _nf_forward(module, window, mask, horizon, input_size, len(hist), len(futr))

    predict.fitted = fitted
    return predict


def lightgbm_predictor(frame: pd.DataFrame, roles: dict, horizon: int, gpu: int,
                       params: dict | None, seed: int = 0, log=None):
    """One booster on the row's own covariates: a coalition is a new design matrix.

    Mirrors ``models/models_statsml.py``: measured columns are lagged by one
    horizon (the row for t may only carry what a forecaster had at the origin),
    announced ones are left alone, and the model is fitted on the context rows.
    """
    import lightgbm as lgb

    from utils.covariates import shift_past_only

    feature_columns = [c for c in frame.columns if c not in LABELS]
    design = shift_past_only(frame[feature_columns], roles, horizon) if roles.get("past_only") \
        else frame[feature_columns]
    context_rows = len(frame) - horizon
    booster_params = {"random_state": seed, "bagging_fraction": 0.5, "feature_fraction": 0.5,
                      "bagging_freq": 1, "verbosity": -1, "n_jobs": -1, "deterministic": False}
    booster_params["device_type"] = "gpu" if gpu in (0, 1) else "cpu"
    protected = {"random_state", "n_jobs", "verbosity", "device_type", "objective"}
    applied = {k: v for k, v in dict(params or {}).items() if k not in protected}
    booster_params.update(applied)
    if applied and log:
        log.info("LightGBM: tuned parameters %s", applied)
    booster = lgb.LGBMRegressor(**booster_params).fit(
        design.iloc[:context_rows], frame["y"].iloc[:context_rows])

    def predict(perturbed: pd.DataFrame) -> np.ndarray:
        features = [c for c in perturbed.columns if c not in LABELS]
        matrix = (shift_past_only(perturbed[features], roles, horizon)
                  if roles.get("past_only") else perturbed[features])
        return booster.predict(matrix.iloc[-horizon:])

    predict.booster = booster
    return predict


def tft_predictor(frame: pd.DataFrame, roles: dict, horizon: int, freq: str, epochs: int,
                  ctx, params: dict | None, level: str, log=None):
    """GluonTS TFT: train once, then predict on a new ``ListDataset`` per coalition.

    The estimator is built from ``models/models_gluonts.py``'s own helpers, so the
    covariate fields, the categorical cardinalities and the Trainer split are the
    ones the benchmark cell uses.
    """
    import mxnet as mx
    from gluonts.mx.model.tft import TemporalFusionTransformerEstimator
    from gluonts.mx.trainer import Trainer

    from models.models_gluonts import covariate_spec, split_params, to_gluonts_dataset

    context_rows = len(frame) - horizon
    context = frame.iloc[:context_rows]
    train_entry, covariate_kwargs = covariate_spec(context, roles, level, train_end=context_rows)
    estimator_kwargs, trainer_kwargs, _ignored = split_params(
        TemporalFusionTransformerEstimator, params)
    estimator = TemporalFusionTransformerEstimator(
        freq=freq, prediction_length=horizon, context_length=horizon,
        trainer=Trainer(ctx=ctx, epochs=int(epochs), num_batches_per_epoch=1, **trainer_kwargs),
        **covariate_kwargs, **estimator_kwargs)
    predictor = estimator.train(training_data=to_gluonts_dataset(
        context, freq, pd.Timestamp(context["ds"].iloc[0]), train_entry))

    def predict(perturbed: pd.DataFrame) -> np.ndarray:
        # Target ends at the forecast origin; the covariate arrays must span
        # context + horizon, which is what ``covariate_spec`` builds from the
        # perturbed frame with the training bounds.
        entry, _ = covariate_spec(perturbed, roles, level, train_end=context_rows)
        dataset = to_gluonts_dataset(perturbed.iloc[:context_rows], freq,
                                     pd.Timestamp(perturbed["ds"].iloc[0]), entry)
        forecast = next(iter(predictor.predict(dataset)))
        return np.asarray(forecast.mean, dtype=float).reshape(-1)[:horizon]

    predict.predictor = predictor
    return predict
