"""Predictor for the TimeSHAP game of the LightGBM baseline.

The zero-shot checkpoints accept a new input mapping, so a coalition is just
another array.  A tabular baseline needs the same ability, and a refit per
coalition is not an option (2^N fits is days, not minutes):

* ``LightGBM`` -- one fitted booster, so a coalition is a new design matrix.  The
  design matrix is rebuilt with the same helpers the reported cell uses
  (``shift_past_only`` for the measured covariates, ``attach_target_features`` for
  the target's own history), so the booster is asked about the input mapping it was
  trained on -- and a coalition that zeroes part of the history moves the lag and
  rolling columns instead of leaving them untouched.

``predict(perturbed_frame) -> np.ndarray`` of ``horizon`` values, the same contract
the checkpoint predictors have.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ("unique_id", "ds", "y")


def lightgbm_predictor(frame: pd.DataFrame, roles: dict, horizon: int, gpu: int,
                       params: dict | None, seed: int = 0, log=None):
    """One booster on the row's own covariates and its own target history.

    Mirrors ``models/models_statsml.py``: measured columns are lagged by one
    horizon (the row for t may only carry what a forecaster had at the origin),
    announced ones are left alone, the target lags and rolling statistics are
    attached on top, and the model is fitted on the context rows.
    """
    import lightgbm as lgb

    from utils.covariates import attach_target_features, shift_past_only

    feature_columns = [c for c in frame.columns if c not in LABELS]
    design = shift_past_only(frame[feature_columns], roles, horizon) if roles.get("past_only") \
        else frame[feature_columns]
    design = attach_target_features(design, frame["y"], horizon)
    context_rows = len(frame) - horizon
    booster_params = {"random_state": seed, "bagging_fraction": 0.5, "feature_fraction": 0.5,
                      "bagging_freq": 1, "verbosity": -1, "n_jobs": -1, "deterministic": False}
    booster_params["device_type"] = "gpu" if gpu in (0, 1) else "cpu"
    # ``objective`` is not protected: the cell trains with the tuned objective (L1 by
    # default), so the explained booster has to use the same loss.
    protected = {"random_state", "n_jobs", "verbosity", "device_type"}
    booster_params.setdefault("objective", "regression_l1")
    booster_params.setdefault("metric", "mae")
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
        # Rebuilt from the *perturbed* target, so a history player that zeroes a day
        # also moves every lag and rolling statistic that reads it.
        matrix = attach_target_features(matrix, perturbed["y"], horizon)
        return booster.predict(matrix.iloc[-horizon:])

    predict.booster = booster
    return predict
