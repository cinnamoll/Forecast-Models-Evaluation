"""Input construction shared by the zero-shot foundation models.

Chronos-2, Timestral and TimesFM 3 all want the same three things: a context,
covariates that stop at the forecast origin, and covariates that extend over the
horizon.  Building them in one place is what keeps those scripts — and their
TimeSHAP explanations — measuring the same object.

Roles decide the shape, not the file:

* ``known_future`` (calendar, day-ahead price) is passed over ``context + horizon``;
  the clock and the auction are known in advance, so this is not leakage.
* ``past_only`` (weather, state of charge) is passed over the context only.  No
  value after the origin is handed over, so the horizon behaves like persistence
  rather than like a perfect weather forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.covariates import scale_covariates

def prepare_inputs(y_df: pd.DataFrame, roles: dict, level: str, horizon: int, scale_categorical: bool = False) -> dict:
    """Per-column arrays for one forecast of one series.

    ``scale_categorical`` asks for a checkpoint with no integer-input channel
    (TimesFM 3): the calendar codes are then min-max scaled like any other
    continuous covariate instead of being kept as codes of an embedding table.
    Chronos-2 and Timestral embed them, so they keep the default.
    """
    feature_columns = [c for c in y_df.columns if c not in ("unique_id", "ds", "y")]
    context_end = len(y_df) - horizon
    if context_end <= 0:
        raise ValueError(f"series of {len(y_df)} points cannot hold horizon {horizon}")

    frame = y_df[feature_columns] if feature_columns else pd.DataFrame(index=y_df.index)
    if feature_columns and level != "L0":
        # A scaled code is what a checkpoint without an embedding table needs:
        # scale_covariates skips ``roles["categorical"]``, so that list is
        # emptied for it and a raw ghi in W/m2 cannot dominate a
        # temperature in Celsius.
        scaling_roles = dict(roles, categorical=[]) if scale_categorical else roles
        frame = scale_covariates(frame, scaling_roles, train_end=context_end)

    categorical = set(roles["categorical"]) if level != "L0" and not scale_categorical else set()
    payload = {
        "context": y_df["y"].to_numpy(dtype=np.float32)[:context_end],
        "context_end": context_end,
        "horizon": horizon,
        "known_numeric": {},
        "known_categorical": {},
        "past_numeric": {},
        "past_categorical": {},
        "report": {
            "level": level,
            "n_context": context_end,
        },
    }
    if level == "L0" or not feature_columns:
        return payload

    for column in roles["known_future"]:
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype=np.float32)
        if column in categorical:
            payload["known_categorical"][column] = np.rint(values)
        else:
            payload["known_numeric"][column] = values

    for column in roles["past_only"]:
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype=np.float32)[:context_end]
        if column in categorical:
            payload["past_categorical"][column] = np.rint(values)
        else:
            payload["past_numeric"][column] = values

    payload["report"]["n_known_future"] = len(payload["known_numeric"]) + len(payload["known_categorical"])
    payload["report"]["n_past_only"] = len(payload["past_numeric"]) + len(payload["past_categorical"])
    return payload

def _stack(block: dict, expected_length: int) -> np.ndarray | None:
    """``{name: array}`` -> ``(C, length)``, or ``None`` when the block is empty."""
    if not block:
        return None
    arrays = []
    for name, values in block.items():
        values = np.asarray(values, dtype=np.float32)
        if values.size != expected_length:
            raise ValueError(f"covariate {name!r} has {values.size} points, expected {expected_length}")
        arrays.append(values)
    return np.stack(arrays)


def chronos2_input(payload: dict) -> dict:
    """One ``Chronos2Pipeline`` input mapping.

    ``future_covariates`` keys must be a subset of the ``past_covariates`` keys,
    so the announced columns appear in both blocks and the measured ones only in
    the past block.
    """
    context_end = payload["context_end"]
    past_covariates = {}
    for block in ("past_numeric", "past_categorical"):
        past_covariates.update(payload[block])
    for block in ("known_numeric", "known_categorical"):
        for name, values in payload[block].items():
            past_covariates[name] = np.asarray(values, dtype=np.float32)[:context_end]

    future_covariates = {}
    for block in ("known_numeric", "known_categorical"):
        for name, values in payload[block].items():
            future_covariates[name] = np.asarray(values, dtype=np.float32)[context_end:]

    entry = {"target": payload["context"], "past_covariates": past_covariates}
    if future_covariates:
        entry["future_covariates"] = future_covariates
    return entry


def timesfm3_covariates(payload: dict) -> dict:
    """``predict`` keyword arguments carrying the TimesFM 3 covariate channels.

    TimesFM 3 has no embedding table and no ``future_covariates`` block: the
    covariates enter as extra variates, ``(channels, context)`` for the
    past-only columns and ``(channels, context + horizon)`` for the columns that
    are known over the horizon.  An empty block is left out instead of being
    passed as an empty array.
    """
    arguments = {}
    past_only = {**payload["past_numeric"], **payload["past_categorical"]}
    stacked = _stack(past_only, payload["context_end"])
    if stacked is not None:
        arguments["past_only_covariates"] = stacked
    past_future = {**payload["known_numeric"], **payload["known_categorical"]}
    stacked = _stack(past_future, payload["context_end"] + payload["horizon"])
    if stacked is not None:
        arguments["past_future_covariates"] = stacked
    return arguments


def timestral_arguments(payload: dict) -> dict:
    """Keyword arguments of ``TimestralForecaster.forecast`` for a batch of one."""
    total = payload["context_end"] + payload["horizon"]
    arguments = {"context": payload["context"].reshape(1, 1, -1)}

    past_numeric = _stack(payload["past_numeric"], payload["context_end"])
    if past_numeric is not None:
        arguments["past_covariates"] = past_numeric[None, ...]
    known_numeric = _stack(payload["known_numeric"], total)
    if known_numeric is not None:
        arguments["known_covariates"] = known_numeric[None, ...]
    past_categorical = _stack(payload["past_categorical"], payload["context_end"])
    if past_categorical is not None:
        arguments["categorical_past_covariates"] = past_categorical[None, ...]
    known_categorical = _stack(payload["known_categorical"], total)
    if known_categorical is not None:
        arguments["categorical_known_covariates"] = known_categorical[None, ...]
    return arguments


def to_numpy(values, dtype=float):
    """``Tensor`` or array -> host ``numpy`` array.

    A CUDA tensor cannot go through ``np.asarray`` (``TypeError: can't convert
    cuda:0 device type tensor to numpy``), so every point-forecast extraction
    goes through here: the same code then works on CPU and on GPU.
    """
    if hasattr(values, "detach"):
        values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
    return np.asarray(values, dtype=dtype)


def point_from_quantiles(quantiles, level_index: int = 1):
    """Median row of a ``(n_variates, horizon, n_levels)`` quantile block."""
    values = to_numpy(quantiles)
    if values.ndim == 3:
        return values[0, :, level_index]
    if values.ndim == 2:
        return values[0]
    return values
