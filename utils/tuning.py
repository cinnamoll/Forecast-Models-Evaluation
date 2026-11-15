"""Hyper-parameter search for the four trainable models (strategy B).

plan_updated.md section 5 fixes the contract:

* the dataset's own validation block -- 14 days, 28 on Zonnedael -- sitting
  immediately before the test window;
* the metric is MAE, because MAE is the table's metric;
* at most 30 trials and 3 hours per model, whichever comes first, with Optuna's
  median pruner and an SQLite study so a Kaggle session can be resumed;
* selection never touches the test window and never picks the best seed;
* parameters tuned on LCL/WPuQ are *transferred* to Belgium/Zonnedael, so a
  parameter sitting on the edge of its search space has to be reported.

The two TSFM columns are not tuned: Chronos-2 and Timestral-123M stay zero-shot
(section 4), so nothing here accepts them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from utils.protocol import (
    PROJECT_ROOT,
    TUNING_MAX_HOURS,
    TUNING_MAX_TRIALS,
    horizon_for,
    points_per_day,
    sampling_tag,
    validation_days as protocol_validation_days,
    validation_windows,
)

TUNED_MODELS = ("LightGBM", "DeepNPTS", "PatchTST", "TFT")

# Datasets the plan tunes on, and the two it transfers to.
TUNING_DATASETS = {"LCL": "london", "WPuQ": "germany"}
TRANSFER_DATASETS = ("belgium", "zonnedael")


# --------------------------------------------------------------------------- spaces

def _suggest_lightgbm(trial) -> Dict[str, object]:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "objective": "l1",
    }


def _suggest_deepnpts(trial) -> Dict[str, object]:
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


def _suggest_patchtst(trial) -> Dict[str, object]:
    return {
        "patch_len": trial.suggest_categorical("patch_len", [16, 32, 48]),
        "stride": trial.suggest_categorical("stride", [8, 12, 16, 24]),
        "encoder_layers": trial.suggest_int("encoder_layers", 1, 4),
        "n_heads": trial.suggest_categorical("n_heads", [4, 8, 16]),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
    }


def _suggest_tft(trial) -> Dict[str, object]:
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "n_head": trial.suggest_categorical("n_head", [2, 4, 8]),
        "n_rnn_layers": trial.suggest_int("n_rnn_layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }


SUGGESTERS: Dict[str, Callable] = {
    "LightGBM": _suggest_lightgbm,
    "DeepNPTS": _suggest_deepnpts,
    "PatchTST": _suggest_patchtst,
    "TFT": _suggest_tft,
}

# Numeric bounds of each continuous parameter, for the transfer warning of
# section 5: a parameter that ends on an edge was not searched, it was clipped.
NUMERIC_BOUNDS: Dict[str, Dict[str, tuple]] = {
    "LightGBM": {
        "num_leaves": (31, 255),
        "learning_rate": (0.01, 0.1),
        "n_estimators": (200, 2000),
        "min_data_in_leaf": (20, 200),
        "feature_fraction": (0.6, 1.0),
        "bagging_fraction": (0.6, 1.0),
        "lambda_l1": (0.0, 10.0),
        "lambda_l2": (0.0, 10.0),
    },
    "DeepNPTS": {
        "n_layers": (1, 3),
        "dropout": (0.0, 0.5),
        "learning_rate": (1e-4, 1e-2),
    },
    "PatchTST": {
        "encoder_layers": (1, 4),
        "dropout": (0.0, 0.4),
        "learning_rate": (1e-5, 1e-3),
    },
    "TFT": {
        "n_rnn_layers": (1, 2),
        "dropout": (0.0, 0.4),
        "learning_rate": (1e-4, 1e-2),
    },
}

CATEGORICAL_CHOICES: Dict[str, Dict[str, Sequence]] = {
    "DeepNPTS": {"hidden_size": (32, 64, 128, 256), "batch_size": (32, 64, 128, 256)},
    "PatchTST": {"patch_len": (16, 32, 48), "stride": (8, 12, 16, 24),
                 "n_heads": (4, 8, 16), "hidden_size": (64, 128, 256)},
    "TFT": {"hidden_size": (32, 64, 128, 256), "n_head": (2, 4, 8)},
}


def boundary_hits(model: str, params: Dict[str, object], rtol: float = 1e-9) -> List[str]:
    """Parameters sitting on an edge of their search space."""
    hits: List[str] = []
    for name, value in params.items():
        if name in NUMERIC_BOUNDS.get(model, {}):
            low, high = NUMERIC_BOUNDS[model][name]
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isclose(number, low, rtol=rtol, atol=0.0) or np.isclose(number, high, rtol=rtol, atol=0.0):
                hits.append(name)
        elif name in CATEGORICAL_CHOICES.get(model, {}):
            choices = list(CATEGORICAL_CHOICES[model][name])
            if value in (choices[0], choices[-1]):
                hits.append(name)
    return sorted(hits)


# --------------------------------------------------------------------------- study

@dataclass
class TuningCell:
    """One tuning cell: a model on one dataset at one exogenous level."""

    model: str
    dataset: str
    level: str = "L0"
    sampling: float = 100.0
    epochs: int = 100
    seed: int = 0
    gpu: int = -1
    validation_days: Optional[int] = None
    max_trials: int = TUNING_MAX_TRIALS
    max_hours: float = TUNING_MAX_HOURS
    output_dir: Optional[Path] = None

    def __post_init__(self):
        if self.model not in TUNED_MODELS:
            raise ValueError(
                f"{self.model!r} is not tuned: plan_updated.md section 4 keeps "
                f"Chronos-2 and Timestral zero-shot. Tuned models: {TUNED_MODELS}"
            )
        # 14 days on Belgium/Germany/London, 28 on Zonnedael: the block is the
        # one reserved by data/README.md, never a flat default.

    @property
    def validation_days_used(self) -> int:
        """Days of the validation block: the override, else the dataset's own."""
        if self.validation_days is None:
            return int(protocol_validation_days(self.dataset))
        return int(self.validation_days)

    @property
    def horizon(self) -> int:
        """Points of the test block at this sampling rate (192 on 15-min data)."""
        return horizon_for(self.sampling, self.dataset)

    @property
    def per_day(self) -> int:
        """Points in one day of the validation block after subsampling."""
        return points_per_day(self.dataset, self.sampling)

    @property
    def directory(self) -> Path:
        return Path(self.output_dir) if self.output_dir else PROJECT_ROOT / "results" / "tuning"

    @property
    def storage(self) -> str:
        path = self.directory
        path.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path / 'optuna.db'}"

    @property
    def study_name(self) -> str:
        return f"{self.model}_{self.dataset}_{self.level}_{sampling_tag(self.sampling)}"

    @property
    def output_path(self) -> Path:
        return self.directory / f"{self.study_name}.json"


def build_study(cell: TuningCell, sampler_seed: Optional[int] = None):
    """Optuna study with SQLite storage, median pruning and a fixed seed."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=cell.seed if sampler_seed is None else sampler_seed,
                                         multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1,
                                         n_min_trials=3)
    return optuna.create_study(
        study_name=cell.study_name,
        storage=cell.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def time_limit_callback(max_hours: float, started_at: Optional[float] = None):
    """Stop the study once ``max_hours`` of wall clock have elapsed."""
    start = time.time() if started_at is None else started_at

    def callback(study, trial):  # noqa: ARG001 - Optuna callback signature
        if time.time() - start >= float(max_hours) * 3600.0:
            study.stop()

    return callback


def trial_table(study) -> List[Dict[str, object]]:
    """``[{number, value, state, params, duration_s}]`` for the appendix figure."""
    rows = []
    for trial in study.trials:
        rows.append({
            "number": trial.number,
            "value": trial.value,
            "state": str(trial.state.name),
            "params": dict(trial.params),
            "duration_s": (
                (trial.datetime_complete - trial.datetime_start).total_seconds()
                if trial.datetime_complete and trial.datetime_start else None
            ),
        })
    return rows


def best_value_curve(study) -> List[List[float]]:
    """Running best value per completed trial, for the saturation check."""
    curve, best = [], float("inf")
    for trial in study.trials:
        if trial.value is None:
            continue
        best = min(best, float(trial.value))
        curve.append([int(trial.number), float(best)])
    return curve


def is_saturated(curve: List[List[float]], tail_fraction: float = 0.3,
                 improvement_tolerance: float = 0.01) -> bool:
    """True when the last third of the trials improved the best value < 1 %.

    plan_updated.md section 5 asks the appendix to say so when the budget was not
    enough; this is the mechanical test behind that sentence.
    """
    if len(curve) < 5:
        return False
    values = np.array([point[1] for point in curve], dtype=float)
    tail_start = max(0, int(np.floor(len(values) * (1.0 - tail_fraction))))
    head_best = float(np.min(values[: max(tail_start, 1)]))
    final_best = float(np.min(values))
    if head_best <= 0:
        return final_best <= head_best
    return (head_best - final_best) / head_best < improvement_tolerance


def save_result(cell: TuningCell, study, hours_used: float,
                extra: Optional[Dict[str, object]] = None) -> Path:
    """Write the best parameters and the budget actually used to JSON."""
    best_params = dict(study.best_params)
    payload = {
        "model": cell.model,
        "dataset": cell.dataset,
        "level": cell.level,
        "sampling": float(cell.sampling),
        "horizon": cell.horizon,
        "validation_days": int(cell.validation_days_used),
        "budget": {"max_trials": int(cell.max_trials), "max_hours": float(cell.max_hours)},
        "used": {"trials": len(study.trials), "hours": round(float(hours_used), 4)},
        "best_value": float(study.best_value),
        "best_params": best_params,
        "boundary_hits": boundary_hits(cell.model, best_params),
        "best_value_curve": best_value_curve(study),
        "saturated": is_saturated(best_value_curve(study)),
        "trials": trial_table(study),
        "strategy": "B: trainable models tuned with one shared budget; TSFM zero-shot",
        "selection_metric": "MAE on the validation block before the test window",
    }
    if extra:
        payload.update(extra)
    path = cell.output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_best_params(path) -> Dict[str, object]:
    """Parameters from a tuning JSON, or an empty dict when the file is absent."""
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "best_params" in payload:
        return dict(payload["best_params"])
    return dict(payload)


def transfer_report(source_path, model: str) -> Dict[str, object]:
    """Summary of what a transferred parameter set implies (section 5 warning)."""
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    params = payload.get("best_params", {})
    hits = boundary_hits(model, params)
    return {
        "source": str(source_path),
        "model": model,
        "params": params,
        "boundary_hits": hits,
        "needs_retuning": bool(hits),
        "advice": (
            "parameter(s) on the edge of the search space: widen the space or tune "
            f"this dataset separately ({', '.join(hits)})" if hits
            else "no parameter sits on an edge; transfer is admissible"
        ),
    }


# --------------------------------------------------------------------------- objectives

def lightgbm_windows_mae(params: Dict[str, object], X, y, windows: Sequence[tuple],
                         seed: int = 0, trial=None) -> float:
    """Mean MAE over the validation windows, training only before each window.

    Refitting per window is what keeps the validation honest for a direct
    multi-step regressor: the model that scores window k has never seen it.
    """
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


def neuralforecast_windows_mae(model_class, params: Dict[str, object], frame,
                               horizon: int, freq: str, windows: Sequence[tuple],
                               epochs: int, seed: int, gpu: int = -1,
                               windows_to_use: int = 1) -> float:
    """MAE of one (or more) validation window for a NeuralForecast model.

    Each window costs a full fit, so the default uses the first window only: the
    budget in section 5 is per model, not per window, and a single window as long
    as the test block already matches the test metric length.  ``windows_to_use``
    can be raised when the budget allows it.
    """
    from neuralforecast import NeuralForecast
    from utils.device import lightning_device_config

    maes = []
    for train_end, eval_end in list(windows)[:max(1, int(windows_to_use))]:
        train = frame.iloc[:train_end]
        truth = frame.iloc[train_end:eval_end]["y"].to_numpy(dtype=float)
        kwargs = dict(
            h=horizon,
            input_size=horizon * 2,
            max_steps=int(epochs),
            random_seed=seed,
            **lightning_device_config(gpu),
            **params,
        )
        forecast = NeuralForecast(models=[model_class(**kwargs)], freq=freq)
        forecast.fit(df=train, val_size=horizon)
        predicted = forecast.predict()
        column = model_class.__name__ if model_class.__name__ in predicted.columns else [
            c for c in predicted.columns if c not in ("unique_id", "ds")
        ][0]
        prediction = predicted.iloc[-len(truth):][column].to_numpy(dtype=float)
        maes.append(float(np.mean(np.abs(prediction - truth))))
    return float(np.mean(maes))


def validate_window_alignment(windows: Sequence[tuple], n_points: int, horizon: int) -> None:
    """Guard: no window may reach into the test block."""
    if not windows:
        raise ValueError("no validation windows")
    last_eval_end = max(eval_end for _, eval_end in windows)
    if last_eval_end > n_points - horizon:
        raise ValueError(
            f"validation window ends at {last_eval_end} but the test block starts at "
            f"{n_points - horizon}"
        )
