#!/usr/bin/env python
"""Optuna hyper-parameter search for the four trainable models.

Spaces are the ones of ``plan/plan_updated.md`` section 5, adjusted where the
script itself was hiding a result.  The contract:

* the metric is MAE, because MAE is the table's metric -- the *loss* stays the
  model's own: LightGBM keeps the paper's default L2, and the search ranks those
  models by MAE;
* the search runs on a 14-day validation block that ends exactly where the test
  window starts, and the windows are anchored backwards from that edge, so the
  window next to the test block is always evaluated;
* at most 30 trials and 3 hours per session (whichever comes first), with
  Optuna's median pruner and an SQLite study so a session that dies can be
  resumed;
* the search runs on ``germany`` (WPuQ), ``zonnedael9`` (Klant 9) and
  ``belgium_pv1`` (Belgium PV house 1) only; ``utils/tuning.py`` maps the ten
  reported cells that borrow those parameters, so a parameter resting on the edge
  of its space is reported instead of being silently transferred.

Only the LightGBM baseline is tuned: the three zero-shot checkpoints stay
zero-shot, and the deep baselines they are compared against are no longer part of
this study.  LightGBM needs no GPU, so the script runs in the environment that
holds it (``statsml``).

Example
-------
    conda run -n statsml python scripts/tune_optuna.py \
        --model LightGBM --dataset germany --level COV --sampling 25 --trials 30
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.covariates import attach_target_features, calendar_frame, roles_for, shift_past_only
from utils.protocol import (freq_for, horizon_for, sampling_tag, step_minutes,
                            steps_per_day, validation_windows)
from utils.tuning import SPACE_VERSION, TUNE_SETS, transfer_targets

#: Neural models evaluate the ``DL_WINDOWS`` validation windows closest to the
#: test block (one fit each); LightGBM is cheap enough to use all of them.
DL_WINDOWS = 2

#: The single series each tuning set is made of.
ZONNEDAEL_CUSTOMER = 9
BELGIUM_PV_HOUSE = 1

#: Alias -> tuning set.  Only the three tuning sets are accepted: London is
#: half-hourly and Belgium's other seven series are reported cells, not tuning
#: data, so admitting them here would quietly change what "tuned" means.
DATASET_ALIASES = {
    "germany": "germany",
    "WPuQ": "germany",
    "wpuq": "germany",
    "zonnedael9": "zonnedael9",
    "zonnedael_9": "zonnedael9",
    "zonnedael": "zonnedael9",
    "Zonnedael": "zonnedael9",
    "belgium_pv1": "belgium_pv1",
    "belgium": "belgium_pv1",
    "Belgium": "belgium_pv1",
}

#: Tuning set -> the key ``utils/protocol.py`` uses for its native step.
NATIVE_DATASET = {"germany": "germany", "zonnedael9": "zonnedael", "belgium_pv1": "belgium"}

#: At ``level="L0"`` the loaders return no covariates (that is the paper's
#: baseline for the neural models) and the 1D loaders rebuild the calendar
#: themselves; this is the block they use.
L0_CALENDAR = {
    "germany": ("hour",),
    "zonnedael9": ("hour", "dayofweek"),
    "belgium_pv1": ("hour",),
}


# ----------------------------------------------------------------------- spaces

def suggest_lightgbm(trial) -> dict:
    """LightGBM, now on the loss the tables report.

    ``objective`` *is* searched: the paper's refit left it on the default L2, but
    every table in this repo scores MAE, so L1 is a legitimate candidate and the
    search decides with the same metric the paper uses.  ``lambda_l1``/``lambda_l2``
    stay out: with a min-max target every leaf output is bounded by 1 and LightGBM
    soft-thresholds leaf outputs with ``lambda_l1``, so any value at or above 1
    collapses the tree to a constant -- the dimension was a switch between
    "normal" and "predict the mean", not a regularisation knob.
    """
    return {
        "objective": trial.suggest_categorical("objective", ["regression_l1", "regression"]),
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step=100),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
    }


SUGGESTERS = {
    "LightGBM": suggest_lightgbm,
}

#: Values every trial gets, whatever it sampled.  They are not suggestions, so
#: Optuna never stores them in ``best_params``; writing them into the JSON keeps
#: the file able to reproduce the tuned model on its own.
FIXED_PARAMS = {
    # ``metric`` only steers LightGBM's own logging/early stopping, never the fit;
    # the objective is now searched (``suggest_lightgbm``).  ``bagging_freq`` has to
    # travel with ``bagging_fraction`` or a consumer that builds the estimator
    # from this file alone silently disables bagging.
    "LightGBM": {"metric": "mae", "bagging_freq": 1},
}


def expand_params(model: str, params: dict) -> dict:
    """Sampled names -> the names the model accepts, plus the fixed values."""
    expanded = dict(params)
    expanded.update(FIXED_PARAMS.get(model, {}))
    return expanded


def boundary_hits(study, rel_tol: float = 0.05) -> list:
    """Parameters whose best value sits on the edge of its own distribution.

    Read from ``study.best_trial.distributions`` rather than from a hand-copied
    table of the space: the previous duplicate had already drifted (a categorical
    searched over four values but checked against three, so the fourth was never
    reported as an edge hit).  Categorical spaces with two choices are skipped -- every
    draw is an edge there -- and numeric spaces use a relative tolerance of
    ``rel_tol`` of the range, in log space when the distribution is log-scaled.
    """
    hits = []
    for name, dist in study.best_trial.distributions.items():
        value = study.best_params.get(name)
        if value is None:
            continue
        choices = getattr(dist, "choices", None)
        if choices is not None:
            if len(choices) > 2 and value in (choices[0], choices[-1]):
                hits.append(name)
            continue
        low, high = float(dist.low), float(dist.high)
        if high <= low:
            continue
        if getattr(dist, "log", False) and low > 0:
            position = ((math.log(float(value)) - math.log(low))
                        / (math.log(high) - math.log(low)))
        else:
            position = (float(value) - low) / (high - low)
        if position < rel_tol or position > 1.0 - rel_tol:
            hits.append(name)
    return sorted(hits)


# ------------------------------------------------------------------------- data

def load_frame(dataset: str, level: str):
    """Long frame of the tuning series at the covariate level being searched.

    Every tuning set is one series on a 15-minute grid: a search over four
    Belgian houses at once would spend the budget on the house the parameter set
    fits best, and a half-hourly series would put the tune-time frequency label
    out of step with every other cell.
    """
    from utils.dataset_config import DatasetBelgiumNF, DatasetGermanyNF, DatasetZonnedaelNF

    # The raw 2024 files are UTC (`2024-01-01T00:00:00Z`), so the filter has to
    # compare tz-aware timestamps; ``_attach_covariates`` strips the zone from
    # ``ds`` afterwards, which is why nothing downstream sees it.
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    if dataset == "germany":
        frame = DatasetGermanyNF().get_inputs_for_load(start, end, level=level)
    elif dataset == "zonnedael9":
        frame = DatasetZonnedaelNF().get_inputs_for_zonnedael_consumption(
            ZONNEDAEL_CUSTOMER, level=level)
    elif dataset == "belgium_pv1":
        frame = DatasetBelgiumNF().get_inputs_for_pv(BELGIUM_PV_HOUSE, start, end, level=level)
    else:
        raise ValueError(f"{dataset!r} is not a tuning set; known: {', '.join(TUNE_SETS)}")
    return frame.reset_index(drop=True)


def l0_features(frame, dataset: str):
    """The paper's ``L0`` block for the direct multi-step models.

    ``level="L0"`` returns no covariates from the loaders — that is the paper's
    baseline for the neural models — and the 1D loaders rebuild the calendar
    themselves.  Without this the LightGBM branch would train on zero columns.
    """
    calendar = calendar_frame(pd.DatetimeIndex(frame["ds"]))
    return calendar[list(L0_CALENDAR[dataset])].reset_index(drop=True)



# -------------------------------------------------------------------- objectives

def _report(trial, value: float, step: int) -> None:
    """Feed the pruner; the caller still owns the value it returns."""
    if trial is None:
        return
    import optuna

    trial.report(value, step)
    if trial.should_prune():
        raise optuna.TrialPruned()


def _remember(trial, maes, n_train) -> None:
    """Keep the per-window numbers so a good trial can be audited later."""
    if trial is None:
        return
    trial.set_user_attr("window_mae", [float(m) for m in maes])
    trial.set_user_attr("n_train", int(n_train))


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
        _report(trial, mae, step)
    _remember(trial, maes, windows[-1][0])
    return float(np.mean(maes))


def build_study(storage: str, name: str, seed: int):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        # ``n_warmup_steps=0``: with the training budget no longer searched, the
        # value at a given window is comparable across trials, so the first
        # window of a trial can already prune the rest of it.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0,
                                           n_min_trials=3),
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


def save_result(model: str, dataset: str, level: str, sampling: float, study, freq: str,
                windows, max_trials: int, max_hours: float, max_steps: int,
                session_hours: float, out_dir: Path) -> Path:
    complete = [trial for trial in study.trials if trial.value is not None]
    search_params = dict(study.best_params) if complete else {}
    best = study.best_trial if complete else None
    curve = best_value_curve(study)
    seconds = sum(t.duration.total_seconds() for t in study.trials if t.duration)
    payload = {
        "model": model,
        "space_version": SPACE_VERSION,
        "tune_on": dataset,
        "transfer_to": transfer_targets(dataset),
        "level": level,
        "sampling": float(sampling),
        "frequency": freq,
        "horizon": horizon_for(sampling),
        "validation_windows": [list(map(int, w)) for w in windows],
        "windows_used": len(windows) if model == "LightGBM" else min(DL_WINDOWS, len(windows)),
        "budget": {"max_trials": int(max_trials), "max_hours_per_session": float(max_hours),
                   "max_steps": int(max_steps)},
        "used": {"trials": len(study.trials),
                 "hours": round(float(seconds) / 3600.0, 4),
                 "session_hours": round(float(session_hours), 4)},
        "best_value_mae": float(study.best_value) if complete else None,
        "best_params": expand_params(model, search_params),
        "search_params": search_params,
        "fixed_params": dict(FIXED_PARAMS.get(model, {})),
        "best_trial": None if best is None else {
            "number": int(best.number),
            "window_mae": best.user_attrs.get("window_mae"),
            "n_train": best.user_attrs.get("n_train"),
        },
        "boundary_hits": boundary_hits(study) if complete else [],
        "best_value_curve": curve,
        "saturated": is_saturated(curve),
        "error": None if complete else "no trial completed; the study is kept in optuna.db",
        "selection_metric": ("MAE on the 14-day validation block that ends at the test window; "
                             "windows are anchored backwards so the last one touches test_start"),
        "strategy": ("B: the four trainable models are tuned on three series and their parameters "
                     "are transferred to the ten remaining cells mapped in utils/tuning.py; "
                     "Chronos2, TimesFM3 and Timestral stay zero-shot"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (f"{model}_{dataset}_{level}_{sampling_tag(sampling)}"
                      f"_v{SPACE_VERSION}.json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def time_limit_callback(max_hours: float):
    """Stop the study after ``max_hours`` of **this** session.

    The limit is per session on purpose: a resumed study cannot know what the
    previous sessions spent, so ``used.hours`` in the JSON accumulates and this
    callback only bounds one sitting.
    """
    start = time.time()

    def callback(study, trial):  # noqa: ARG001 - Optuna signature
        if time.time() - start >= float(max_hours) * 3600.0:
            study.stop()

    return callback


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Optuna search for one model (plan section 5).")
    parser.add_argument("--model", required=True, choices=list(SUGGESTERS),
                        help="Tunable models left after the deep baselines were dropped.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_ALIASES),
                        help="Tuning series: germany (WPuQ), zonnedael9 (Klant 9), "
                             "belgium_pv1 (Belgium PV house 1).")
    parser.add_argument("--level", default="COV", choices=list(("L0", "COV")))
    parser.add_argument("--sampling", type=float, default=100.0)
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Unused by LightGBM; kept so older command lines still parse.")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--hours", type=float, default=1.5,
                        help="Wall-clock limit of one session; resuming adds more.")
    parser.add_argument("--seed", type=int, default=0, help="Sampler seed and model seed.")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "tuning")
    parser.add_argument("--print-space", action="store_true",
                        help="Sample one configuration and exit (no training).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("tune")

    if args.print_space:
        import optuna

        print(SUGGESTERS[args.model](optuna.create_study(direction="minimize").ask()))
        return 0

    dataset = DATASET_ALIASES[args.dataset]
    native = NATIVE_DATASET[dataset]
    freq = freq_for(args.sampling, native)
    horizon = horizon_for(args.sampling)
    per_day = steps_per_day(args.sampling, native)

    frame = load_frame(dataset, args.level)
    stride = step_minutes(args.sampling)
    if stride > 1:
        # Subsampling happens once, here, for every family: the validation
        # windows, the horizon and the past-only shift are all in the subsampled
        # grid, and the neural branch used to skip this step entirely, so at
        # 50 % it trained on 15-minute rows while the label and the horizon
        # implied 30.
        frame = frame.iloc[::stride].reset_index(drop=True)
    windows = validation_windows(len(frame), horizon, per_day, 14)
    log.info("%s/%s: %d points, %s, horizon %d, %d validation window(s) of %d steps",
             args.model, dataset, len(frame), freq, horizon, len(windows), horizon)

    if args.model == "LightGBM":
        X = (l0_features(frame, dataset) if args.level == "L0"
             else frame.drop(columns=["unique_id", "ds", "y"]).reset_index(drop=True))
        y = frame["y"].reset_index(drop=True)
        roles = roles_for(X) if len(X.columns) else {
            "known_future": [], "past_only": [], "categorical": []}
        if roles["past_only"]:
            # After subsampling, so ``horizon`` is a horizon of the grid the
            # model is actually trained on: shifting before it, at 33 %, moved
            # 64 full-resolution rows, i.e. 21 subsampled steps out of 64.
            X = shift_past_only(X, roles, horizon)
        # Same feature builder the reported cell uses, otherwise the parameters are
        # searched for a different input mapping.
        X = attach_target_features(X, y, horizon, per_day)
        log.info("%s: %d feature(s), %d past-only lagged by %d steps, "
                 "%d target-history column(s)",
                 args.model, len(X.columns), len(roles["past_only"]), horizon,
                 len([c for c in X.columns if c.startswith("y_")]))

        def objective(trial):
            params = expand_params("LightGBM", SUGGESTERS["LightGBM"](trial))
            return lightgbm_objective(params, X, y, windows, args.seed, trial)

    storage = f"sqlite:///{args.out_dir / 'optuna.db'}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    study = build_study(storage, f"{args.model}_{dataset}_{args.level}_"
                                 f"{sampling_tag(args.sampling)}_v{SPACE_VERSION}", args.seed)

    remaining = max(0, int(args.trials) - len(study.trials))
    session_hours, failure = 0.0, None
    if remaining == 0:
        log.warning("study already used its %d trials; writing the existing result",
                    len(study.trials))
    else:
        started = time.time()
        try:
            study.optimize(objective, n_trials=remaining,
                           callbacks=[time_limit_callback(args.hours)],
                           gc_after_trial=True, catch=(RuntimeError,))
        except Exception as error:  # keep the trials the study holds instead of losing the report
            failure = error
            log.exception("search aborted; writing the result of the trials already completed")
        session_hours = (time.time() - started) / 3600.0

    path = save_result(args.model, dataset, args.level, args.sampling, study, freq, windows,
                       args.trials, args.hours, args.max_steps, session_hours, args.out_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"saved {path}")
    if payload["best_value_mae"] is None:
        print("no trial completed - nothing to select; the study is kept in optuna.db")
        return 1
    print(f"best MAE={payload['best_value_mae']:.6f} after {payload['used']['trials']} trials / "
          f"{payload['used']['hours']:.2f} h (last session {payload['used']['session_hours']:.2f} h)")
    print(f"tuned on {payload['tune_on']} -> {len(payload['transfer_to'])} cell(s): "
          f"{', '.join(payload['transfer_to'])}")
    if payload["boundary_hits"]:
        print("WARNING: parameter(s) on the edge of the space - widen it or tune this dataset "
              "separately: " + ", ".join(payload["boundary_hits"]))
    if not payload["saturated"]:
        print("WARNING: the best value was still improving at the end - the budget was not enough; "
              "say so in the appendix.")
    return 1 if failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
