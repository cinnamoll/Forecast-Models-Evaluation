#!/usr/bin/env python
"""Run the section-5 hyper-parameter search for one model.

Strategy B of plan_updated.md: the four trainable models are tuned with the same
budget on the same 14-day validation block; Chronos-2 and Timestral-123M are not
tuned (section 4).  The study lives in SQLite, so a Kaggle session that dies can
be resumed by re-running the same command: Optuna picks up the finished trials
and only spends the remaining budget.

Examples
--------
    python scripts/tune_models.py --model LightGBM  --dataset LCL
    python scripts/tune_models.py --model PatchTST  --dataset WPuQ --epochs 100

Output: ``results/tuning/<model>_<dataset>_<level>_<sampling>.json`` with the
best parameters, the trials actually used, the hours used, the saturation flag
and the parameters that ended on the edge of their space.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.protocol import EXOG_LEVEL_CHOICES, TUNING_MAX_HOURS, TUNING_MAX_TRIALS, validation_windows
from utils.tuning import (
    SUGGESTERS,
    TUNED_MODELS,
    TUNING_DATASETS,
    TuningCell,
    build_study,
    lightgbm_windows_mae,
    neuralforecast_windows_mae,
    save_result,
    time_limit_callback,
    validate_window_alignment,
)

LOGGER = logging.getLogger("tune")

NEWRALFORECAST_CLASSES = {
    "DeepNPTS": "DeepNPTS",
    "PatchTST": "PatchTST",
    "TFT": "TFT",
}


def load_lightgbm_data(dataset: str, level: str, sampling: float):
    import pandas as pd

    from utils.dataset_config import DatasetGermany1D, DatasetLondon1D

    loader = {"london": DatasetLondon1D, "germany": DatasetGermany1D}[dataset]()
    if dataset == "london":
        X, y = loader.get_load_data(exog_level=level)
    else:
        start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        end = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
        X, y = loader.get_load_data(start, end, exog_level=level)
    step = int(round(100.0 / float(sampling)))
    return X.iloc[::step].reset_index(drop=True), y.iloc[::step].reset_index(drop=True)


def load_neuralforecast_frame(dataset: str, level: str, sampling: float):
    import pandas as pd

    from utils.dataset_config import DatasetGermanyNF, DatasetLondonNF

    loader = {"london": DatasetLondonNF, "germany": DatasetGermanyNF}[dataset]()
    if dataset == "london":
        frame = loader.get_inputs_for_load(exog_level=level)
    else:
        start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        end = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
        frame = loader.get_inputs_for_load(start, end, exog_level=level)
    step = int(round(100.0 / float(sampling)))
    return frame.iloc[::step].reset_index(drop=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tune one trainable model (strategy B).")
    parser.add_argument("--model", required=True, choices=list(TUNED_MODELS))
    parser.add_argument("--dataset", required=True,
                        choices=sorted(set(TUNING_DATASETS) | {"london", "germany"}),
                        help="LCL/WPuQ are the tuning datasets; london/germany are aliases.")
    parser.add_argument("--level", default="L0", choices=list(EXOG_LEVEL_CHOICES),
                        help="Exogenous level the search trains on (default L0, the main table).")
    parser.add_argument("--sampling", type=float, default=100.0,
                        help="Sampling rate of the tuning runs (default 100 %%: the full series).")
    parser.add_argument("--trials", type=int, default=TUNING_MAX_TRIALS,
                        help="Hard trial cap (plan: <= 30).")
    parser.add_argument("--hours", type=float, default=TUNING_MAX_HOURS,
                        help="Hard wall-clock cap for this model (plan: <= 3 h).")
    parser.add_argument("--epochs", type=int, default=100, help="Training steps for NF models.")
    parser.add_argument("--seed", type=int, default=0, help="Sampler seed and model seed.")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--windows", type=int, default=1,
                        help="Validation windows per trial for NF models (each window is a full fit).")
    parser.add_argument("--print-space", action="store_true",
                        help="Only print the search space and exit (no training).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    dataset = {"LCL": "london", "WPuQ": "germany"}.get(args.dataset, args.dataset)
    cell = TuningCell(model=args.model, dataset=dataset, level=args.level,
                      sampling=args.sampling, epochs=args.epochs, seed=args.seed,
                      gpu=args.gpu, max_trials=args.trials, max_hours=args.hours)

    horizon = cell.horizon
    if args.print_space:
        import optuna

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        print(f"{cell.model}: {SUGGESTERS[cell.model](trial)}")
        print(f"study={cell.study_name} storage={cell.storage}")
        return 0

    if args.model == "LightGBM":
        X, y = load_lightgbm_data(dataset, args.level, args.sampling)
        n_points = len(y)
        windows = validation_windows(n_points, horizon, args.sampling, cell.validation_days)
        validate_window_alignment(windows, n_points, horizon)

        def objective(trial):
            params = SUGGESTERS["LightGBM"](trial)
            params.setdefault("bagging_freq", 1)
            return lightgbm_windows_mae(params, X, y, windows, seed=args.seed, trial=trial)
    else:
        import neuralforecast.models as nf_models

        frame = load_neuralforecast_frame(dataset, args.level, args.sampling)
        n_points = len(frame)
        windows = validation_windows(n_points, horizon, args.sampling, cell.validation_days)
        validate_window_alignment(windows, n_points, horizon)
        model_class = getattr(nf_models, NEWRALFORECAST_CLASSES[args.model])

        def objective(trial):
            params = SUGGESTERS[args.model](trial)
            return neuralforecast_windows_mae(
                model_class, params, frame, horizon, f"{int(15 * (100 / args.sampling))}min",
                windows, epochs=args.epochs, seed=args.seed, gpu=args.gpu,
                windows_to_use=args.windows,
            )

    study = build_study(cell)
    remaining = max(0, int(cell.max_trials) - len(study.trials))
    if remaining == 0:
        LOGGER.warning("study %s already used its %d trials; writing the existing result",
                       cell.study_name, len(study.trials))
    else:
        started = time.time()
        study.optimize(objective, n_trials=remaining,
                       callbacks=[time_limit_callback(cell.max_hours, started)],
                       gc_after_trial=True)
        used_hours = (time.time() - started) / 3600.0
        path = save_result(cell, study, used_hours,
                           extra={"windows": [list(w) for w in windows],
                                  "n_points": int(n_points)})
        print(f"saved {path}")
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        print(f"best MAE={payload['best_value']:.6f} after "
              f"{payload['used']['trials']} trials / {payload['used']['hours']:.2f} h")
        if payload["boundary_hits"]:
            print("WARNING: parameters on the edge of the space (transfer is inadmissible): "
                  + ", ".join(payload["boundary_hits"]))
        if not payload["saturated"]:
            print("WARNING: the best value was still improving at the end - the budget "
                  "was not enough, say so in the appendix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
