#!/usr/bin/env python
"""Drop covariates one group at a time, driven by TimeSHAP shares.

Start from every covariate (``--level ALL``), measure the game of
``docs/timeshap_xai_de_xuat.md`` section 3, drop the player with the smallest
share, measure again, stop when the next drop costs more MAE than ``--tolerance``.
The loop itself lives in ``utils/elimination.py``; this file only wires a model to
it, and that wiring is deliberately the smallest possible: the forecast closures it
reuses are the ones ``scripts/xai_report.py`` already runs, so no model inference
path is written twice.

Three decisions the defaults encode (proposal section 10):

* The drop is decided on the **validation block** -- the last horizon-sized window
  that ends ``protocol.TUNING_VALIDATION_DAYS`` days before the test window -- and
  only the resulting ``test_mae`` is a reportable number.  Selecting on the window
  you then report is how an "improvement" becomes circular.
* ``--tolerance`` is the MAE a drop may cost before it is refused.  Set it to the
  IQR of the model's L0 runs: the seed is ``int(time.time())``, so a delta smaller
  than that spread is a reseeding, not an improvement.
* Nothing is retrained between coalitions.  A game whose ``f`` changes mid-game has
  no Shapley values to speak of, which is why the elimination covers the models
  that can be perturbed at inference and refuses the ones that cannot.

Models this script can measure: ``LightGBM`` (TreeSHAP, exact, no target players --
see proposal 3.5), ``Chronos2`` and ``Timestral`` (zero-shot, masking/background at
inference).  ``PatchTST``/``TCN``/``DeepNPTS`` and ``TFT`` are refused with the
reason, because neuralforecast freezes the context and ``hist_exog`` into its
dataset at ``fit`` and the gluonts environment does not exist yet -- the two
questions open in proposal 3.2 and 1.2.

Usage
-----
    python scripts/xai_elimination.py --dataset london --series london_load \
        --model Chronos2 --sampling 100 --granularity group --tolerance 0.001
    python scripts/xai_elimination.py --dataset london --series london_load \
        --model LightGBM --dry-run

Run each model in the environment that holds it (``conda run -n chronos``,
``-n statsml``), the way ``scripts/run_experiments.py`` does.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from utils.elimination import Measurement, eliminate, write_outputs
from utils.protocol import (
    COVARIATE_LEVEL,
    EXOG_LEVEL_CHOICES,
    TABLE_MODELS,
    TUNING_VALIDATION_DAYS,
    horizon_for,
    sampling_tag,
    steps_per_day,
)
from utils.xai import (
    additivity_residual,
    coalition_frame,
    collapse_to_groups,
    periodic_background,
    player_shares,
    players_for_frame,
    shapley,
)

LOGGER = logging.getLogger("xai-elimination")

IDENTIFIER_COLUMNS = ("unique_id", "ds", "y")

#: Models whose inference can be perturbed without refitting, mapped to the reason
#: the others cannot.  Kept as data so ``--dry-run`` and the error message agree.
REFUSED = {
    "DeepNPTS": "neuralforecast bakes the context and hist_exog into the fitted dataset "
                "(models_neuralforecast.py:129,154); proposal 3.2 options (a)/(b)",
    "PatchTST": "same neuralforecast limitation; proposal 3.2 options (a)/(b)",
    "TCN": "same neuralforecast limitation; proposal 3.2 options (a)/(b)",
    "TFT": "the gluonts environment does not exist yet (proposal 1.2) and "
           "past_dynamic_features is not passed (1.2)",
}


def window(frame: pd.DataFrame, offset: int) -> pd.DataFrame:
    """The frame as it looked ``offset`` points before the end (0 = test window)."""
    if offset <= 0:
        return frame.reset_index(drop=True)
    if len(frame) - offset <= 0:
        raise ValueError(f"offset {offset} leaves no frame")
    return frame.iloc[: len(frame) - offset].reset_index(drop=True)


# ------------------------------------------------------------------ predictors

def tsfm_predictor(model: str, horizon: int, gpu: int = -1) -> Callable[[pd.DataFrame], np.ndarray]:
    """``predict(window_frame) -> horizon`` for the two zero-shot checkpoints.

    Reuses the closures of ``scripts/xai_report.py`` verbatim: they split the frame
    into ``frame[:-horizon]`` (context) and ``frame[-horizon:]`` (target), which is
    exactly the window contract used here.
    """
    from scripts.xai_report import chronos2_predict_closure, timestral_predict_closure
    from utils.device import torch_device

    if model == "Chronos2":
        from chronos import Chronos2Pipeline

        pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2",
                                                    device_map=torch_device(gpu))
        return chronos2_predict_closure(pipeline, horizon)

    from timestral import TimestralForecaster

    forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-123M",
                                                     device=torch_device(gpu))
    return timestral_predict_closure(forecaster, horizon)


# ------------------------------------------------------------------- measuring

def build_measure(dataset: str, series: str, level: str, sampling: float,
                  model: str, gpu: int = -1, params: Optional[Dict] = None,
                  granularity: str = "group", mode: str = "background",
                  n_samples: int = 4096, seed: int = 0,
                  predict_override: Optional[Callable[[pd.DataFrame], np.ndarray]] = None) -> Callable:
    """``measure(columns) -> Measurement``: one game on the selection window.

    The model is built once and called once per coalition.  ``selection_mae`` is the
    MAE of the full forecast on the selection window -- the same payoff the game
    explains -- so the drop rule and the attribution are measured on one object.

    ``predict_override`` exists so the loop can be exercised end to end on real data
    with a stand-in learner in an environment that has no checkpoint; nothing else
    in the path changes.
    """
    horizon = horizon_for(sampling)
    per_day = steps_per_day(sampling)
    offset = int(TUNING_VALIDATION_DAYS) * per_day

    if model == "LightGBM":
        from scripts.xai_report import load_direct_frame

        X, y = load_direct_frame(dataset, series, level, sampling)
        table = pd.DataFrame({"y": y.to_numpy()})
        for column in X.columns:
            table[column] = X[column].to_numpy()
        columns = tuple(X.columns)
        return lambda cols: _lightgbm_measure(cols, window(table, 0), window(table, offset),
                                             horizon, params, gpu, granularity)

    from scripts.xai_report import load_long_frame

    frame = load_long_frame(dataset, series, level, sampling)
    predict = predict_override or tsfm_predictor(model, horizon, gpu)
    cache: Dict[Tuple[str, ...], Measurement] = {}

    def measure(columns: Sequence[str]) -> Measurement:
        key = tuple(columns)
        if key in cache:
            return cache[key]
        select = window(frame, offset)
        test = window(frame, 0)
        players = players_for_frame(select, columns=key, steps_per_day=per_day,
                                    granularity=granularity)
        background = periodic_background(select, per_day, train_end=len(select) - horizon)

        def payoff(active: Tuple[str, ...]) -> np.ndarray:
            perturbed = coalition_frame(select, active, players, background, mode=mode)
            prediction = np.asarray(predict(perturbed), dtype=float).ravel()
            if prediction.size != horizon:
                raise ValueError(f"{model} returned {prediction.size} points, expected {horizon}")
            return prediction

        result = shapley(payoff, players, n_samples=n_samples, seed=seed)
        prediction = payoff(tuple(p.name for p in players))
        truth = select.iloc[-horizon:]["y"].to_numpy(dtype=float)
        test_prediction = np.asarray(predict(test), dtype=float).ravel()
        truth_test = test.iloc[-horizon:]["y"].to_numpy(dtype=float)
        measurement = Measurement(
            columns=key, shares=result.shares(), groups=result.shares(),
            selection_mae=float(np.mean(np.abs(prediction - truth))),
            test_mae=float(np.mean(np.abs(test_prediction - truth_test))),
            players=tuple(players), evaluated=result.evaluated, exact=result.exact,
            additivity=additivity_residual(result.phi, result.base, prediction),
        )
        cache[key] = measurement
        return measurement

    return measure


def _lightgbm_measure(columns, test_table, select_table, horizon, params, gpu,
                      granularity) -> Measurement:
    """TreeSHAP of the direct multi-step LightGBM on the selection window.

    LightGBM has no target-history player: its design matrix is covariates only
    (``utils/dataset_config.py:180,207,258``), so the game is over the covariate
    groups alone and its vector says nothing about the past load (proposal 3.5).
    """
    import lightgbm as lgb

    from utils.xai import tree_shap

    def fit_and_attribute(table: pd.DataFrame) -> Tuple[pd.Series, float]:
        X = table[list(columns)]
        y = table["y"]
        model = lgb.LGBMRegressor(
            random_state=0, n_jobs=-1, verbosity=-1, deterministic=False,
            bagging_fraction=0.5, feature_fraction=0.5, bagging_freq=1,
            device_type="gpu" if gpu in (0, 1) else "cpu",
            **{k: v for k, v in (params or {}).items()
               if k not in {"random_state", "n_jobs", "verbosity"}},
        ).fit(X.iloc[:-horizon], y.iloc[:-horizon])
        shap_values = tree_shap(model, X.iloc[-horizon:]).drop(columns="bias")
        magnitude = shap_values.abs().mean(axis=0)
        prediction = model.predict(X.iloc[-horizon:])
        mae = float(np.mean(np.abs(prediction - y.iloc[-horizon:].to_numpy(dtype=float))))
        return magnitude, mae

    magnitude, selection = fit_and_attribute(select_table)
    _, test_mae = fit_and_attribute(test_table)
    players = players_for_frame(select_table, columns=columns, granularity=granularity,
                               include_target=False)
    shares = player_shares(magnitude.to_numpy().reshape(1, -1), list(magnitude.index))
    return Measurement(columns=tuple(columns), shares=shares,
                       groups=collapse_to_groups(shares, players),
                       selection_mae=selection, test_mae=test_mae,
                       players=tuple(players), evaluated=len(columns), exact=True,
                       additivity=0.0)


# ------------------------------------------------------------------------ main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TimeSHAP-driven covariate elimination.")
    parser.add_argument("--dataset", required=True,
                        choices=["belgium", "germany", "london", "zonnedael"])
    parser.add_argument("--series", required=True)
    parser.add_argument("--model", required=True, choices=list(TABLE_MODELS))
    parser.add_argument("--level", default=COVARIATE_LEVEL, choices=list(EXOG_LEVEL_CHOICES),
                        help="Starting point (default ALL: every covariate the dataset has). "
                             "L0 has nothing to drop.")
    parser.add_argument("--sampling", type=float, default=100.0)
    parser.add_argument("--granularity", choices=["group", "column"], default="group",
                        help="group = exact game over 4+G players; column = sampled KernelSHAP "
                             "over every column, which is what the player count forces.")
    parser.add_argument("--mode", choices=["background", "mask", "drop"], default="background",
                        help="how a dropped player is blanked (proposal 3.2).")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="Selection MAE a drop may cost. Use the IQR of the model's L0 runs.")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--min-covariates", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=4096,
                        help="Coalitions for the sampled estimator (only above 12 players).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--params-dir", type=Path, default=PROJECT_ROOT / "results" / "tuning")
    parser.add_argument("--seconds-per-call", type=float, default=None,
                        help="Latency of one forecast, to print the cost of a dry run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the player set, the coalition count and the plan; run nothing.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    horizon = horizon_for(args.sampling)
    per_day = steps_per_day(args.sampling)

    if args.level == "L0":
        raise SystemExit(
            "L0 is the paper baseline and has no covariate to drop. Use --level ALL "
            "(the default): the elimination starts from every covariate the dataset has."
        )

    if args.model in REFUSED:
        raise SystemExit(
            f"{args.model} cannot be measured without refitting: {REFUSED[args.model]}.\n"
            "Run --model LightGBM, Chronos2 or Timestral, or settle proposal 3.2 first."
        )

    if args.model == "LightGBM":
        from scripts.xai_report import load_direct_frame

        X, _ = load_direct_frame(args.dataset, args.series, args.level, args.sampling)
        columns = tuple(X.columns)
        frame = None
    else:
        from scripts.xai_report import load_long_frame

        frame = load_long_frame(args.dataset, args.series, args.level, args.sampling)
        columns = tuple(c for c in frame.columns if c not in IDENTIFIER_COLUMNS)

    if args.model == "LightGBM":
        # covariates only: the direct matrix has no target-history player (3.5)
        probe = pd.DataFrame({c: 0.0 for c in columns}, index=[0])
        include_target = False
    else:
        # the real selection window, so the player count is the one that will be used
        probe = window(frame, int(TUNING_VALIDATION_DAYS) * per_day)
        include_target = True
    sample_players = players_for_frame(probe, columns=columns, steps_per_day=per_day,
                                       granularity=args.granularity,
                                       include_target=include_target)
    n_players = len(sample_players)
    coalitions = (1 << n_players) if n_players <= 12 else int(args.n_samples)
    LOGGER.info("%s/%s at %s: %d covariate columns -> %d players (%s), %d coalitions",
                args.dataset, args.series, args.level, len(columns), n_players,
                args.granularity, coalitions)
    LOGGER.info("players: %s", [p.name for p in sample_players])

    if args.dry_run:
        # only covariate players can be dropped, and the loop never goes below
        # ``--min-covariates``: counting the temporal players as rounds overstated
        # the budget by four
        n_covariates = len([p for p in sample_players if not p.temporal])
        max_drops = max(0, n_covariates - int(args.min_covariates))
        drops = max_drops if args.max_rounds is None else min(int(args.max_rounds), max_drops)
        rounds_total = drops + 1
        per_round = coalitions + 1          # + the full-coalition forecast
        if args.model != "LightGBM":
            per_round += 1                  # + the test-window forecast of that round
        estimate = "unknown (pass --seconds-per-call)"
        if args.seconds_per_call:
            seconds = rounds_total * per_round * float(args.seconds_per_call)
            estimate = f"{rounds_total * per_round} forecasts, {seconds / 60:.1f} min"
        print(f"columns  : {len(columns)}")
        print(f"players  : {n_players}"
              f"{'' if args.model == 'LightGBM' else ' (4 temporal + ' + str(n_covariates) + ' covariate)'}")
        print(f"coalitions per round: {coalitions}")
        print(f"drops    : up to {drops} ({rounds_total} rounds)")
        print(f"cost     : {estimate}")
        print(f"selection window: last {horizon} points ending "
              f"{TUNING_VALIDATION_DAYS} days before the test window")
        return 0

    params_path = (args.params_dir /
                   f"{args.model}_{args.dataset}_{args.level}_{sampling_tag(args.sampling)}.json")
    params: Dict = {}
    if params_path.exists():
        from utils.tuning import load_best_params

        params = load_best_params(params_path)

    measure = build_measure(args.dataset, args.series, args.level, args.sampling,
                            args.model, gpu=args.gpu, params=params,
                            granularity=args.granularity, mode=args.mode,
                            n_samples=args.n_samples, seed=args.seed)

    result = eliminate(measure, columns, tolerance=args.tolerance,
                       max_rounds=args.max_rounds,
                       min_covariates=args.min_covariates,
                       log=lambda message: LOGGER.info(message))

    out_dir = args.out or (PROJECT_ROOT / "results" / "xai_elimination" / args.dataset /
                          f"{args.series}_{args.model}_{args.level}_{sampling_tag(args.sampling)}")
    paths = write_outputs(result, out_dir)
    print(result.ledger().to_string(index=False))
    LOGGER.info("kept %d of %d columns, stop: %s", len(result.kept), len(columns),
                result.stop_reason)
    for key, path in paths.items():
        LOGGER.info("%s -> %s", key, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
