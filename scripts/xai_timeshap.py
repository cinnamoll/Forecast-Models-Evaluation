#!/usr/bin/env python
"""TimeSHAP explanation of one forecast, one model, one series.

Players: four one-day players of the target history plus one player per covariate
block (``utils/covariates.group_columns``), which keeps ``2^N <= 4096`` for every
dataset here — so the game is enumerated exactly and the numbers carry no sampling
error (``plan/timeshap_xai_de_xuat.md`` sections 3.1-3.3).

Payoff: the whole horizon vector.  One coalition therefore explains all 192 (or
96 / 64 / 48) steps, and the explanation can say *when* a block matters.

Perturbation: ``h_X(z) = X ⊙ D_z + B ⊙ (1 - D_z)`` with ``B`` the hour-of-day mean
of the training rows.  ``--mode mean`` replaces it with one constant per column
and is the robustness check the proposal asks for.

Which models can be explained, and how:

* ``LightGBM`` — one fitted booster over the row's own design matrix (covariates
  shifted back one horizon, plus the target's own lags and rolling statistics), so
  a coalition is a new matrix.  Its history channel is real now, which is why the
  temporal players are no longer structurally zero.
* ``Chronos2`` / ``Timestral`` / ``TimesFM3`` — zero-shot checkpoints: a coalition
  is another covariate array, so nothing is fitted per coalition.

Usage
-----
    conda run -n statsml python scripts/xai_timeshap.py \
        --dataset london --series london_load --model LightGBM

    conda run -n chronos python scripts/xai_timeshap.py \
        --dataset germany --series germany_load --model Chronos2 Timestral TimesFM3
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

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.covariates import (CYCLIC_CALENDAR, TIMESHAP_COLUMNS, attach_block, cyclic_calendar,
                              cyclic_columns, expand_cyclic_names, group_columns,
                              restrict_columns, roles_for, scale_covariates, shift_past_only)
from utils.protocol import horizon_for, sampling_tag,steps_per_day
from utils.timeshap import (apply_coalition, additivity_residual, covariate_players,
                            drop_noop_players, enumerate_game, group_shares,
                            periodic_background, players_per_variable, shapley_exact,
                            shares_by_player, signed_by_player, temporal_players,
                            temporal_pruning_report)

#: Models whose explanation needs no refit: a new input mapping for the zero-shot
#: checkpoints, exact TreeSHAP for LightGBM.
CLEAN_GAME_MODELS = ("LightGBM", "Chronos2", "Timestral", "TimesFM3")

#: Models whose input mapping has no target-history channel (a direct regressor reads the
#: row's covariates only).  Perturbing a day of the target history cannot change their
#: forecast, so that player's phi is 0 by construction; a game that has to be comparable
#: with the checkpoints therefore runs with covariate players only for *every* model in
#: the run instead of handing LightGBM free zeros.
NO_HISTORY_CHANNEL = ("LightGBM",)

#: Default model set: the covariate flow of ``plan/scripts.md`` covers these three
#: zero-shot checkpoints only.  Nothing is trained, so there is nothing to tune.
DEFAULT_MODELS = ("Chronos2", "Timestral", "TimesFM3")


def series_prefix(series: str) -> str:
    head, _, tail = series.rpartition("_")
    return head if head and tail.isdigit() else series


def series_entity(series: str):
    """Numeric suffix of a series id, when it has one."""
    head, _, tail = series.rpartition("_")
    return int(tail) if head and tail.isdigit() else None


def load_long_frame(dataset: str, series: str, level: str, sampling: float) -> pd.DataFrame:
    """The long frame every model consumes, subsampled like the benchmark does."""
    from utils.dataset_config import (DatasetBelgiumNF, DatasetGermanyNF,
                                      DatasetLondonNF, DatasetZonnedaelNF)

    classes = dict(belgium=DatasetBelgiumNF, germany=DatasetGermanyNF,
                   london=DatasetLondonNF, zonnedael=DatasetZonnedaelNF)
    loader = classes[dataset]()
    prefix = series_prefix(series)
    entity = series_entity(series)
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    if prefix in ("pv_house", "bess_house"):
        # pv -> get_inputs_for_pv, bess -> get_inputs_for_battery.
        frame = getattr(loader, {"pv_house": "get_inputs_for_pv",
                                 "bess_house": "get_inputs_for_battery"}[prefix])(
            entity, start, end, level=level)
    elif prefix == "zonnedael_customer":
        frame = loader.get_inputs_for_zonnedael_consumption(entity, level=level)
    elif dataset == "germany":
        frame = loader.get_inputs_for_load(start, end, level=level)
    else:
        frame = loader.get_inputs_for_load(level=level)
    step = int(round(100.0 / float(sampling)))
    return frame.iloc[::step].reset_index(drop=True)


def scaled_frame(frame: pd.DataFrame, level: str, horizon: int):
    """Frame as a neural model sees it: covariates min-max scaled on the context."""
    feature_columns = [c for c in frame.columns if c not in ("unique_id", "ds", "y")]
    if level == "L0" or not feature_columns:
        empty = {"known_future": [], "past_only": [], "static": [], "categorical": []}
        return frame, empty
    roles = roles_for(frame[feature_columns])
    scaled = scale_covariates(frame[feature_columns], roles, train_end=len(frame) - horizon)
    return pd.concat([frame.drop(columns=feature_columns), scaled], axis=1), roles


def build_players(frame: pd.DataFrame, roles: dict, horizon: int, per_day: int,
                  include_temporal: bool = True, per_variable: bool = False,
                  cycles: dict | None = None):
    """Temporal players + one player per covariate block (or per variable).

    Returns ``(players, dropped)``.  Players whose perturbation would reproduce
    the frame exactly (a column equal to its own phase profile, i.e. a no-op
    under ``mode="background"``) are dropped: their ``phi`` is identically zero,
    so they only double the game.
    """
    n_context = len(frame) - horizon
    players = temporal_players(n_context, per_day) if include_temporal else []
    if per_variable:
        players += players_per_variable(frame, roles, n_context)
    else:
        groups = group_columns([c for c in frame.columns if c not in ("unique_id", "ds", "y")])
        players += covariate_players(frame, groups, roles, n_context)
    background = periodic_background(frame, per_day, train_end=n_context, cycles=cycles)
    return drop_noop_players(frame, players, background)


# ------------------------------------------------------------------ predictors
def chronos2_predictor(frame: pd.DataFrame, roles: dict, horizon: int, gpu: int, level: str):
    import torch
    from chronos import Chronos2Pipeline

    from utils.device import torch_device
    from utils.tsfm import chronos2_input, point_from_quantiles, prepare_inputs

    device = torch_device(gpu)
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", device_map=device,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)

    def predict(perturbed: pd.DataFrame) -> np.ndarray:
        payload = prepare_inputs(perturbed, roles, level, horizon)
        quantiles, _mean = pipeline.predict_quantiles(
            inputs=[chronos2_input(payload)], prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9])
        return point_from_quantiles(quantiles[0], 1)

    return predict

def timestral_predictor(frame: pd.DataFrame, roles: dict, horizon: int, gpu: int, level: str):
    from timestral import TimestralForecaster

    from utils.device import torch_device
    from utils.tsfm import prepare_inputs, timestral_arguments, to_numpy

    forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-123M",
                                                     device=torch_device(gpu))

    def predict(perturbed: pd.DataFrame) -> np.ndarray:
        payload = prepare_inputs(perturbed, roles, level, horizon)
        forecast = forecaster.forecast(prediction_length=horizon, **timestral_arguments(payload))
        return to_numpy(forecast.median).reshape(-1)[:horizon]

    return predict


def timesfm3_predictor(frame: pd.DataFrame, roles: dict, horizon: int, gpu: int, level: str,
                       model_id: str = "google/timesfm-3.0-pytorch", batch_size: int = 4):
    """TimesFM 3: covariates are extra variates, so a coalition is a new stack.

    ``scale_categorical`` is on because this checkpoint has no integer embedding
    channel: the calendar codes are min-max scaled like any other covariate
    (``utils/tsfm.timesfm3_covariates``).  Static metadata has no channel and is
    dropped by ``prepare_inputs`` before it ever reaches the game.
    """
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    from utils.device import torch_device
    from utils.tsfm import prepare_inputs, timesfm3_covariates, to_numpy

    forecaster = TimesFM3Evaluator(ModelConfig(
        checkpoint_path=model_id, per_core_batch_size=batch_size, device=torch_device(gpu)))

    def predict(perturbed: pd.DataFrame) -> np.ndarray:
        payload = prepare_inputs(perturbed, roles, level, horizon, scale_categorical=True)
        output = forecaster.predict(
            context=payload["context"], horizon=horizon, return_quantiles=True,
            use_symmetric_averaging=True, make_positive=True, **timesfm3_covariates(payload))
        return to_numpy(output.forecast).reshape(-1)[:horizon]

    return predict



def make_predictor(model: str, frame: pd.DataFrame, roles: dict, horizon: int, freq: str,
                   epochs: int, gpu: int, params: dict | None, level: str):
    if model == "Chronos2":
        return chronos2_predictor(frame, roles, horizon, gpu, level), True
    if model == "Timestral":
        return timestral_predictor(frame, roles, horizon, gpu, level), True
    if model == "TimesFM3":
        return timesfm3_predictor(frame, roles, horizon, gpu, level), True

    # Trainable baseline: fitted once, scored per coalition without refit
    # (utils/xai_predictors.py).
    from utils.xai_predictors import lightgbm_predictor

    log = logging.getLogger("timeshap")
    if model == "LightGBM":
        return lightgbm_predictor(frame, roles, horizon, gpu, params, log=log), True
    raise ValueError(f"no predictor for {model!r}")

def explain(model: str, frame: pd.DataFrame, roles: dict, horizon: int, per_day: int,
            mode: str, predict, include_temporal: bool, out_dir: Path,
            sampling: float, level: str, per_variable: bool = False,
            max_players: int = 12, cycles: dict | None = None) -> dict:
    players, dropped = build_players(frame, roles, horizon, per_day,
                                     include_temporal=include_temporal,
                                     per_variable=per_variable, cycles=cycles)
    if dropped:
        logging.getLogger("timeshap").info(
            "%s: dropped no-op player(s) %s (replacing them with the hour-of-day "
            "background reproduces the column, so phi would be exactly 0)",
            model, ", ".join(dropped))
    if len(players) > max_players:
        raise ValueError(f"{model}: {len(players)} players -> {2 ** len(players)} coalitions; "
                         f"raise --max-players above {max_players} only if that cost is affordable, "
                         "or drop --per-covariate")
    n_context = len(frame) - horizon
    background = periodic_background(frame, per_day, train_end=n_context, cycles=cycles)

    started = time.time()
    values = enumerate_game(
        lambda active: predict(apply_coalition(frame, background, players, active, mode=mode)),
        players, max_players=max_players)
    phi, base, prediction = shapley_exact(values, players)
    residual = additivity_residual(phi, base, prediction)
    elapsed = time.time() - started
    if residual > 1e-6:
        raise AssertionError(
            f"{model}: additivity residual {residual:.3e} exceeds 1e-6; the estimator or the "
            "payoff is wrong and the explanation must not be reported")

    shares = shares_by_player(phi, players)
    grouped = group_shares(shares)
    signed = signed_by_player(phi, players)

    out_dir.mkdir(parents=True, exist_ok=True)
    step_rows = []
    for index, player in enumerate(players):
        group = "target_history" if player.kind == "temporal" else player.name
        for step, value in enumerate(phi[index]):
            step_rows.append({"model": model, "player": player.name, "group": group,
                              "step": step, "phi": float(value)})
    pd.DataFrame(step_rows).to_csv(out_dir / "shap_values.csv", index=False, float_format="%.8f")
    pd.DataFrame({"model": model, "step": np.arange(horizon), "base": base,
                  "prediction": prediction, "residual": phi.sum(axis=0) + base - prediction}
                 ).to_csv(out_dir / "base_value.csv", index=False, float_format="%.8f")
    pd.DataFrame({"model": model, "group": grouped.index,
                  "share": grouped.to_numpy()}).to_csv(out_dir / "group_shares.csv", index=False, float_format="%.8f")
    pd.DataFrame({"model": model, "player": shares.index, "share": shares.to_numpy(),
                  "signed_mean": signed.to_numpy()}).to_csv(
        out_dir / "player_shares.csv", index=False, float_format="%.8f")
    temporal_pruning_report(phi, players).to_csv(out_dir / "temporal_pruning.csv", index=False, float_format="%.8f")
    (out_dir / "coalitions.json").write_text(json.dumps({
        "model": model, "level": level, "sampling": float(sampling),
        "players": [p.name for p in players], "n_players": len(players),
        "coalitions": 2 ** len(players), "estimator": "exact (full enumeration)",
        "mode": mode, "horizon": horizon, "context": n_context,
        "background_rotation": per_day // 2,
        "additivity_residual": residual, "seconds": round(elapsed, 3),
        "temporal_players": include_temporal, "per_variable": per_variable,
        "noop_dropped": dropped,
        "cyclic": bool(cycles), "cycles": dict(cycles or {}),
    }, indent=2), encoding="utf-8")

    logging.getLogger("timeshap").info("%s: %d players, %d coalitions in %.1fs, residual %.2e",
                                       model, len(players), 2 ** len(players), elapsed, residual)
    return {"shares": grouped, "players": shares, "residual": residual,
            "seconds": elapsed, "n_players": len(players)}



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TimeSHAP explanation of the covariate models.")
    parser.add_argument("--dataset", required=True, choices=["belgium", "germany", "london", "zonnedael"])
    parser.add_argument("--series", required=True)
    parser.add_argument("--models", nargs="+", choices=list(CLEAN_GAME_MODELS), default=list(DEFAULT_MODELS))
    parser.add_argument("--level", default="COV", choices=list(("L0", "COV")))
    parser.add_argument("--sampling", type=float, default=100.0)
    parser.add_argument("--mode", choices=["background", "mean", "zero"], default="background",
                        help="Perturbation: hour-of-day background (main), column mean, or zero.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--params-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "tuning")
    parser.add_argument("--allow-refit", action="store_true",
                        help="Accepted for older command lines; every model that is left "
                             "(covariate players only; expensive but clean).")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--players", choices=("auto", "covariates", "temporal"), default="auto",
                        help="Player set, identical for every model in the run.  "
                             "'covariates' = one player per covariate variable (5 -> 32 coalitions); "
                             "'temporal' = those plus hist_d1..hist_d4 (9 -> 512); "
                             "'auto' (default) picks 'covariates' when the model list contains one "
                             "with no target-history channel (LightGBM), because its temporal phi is "
                             "exactly 0 there.")
    parser.add_argument("--covariate-only", dest="players", action="store_const", const="covariates",
                        help="Old name for --players covariates.")
    parser.add_argument("--per-covariate", "--per-variable", dest="per_variable",
                        action="store_true",
                        help="One player per covariate variable, on top of hist_d1..hist_d4 "
                             "(5 variables -> 9 players -> 512 coalitions, so a full game stays "
                             "affordable).  --per-variable is the old name, kept as an alias.")
    parser.add_argument("--cyclic", "--harmonic", dest="cyclic", action="store_true",
                        help="Carry hour and dayofweek as (sin, cos) pairs on their natural "
                             "period instead of integer codes: a numeric channel for every "
                             "model, so Timestral stops target-encoding the clock into a "
                             "climatology (which made its hour player phi == 0 exactly, "
                             "because a relabeling of the codes is invisible to a per-category "
                             "mean of the target).  Each pair stays ONE player, and its "
                             "counterfactual is a shift of half a cycle (12 h / 3.5 days).")
    parser.add_argument("--covariates", nargs="+", default=list(TIMESHAP_COLUMNS),
                        metavar="COLUMN",
                        help="Covariate columns the game and the model input use.  Default: "
                             + " ".join(TIMESHAP_COLUMNS)
                             + ".  The default set is the one that survived ablation: it drops "
                               "quarter_hour/hour*4+minute//15 and is_weekend/dayofweek>=5 "
                               "(exact functions of kept columns), dni (r=0.82-0.87 with ghi), "
                               "precipitation (near-constant), and temperature + wind_speed "
                               "(< 4 %% horizon-MAE effect, and temperature hurts more often "
                               "than it helps).  Pass every column to reproduce the older "
                               "12-player game.")
    parser.add_argument("--all-covariates", action="store_true",
                        help="Take every covariate the COV frame carries instead of --covariates.  "
                             "This is the benchmark's own input set (calendar + weather + price + "
                             "state), and with block players it still fits: 12-14 columns form "
                             "5-7 blocks, i.e. 32-128 coalitions.")
    parser.add_argument("--max-players", type=int, default=12,
                        help="Refuse a game larger than this many players (2^N coalitions).")
    parser.add_argument("--dry-run", action="store_true", help="Print the player set and exit.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # One player set for the whole run.  'temporal' is allowed even for a model without a
    # history channel: its four hist_d players then score exactly 0, which is reported
    # rather than hidden, so the player list stays comparable across models.
    covariate_only = args.players == "covariates"
    no_history = [model for model in args.models if model in NO_HISTORY_CHANNEL]
    if args.players == "auto":
        covariate_only = bool(no_history)
        if no_history:
            logging.getLogger("timeshap").warning(
                "%s has no target-history channel, so a temporal player scores exactly 0 there; "
                "using --players covariates for every model in this run (%d variables). "
                "Pass --players temporal to keep the 9-player game anyway.",
                ", ".join(no_history), len(args.covariates))
    elif no_history and not covariate_only:
        logging.getLogger("timeshap").warning(
            "%s has no target-history channel: its hist_d players will be exactly 0 in this "
            "9-player game (the player set still matches the other models)",
            ", ".join(no_history))
    horizon = horizon_for(args.sampling)
    per_day = steps_per_day(args.sampling, args.dataset)
    frame = load_long_frame(args.dataset, args.series, args.level, args.sampling)
    prefix = series_prefix(args.series)
    task = {"pv_house": "solar", "bess_house": "battery"}.get(prefix, "load")
    for column in ([] if args.all_covariates else [c for c in args.covariates if c not in frame.columns]):
        frame = attach_block(frame, args.dataset, series_entity(args.series), task, column)
        logging.getLogger("timeshap").info(
            "attached %r to %s/%s - the loader scopes it to another task, so this game explains a "
            "model WITH that channel, not the benchmark's input mapping for this series",
            column, args.dataset, args.series)

    cycles: dict = {}
    requested = [] if args.all_covariates else list(args.covariates)
    if args.cyclic:
        frame, cycles = cyclic_calendar(frame, steps_per_day=per_day)
        requested = expand_cyclic_names(requested, cycles)
        replaced = [base for base in CYCLIC_CALENDAR if cyclic_columns(base)[0] in cycles]
        logging.getLogger("timeshap").info(
            "cyclic calendar: %s replaced by %s; each pair is one player and its "
            "counterfactual is a shift of half a cycle (%s rows)",
            ", ".join(replaced),
            ", ".join(name for base in replaced for name in cyclic_columns(base)),
            ", ".join(str(cycles[cyclic_columns(base)[0]]) for base in replaced))

    scaled, roles = scaled_frame(frame, args.level, horizon)
    if requested:
        scaled, roles = restrict_columns(scaled, roles, requested)
    freq = f"{int(15 * (100 / args.sampling))}min"

    used = requested or [c for c in frame.columns if c not in ("unique_id", "ds", "y")]
    logging.getLogger("timeshap").info(
        "covariates (%d): %s", len(used), ", ".join(used))

    players, dropped = build_players(scaled, roles, horizon, per_day,
                                     include_temporal=not covariate_only,
                                     per_variable=args.per_variable, cycles=cycles)
    logging.getLogger("timeshap").info(
        "%s/%s: %d points, horizon %d, %d players -> %d coalitions",
        args.dataset, args.series, len(scaled), horizon, len(players), 2 ** len(players))
    if dropped:
        logging.getLogger("timeshap").info(
            "dropped no-op player(s) %s (no-op means replacing the column with the "
            "hour-of-day background reproduces it, so phi would be exactly 0)",
            ", ".join(dropped))
    if args.dry_run:
        for player in players:
            print(f"  {player.name:14s} {player.kind:10s} rows={player.size:5d} columns={len(player.columns)}")
        return 0

    tag = sampling_tag(args.sampling) + ("_harmonic" if args.cyclic else "")
    out_dir = args.out or (Path(__file__).resolve().parents[1] / "results_covariate" / "xai_timeshap" / args.dataset /
                           f"{args.series}_{args.level}_{tag}")
    shares_rows, notes = {}, {}
    for model in args.models:
        # The tuned file is named after the *tuning* series, not the reported dataset
        # (london borrows zonnedael9, belgium borrows belgium_pv1), so ask
        # utils.tuning for the resolution the benchmark cells used; the old
        # {model}_{dataset}_... name stays as a fallback for hand-written files.
        params_path = args.params_dir / f"{model}_{args.dataset}_{args.level}_{sampling_tag(args.sampling)}.json"
        try:
            from utils.tuning import tuned_resolution
            resolved, _tuned_at = tuned_resolution(Path(args.params_dir).parent, model,
                                                   args.dataset, args.level, args.sampling)
        except (KeyError, ImportError):
            resolved = params_path
        if resolved.exists():
            params_path = resolved
        params = {}
        if params_path.exists():
            payload = json.loads(params_path.read_text(encoding="utf-8"))
            params = dict(payload.get("best_params", payload))
            logging.getLogger("timeshap").info("%s: tuned parameters from %s", model, params_path.name)
        include_temporal = not covariate_only   # one player set for every model in the run
        if model not in CLEAN_GAME_MODELS and not args.allow_refit:
            notes[model] = ("skipped: NeuralForecast freezes the context and hist_exog into the "
                            "fitted dataset, so a clean coalition game needs --allow-refit")
            logging.getLogger("timeshap").warning(
                "%s skipped (no clean game without refit); pass --allow-refit", model)
            continue
        if model not in CLEAN_GAME_MODELS:
            include_temporal = False
            notes[model] = "refit per coalition; covariate players only"
        try:
            predict, _ = make_predictor(model, scaled, roles, horizon, freq, args.epochs,
                                        args.gpu, params, args.level)
            result = explain(model, scaled, roles, horizon, per_day, args.mode, predict,
                             include_temporal, out_dir / model, args.sampling, args.level,
                             per_variable=args.per_variable, max_players=args.max_players,
                             cycles=cycles)
        except Exception as error:  # one model must not kill the whole report
            logging.getLogger("timeshap").error("%s failed: %s", model, error, exc_info=True)
            notes[model] = f"failed: {error}"
            continue
        shares_rows[model] = result["shares"]

    if not shares_rows:
        print("no model produced an explanation - nothing written")
        return 1
    table = pd.DataFrame(shares_rows)
    table.to_csv(out_dir / "group_shares_all_models.csv", float_format="%.8f")
    (out_dir / "notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(table.round(4).to_string())
    print(f"written -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
