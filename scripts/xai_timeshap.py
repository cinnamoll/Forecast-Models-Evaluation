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

* ``LightGBM`` — TreeSHAP is exact, but its design matrix has **no lag**, so its
  game has covariate players only.  That is reported, not hidden.
* ``TFT`` — the GluonTS predictor accepts a new ``ListDataset`` per coalition, so
  the game is clean without refitting.
* ``Chronos2`` / ``Timestral`` — zero-shot, covariates are plain arrays, so a
  coalition is a new input mapping.
* ``DeepNPTS`` / ``PatchTST`` / ``TCN`` — NeuralForecast freezes the context and
  ``hist_exog`` into the fitted dataset, so a clean game needs a refit per
  coalition.  That is enabled explicitly with ``--allow-refit`` (and then the
  player set is restricted to covariate blocks to keep the cost finite); without
  it the model is skipped with the reason.

Usage
-----
    conda run -n statsml python scripts/xai_timeshap.py \
        --dataset london --series london_load --model LightGBM

    conda run -n gluonts python scripts/xai_timeshap.py \
        --dataset belgium --series pv_house_1 --model TFT

    conda run -n chronos python scripts/xai_timeshap.py \
        --dataset germany --series germany_load --model Chronos2 Timestral
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

from utils.covariates import group_columns, roles_for, scale_covariates, shift_past_only
from utils.protocol import horizon_for, sampling_tag,steps_per_day
from utils.timeshap import (apply_coalition, additivity_residual, covariate_players,
                            enumerate_game, group_shares, periodic_background,
                            shapley_exact, shares_by_player, signed_by_player,
                            temporal_players, temporal_pruning_report)


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
                  include_temporal: bool = True):
    """Temporal players + one player per covariate block."""
    n_context = len(frame) - horizon
    players = temporal_players(n_context, per_day) if include_temporal else []
    groups = group_columns([c for c in frame.columns if c not in ("unique_id", "ds", "y")])
    players += covariate_players(frame, groups, roles, n_context)
    return players


# ------------------------------------------------------------------ predictors

def lightgbm_predictor(frame: pd.DataFrame, horizon: int, params: dict | None):
    """Direct multi-step regressor: predict the horizon rows from their covariates."""
    import lightgbm as lgb

    columns = [c for c in frame.columns if c not in ("unique_id", "ds", "y")]
    train_end = len(frame) - horizon
    model = lgb.LGBMRegressor(random_state=0, n_jobs=-1, verbosity=-1, deterministic=False,
                              bagging_fraction=0.5, feature_fraction=0.5, bagging_freq=1,
                              **(params or {})).fit(frame[columns].iloc[:train_end],
                                                    frame["y"].iloc[:train_end])

    def predict(perturbed: pd.DataFrame) -> np.ndarray:
        return model.predict(perturbed[columns].iloc[-horizon:])

    return predict


def tft_predictor(frame: pd.DataFrame, roles: dict, horizon: int, freq: str, epochs: int,
                  gpu: int, params: dict | None, level: str):
    """GluonTS TFT: one fit, then a new dataset per coalition."""
    import mxnet as mx
    from gluonts.mx.model.tft import TemporalFusionTransformerEstimator
    from gluonts.mx.trainer import Trainer

    from models.models_gluonts import covariate_spec, to_gluonts_dataset
    from utils.device import uses_gpu

    train_end = len(frame) - horizon
    entry_extra, covariate_kwargs = covariate_spec(frame, roles, level, train_end=train_end)
    start = pd.Timestamp(frame["ds"].iloc[0])
    train_ds = to_gluonts_dataset(frame.iloc[:train_end], freq, start, entry_extra)
    ctx = mx.gpu(gpu) if uses_gpu(gpu) else mx.cpu()

    accepted = {k: v for k, v in (params or {}).items()
                if k in {"hidden_dim", "num_heads", "dropout_rate"}}
    trainer_kwargs = {k: v for k, v in (params or {}).items()
                      if k in {"learning_rate", "num_batches_per_epoch"}}
    trainer_kwargs.setdefault("num_batches_per_epoch", 1)
    estimator = TemporalFusionTransformerEstimator(
        freq=freq, prediction_length=horizon, **covariate_kwargs, **accepted,
        trainer=Trainer(epochs=epochs, ctx=ctx, **trainer_kwargs))
    predictor = estimator.train(training_data=train_ds)

    def predict(perturbed: pd.DataFrame) -> np.ndarray:
        dataset = to_gluonts_dataset(perturbed, freq, start, entry_extra)
        forecast = next(iter(predictor.predict(dataset)))
        return np.asarray(forecast.mean[:horizon], dtype=float)

    return predict


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
    from utils.tsfm import prepare_inputs, timestral_arguments

    forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-123M",
                                                     device=torch_device(gpu))

    def predict(perturbed: pd.DataFrame) -> np.ndarray:
        payload = prepare_inputs(perturbed, roles, level, horizon)
        forecast = forecaster.forecast(prediction_length=horizon, **timestral_arguments(payload))
        return np.asarray(forecast.median, dtype=float).reshape(-1)[:horizon]

    return predict


def neuralforecast_refit_predictor(frame: pd.DataFrame, roles: dict, horizon: int, freq: str,
                                   epochs: int, gpu: int, params: dict | None, model_name: str):
    """NeuralForecast: refit on the perturbed frame for every coalition.

    Expensive, and offered on purpose: without it those models have no Shapley
    value at all, and a table that silently omits them reads as "no covariate
    dependence" instead of "not measured".
    """
    import neuralforecast.models as nf_models
    from neuralforecast import NeuralForecast

    from utils.device import lightning_device_config, uses_gpu
    import torch

    model_class = getattr(nf_models, model_name)
    protected = {"h", "input_size", "max_steps", "random_seed", "accelerator", "devices",
                 "hist_exog_list", "futr_exog_list", "stat_exog_list"}

    def predict(perturbed: pd.DataFrame) -> np.ndarray:
        train_end = len(perturbed) - horizon
        exog = {}
        if roles["past_only"]:
            exog["hist_exog_list"] = list(roles["past_only"])
        if roles["known_future"]:
            exog["futr_exog_list"] = list(roles["known_future"])
        if roles["static"]:
            exog["stat_exog_list"] = list(roles["static"])
        kwargs = dict(h=horizon, input_size=horizon * 2, max_steps=int(epochs),
                      random_seed=0, **lightning_device_config(gpu), **exog)
        kwargs.update({k: v for k, v in (params or {}).items() if k not in protected})
        static_df = None
        if exog.get("stat_exog_list"):
            static_df = perturbed.groupby("unique_id")[exog["stat_exog_list"]].first().reset_index()
        nf = NeuralForecast(models=[model_class(**kwargs)], freq=freq)
        nf.fit(df=perturbed.iloc[:train_end], val_size=horizon, static_df=static_df)
        if exog.get("futr_exog_list"):
            futr_df = perturbed.iloc[train_end:][["unique_id", "ds", *exog["futr_exog_list"]]]
            forecast = nf.predict(futr_df=futr_df)
        else:
            forecast = nf.predict()
        return forecast.iloc[-horizon:][model_name].to_numpy(dtype=float)

    return predict


def make_predictor(model: str, frame: pd.DataFrame, roles: dict, horizon: int, freq: str,
                   epochs: int, gpu: int, params: dict | None, level: str):
    if model == "LightGBM":
        return lightgbm_predictor(frame, horizon, params), False
    if model == "TFT":
        return tft_predictor(frame, roles, horizon, freq, epochs, gpu, params, level), True
    if model == "Chronos2":
        return chronos2_predictor(frame, roles, horizon, gpu, level), True
    if model == "Timestral":
        return timestral_predictor(frame, roles, horizon, gpu, level), True
    return neuralforecast_refit_predictor(frame, roles, horizon, freq, epochs, gpu, params, model), False


# ----------------------------------------------------------------------- driver

def explain(model: str, frame: pd.DataFrame, roles: dict, horizon: int, per_day: int,
            mode: str, predict, include_temporal: bool, out_dir: Path,
            sampling: float, level: str) -> dict:
    players = build_players(frame, roles, horizon, per_day, include_temporal=include_temporal)
    if len(players) > 12:
        raise ValueError(f"{model}: {len(players)} players -> {2 ** len(players)} coalitions; "
                         "group the covariates further or lower --max-players")
    n_context = len(frame) - horizon
    background = periodic_background(frame, per_day, train_end=n_context)

    started = time.time()
    values = enumerate_game(
        lambda active: predict(apply_coalition(frame, background, players, active, mode=mode)),
        players)
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
        "additivity_residual": residual, "seconds": round(elapsed, 3),
        "temporal_players": include_temporal,
    }, indent=2), encoding="utf-8")

    logging.getLogger("timeshap").info("%s: %d players, %d coalitions in %.1fs, residual %.2e",
                                       model, len(players), 2 ** len(players), elapsed, residual)
    return {"shares": grouped, "players": shares, "residual": residual,
            "seconds": elapsed, "n_players": len(players)}


def spearman_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Rank agreement between models over the reporting blocks."""
    ranked = rows.rank(axis=1)
    return ranked.T.corr(method="spearman")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TimeSHAP explanation of the covariate models.")
    parser.add_argument("--dataset", required=True, choices=["belgium", "germany", "london", "zonnedael"])
    parser.add_argument("--series", required=True)
    parser.add_argument("--models", nargs="+",
                        choices=list(("LightGBM", "DeepNPTS", "PatchTST", "TCN", "TFT",
                                      "Chronos2", "Timestral")),
                        default=list(("LightGBM", "DeepNPTS", "PatchTST", "TCN", "TFT",
                                      "Chronos2", "Timestral")))
    parser.add_argument("--level", default="COV", choices=list(("L0", "COV")))
    parser.add_argument("--sampling", type=float, default=100.0)
    parser.add_argument("--mode", choices=["background", "mean", "zero"], default="background",
                        help="Perturbation: hour-of-day background (main), column mean, or zero.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--params-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "tuning")
    parser.add_argument("--allow-refit", action="store_true",
                        help="Explain DeepNPTS/PatchTST/TCN by refitting per coalition "
                             "(covariate players only; expensive but clean).")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the player set and exit.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    horizon = horizon_for(args.sampling)
    per_day = steps_per_day(args.sampling, args.dataset)
    frame = load_long_frame(args.dataset, args.series, args.level, args.sampling)
    scaled, roles = scaled_frame(frame, args.level, horizon)
    freq = f"{int(15 * (100 / args.sampling))}min"

    players = build_players(scaled, roles, horizon, per_day, include_temporal=True)
    logging.getLogger("timeshap").info(
        "%s/%s: %d points, horizon %d, %d players -> %d coalitions",
        args.dataset, args.series, len(scaled), horizon, len(players), 2 ** len(players))
    if args.dry_run:
        for player in players:
            print(f"  {player.name:14s} {player.kind:10s} rows={player.size:5d} columns={len(player.columns)}")
        return 0

    out_dir = args.out or (Path(__file__).resolve().parents[1] / "results" / "xai_timeshap" / args.dataset /
                           f"{args.series}_{args.level}_{sampling_tag(args.sampling)}")
    shares_rows, notes = {}, {}
    for model in args.models:
        params_path = args.params_dir / f"{model}_{args.dataset}_{args.level}_{sampling_tag(args.sampling)}.json"
        params = {}
        if params_path.exists():
            payload = json.loads(params_path.read_text(encoding="utf-8"))
            params = dict(payload.get("best_params", payload))
        include_temporal = model != "LightGBM"   # a direct regressor has no lag
        if model not in ("LightGBM", "TFT", "Chronos2", "Timestral") and not args.allow_refit:
            notes[model] = ("skipped: NeuralForecast freezes the context and hist_exog into the "
                            "fitted dataset, so a clean coalition game needs --allow-refit")
            logging.getLogger("timeshap").warning(
                "%s skipped (no clean game without refit); pass --allow-refit", model)
            continue
        if model not in ("LightGBM", "TFT", "Chronos2", "Timestral"):
            include_temporal = False
            notes[model] = "refit per coalition; covariate players only"
        try:
            predict, _ = make_predictor(model, scaled, roles, horizon, freq, args.epochs,
                                        args.gpu, params, args.level)
            result = explain(model, scaled, roles, horizon, per_day, args.mode, predict,
                             include_temporal, out_dir / model, args.sampling, args.level)
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
    matrix = spearman_matrix(table)
    matrix.to_csv(out_dir / "spearman_matrix.csv", float_format="%.6f")
    (out_dir / "notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(table.round(4).to_string())
    print(f"written -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
