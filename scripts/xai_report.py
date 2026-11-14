#!/usr/bin/env python
"""XAI for the six table models (plan_updated.md section 9).

One row per model, one column per variable group, summing to one -- that is the
object this script produces, plus the 6x6 Spearman matrix and the cross-check
against the L0-L5 ablation.  ``group_shares.csv`` is that object;
``feature_shares.csv`` is the same measurement split down to single columns
(temperature, humidity, ghi, dni, cloud_cover, wind_speed, precipitation, ...),
so the weather share can be read feature by feature.  LightGBM and DeepNPTS are
per feature by construction; the occlusion adapters need
``--per-column-occlusion`` because each column costs another model pass.

Method per model (section 9):

* LightGBM        TreeSHAP via ``LGBMRegressor.predict(pred_contrib=True)``.
* DeepNPTS        block permutation of one group at a time, refitting between
                  permutations (the NF family cannot permute its input window
                  without refitting, so the refit is part of the measurement).
* PatchTST        occlusion of the last context day per group, refitting.
* TFT             occlusion as above, cross-checked against the model's own
                  variable-selection weights when the attribute exists.
* Chronos-2       occlusion over context day windows and per covariate group,
                  no refit (it is zero-shot).
* Timestral-123M  same as Chronos-2, using its own covariate tensors.

Everything a group cannot be measured for is logged as n/a and left out of the
vector; nothing is imputed, because an invented share would travel into the
Spearman matrix and the paper's claim about agreement.

Run from the environment that has the model installed, e.g.
``conda run -n chronos python scripts/xai_report.py --dataset london --series london_load --models Chronos2 Timestral``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from utils.benchmark_series import iter_series
from utils.protocol import (
    COVARIATE_LEVEL,
    EXOG_LEVEL_CHOICES,
    TABLE_MODELS,
    horizon_for,
    sampling_tag,
    steps_per_day,
)
from utils.tuning import load_best_params
from utils.xai import (
    column_groups,
    column_shares_frame,
    compare_with_ablation,
    day_windows,
    importance_by_group,
    occlusion_importance,
    permutation_importance,
    plot_group_shares,
    plot_spearman_heatmap,
    shares_frame,
    spearman_matrix,
    tree_shap,
)

LOGGER = logging.getLogger("xai")

#: Models whose attribution is already per feature: TreeSHAP (LightGBM) and
#: block permutation (DeepNPTS) perturb/score one column at a time.  The
#: occlusion adapters score a whole group per call, so splitting them costs one
#: pass per column and is only done when ``--per-column-occlusion`` is given.
PER_COLUMN_MODELS = ("LightGBM", "DeepNPTS")

#: series prefix -> 1D loader method (the direct multi-step matrix of LightGBM)
DIRECT_LOADERS = {
    "pv_house": "get_pv_data",
    "bess_house": "get_battery_data",
    "germany_load": "get_load_data",
    "london_load": "get_load_data",
    "zonnedael_customer": "get_inputs_for_zonnedael_consumption",
}


def series_prefix(series: str) -> str:
    """Loader key of a series id.

    A series carries its entity as a numeric suffix (``pv_house_1`` ->
    ``pv_house``) only when it has one at all: ``london_load`` and
    ``germany_load`` are single-entity datasets and their own name is the key, so
    stripping the last underscore segment turned them into ``london`` and
    ``germany`` and the lookup below raised ``KeyError``.
    """
    head, _, tail = series.rpartition("_")
    return head if head and tail.isdigit() else series


def load_direct_frame(dataset: str, series: str, level: str, sampling: float):
    """(X, y) exactly as ``models/models_statsml.py`` builds them."""
    from utils.dataset_config import DatasetBelgium1D, DatasetGermany1D, DatasetLondon1D, DatasetZonnedael1D

    classes = dict(belgium=DatasetBelgium1D, germany=DatasetGermany1D,
                   london=DatasetLondon1D, zonnedael=DatasetZonnedael1D)
    loader = classes[dataset]()
    method = DIRECT_LOADERS[series_prefix(series)]
    kwargs: Dict[str, object] = {"exog_level": level}
    if series_prefix(series) in ("pv_house", "bess_house"):
        kwargs["house"] = int(series.rsplit("_", 1)[1])
        kwargs.update(start_dt=pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                      end_dt=pd.Timestamp("2024-04-01 00:00:00", tz="UTC"))
    elif series_prefix(series) == "zonnedael_customer":
        kwargs["customer_number"] = int(series.rsplit("_", 1)[1])
    elif dataset == "germany":
        kwargs.update(start_dt=pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
                      end_dt=pd.Timestamp("2024-04-01 00:00:00", tz="UTC"))
    X, y = getattr(loader, method)(**kwargs)
    step = int(round(100.0 / float(sampling)))
    return X.iloc[::step].reset_index(drop=True), y.iloc[::step].reset_index(drop=True)


def load_long_frame(dataset: str, series: str, level: str, sampling: float) -> pd.DataFrame:
    """The long (unique_id, ds, y, covariates) frame every other model consumes."""
    from utils.dataset_config import DatasetBelgiumNF, DatasetGermanyNF, DatasetLondonNF, DatasetZonnedaelNF

    classes = dict(belgium=DatasetBelgiumNF, germany=DatasetGermanyNF,
                   london=DatasetLondonNF, zonnedael=DatasetZonnedaelNF)
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
    for series_id, frame in iter_series(dataset, classes[dataset](), start, end, exog_level=level):
        if series_id == series:
            step = int(round(100.0 / float(sampling)))
            return frame.iloc[::step].reset_index(drop=True)
    raise ValueError(f"series {series!r} not found in {dataset}")


def compare_with_elimination(shares: pd.DataFrame, args):
    """Attribution shares against what dropping each group actually cost.

    The ladder is gone, so the ablation is the elimination path: per model, the
    ledger of ``scripts/xai_elimination.py`` says which group went and what MAE
    that cost.  A group with a large share that was free to remove is the
    contradiction section 9 requires an explanation for.
    """
    from utils.elimination import load_effects

    frames = []
    for model in shares.index:
        ledger = (PROJECT_ROOT / "results" / "xai_elimination" / args.dataset /
                  f"{args.series}_{model}_{args.level}_{sampling_tag(args.sampling)}" /
                  "elimination_rounds.csv")
        effects = load_effects(ledger)
        if effects.empty:
            continue
        effects.insert(0, "model", model)
        frames.append(effects)
    if not frames:
        return None
    ablation = pd.concat(frames, ignore_index=True)
    return compare_with_ablation(shares[shares.index.isin(ablation["model"])], ablation)


def feature_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c not in ("unique_id", "ds", "y")]


def attribution_groups(columns: List[str], per_column: bool = False) -> Dict[str, List[str]]:
    """Grouping handed to the occlusion adapter.

    By default a group of columns is blanked per call, which is what the group
    share vector of section 9 needs.  ``per_column`` maps every column to itself
    so the measured vector is per feature (temperature, humidity, ghi, ...);
    that is one extra pass per column, hence opt-in.
    """
    if per_column:
        return {column: [column] for column in columns}
    grouped: Dict[str, List[str]] = {}
    for column, group in column_groups(columns).items():
        grouped.setdefault(group, []).append(column)
    return grouped


# ------------------------------------------------------------------ LightGBM

def lightgbm_attributions(dataset, series, level, sampling, horizon, params_path=None,
                          gpu: int = -1) -> Dict[str, float]:
    import lightgbm as lgb

    X, y = load_direct_frame(dataset, series, level, sampling)
    X_train, X_test = X.iloc[:-horizon], X.iloc[-horizon:]
    y_train = y.iloc[:-horizon]
    tuned = load_best_params(params_path) if params_path else {}
    model = lgb.LGBMRegressor(
        random_state=0, n_jobs=-1, verbosity=-1, deterministic=False,
        bagging_fraction=0.5, feature_fraction=0.5, bagging_freq=1,
        device_type="gpu" if gpu in (0, 1) else "cpu",
        **{k: v for k, v in tuned.items() if k not in {"random_state", "n_jobs", "verbosity"}},
    ).fit(X_train, y_train)
    shap_values = tree_shap(model, X_test).drop(columns="bias")
    # Mean *absolute* contribution per feature over the horizon.  The question a
    # share vector answers is how much a group moves the forecast; a signed mean
    # cancels exactly the drivers that push it both ways inside a day (sun
    # position, price), which made a diurnal feature look unimportant on PV.  The
    # permutation and occlusion adapters below already measure magnitude (MAE
    # increase), so this keeps all six models on the same definition.
    return shap_values.abs().mean(axis=0).to_dict()


# ------------------------------------------------------------------ NF models

def _nf_predict_closure(model_class, params, freq, horizon, epochs, gpu, exog_kwargs=None):
    """``predict(frame) -> ndarray`` that refits on frame[:-horizon] each call."""
    import inspect

    from neuralforecast import NeuralForecast
    from utils.device import lightning_device_config

    accepted = set(inspect.signature(model_class.__init__).parameters)
    exog_kwargs = dict(exog_kwargs or {})

    def predict(frame: pd.DataFrame) -> np.ndarray:
        train = frame.iloc[:-horizon]
        truth_length = horizon
        kwargs = dict(h=horizon, input_size=horizon * 2, max_steps=int(epochs),
                      random_seed=0, **lightning_device_config(gpu))
        kwargs.update({k: v for k, v in (params or {}).items() if k in accepted})
        # Covariates must be declared on the model or NeuralForecast ignores the
        # columns entirely, and a permutation of an ignored column always looks
        # like zero importance.
        kwargs.update(exog_kwargs)
        forecast = NeuralForecast(models=[model_class(**kwargs)], freq=freq)
        forecast.fit(df=train, val_size=horizon)
        predicted = forecast.predict()
        column = model_class.__name__ if model_class.__name__ in predicted.columns else [
            c for c in predicted.columns if c not in ("unique_id", "ds")
        ][0]
        return predicted.iloc[-truth_length:][column].to_numpy(dtype=float)

    return predict


def neural_attributions(dataset, series, level, sampling, horizon, model_name,
                        params_path=None, gpu: int = -1, epochs: int = 100,
                        n_repeats: int = 3, method: str = "occlusion",
                        per_column: bool = False) -> Dict[str, float]:
    import neuralforecast.models as nf_models

    from utils.exogenous.feature_store import partition_for_model

    model_class = getattr(nf_models, model_name)
    frame = load_long_frame(dataset, series, level, sampling)
    columns = feature_columns(frame)
    if not columns:
        LOGGER.warning("%s/%s at %s has no covariates; nothing to attribute",
                       dataset, series, level)
        return {}

    exog_kwargs, exog_report = partition_for_model(model_class, columns)
    if not exog_kwargs:
        # The architecture has no covariate channel (PatchTST, NBEATS, NLinear,
        # iTransformer).  Reporting an even split would look like agreement, so the
        # model is left out of the matrix and the reason is logged.
        LOGGER.warning("%s supports no exogenous inputs (%s); no attribution to measure "
                       "for %s/%s at %s", model_name, exog_report.get("capabilities"),
                       dataset, series, level)
        return {}
    if exog_report.get("n_dropped"):
        LOGGER.info("%s uses %s and drops %d column(s) (%s)", model_name,
                    {k: len(v) for k, v in exog_kwargs.items()}, exog_report["n_dropped"],
                    {k: len(v) for k, v in exog_report["dropped"].items()})

    truth = frame.iloc[-horizon:]["y"].to_numpy(dtype=float)
    predict = _nf_predict_closure(model_class, load_best_params(params_path) if params_path else {},
                                  f"{int(15 * (100 / sampling))}min", horizon, epochs, gpu,
                                  exog_kwargs=exog_kwargs)
    if method == "permutation":
        # Permutation perturbs one column at a time, so this vector is per
        # feature whatever ``per_column`` says.
        return permutation_importance(predict, frame, truth, columns,
                                      block=steps_per_day(sampling),
                                      n_repeats=n_repeats)
    context = frame.iloc[:-horizon]
    windows = day_windows(context.index, steps_per_day(sampling), last_n_days=1)
    return occlusion_importance(predict, frame, truth, windows,
                                attribution_groups(columns, per_column))


# NeuralForecast 3.0.2's TFT keeps attention weights over time steps but does not
# store the variable-selection softmax it computes inside the VSN block, so the
# "TFT has ready-made variable-selection weights" assumption of plan_updated.md
# section 9 cannot be met by this build.  TFT is therefore attributed by
# occlusion only, and that limitation is recorded in the paper.
TFT_SELECTION_WEIGHTS_AVAILABLE = False


# ------------------------------------------------------------------ TSFM

def chronos2_predict_closure(pipeline, horizon, context_length=None):
    from models.fmodel_chronos2 import build_inputs

    def predict(frame: pd.DataFrame) -> np.ndarray:
        payload, _, _ = build_inputs(frame, horizon, context_length=context_length)
        quantiles, _ = pipeline.predict_quantiles(
            inputs=[payload], prediction_length=horizon, quantile_levels=[0.5],
        )
        return np.asarray(quantiles[0])[0, :horizon, 0]

    return predict


def timestral_predict_closure(forecaster, horizon, context_length=None):
    from models.fmodel_timestral import build_covariates

    def predict(frame: pd.DataFrame) -> np.ndarray:
        payload = build_covariates(frame, horizon, context_length=context_length)
        forecast = forecaster.forecast(
            payload["context"], prediction_length=horizon,
            past_covariates=payload["past_covariates"],
            known_covariates=payload["known_covariates"],
            categorical_past_covariates=payload["categorical_past_covariates"],
            categorical_known_covariates=payload["categorical_known_covariates"],
        )
        return np.asarray(forecast.median)[0, 0, :horizon]

    return predict


def tsfm_attributions(frame, horizon, sampling, predict, last_n_days: int = 2,
                      per_column: bool = False) -> Dict[str, float]:
    """Occlusion of the last context days, one variable group (or column) at a time."""
    columns = feature_columns(frame)
    truth = frame.iloc[-horizon:]["y"].to_numpy(dtype=float)
    context = frame.iloc[:-horizon]
    windows = day_windows(context.index, steps_per_day(sampling), last_n_days=last_n_days)
    return occlusion_importance(predict, frame, truth, windows,
                                attribution_groups(columns, per_column))


# ------------------------------------------------------------------ driver

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Attribution report for the six table models.")
    parser.add_argument("--dataset", required=True, choices=["belgium", "germany", "london", "zonnedael"])
    parser.add_argument("--series", required=True)
    parser.add_argument("--level", default=None, choices=list(EXOG_LEVEL_CHOICES),
                        help="Level the attribution is computed at (default L0; use L2/L4/L5 "
                             "to attribute weather, metadata or neighbours, ALL to attribute "
                             "every covariate at once).")
    parser.add_argument("--use_covariates", "--use-covariates", action="store_true",
                        help="Attribute the all-covariates block (same as --level ALL; "
                             "it is the default, since L0 has nothing to attribute).")
    parser.add_argument("--sampling", type=float, default=100.0)
    parser.add_argument("--models", nargs="+", default=list(TABLE_MODELS), choices=list(TABLE_MODELS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--per-column-occlusion", action="store_true",
                        help="Split the occlusion adapters (PatchTST, TFT, Chronos2, "
                             "Timestral) down to one column per measurement instead of one "
                             "group; costs one extra model pass per column.")
    parser.add_argument("--params-dir", type=Path, default=PROJECT_ROOT / "results" / "tuning")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    # Same rule as the model scripts: the shorthand and an explicit level must
    # agree, and an omitted level means L0 (the paper baseline).
    if args.level is None:
        # attribution only means something with covariates present: the baseline
        # (L0) has none, so the default is the all-covariates run
        args.level = COVARIATE_LEVEL
    elif args.use_covariates and args.level != COVARIATE_LEVEL:
        parser.error(f"--use_covariates and --level {args.level} disagree; drop one")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    horizon = horizon_for(args.sampling)
    out_dir = args.out or (PROJECT_ROOT / "results" / "xai" / args.dataset
                           / f"{args.series}_{args.level}_{sampling_tag(args.sampling)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model: Dict[str, Dict[str, float]] = {}
    feature_level: Dict[str, Dict[str, float]] = {}
    detail: Dict[str, object] = {}

    for model in args.models:
        params_path = args.params_dir / f"{model}_{args.dataset}_{args.level}_{sampling_tag(args.sampling)}.json"
        # TreeSHAP and block permutation already score one column at a time; the
        # occlusion adapters do so only when asked, so a model's granularity has
        # to travel with its result instead of being re-inferred later.
        per_column = model in PER_COLUMN_MODELS or args.per_column_occlusion
        try:
            if model == "LightGBM":
                attributions = lightgbm_attributions(args.dataset, args.series, args.level,
                                                     args.sampling, horizon, params_path, args.gpu)
            elif model in ("DeepNPTS", "PatchTST", "TFT"):
                method = "permutation" if model == "DeepNPTS" else "occlusion"
                attributions = neural_attributions(
                    args.dataset, args.series, args.level, args.sampling, horizon, model,
                    params_path, args.gpu, epochs=args.epochs,
                    n_repeats=args.n_repeats, method=method,
                    per_column=args.per_column_occlusion,
                )
                if model == "TFT" and not TFT_SELECTION_WEIGHTS_AVAILABLE:
                    LOGGER.warning(
                        "this NeuralForecast build exposes no TFT variable-selection "
                        "weights; occlusion is the only TFT attribution (limitation)"
                    )
            elif model == "Chronos2":
                from chronos import Chronos2Pipeline
                from utils.device import torch_device

                pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2",
                                                            device_map=torch_device(args.gpu))
                frame = load_long_frame(args.dataset, args.series, args.level, args.sampling)
                attributions = tsfm_attributions(
                    frame, horizon, args.sampling,
                    chronos2_predict_closure(pipeline, horizon),
                    per_column=args.per_column_occlusion,
                )
            else:
                from timestral import TimestralForecaster
                from utils.device import torch_device

                forecaster = TimestralForecaster.from_pretrained(
                    "mistralai/Timestral-123M", device=torch_device(args.gpu))
                frame = load_long_frame(args.dataset, args.series, args.level, args.sampling)
                attributions = tsfm_attributions(
                    frame, horizon, args.sampling,
                    timestral_predict_closure(forecaster, horizon),
                    per_column=args.per_column_occlusion,
                )
        except Exception as error:  # one model must not kill the whole report
            LOGGER.error("%s failed: %s", model, error, exc_info=True)
            continue
        if not attributions:
            LOGGER.warning("%s produced no attribution; left out of the matrix", model)
            continue
        per_model[model] = attributions
        if per_column:
            feature_level[model] = attributions
        detail[model] = {k: float(v) for k, v in attributions.items()}

    if not per_model:
        print("no model produced an attribution - nothing written")
        return 1

    shares = shares_frame(per_model)
    shares.to_csv(out_dir / "group_shares.csv", float_format="%.6f")
    pd.DataFrame(detail).T.to_csv(out_dir / "raw_importances.csv", float_format="%.6f")
    plot_group_shares(shares, out_dir / "group_shares.pdf")

    # Per-feature shares sit next to the group vector: same measurement, split
    # down to the single column (temperature vs. ghi vs. cloud cover) instead of
    # the weather block as one number.
    if feature_level:
        features = column_shares_frame(feature_level)
        features.to_csv(out_dir / "feature_shares.csv", float_format="%.6f")
        group_only = [model for model in per_model if model not in feature_level]
        if group_only:
            LOGGER.info("feature_shares.csv covers %s; %s are group-level only "
                        "(pass --per-column-occlusion to split them)",
                        list(feature_level), group_only)
        print(features.round(4).to_string())

    matrix = spearman_matrix(shares)
    matrix.to_csv(out_dir / "spearman_matrix.csv", float_format="%.6f")
    plot_spearman_heatmap(matrix, out_dir / "spearman_matrix.pdf")

    cross = compare_with_elimination(shares, args)
    if cross is not None:
        cross.to_csv(out_dir / "ablation_cross_check.csv", index=False, float_format="%.6f")
        flagged = cross[cross["flag"] != ""]
        if len(flagged):
            LOGGER.warning("%d group(s) have a high share but no measurable cost when "
                           "removed; they need an explanation in the paper", len(flagged))
    else:
        LOGGER.warning("no elimination ledger for this dataset/series/level; run "
                       "scripts/xai_elimination.py first to enable the cross-check")

    (out_dir / "attributions.json").write_text(
        json.dumps({"dataset": args.dataset, "series": args.series, "level": args.level,
                    "sampling": args.sampling,
                    "granularity": {model: ("column" if model in feature_level else "group")
                                    for model in per_model},
                    "models": detail},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(shares.round(4).to_string())
    print(f"written -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
