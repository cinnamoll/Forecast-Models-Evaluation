"""Aggregate the ten seeds of a cell into median + IQR, and load losses.

plan_updated.md section 8 reports MAE/RMSE as median with IQR across the ten
seeds, on the min-max scale, and section 9 compares models per variable group.
Both need one place that knows how the results tree is laid out:

    results/results_<dataset>/<model>/Sampling_<tag>/[<level>/]Run_<n>/
    results/results_<dataset>/<model>/Sampling_<tag>/[<level>/]Epochs_<e>_<n>/

``L0`` has no level directory (it is the paper's original tree); ``L1``-``L5``
carry an extra ``Lx`` component so an ablation can never overwrite the baseline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.benchmark_series import SERIES_ORDER, display_name
from utils.protocol import PROJECT_ROOT, sampling_tag

_RUN_DIR = re.compile(r"^(?:Run_(?P<run>\d+)|Epochs_(?P<epochs>\d+)_(?P<run_e>\d+))$")

SUMMARY_COLUMNS = [
    "dataset", "model", "sampling", "level", "series",
    "mae_median", "mae_q1", "mae_q3", "mae_iqr", "mae_mean", "mae_std",
    "rmse_median", "rmse_q1", "rmse_q3", "rmse_iqr", "n_runs",
]


def _sampling_dir(root, dataset: str, model: str, sampling: float, level: str) -> Path:
    """``<root>/results/results_<dataset>/<model>/Sampling_<tag>[/Lx]``.

    ``root`` is the project root (the directory that holds ``results/``), so the
    same function works for the live tree and for a snapshot tree.
    """
    base = Path(root) / "results" / f"results_{dataset}" / model / f"Sampling_{sampling_tag(sampling)}"
    return base / level if level != "L0" else base


def iter_run_dirs(run_parent: Path) -> List[Path]:
    """Run directories under ``.../Sampling_x`` (or ``.../Sampling_x/L2``)."""
    if not run_parent.exists():
        return []
    found = []
    for child in sorted(run_parent.iterdir()):
        if child.is_dir() and _RUN_DIR.match(child.name):
            found.append(child)
    return found


def read_run_metrics(run_dir: Path) -> Optional[pd.DataFrame]:
    """``model_metrics_summary.csv`` of one run, or None when it is missing."""
    path = Path(run_dir) / "model_metrics_summary.csv"
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if not {"model", "MAE", "RMSE"}.issubset(frame.columns):
        return None
    frame = frame.rename(columns={"model": "series"})
    return frame[["series", "MAE", "RMSE"]]


def collect_cell_metrics(root, dataset: str, model: str, sampling: float,
                         level: str = "L0") -> pd.DataFrame:
    """Per-run metrics of one (dataset, model, sampling, level) cell."""
    run_parent = _sampling_dir(root, dataset, model, sampling, level)
    frames = []
    for run_dir in iter_run_dirs(run_parent):
        run_metrics = read_run_metrics(run_dir)
        if run_metrics is None:
            continue
        run_metrics = run_metrics.copy()
        run_metrics["run"] = run_dir.name
        frames.append(run_metrics)
    if not frames:
        return pd.DataFrame(columns=["series", "MAE", "RMSE", "run"])
    return pd.concat(frames, ignore_index=True)


def summarize_cell(metrics: pd.DataFrame) -> pd.DataFrame:
    """Median + IQR per series over the runs of one cell."""
    if metrics.empty:
        return pd.DataFrame(columns=["series", "mae_median", "mae_q1", "mae_q3",
                                     "mae_iqr", "mae_mean", "mae_std",
                                     "rmse_median", "rmse_q1", "rmse_q3",
                                     "rmse_iqr", "n_runs"])
    grouped = metrics.groupby("series")
    rows = []
    for series, block in grouped:
        mae = block["MAE"].to_numpy(dtype=float)
        rmse = block["RMSE"].to_numpy(dtype=float)
        mae_q1, mae_q3 = np.quantile(mae, [0.25, 0.75])
        rmse_q1, rmse_q3 = np.quantile(rmse, [0.25, 0.75])
        rows.append({
            "series": series,
            "mae_median": float(np.median(mae)),
            "mae_q1": float(mae_q1),
            "mae_q3": float(mae_q3),
            "mae_iqr": float(mae_q3 - mae_q1),
            "mae_mean": float(np.mean(mae)),
            "mae_std": float(np.std(mae, ddof=1)) if len(mae) > 1 else 0.0,
            "rmse_median": float(np.median(rmse)),
            "rmse_q1": float(rmse_q1),
            "rmse_q3": float(rmse_q3),
            "rmse_iqr": float(rmse_q3 - rmse_q1),
            "n_runs": int(block["run"].nunique()),
        })
    order = {series: i for i, series in enumerate(
        sum((SERIES_ORDER[name] for name in SERIES_ORDER), [])
    )}
    summary = pd.DataFrame(rows)
    summary["_order"] = summary["series"].map(order).fillna(len(order))
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def build_summary_table(
    root,
    datasets: Sequence[str] = ("belgium", "germany", "london", "zonnedael"),
    models: Sequence[str] = (),
    samplings: Sequence[float] = (25.0, 100.0 / 3.0, 50.0, 100.0),
    levels: Sequence[str] = ("L0",),
) -> pd.DataFrame:
    """Long table over every cell that actually has results on disk."""
    frames = []
    for dataset in datasets:
        dataset_dir = Path(root) / f"results_{dataset}"
        available_models = [m for m in models] if models else (
            [p.name for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
            if dataset_dir.exists() else []
        )
        for model in available_models:
            for sampling in samplings:
                for level in levels:
                    metrics = collect_cell_metrics(root, dataset, model, sampling, level)
                    if metrics.empty:
                        continue
                    summary = summarize_cell(metrics)
                    summary.insert(0, "level", level)
                    summary.insert(0, "sampling", sampling_tag(sampling))
                    summary.insert(0, "model", model)
                    summary.insert(0, "dataset", dataset)
                    frames.append(summary)
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    table = pd.concat(frames, ignore_index=True)
    return table[SUMMARY_COLUMNS]


def read_forecast_frame(run_dir: Path, series_id: str) -> Optional[pd.DataFrame]:
    """``Actual``/``Forecast`` of one series, indexed by datetime."""
    path = Path(run_dir) / f"{display_name(series_id)}_forecast_vs_actual.csv"
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    renamed = {column.lower().strip(): column for column in frame.columns}
    if "actual" not in renamed or "forecast" not in renamed:
        return None
    if "datetime" in renamed:
        index = pd.to_datetime(frame[renamed["datetime"]], errors="coerce")
        frame = frame.assign(datetime=index).set_index("datetime")
    frame = frame.rename(columns={renamed["actual"]: "actual", renamed["forecast"]: "forecast"})
    frame = frame[~frame.index.isna()].sort_index()
    return frame[["actual", "forecast"]].astype(float)


def mean_forecast(
    root, dataset: str, model: str, sampling: float, series_id: str,
    level: str = "L0",
) -> Optional[pd.DataFrame]:
    """Average forecast across the runs of a cell, on their common timestamps.

    Averaging the seeds before computing the loss is what makes the DM/bootstrap
    inputs stable: the ten runs share one test window, so the averaged forecast
    is the model's expected output rather than one lucky seed.
    """
    run_parent = _sampling_dir(root, dataset, model, sampling, level)
    frames = []
    for run_dir in iter_run_dirs(run_parent):
        frame = read_forecast_frame(run_dir, series_id)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return None
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    if len(common) == 0:
        return None
    actual = np.mean([frame["actual"].reindex(common).to_numpy() for frame in frames], axis=0)
    forecast = np.mean([frame["forecast"].reindex(common).to_numpy() for frame in frames], axis=0)
    return pd.DataFrame({"actual": actual, "forecast": forecast},
                        index=pd.DatetimeIndex(common).sort_values()).sort_index()


def loss_series(frame: pd.DataFrame, kind: str = "mae") -> pd.Series:
    """Per-step loss on the min-max scale: absolute error or squared error."""
    error = frame["actual"].to_numpy(dtype=float) - frame["forecast"].to_numpy(dtype=float)
    values = np.abs(error) if kind == "mae" else error ** 2
    return pd.Series(values, index=frame.index)


def load_losses(
    root, dataset: str, models: Iterable[str], sampling: float,
    series_id: str, level: str = "L0", kind: str = "mae",
) -> dict:
    """``{model: loss Series}`` for the models that have forecasts on disk."""
    losses = {}
    for model in models:
        frame = mean_forecast(root, dataset, model, sampling, series_id, level)
        losses[model] = None if frame is None else loss_series(frame, kind)
    return losses


def ablation_table(root, models: Iterable[str], dataset: str, sampling: float,
                   series_id: str,
                   levels: Sequence[str] = ("L0", "ALL")) -> pd.DataFrame:
    """MAE per level for the ablation, one row per level that has results."""
    rows = []
    for model in models:
        for level in levels:
            metrics = collect_cell_metrics(root, dataset, model, sampling, level)
            if metrics.empty:
                continue
            try:
                cell = metrics[metrics["series"] == series_id]
            except KeyError:
                continue
            if cell.empty:
                continue
            rows.append({
                "dataset": dataset, "model": model, "sampling": sampling_tag(sampling),
                "series": series_id, "level": level,
                "mae_median": float(cell["MAE"].median()),
                "n_runs": int(cell["run"].nunique()),
            })
    return pd.DataFrame(rows, columns=["dataset", "model", "sampling", "series",
                                       "level", "mae_median", "n_runs"])
