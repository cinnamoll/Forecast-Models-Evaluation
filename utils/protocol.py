"""Evaluation protocol: windows, horizon, sampling grid, result paths.

Every number that ends up in the comparison table comes from this module, so a
new run can always be placed next to the paper's table:

* the evaluation window is the one in ``data/README.md``;
* the horizon is 192 points at 100 % (96 / 64 / 48 at 50 / 33 / 25 %), i.e. the
  horizon shrinks with the sampling rate so it still covers two days **at 15
  minutes**.  For London the native step is 30 minutes, so 192 points cover four
  days; the loaders keep the paper's handling (see ``freq_for``) and this is
  recorded as a limitation rather than silently "fixed";
* the frequency label handed to a model is derived from the dataset's **native**
  step (``freq_for``), because the published labels were always the 15-minute
  grid and on London that made every generated clock run at half speed;
* the result tree keeps the paper's spelling, ``<Model>/Sampling_<tag>/Run_<n>``
  (statistical / TSFM) and ``<Model>/Sampling_<tag>/Epochs_<n>_<run>`` (the two
  framework scripts), so the new columns join the old ones.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Evaluation window per dataset, exactly as ``data/README.md`` describes it.
#: Belgium and WPuQ ship a three-month file (Jan-Mar 2024); LCL covers all of
#: 2013 and Zonnedael January-June 2013.  ``end`` is exclusive.
DATASET_WINDOWS = {
    "belgium": ("2024-01-01 00:00:00", "2024-04-01 00:00:00"),
    "germany": ("2024-01-01 00:00:00", "2024-04-01 00:00:00"),
    "london": ("2013-01-01 00:00:00", "2014-01-01 00:00:00"),
    "zonnedael": ("2013-01-01 00:00:00", "2013-07-01 00:00:00"),
}

#: Native step of the series, in minutes.  London LCL is half-hourly; everything
#: else is quarter-hourly.
DATASET_STEP_MINUTES = {
    "belgium": 15,
    "germany": 15,
    "london": 30,
    "zonnedael": 15,
}


def window(dataset: str):
    """``(start, end)`` timestamps of a dataset's evaluation window."""
    start, end = DATASET_WINDOWS[dataset]
    return pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")


def step_minutes(sampling_rate: float) -> int:
    """Subsampling factor of the paper: 100 % keeps every point, 25 % every fourth."""
    return max(1, int(round(100.0 / float(sampling_rate))))


def horizon_for(sampling_rate: float) -> int:
    """Horizon in points: 192 at 100 %, 48 at 25 %."""
    return 192 // step_minutes(sampling_rate)


def freq_for(sampling_rate: float, dataset: str | None = None, style: str = "min") -> str:
    """Frequency label handed to the model, on the dataset's **native** step.

    ``freq_for(100, "london")`` is ``30min``: LCL is half-hourly, and labelling
    it ``15min`` -- which is what every published run did, since the label was
    the paper's 15-minute grid for every dataset -- makes each generated clock
    run at half speed.  GluonTS builds its time features from
    ``start + i * freq``, so on London that drifts by ~12 hours across 2013, and
    any model that resamples onto the label (``fmodel_timegpt.py``) doubles the
    series.  Datasets that are already on 15 minutes are unaffected.

    ``dataset=None`` keeps the old 15-minute grid, which is what the model
    scripts that never read the label still pass.
    """
    base = DATASET_STEP_MINUTES[dataset] if dataset else 15
    minutes = int(base * step_minutes(sampling_rate))
    return f"{minutes}T" if style == "T" else f"{minutes}min"


def native_step_minutes(dataset: str) -> int:
    return DATASET_STEP_MINUTES[dataset]


def steps_per_day(sampling_rate: float, dataset: str | None = None) -> int:
    """Points in one wall-clock day after subsampling."""
    # 15 minutes is the paper's grid, used for every label.
    base = DATASET_STEP_MINUTES[dataset] if dataset else 15
    return int(round(1440.0 / (base * step_minutes(sampling_rate))))


def sampling_tag(sampling_rate: float) -> str:
    """Directory tag the paper's tree already uses (25, 33, 50, 100)."""
    return f"{float(sampling_rate):.0f}"


def runs_for(model: str) -> int:
    """Runs per cell: one for a deterministic model, ten when seeds are averaged."""
    return 1 if model in ("LightGBM", "Chronos2", "Timestral") else 10


def run_dir(root, dataset: str, model: str, sampling_rate: float, run: int = 1,
            epochs: int | None = None, style: str = "run", exog_level: str = "L0"):
    """Result directory of one run.

    ``style="run"`` is the statistical/zero-shot layout (``Run_<n>``);
    ``style="epochs"`` is the NeuralForecast/GluonTS layout
    (``Epochs_<epochs>_<n>``).  The ``COV`` level is inserted as an extra
    component so an improved run can never overwrite the paper's tree; ``L0``
    keeps the original layout byte for byte.
    """
    base = Path(root) / f"results_{dataset}" / model / f"Sampling_{sampling_tag(sampling_rate)}"
    if exog_level != "L0":
        if exog_level not in ("L0", "COV"):
            raise ValueError(f"unknown exog level {exog_level!r}; known: {('L0', 'COV')}")
        base = base / exog_level
    if style == "epochs":
        return base / f"Epochs_{epochs}_{run}"
    return base / f"Run_{run}"


def validation_bounds(test_start, days: int = 14):
    """The ``days``-day block that ends exactly where the test window starts."""
    test_start = pd.Timestamp(test_start)
    return test_start - pd.Timedelta(days=days), test_start


def validation_windows(n_points: int, horizon: int, steps_per_day_: int,
                       days: int = 14) -> list[tuple[int, int]]:
    """Consecutive ``horizon``-long windows inside the tuning block.

    Returns ``[(train_end, eval_end)]``; rows ``< train_end`` are the only ones a
    trial may train on, so the last window ends exactly where the test begins.

    The windows are anchored **backwards** from that edge.  Stepping forwards
    from the start of the block (the previous behaviour) left the remainder on
    the floor next to the test window whenever ``days * steps_per_day`` was not a
    multiple of ``horizon`` -- on LCL the 14-day block is 672 points and three
    192-point windows consume 576, so 96 points (two days) went unevaluated and
    the window closest to the test block was never used.  Anchoring backwards
    keeps every divisible case byte-identical and drops the remainder at the far
    end instead.
    """
    if n_points <= horizon:
        raise ValueError(f"series of {n_points} points cannot hold horizon {horizon}")
    block = int(days) * steps_per_day_
    val_end = n_points - horizon
    val_start = max(horizon, val_end - block)
    windows, end = [], val_end
    while end - horizon >= val_start:
        windows.append((end - horizon, end))
        end -= horizon
    if not windows:
        raise ValueError("validation block is shorter than one horizon")
    return list(reversed(windows))
