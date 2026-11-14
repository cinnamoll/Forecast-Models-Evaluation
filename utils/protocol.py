"""The paper's evaluation protocol, in one place.

Every script that produces a number in the comparison table must read the
window, the horizon and the sampling grid from here.  plan_updated.md section 0
fixes them: a new column is only comparable to the original table if the test
window, the horizon after sampling and the number of seeds are unchanged.

Nothing in this module tunes anything: `TUNING_VALIDATION_DAYS` only describes
the block of data reserved for hyper-parameter search, and that block always sits
strictly before the test window.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAPER_START = "2024-01-01 00:00:00"
PAPER_END = "2024-04-01 00:00:00"
PAPER_TIMEZONE = "UTC"

# 192 points at 100 %, 96 at 50 %, 64 at 33 %, 48 at 25 % (plan_updated.md sec. 0).
BASE_HORIZON_POINTS = 192
BASE_STEP_MINUTES = 15

SAMPLING_RATES = (25.0, 100.0 / 3.0, 50.0, 100.0)
DEFAULT_RUNS = 10

#: The two run modes of the benchmark.  There is no ladder of exogenous levels:
#: every model runs the paper baseline (``L0``, kept for parity with the original
#: table) and once with **every** covariate its dataset has (``ALL``).  What the
#: ladder used to measure -- which block earns its place -- is measured instead by
#: the TimeSHAP elimination of ``scripts/xai_elimination.py``: it drops one group
#: at a time on the validation block and records what each drop costs.  That is a
#: stronger answer than three fixed rungs, because the groups are dropped in the
#: order the model actually uses them rather than in the order they were written.
PAPER_BASELINE_LEVEL = "L0"
COVARIATE_LEVEL = "ALL"
RUN_LEVELS = (PAPER_BASELINE_LEVEL, COVARIATE_LEVEL)
EXOG_LEVEL_CHOICES = RUN_LEVELS

#: Blocks that used to be rungs of the ladder.  L1-L4 are not addressable levels
#: any more, but their builders still feed ``ALL``; L5 and L6 were removed from the
#: pipeline.  The reasons are kept here so the paper can state why the design has
#: two modes instead of six rungs, and so that results still sitting under
#: ``results/.../L1``-``L6`` stay interpretable.
BLOCKS_ONLY_IN_ALL = {
    "L1": "detailed calendar (holidays, bridge days, DST, cyclic encodings)",
    "L2": "physical weather (temperature, HDD/CDD, irradiance, rolling means)",
    "L3": "day-ahead price, where the dataset has a series covering its window",
    "L4": "static metadata, where the dataset ships a table",
}
REMOVED_BLOCKS = {
    "L5": "neighbour aggregate: only two datasets could build it, and both peer "
          "matrices are missing under the name the code asks for",
    "L6": "measured plant state: Belgium only, and its state file was never built",
}

#: Sampling rates of the all-covariates run: the elimination and its XAI are the
#: expensive part, so it runs at 100 % and 25 % only.  The baseline keeps all four
#: rates of the original table.
ALL_COVARIATES_SAMPLING_RATES = (25.0, 100.0)

TUNING_VALIDATION_DAYS = 14
TUNING_MAX_TRIALS = 30
TUNING_MAX_HOURS = 3.0

# plan_updated.md section 8: the block bootstrap resamples whole days, declared
# in days so the block follows the sampling rate instead of assuming 15-minute
# steps.  (The DM test itself is kept as published, by request.)
BOOTSTRAP_BLOCK_DAYS = 1.0

# Models of the comparison table (plan_updated.md section 1).
TRAINABLE_MODELS = ("LightGBM", "DeepNPTS", "PatchTST", "TFT")
ZERO_SHOT_MODELS = ("Chronos2", "Timestral")
TABLE_MODELS = TRAINABLE_MODELS + ZERO_SHOT_MODELS


def paper_bounds():
    """Return the (start, end) timestamps of the paper's evaluation window."""
    return (
        pd.Timestamp(PAPER_START, tz=PAPER_TIMEZONE),
        pd.Timestamp(PAPER_END, tz=PAPER_TIMEZONE),
    )


def step_minutes(sampling_rate: float) -> int:
    """Sampling factor of the paper: 100 % keeps 15 min, 25 % keeps 1 h."""
    return max(1, round(100.0 / float(sampling_rate)))


def horizon_for(sampling_rate: float) -> int:
    """Horizon in points after subsampling: 192 / 96 / 64 / 48."""
    return BASE_HORIZON_POINTS // step_minutes(sampling_rate)


def freq_for(sampling_rate: float, style: str = "min", base_minutes: int = BASE_STEP_MINUTES) -> str:
    """Pandas frequency string for a sampling rate (``min`` or ``T`` style)."""
    minutes = int(base_minutes * step_minutes(sampling_rate))
    return f"{minutes}{style}" if style == "T" else f"{minutes}min"


def sampling_tag(sampling_rate: float) -> str:
    """Directory tag the original results already use (25, 33, 50, 100)."""
    return f"{float(sampling_rate):.0f}"


def steps_per_day(sampling_rate: float, base_minutes: int = BASE_STEP_MINUTES) -> int:
    """Points in one wall-clock day after subsampling (96 at 100 %, 24 at 25 %)."""
    return int(round(1440.0 / (base_minutes * step_minutes(sampling_rate))))


def run_dir(root, dataset: str, model: str, sampling_rate: float, run: int = 1,
            epochs: int | None = None, style: str = "run", exog_level: str = "L0"):
    """Build the results directory of one run.

    ``style="run"`` matches the Chronos/stat-ML layout (``Run_{n}``);
    ``style="epochs"`` matches the NeuralForecast/GluonTS layout
    (``Epochs_{epochs}_{n}``).  Keeping both identical to the original tree is
    what lets the new columns be joined to the old ones.

    The exogenous level is inserted only for ``L1``-``L6`` and for ``ALL``: the
    ``L0`` tree stays byte-identical to the paper's, and an ablation run can never
    overwrite it.
    """
    base = Path(root) / f"results_{dataset}" / model / f"Sampling_{sampling_tag(sampling_rate)}"
    if exog_level != "L0":
        if exog_level not in EXOG_LEVEL_CHOICES:
            raise ValueError(f"Unknown exog level {exog_level!r}; known: {EXOG_LEVEL_CHOICES}")
        base = base / exog_level
    if style == "epochs":
        return base / f"Epochs_{epochs}_{run}"
    return base / f"Run_{run}"


def validation_bounds(start, test_start, days: int = TUNING_VALIDATION_DAYS):
    """Validation block of ``days`` days that ends exactly where the test starts."""
    test_start = pd.Timestamp(test_start)
    return test_start - pd.Timedelta(days=days), test_start


def validation_windows(n_points: int, horizon: int, sampling_rate: float,
                       days: int = TUNING_VALIDATION_DAYS) -> list[tuple[int, int]]:
    """Consecutive evaluation windows inside the tuning block.

    The block is the last ``days`` days before the test window, and it is cut
    into windows of exactly ``horizon`` points: the metric reported for a trial
    is then an MAE over the same forecast length as the test metric, not over a
    single two-day slice that happens to be calm.

    Returns ``[(train_end, eval_end)]`` pairs, where ``train_end`` is the index
    of the first validation point and ``eval_end = train_end + horizon``.  Points
    at indices ``< train_end`` are the only ones a trial may train on.
    """
    if n_points <= horizon:
        raise ValueError(f"series of {n_points} points cannot hold horizon {horizon}")
    per_day = steps_per_day(sampling_rate)
    block = int(days) * per_day
    val_end = n_points - horizon
    val_start = max(horizon, val_end - block)
    windows = []
    start = val_start
    while start + horizon <= val_end:
        windows.append((start, start + horizon))
        start += horizon
    if not windows:
        raise ValueError("validation block is shorter than one horizon")
    return windows


def run_mode_scale(level: str) -> int:
    """Ordinal of a run mode, for ordering tables that list both."""
    return RUN_LEVELS.index(level)
