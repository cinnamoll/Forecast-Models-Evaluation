"""Which tuning run supplies the parameters of a reported cell.

The search is cheap enough to run on three series only, and every other cell of
the table borrows those parameters.  Keeping the mapping in one place means the
tuner (which writes ``transfer_to`` into its JSON) and the runner (which has to
find the file again) cannot disagree, and a dataset that is not mapped raises
instead of silently training on defaults.

Tuning sets, one series each:

* ``germany``     -- WPuQ ``SFH19`` load, 15 minutes, Jan-Mar 2024;
* ``zonnedael9``  -- Zonnedael ``Klant 9``, 15 minutes, Jan-Jun 2013;
* ``belgium_pv1`` -- Belgium PV house 1, 15 minutes, Jan-Mar 2024.

The three were chosen for being single series on a 15-minute grid: a search
over four Belgium houses at once would spend the budget on the house the
parameter set fits best, and a half-hourly series would put the tune-time
frequency label out of step with every other cell.
"""

from __future__ import annotations

from pathlib import Path

from utils.protocol import sampling_tag

#: Bumped whenever the search space changes.  The tuner names its study and its
#: JSON with it and the runner looks the JSON up with it, so an old run stays
#: readable next to a new one instead of being silently reused.
SPACE_VERSION = 2

#: Series the search is allowed to run on, in the spelling the JSON files use.
TUNE_SETS = ("germany", "zonnedael9", "belgium_pv1")

#: Reported cell -> the tuning set its parameters come from.  The three tuning
#: cells appear on purpose: they are reported too, so their row is selected on
#: itself and the appendix has to say so.
TRANSFER = {
    "germany": "germany",
    "zonnedael9": "zonnedael9",
    "belgium_pv1": "belgium_pv1",
    "london": "zonnedael9",
    "zonnedael8": "zonnedael9",
    "zonnedael43": "zonnedael9",
    "belgium_pv2": "belgium_pv1",
    "belgium_pv3": "belgium_pv1",
    "belgium_pv4": "belgium_pv1",
    "belgium_bess1": "belgium_pv1",
    "belgium_bess2": "belgium_pv1",
    "belgium_bess3": "belgium_pv1",
    "belgium_bess4": "belgium_pv1",
}

#: Dataset spelling of ``scripts/run_benchmark.py`` (one entry per directory of
#: ``results_<dataset>``) -> tuning set.  A dataset is one run covering every
#: series it holds, so all of Belgium's eight series point at the same search.
RUN_SOURCE = {
    "germany": "germany",
    "london": "zonnedael9",
    "zonnedael": "zonnedael9",
    "belgium": "belgium_pv1",
}


def tune_source(dataset: str) -> str:
    """Tuning set that supplies ``dataset`` (a tuning set supplies itself)."""
    source = RUN_SOURCE.get(dataset, dataset)
    if source not in TUNE_SETS:
        raise KeyError(f"{dataset!r} has no tuning source; known: {', '.join(sorted(RUN_SOURCE))}")
    return source


def transfer_targets(source: str) -> list[str]:
    """Reported cells that borrow the parameters tuned on ``source``."""
    return sorted(cell for cell, origin in TRANSFER.items() if origin == source)


def _tuned_file(root, model: str, source: str, level: str, sampling_rate: float,
                space_version: int) -> Path:
    return (Path(root) / "tuning"
            / f"{model}_{source}_{level}_{sampling_tag(sampling_rate)}_v{space_version}.json")


def tuned_resolution(root, model: str, dataset: str, level: str, sampling_rate: float,
                     space_version: int = SPACE_VERSION):
    """Path of the tuned JSON for a cell, plus the sampling rate it was tuned at.

    A search run at one sampling rate is reused at the others.  The tuned values
    are architecture and optimiser choices, which do not depend on the sampling
    grid, and the published flow copied one JSON onto every tag for exactly this
    reason.  So: prefer the file of the exact rate, otherwise fall back to 100 %,
    which is the rate every search is run at.  ``(path, None)`` means "no tuned
    parameters for this cell" and the caller trains on the paper's defaults.

    The second element lets a caller (``scripts/run_benchmark.py``) say out loud
    that a 50 % cell is running on parameters searched at 100 %.
    """
    source = tune_source(dataset)
    exact = _tuned_file(root, model, source, level, sampling_rate, space_version)
    if exact.exists():
        return exact, float(sampling_rate)
    fallback = _tuned_file(root, model, source, level, 100.0, space_version)
    if fallback.exists():
        return fallback, 100.0
    return exact, None

