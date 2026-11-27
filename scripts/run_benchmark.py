#!/usr/bin/env python
"""Run plan: seven models, four datasets, up to four sampling rates, two levels.

The models live in three conda environments, so this script never imports torch,
mxnet or lightgbm — it builds the command line of each run and executes it in the
right environment, the same way the model scripts are meant to be called.

A cell that already has its ``model_metrics_summary.csv`` is skipped, so a crash
costs only the run that was in flight.

Usage
-----
    python scripts/run_benchmark.py --dry-run
    python scripts/run_benchmark.py --models LightGBM TCN --datasets london
    python scripts/run_benchmark.py --level L0 --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.protocol import DATASET_WINDOWS, runs_for, sampling_tag
from utils.tuning import tuned_resolution

#: Every model a cell may run: the five trainable ones of the transfer flow
#: (LightGBM, DeepNPTS, TCN, TFT, TimeXer) and the three zero-shot checkpoints.
MODEL_ORDER = ("LightGBM", "DeepNPTS", "TCN", "TFT", "TimeXer",
               "Chronos2", "Timestral", "TimesFM3")

#: Models whose refit accepts ``--params_json``; the zero-shot ones take none.
#: Their parameters come from ``scripts/tune_optuna.py`` and are looked up through
#: ``utils/tuning.py``: London runs on Zonnedael's search, Belgium's eight series
#: on PV house 1's.
TUNED_MODELS = ("LightGBM", "DeepNPTS", "TCN", "TFT")

def cell_dir(dataset: str, model: str, sampling: float, level: str, results_dir) -> Path:
    base = PROJECT_ROOT / results_dir / f"results_{dataset}" / model / f"Sampling_{sampling_tag(sampling)}"
    if level != "L0":
        base = base / level
    return base


def completed_runs(dataset: str, model: str, sampling: float, level: str, results_dir) -> int:
    """How many runs of a cell already hold a metrics file."""
    root = cell_dir(dataset, model, sampling, level, results_dir)
    if not root.exists():
        return 0
    return sum(1 for child in root.iterdir()
               if child.is_dir() and (child / "model_metrics_summary.csv").exists())


def command_for(dataset: str, model: str, sampling: float, level: str, results_dir,
                covariates=None) -> list[str]:
    # model -> (conda env, script, extra flags); the run is spawned in that env.
    specs = {
        "LightGBM": {"env": "statsml", "script": "models/models_statsml.py",
                     "flags": ["--models", "LightGBM"]},
        "DeepNPTS": {"env": "neuralforecast", "script": "models/models_neuralforecast.py",
                     "flags": ["--models", "DeepNPTS", "--epochs", "100"]},
        "TCN": {"env": "neuralforecast", "script": "models/models_neuralforecast.py",
                "flags": ["--models", "TCN", "--epochs", "100"]},
        "TFT": {"env": "gluonts", "script": "models/models_gluonts.py",
                "flags": ["--models", "TemporalFusionTransformer", "--epochs", "100"]},
        "TimeXer": {"env": "neuralforecast", "script": "models/model_timexer.py",
                    "flags": ["--epochs", "100"]},
        "Chronos2": {"env": "chronos", "script": "models/fmodel_chronos2.py", "flags": []},
        "Timestral": {"env": "chronos", "script": "models/fmodel_timestral.py", "flags": []},
        "TimesFM3": {"env": "timesfm3", "script": "models/fmodel_timesfm3.py", "flags":[]}
    }
    spec = specs[model]
    parts = ["conda", "run", "--no-capture-output", "-n", spec["env"], "python", spec["script"],
             "--dataset", dataset, "--sampling_rates", f"{sampling:g}",
             "--runs", str(runs_for(model)), "--exog_level", level, "--results_dir", str(results_dir)]
    parts.extend(spec["flags"])
    if covariates:
        parts.extend(["--covariates", *covariates])
    # The parameters come from the tuning run of the *source* dataset, not of the
    # dataset being run: London borrows Zonnedael's, Belgium's eight series
    # borrow PV house 1's.  ``utils/tuning.py`` holds that map, and also decides
    # whether the file of this sampling rate exists or the 100 % search is
    # borrowed, so a cell with no source raises instead of training on defaults.
    tuned, _tuned_at = tuned_resolution(PROJECT_ROOT/results_dir, model, dataset, level, sampling)
    if model in TUNED_MODELS and tuned.exists():
        parts.extend(["--params_json", str(tuned)])
    return parts


def tuned_note(dataset: str, model: str, sampling: float, level: str, results_dir):
    """One line saying where a cell's parameters come from, or why it has none."""
    if model not in TUNED_MODELS:
        return None
    tuned, tuned_at = tuned_resolution(PROJECT_ROOT / results_dir, model, dataset, level, sampling)
    if tuned_at is None:
        return f"no tuned parameters for {model} on this dataset: paper defaults"
    if float(tuned_at) != float(sampling):
        return f"parameters borrowed from the {tuned_at:g} % search ({tuned.name})"
    return None


def main(argv=None) -> int:
    datasets = tuple(DATASET_WINDOWS)
    sampling_rates = (25.0, 100.0 / 3.0, 50.0, 100.0)

    parser = argparse.ArgumentParser(description="Run the covariate benchmark.")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_ORDER), default=None,
                        help="Default: every model in MODEL_ORDER.")
    parser.add_argument("--datasets", nargs="+", choices=list(datasets), default=list(datasets))
    parser.add_argument("--samplings", nargs="+", type=float, default=list(sampling_rates))
    parser.add_argument("--level", choices=list(("L0", "COV")), default="COV")
    parser.add_argument("--force", action="store_true", help="Re-run cells that are complete.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned commands and exit.")
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results", help="Root directory for generated results.")
    parser.add_argument("--covariates", nargs="+", default=None,
                        help="Restrict the COV block to these columns, e.g. the TimeSHAP set "
                             "(hour dayofweek humidity temperature cloud_cover) plus price/soc on Belgium. "
                             "Omit to keep every block the dataset has.")
    args = parser.parse_args(argv)
    results_dir = args.results_dir.resolve()

    # Every model may run at either level: the four trainable models are the
    # transfer flow (COV = covariates + tuned parameters, L0 = the paper's
    # baseline), the three checkpoints are zero-shot at both.
    models = list(args.models) if args.models else list(MODEL_ORDER)

    executed = skipped = 0
    for dataset in args.datasets:
        for model in models:
            for sampling in args.samplings:
                done = completed_runs(dataset, model, sampling, args.level, results_dir)
                expected = runs_for(model)
                label = f"{dataset}/{model}/Sampling_{sampling_tag(sampling)}/{args.level}"
                if done >= expected and not args.force:
                    print(f"SKIP  {label}: {done}/{expected} runs on disk")
                    skipped += 1
                    continue
                command = command_for(dataset, model, sampling, args.level, results_dir,
                                     covariates=args.covariates)
                print(f"RUN   {label} ({done}/{expected} runs)")
                note = tuned_note(dataset, model, sampling, args.level, results_dir)
                if note:
                    print(f"      note: {note}")
                print("      " + " ".join(command))
                if args.dry_run:
                    executed += 1
                    continue
                result = subprocess.run(command, cwd=str(PROJECT_ROOT))
                if result.returncode != 0:
                    print(f"FAIL  {label}: exit {result.returncode}")
                else:
                    executed += 1
    print(f"done: {executed} command(s) executed, {skipped} cell(s) skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
