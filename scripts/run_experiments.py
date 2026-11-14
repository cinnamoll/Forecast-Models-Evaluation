#!/usr/bin/env python
"""Run plan of the benchmark: the paper baseline and the all-covariates run.

There is no ladder of exogenous levels.  Two modes:

* ``L0`` -- the paper baseline: the original table's calendar, the four sampling
  rates of the paper and ten runs.  It exists to reproduce that table and to be
  the reference every new column is compared against.
* ``ALL`` -- every covariate the dataset has, at 100 % and 25 % only (the
  elimination and its XAI are the expensive part).  Which of those covariates
  earns its place is decided afterwards by ``scripts/xai_elimination.py``, which
  drops one group at a time on the validation block and records what each drop
  costs.

Rules this script encodes:

* A cell that already has its results is skipped, so a crash costs only the cell
  that was running.
* ``ALL`` always runs; what varies per dataset is which blocks it contains.  The
  script prints the ``scripts/exogenous_availability.py`` picture for the dataset
  it is about to launch, so the appendix can state it next to the numbers.
* The models are launched in their own conda environments (this repository keeps
  one per framework), so nothing here imports torch, mxnet or lightgbm.

Usage
-----
    python scripts/run_experiments.py --dry-run                 # print the plan
    python scripts/run_experiments.py --models LightGBM ALL     # one model, both modes
    python scripts/run_experiments.py --levels ALL --force
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.benchmark_series import SERIES_ORDER
from utils.protocol import (
    ALL_COVARIATES_SAMPLING_RATES,
    COVARIATE_LEVEL,
    DEFAULT_RUNS,
    EXOG_LEVEL_CHOICES,
    PAPER_BASELINE_LEVEL,
    RUN_LEVELS,
    SAMPLING_RATES,
    paper_bounds,
    run_dir,
    sampling_tag,
)

#: The four datasets of data/README.md, taken from the series list itself so the
#: plan cannot launch a dataset the data documentation does not describe.
DATASETS = tuple(SERIES_ORDER)

#: model -> (conda env, script, extra CLI args, result-dir style)
MODEL_SPECS: Dict[str, dict] = {
    "LightGBM": {
        "env": "statsml",
        "script": "models/models_statsml.py",
        "style": "run",
        "flags": ["--models", "LightGBM"],
    },
    "DeepNPTS": {
        "env": "neuralforecast",
        "script": "models/models_neuralforecast.py",
        "style": "epochs",
        "epochs": 100,
        "flags": ["--models", "DeepNPTS", "--epochs", "100"],
    },
    "PatchTST": {
        "env": "neuralforecast",
        "script": "models/models_neuralforecast.py",
        "style": "epochs",
        "epochs": 100,
        "flags": ["--models", "PatchTST", "--epochs", "100"],
    },
    "TCN": {
        "env": "neuralforecast",
        "script": "models/models_neuralforecast.py",
        "style": "epochs",
        "epochs": 100,
        "flags": ["--models", "TCN", "--epochs", "100"],
    },
    "TFT": {
        # plan_updated.md 1.4: only the GluonTS MX TemporalFusionTransformer is
        # part of the comparison.  The NeuralForecast TFT was removed.
        "env": "gluonts",
        "script": "models/models_gluonts.py",
        "style": "epochs",
        "epochs": 100,
        "flags": ["--models", "TemporalFusionTransformer", "--epochs", "100"],
    },
    "Chronos2": {
        "env": "chronos",
        "script": "models/fmodel_chronos2.py",
        "style": "run",
        "flags": [],
    },
    "Timestral": {
        "env": "chronos",
        "script": "models/fmodel_timestral.py",
        "style": "run",
        "flags": [],
    },
}

MODELS = tuple(MODEL_SPECS)

#: Representative entity per dataset, used only to ask the availability report.
ENTITY_FOR = {
    "belgium": "house_1",
    "germany": "SFH19",
    "london": "MAC000033",
    "zonnedael": "Klant 8",
}


def samplings_for(level: str) -> Sequence[float]:
    """The baseline keeps the paper's four rates; the all-covariates run keeps two."""
    return SAMPLING_RATES if level == PAPER_BASELINE_LEVEL else ALL_COVARIATES_SAMPLING_RATES


@dataclass
class PlannedRun:
    dataset: str
    model: str
    level: str
    samplings: Sequence[float]
    status: str = "planned"
    reason: str = ""
    command: List[str] = field(default_factory=list)


def tuning_file(model: str, dataset: str, level: str, sampling: float) -> Path:
    return PROJECT_ROOT / "results" / "tuning" / f"{model}_{dataset}_{level}_{sampling_tag(sampling)}.json"


def existing_runs(dataset: str, model: str, level: str, sampling: float) -> int:
    """How many run directories already hold a metrics file."""
    spec = MODEL_SPECS[model]
    count = 0
    for run in range(1, DEFAULT_RUNS + 1):
        path = run_dir(PROJECT_ROOT / "results", dataset, model, sampling, run,
                       epochs=spec.get("epochs"), style=spec["style"], exog_level=level)
        if (path / "model_metrics_summary.csv").exists():
            count += 1
    return count


def blocks_of(dataset: str, level: str) -> Dict[str, str]:
    """Which blocks ``ALL`` contains here (empty for the baseline)."""
    if level == PAPER_BASELINE_LEVEL:
        return {}
    from utils.exogenous.sources import availability

    start, end = paper_bounds()
    report = availability.report(dataset, ENTITY_FOR[dataset], start, end)
    return {block: ("in ALL" if ok else f"absent: {detail}")
            for block, (ok, detail) in report.items()}


def build_command(run: PlannedRun) -> List[str]:
    spec = MODEL_SPECS[run.model]
    sampling_arg = " ".join(f"{rate:g}" for rate in run.samplings)
    parts = [
        "conda", "run", "--no-capture-output", "-n", spec["env"],
        "python", spec["script"],
        "--dataset", run.dataset,
        "--sampling_rates", sampling_arg,
        "--runs", str(DEFAULT_RUNS),
        "--exog_level", run.level,
    ]
    parts.extend(spec["flags"])
    params = tuning_file(run.model, run.dataset, run.level, run.samplings[0])
    if run.model in ("LightGBM", "DeepNPTS", "PatchTST", "TFT", "TCN") and params.exists():
        parts.extend(["--params_json", str(params)])
    return parts


def plan(levels: Sequence[str], models: Sequence[str], datasets: Sequence[str]) -> List[PlannedRun]:
    planning = []
    for dataset in datasets:
        for model in models:
            for level in levels:
                run = PlannedRun(dataset=dataset, model=model, level=level,
                                 samplings=tuple(samplings_for(level)))
                done = sum(existing_runs(dataset, model, level, rate) >= DEFAULT_RUNS
                           for rate in run.samplings)
                if done == len(run.samplings):
                    run.status = "complete"
                    run.reason = "all runs already on disk"
                elif done:
                    run.status = "partial"
                    run.reason = f"{done}/{len(run.samplings)} sampling rates complete"
                run.command = build_command(run)
                planning.append(run)
    return planning


def write_manifest(planning: Sequence[PlannedRun], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "model", "level", "samplings", "status", "reason", "command"])
        for run in planning:
            writer.writerow([
                run.dataset, run.model, run.level,
                " ".join(f"{rate:g}" for rate in run.samplings),
                run.status, run.reason, " ".join(run.command),
            ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the baseline and all-covariates cells.")
    parser.add_argument("--levels", nargs="+", choices=list(EXOG_LEVEL_CHOICES), default=list(RUN_LEVELS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--dry-run", action="store_true", help="Print the plan, run nothing.")
    parser.add_argument("--force", action="store_true", help="Re-run cells that are already complete.")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "results" / "run_plan_manifest.csv")
    args = parser.parse_args(argv)

    planning = plan(args.levels, args.models, args.datasets)
    if args.force:
        for run in planning:
            if run.status == "complete":
                run.status, run.reason = "planned", "forced re-run"
    write_manifest(planning, args.manifest)
    print(f"manifest -> {args.manifest}")

    reported = set()
    executed = 0
    for run in planning:
        label = f"{run.dataset}/{run.model}/{run.level}"
        if run.status == "complete" and not args.force:
            print(f"SKIP  {label}: {run.reason}")
            continue
        if run.dataset not in reported:
            reported.add(run.dataset)
            blocks = blocks_of(run.dataset, COVARIATE_LEVEL)
            print(f"ALL for {run.dataset}: "
                  + ", ".join(f"{name} {state}" for name, state in blocks.items()))
        print(f"RUN   {label} ({run.status})")
        print("      " + " ".join(run.command))
        if args.dry_run:
            continue
        result = subprocess.run(run.command, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"FAIL  {label}: exit {result.returncode}")
        else:
            executed += 1
    print(f"done: {executed} command(s) executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
