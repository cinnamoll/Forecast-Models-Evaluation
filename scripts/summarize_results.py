#!/usr/bin/env python
"""Build the section-8 tables from the result tree.

Outputs (all under ``results/``):

* ``summary_median_iqr.csv`` -- MAE/RMSE median and IQR over the ten seeds, per
  dataset, model, sampling rate and series.
* ``ablation_ladder.csv`` -- MAE per exogenous level, the L0-L5 ablation table.
* ``elimination_effects.csv`` -- per-group effect of *removing* the group, taken
  from the TimeSHAP elimination ledgers (there is no ladder to difference),
  which is what the XAI cross-check consumes.
* ``bootstrap/<dataset>__<series>.csv`` -- block-bootstrap CI of the MAE gap for
  every model pair, blocks of one day.
* ``dm_test/<dataset>/`` -- DM heatmaps and p-value CSVs from ``utils/dm_test.py``
  (the original test, unchanged).

The DM test is the published implementation from the paper; the autocorrelation
correction asked for in section 8 lives in the block-bootstrap CI instead, so the
two stay separable in the appendix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from utils.block_bootstrap import DEFAULT_N_BOOT, pairwise_bootstrap_table
from utils.dm_test import run_dm_for_dataset
from utils.protocol import (
    ALL_COVARIATES_SAMPLING_RATES,
    BOOTSTRAP_BLOCK_DAYS,
    EXOG_LEVEL_CHOICES,
    RUN_LEVELS,
    SAMPLING_RATES,
    TABLE_MODELS,
    sampling_tag,
)
from utils.seed_aggregation import (
    ablation_table,
    build_summary_table,
    load_losses,
)

DATASETS = ("belgium", "germany", "london", "zonnedael")
SERIES_FOR = {
    "belgium": ["pv_house_1", "bess_house_1"],
    "germany": ["germany_load"],
    "london": ["london_load"],
    "zonnedael": ["zonnedael_customer_8"],
}

LEGEND = {"LightGBM": "LightGBM", "DeepNPTS": "DeepNPTS", "PatchTST": "PatchTST",
          "TFT": "TFT", "Chronos2": "Chronos-2 (zero-shot)",
          "Timestral": "Timestral-123M (zero-shot)"}


def tuning_appendix(root: Path) -> pd.DataFrame:
    """The section-5 appendix table: trials used, hours used, saturation, edges."""
    rows = []
    for path in sorted((Path(root) / "results" / "tuning").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "best_params" not in payload:
            continue
        rows.append({
            "model": payload.get("model"),
            "dataset": payload.get("dataset"),
            "level": payload.get("level"),
            "sampling": payload.get("sampling"),
            "trials_used": payload.get("used", {}).get("trials"),
            "hours_used": payload.get("used", {}).get("hours"),
            "max_trials": payload.get("budget", {}).get("max_trials"),
            "max_hours": payload.get("budget", {}).get("max_hours"),
            "best_mae": payload.get("best_value"),
            "saturated": payload.get("saturated"),
            "boundary_hits": ", ".join(payload.get("boundary_hits", [])),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate results into the paper tables.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=list(TABLE_MODELS))
    parser.add_argument("--samplings", nargs="+", type=float, default=list(SAMPLING_RATES))
    parser.add_argument("--all-covariate-samplings", nargs="+", type=float,
                        default=list(ALL_COVARIATES_SAMPLING_RATES))
    parser.add_argument("--levels", nargs="+", choices=list(EXOG_LEVEL_CHOICES), default=list(RUN_LEVELS))
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--bootstrap-block-days", type=float, default=BOOTSTRAP_BLOCK_DAYS)
    parser.add_argument("--skip-dm", action="store_true", help="Do not rebuild the DM heatmaps.")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args(argv)

    results_root = args.root / "results"

    # ---------------------------------------------------------------- summary
    summary = build_summary_table(args.root, args.datasets, args.models,
                                  args.samplings, levels=("L0",))
    summary_path = results_root / "summary_median_iqr.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"summary -> {summary_path} ({len(summary)} rows)")

    # ------------------------------------------------------------- run modes
    mode_frames = []
    for dataset in args.datasets:
        for series in SERIES_FOR[dataset]:
            mode_frames.append(ablation_table(args.root, args.models, dataset,
                                              args.all_covariate_samplings[-1], series,
                                              levels=args.levels))
    modes = pd.concat(mode_frames, ignore_index=True) if mode_frames else pd.DataFrame()
    modes_path = results_root / "run_modes.csv"
    modes.to_csv(modes_path, index=False, float_format="%.6f")
    print(f"modes   -> {modes_path} ({len(modes)} rows)")

    # ------------------------------------------------- elimination (ablation)
    # The ladder is gone: what a group is worth is the MAE the elimination paid
    # when it dropped that group, per model, on the validation block.
    from utils.elimination import load_effects

    effect_frames = []
    for dataset in args.datasets:
        for series in SERIES_FOR[dataset]:
            for model in args.models:
                ledger = (PROJECT_ROOT / "results" / "xai_elimination" / dataset /
                          f"{series}_{model}_ALL_{sampling_tag(args.all_covariate_samplings[-1])}" /
                          "elimination_rounds.csv")
                effects = load_effects(ledger)
                if effects.empty:
                    continue
                effects.insert(0, "model", model)
                effects.insert(0, "series", series)
                effects.insert(0, "dataset", dataset)
                effect_frames.append(effects)
    effects = pd.concat(effect_frames, ignore_index=True) if effect_frames else pd.DataFrame(
        columns=["dataset", "series", "model", "group", "delta_mae", "from_round", "to_round"])
    effects_path = results_root / "elimination_effects.csv"
    effects.to_csv(effects_path, index=False, float_format="%.6f")
    print(f"effects  -> {effects_path} ({len(effects)} rows)")

    # ---------------------------------------------------------------- tuning
    tuning = tuning_appendix(args.root)
    tuning_path = results_root / "tuning_summary.csv"
    tuning.to_csv(tuning_path, index=False, float_format="%.4f")
    print(f"tuning   -> {tuning_path} ({len(tuning)} study/studies)")
    if not tuning.empty and (tuning["saturated"] == False).any():  # noqa: E712
        print("WARNING: some studies had not saturated - say in the appendix that the "
              "budget was not enough")

    # ---------------------------------------------------------------- bootstrap
    if not args.skip_bootstrap:
        boot_dir = results_root / "bootstrap"
        boot_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for dataset in args.datasets:
            for series in SERIES_FOR[dataset]:
                losses = load_losses(args.root, dataset, args.models,
                                     args.samplings[-1], series)
                table = pairwise_bootstrap_table(losses, n_boot=args.bootstrap_n,
                                                 block_days=args.bootstrap_block_days,
                                                 model_order=args.models)
                if table.empty:
                    continue
                path = boot_dir / f"{dataset}__{series}.csv"
                table.to_csv(path, index=False, float_format="%.6f")
                written += 1
        print(f"bootstrap -> {boot_dir} ({written} file(s))")

    # ---------------------------------------------------------------- DM test
    if not args.skip_dm:
        for dataset in args.datasets:
            paths = run_dm_for_dataset(dataset, args.root, legend_names=LEGEND)
            print(f"dm       -> {dataset}: {len(paths)} file(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
