#!/usr/bin/env python
"""L0 parity check (section 7): stop if a re-run drifted from the paper's table.

The original models must reproduce their published numbers.  This script compares
the median MAE/RMSE of two result trees, cell by cell, and fails when the relative
difference exceeds 2 %.  Run it before the ladder, not after: the point is to
catch a broken loader or a changed window before every level inherits the fault.

Typical use
-----------
    # 1. snapshot the current L0 tree (from the committed results)
    python scripts/check_l0_parity.py --snapshot

    # 2. re-run L0 (scripts/run_experiments.py --levels L0)

    # 3. compare
    python scripts/check_l0_parity.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from utils.protocol import DEFAULT_RUNS, SAMPLING_RATES, sampling_tag
from utils.seed_aggregation import collect_cell_metrics, summarize_cell

L0_MODELS = ("LightGBM", "DeepNPTS", "PatchTST", "TFT", "Chronos", "Chronos2", "Timestral")
DATASETS = ("belgium", "germany", "london", "zonnedael")


def snapshot(baseline_root: Path, datasets, models, samplings) -> int:
    """Copy the current ``model_metrics_summary.csv`` files into ``baseline_root``."""
    copied = 0
    for dataset in datasets:
        for model in models:
            for sampling in samplings:
                source = PROJECT_ROOT / "results" / f"results_{dataset}" / model / f"Sampling_{sampling_tag(sampling)}"
                if not source.exists():
                    continue
                for run_dir in sorted(source.iterdir()):
                    metrics = run_dir / "model_metrics_summary.csv"
                    if not run_dir.is_dir() or not metrics.exists():
                        continue
                    target = baseline_root / "results" / f"results_{dataset}" / model / f"Sampling_{sampling_tag(sampling)}" / run_dir.name
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(metrics, target / "model_metrics_summary.csv")
                    copied += 1
    return copied


def compare(baseline_root: Path, new_root: Path, datasets, models, samplings,
            tolerance: float) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        for model in models:
            for sampling in samplings:
                base = summarize_cell(collect_cell_metrics(baseline_root, dataset, model, sampling))
                new = summarize_cell(collect_cell_metrics(new_root, dataset, model, sampling))
                cell_key = {"dataset": dataset, "model": model, "sampling": sampling_tag(sampling)}
                if base.empty and new.empty:
                    # The model was never run for this dataset: not a drift.
                    rows.append({**cell_key, "series": "*", "status": "absent",
                                 "detail": "no results in either tree"})
                    continue
                if base.empty:
                    rows.append({**cell_key, "series": "*", "status": "new-only",
                                 "detail": "no reference for this cell; nothing to compare"})
                    continue
                if new.empty:
                    rows.append({**cell_key, "series": "*", "status": "missing-in-rerun",
                                 "detail": "reference exists but the re-run wrote no metrics"})
                    continue
                merged = base.merge(new, on="series", suffixes=("_base", "_new"))
                for _, cell in merged.iterrows():
                    base_mae = float(cell["mae_median_base"])
                    new_mae = float(cell["mae_median_new"])
                    base_rmse = float(cell["rmse_median_base"])
                    new_rmse = float(cell["rmse_median_new"])
                    rel_mae = abs(new_mae - base_mae) / base_mae if base_mae else float("inf")
                    rel_rmse = abs(new_rmse - base_rmse) / base_rmse if base_rmse else float("inf")
                    worst = max(rel_mae, rel_rmse)
                    rows.append({
                        **cell_key, "series": cell["series"],
                        "mae_base": base_mae, "mae_new": new_mae, "mae_rel_diff": rel_mae,
                        "rmse_base": base_rmse, "rmse_new": new_rmse, "rmse_rel_diff": rel_rmse,
                        "status": "fail" if worst > tolerance else "ok",
                        "detail": f"worst relative change {worst:.2%} (tolerance {tolerance:.0%})",
                    })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare a re-run of L0 with the paper's results.")
    parser.add_argument("--baseline-root", type=Path, default=PROJECT_ROOT / "results" / "baseline_l0",
                        help="Tree holding the reference L0 metrics (default results/baseline_l0).")
    parser.add_argument("--new-root", type=Path, default=PROJECT_ROOT,
                        help="Root that contains results/results_<dataset> of the re-run.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=list(L0_MODELS))
    parser.add_argument("--samplings", nargs="+", type=float, default=list(SAMPLING_RATES))
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Maximum relative drift allowed (default 2 %%, as section 7 requires).")
    parser.add_argument("--snapshot", action="store_true",
                        help="Copy the current L0 tree into the baseline root and exit.")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "results" / "l0_parity_report.csv")
    args = parser.parse_args(argv)

    if args.snapshot:
        copied = snapshot(args.baseline_root, args.datasets, args.models, args.samplings)
        print(f"snapshot: {copied} run summaries copied to {args.baseline_root}")
        if copied == 0:
            print("nothing copied - are the model names right? "
                  f"expected e.g. results/results_london/LightGBM/Sampling_100/Run_1")
        return 0

    report = compare(args.baseline_root, args.new_root, args.datasets, args.models,
                     args.samplings, args.tolerance)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False, float_format="%.6f")
    if report.empty:
        print("no comparable cells found - check --baseline-root and the model names")
        return 1
    failures = report[report["status"].isin(["fail", "missing-in-rerun"])]
    print(f"compared {len(report)} cells, {len(failures)} blocking")
    if not failures.empty:
        print(failures[["dataset", "model", "sampling", "series", "status", "detail"]].to_string(index=False))
        print(f"report -> {args.report}")
        return 1
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
