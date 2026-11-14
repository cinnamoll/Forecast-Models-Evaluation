#!/usr/bin/env python3
"""Print what ``ALL`` contains for each dataset, and what was removed.

There is no ladder any more: a model runs ``L0`` (paper baseline) or ``ALL``.
This table states, per dataset, which blocks ``ALL`` actually has -- and which
blocks were removed from the pipeline and why.  Nothing here is filled with a
proxy: a block is either built from a named source or reported absent.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from utils.exogenous.feature_store import availability_report  # noqa: E402
from utils.exogenous.sources.availability import removed_blocks  # noqa: E402

# Benchmark windows, matching data/README.md.
WINDOWS = {
    # data/README.md: Belgium covers 1 January - 31 March 2024 for every task, so
    # the availability table is asked about the window the models actually run.
    "belgium": ("2024-01-01", "2024-04-01", {"house_1": "house_1"}),
    "germany": ("2024-01-01", "2024-04-01", {"SFH19": "SFH19"}),
    "london": ("2013-01-01", "2013-12-31", {"MAC000033": "MAC000033"}),
    "zonnedael": ("2013-01-01", "2013-06-30", {"Klant 8": "Klant 8"}),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=[*WINDOWS, "all"], default=["all"])
    args = parser.parse_args(argv)

    datasets = list(WINDOWS) if "all" in args.dataset else args.dataset
    rows = []
    for dataset in datasets:
        start, end, entities = WINDOWS[dataset]
        entity_id = next(iter(entities.values()))
        report = availability_report(dataset, entity_id, start, end)
        for block in ("calendar", "weather", "price", "static"):
            available, detail = report[block]
            rows.append(
                {
                    "dataset": dataset,
                    "entity": entity_id,
                    "window": f"{start}..{end}",
                    "level": block,
                    "status": "in ALL" if available else "absent",
                    "detail": detail,
                }
            )
        # blocks that were removed from the pipeline: stated every time, so the
        # table cannot be read as "nobody tried"
        for level, detail in removed_blocks().items():
            rows.append(
                {
                    "dataset": dataset,
                    "entity": entity_id,
                    "window": f"{start}..{end}",
                    "level": level,
                    "status": "removed",
                    "detail": detail,
                }
            )

    table = pd.DataFrame(rows)
    width = max(len(str(value)) for value in table["detail"])
    for dataset in table["dataset"].unique():
        print(f"\n{dataset}")
        subset = table[table["dataset"] == dataset]
        for _, row in subset.iterrows():
            detail = str(row["detail"])
            if len(detail) > 120:
                detail = detail[:117] + "..."
            print(f"  {row['level']:14s} {row['status']:5s} {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
