#!/usr/bin/env python3
"""Build the peer matrices used by L5 from files already on disk.

* Zonnedael: the repo file already holds all 56 customers, so the matrix is a
  reshape, no download.
* London: produced by ``fetch_lcl_households.py`` (full LCL download).
* Belgium: no peers in the benchmark window.  All four houses replay the same
  household load (SFH19), so a load peer would be the target itself, and the BESS
  output is driven by whichever EMS ran that day.  PV peers exist only in the
  release's experiment window (2024-04-08 .. 2024-06-17), which this pipeline does
  not read, so L5 is n/a for Belgium by construction.
* Germany: no peers -> L5 is n/a by construction, not by omission.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.exogenous.config import NEIGHBOR_DIR  # noqa: E402

ZONNEDAEL_SOURCE = PROJECT_ROOT / "data" / "zonnedael_dataset" / "liander_zonnedael_2013_original.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=["zonnedael"])
    args = parser.parse_args(argv)

    NEIGHBOR_DIR.mkdir(parents=True, exist_ok=True)
    if "zonnedael" in args.dataset:
        build_zonnedael()
    for dataset in ("london", "belgium", "germany"):
        if dataset in args.dataset:
            if dataset == "london":
                print("[n/a]  london: run scripts/fetch/fetch_lcl_households.py")
            elif dataset == "belgium":
                print("[n/a]  belgium: the only cross-house series in the benchmark window "
                      "is the shared SFH19 load, i.e. the target itself; the release's "
                      "experiment window (Apr-Jun) is not used")
            else:
                print("[n/a]  germany: the source ships no peer households")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
