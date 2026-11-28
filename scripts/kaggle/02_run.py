#!/usr/bin/env python
"""Run scripts/run_benchmark.py on a hosted GPU.

The runner builds each command without ``--gpu``, and ``utils/device.py`` reads
anything except 0/1 as "CPU", so a notebook run would train every cell on CPU
while the notebook looks healthy.  This wrapper imports the runner, appends
the accelerator flag to every command it builds, guards the conda envs it will
shell out to, and forwards the rest of the arguments untouched.

    python scripts/kaggle/02_run.py --dry-run --models LightGBM \
        --datasets germany --samplings 25 --level COV --results-dir /kaggle/working/results

    python scripts/kaggle/02_run.py --models "LightGBM Chronos2 Timestral TimesFM3" --level COV
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(os.environ.get("REPO_DIR") or Path(__file__).resolve().parents[2])
if str(REPO_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "scripts"))

# Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline; the conda envs have
# no matplotlib_inline, so a child that imports pyplot dies on an invalid backend.  The
# model scripts only ever savefig, so the headless backend is the right one here.
os.environ["MPLBACKEND"] = "Agg"

import run_benchmark as runner  # noqa: E402

#: model -> conda env the runner will call, so a missing env fails here instead of
#: after the first import inside a subprocess.
MODEL_ENV = {"LightGBM": "statsml", "Chronos2": "chronos",
             "Timestral": "chronos", "TimesFM3": "timesfm3"}


def conda_envs() -> set[str]:
    try:
        out = subprocess.run(["conda", "env", "list"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {line.split()[0] for line in out.splitlines() if line.strip() and not line.startswith("#")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Hosted-GPU wrapper around the COV benchmark runner.")
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("KAGGLE_GPU", "0")),
                        help="Accelerator index handed to every model script (default 0).")
    parser.add_argument("--models", nargs="+", default=["LightGBM"])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Omit for the runner's default: all four datasets.")
    parser.add_argument("--samplings", nargs="+", type=float, default=None,
                        help="Omit for the runner's default: 25, 100/3, 50, 100.")
    parser.add_argument("--level", default="COV", choices=("COV", "L0"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path(os.environ.get("RESULTS", "results")))
    parser.add_argument("--force", action="store_true", help="Re-run complete cells.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands, run nothing.")
    args = parser.parse_args(argv)

    original = runner.command_for

    def command_for_with_gpu(*call_args, **call_kwargs):
        command = original(*call_args, **call_kwargs)
        if "--gpu" not in command:
            command += ["--gpu", str(args.gpu)]
        return command

    runner.command_for = command_for_with_gpu

    if not args.dry_run:
        available = conda_envs()
        missing = sorted({MODEL_ENV[m] for m in args.models} - available)
        if missing:
            print(f"FAIL: conda env(s) missing: {', '.join(missing)}\n"
                  f"  run: ENVS=\"{' '.join(missing)}\" bash scripts/kaggle/01_setup.sh", file=sys.stderr)
            return 2

    forwarded = ["--models", *args.models, "--level", args.level,
                 "--results_dir", str(args.results_dir)]
    if args.datasets:
        forwarded += ["--datasets", *args.datasets]
    if args.samplings:
        forwarded += ["--samplings", *[f"{s:g}" for s in args.samplings]]
    if args.force:
        forwarded.append("--force")
    if args.dry_run:
        forwarded.append("--dry-run")
    return runner.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
