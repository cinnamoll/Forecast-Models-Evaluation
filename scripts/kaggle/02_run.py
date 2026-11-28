#!/usr/bin/env python
"""Run scripts/run_benchmark.py on a hosted GPU.

The runner builds each command without ``--gpu``, and ``utils/device.py`` reads
anything except 0/1 as "CPU", so a notebook run would train every cell on CPU
while the notebook looks healthy.  This wrapper imports the runner, appends
the accelerator flag to every command it builds, guards the conda envs it will
shell out to, and forwards the rest of the arguments untouched.

    python scripts/kaggle/02_run.py --dry-run --models LightGBM \
        --datasets germany --samplings 25 --level COV --results-dir /kaggle/working/results_covariate

    python scripts/kaggle/02_run.py --models "LightGBM Chronos2 Timestral TimesFM3" --level COV

The results root follows ``--level`` when neither ``--results-dir`` nor ``$RESULTS``
is given: COV writes into ``<repo>/results_covariate``, L0 into ``<repo>/results``.
That is the layout the tuning JSONs and the TimeSHAP wrapper expect too.
"""
from __future__ import annotations

import argparse
import os
import shutil
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

try:
    import run_benchmark as runner  # noqa: E402
except ModuleNotFoundError as exc:
    # The runner's own imports (utils/protocol.py -> pandas) run in *this* interpreter,
    # not in the model envs it shells out to, so the orchestrator python needs them too.
    raise SystemExit(
        f"FAIL: {exc}.  This wrapper imports run_benchmark in the current interpreter, which\n"
        f"  needs pandas: run it with a python that has one, e.g.\n"
        f"    conda run --no-capture-output -n statsml python scripts/kaggle/02_run.py ...\n"
        f"  or re-run bash scripts/kaggle/01_setup.sh (it installs pandas into the conda base)."
    ) from None

#: model -> conda env the runner will call, so a missing env fails here instead of
#: after the first import inside a subprocess.
MODEL_ENV = {"LightGBM": "statsml", "Chronos2": "chronos",
             "Timestral": "chronos", "TimesFM3": "timesfm3"}


def flatten(values):
    """argparse hands back one token per element, so ``--models "A B"`` arrives as a
    single string and ``MODEL_ENV["A B"]`` would raise KeyError.  Split it, so both the
    docstring's quoted form and the plain ``--models A B`` work."""
    return [item for value in values for item in str(value).split()]


def ensure_conda() -> None:
    """The runner shells out to ``conda run -n <env>``, and a fresh Kaggle cell has no
    conda on PATH; same search as scripts/kaggle/_conda.sh, for the python entry point."""
    if shutil.which("conda"):
        return
    for cand in (os.environ.get("CONDA_HOME", "/tmp/miniforge3"), "/opt/conda",
                 "~/miniforge3", "~/anaconda3", "~/miniconda3", "/kaggle/working/miniforge3"):
        exe = Path(cand).expanduser() / "bin" / "conda"
        if exe.is_file():
            os.environ["CONDA_HOME"] = str(exe.parent.parent)
            os.environ["PATH"] = f"{exe.parent}{os.pathsep}{os.environ.get('PATH', '')}"
            return


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
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Results root.  Default: $RESULTS when exported, otherwise "
                             "results_covariate at --level COV and results at L0.")
    parser.add_argument("--force", action="store_true", help="Re-run complete cells.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands, run nothing.")
    args = parser.parse_args(argv)
    args.models = flatten(args.models)
    if args.datasets:
        args.datasets = flatten(args.datasets)
    unknown = [m for m in args.models if m not in MODEL_ENV]
    if unknown:
        print(f"FAIL: unknown model(s): {', '.join(unknown)}\n"
              f"  known: {' '.join(MODEL_ENV)}", file=sys.stderr)
        return 2
    if args.results_dir is None:
        exported = os.environ.get("RESULTS")
        args.results_dir = (Path(exported) if exported else
                            REPO_DIR / ("results_covariate" if args.level == "COV" else "results"))

    original = runner.command_for

    def command_for_with_gpu(*call_args, **call_kwargs):
        command = original(*call_args, **call_kwargs)
        if "--gpu" not in command:
            command += ["--gpu", str(args.gpu)]
        return command

    runner.command_for = command_for_with_gpu

    if not args.dry_run:
        ensure_conda()
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
