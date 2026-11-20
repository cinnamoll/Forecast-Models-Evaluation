#!/usr/bin/env bash
# Setup for a hosted GPU runtime (Colab, Kaggle, or any box with a CUDA torch).
#
# The runtime must already ship a CUDA build of torch - this script never installs
# or upgrades torch.  Replacing torch on a hosted image is the usual way a
# CUDA-enabled runtime silently becomes a CPU one, and then a 2^N loop looks
# "frozen" for hours.
#
#   Colab : bash scripts/colab/01_setup.sh
#   local : PYTHON=/path/to/env/python INSTALL=0 RESULTS=/tmp/results \
#           ENV_FILE=/tmp/env.sh bash scripts/colab/01_setup.sh
#
# Knobs
#   PYTHON             interpreter to use (default: python)
#   INSTALL=1          pip install the inference packages (0 = skip)
#   INSTALL_TIMESFM=1  also install timesfm[torch]==3.0.2 (needed for TimesFM3)
#   RESULTS            result root (Colab: mount Drive first)
#   ENV_FILE           where to write the exports later cells must re-source
set -euo pipefail

PYTHON="${PYTHON:-python}"
_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${REPO_DIR:-$_SCRIPT_ROOT}"
RESULTS="${RESULTS:-/content/drive/MyDrive/timeshap_results}"
ENV_FILE="${ENV_FILE:-/content/timeshap_env.sh}"
INSTALL="${INSTALL:-1}"
INSTALL_TIMESFM="${INSTALL_TIMESFM:-0}"

say() { printf '\n=== %s\n' "$*"; }
die() { printf '\n[FATAL] %s\n' "$*" >&2; exit 1; }

say "1/6 accelerator"
command -v nvidia-smi >/dev/null || die "no nvidia-smi: pick a GPU runtime (Colab: Runtime > Change runtime type > GPU)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

say "2/6 torch must already be a CUDA build"
"$PYTHON" - <<'PY' || die "torch cannot see the GPU. Do NOT pip install torch here: restart on a GPU runtime so the preinstalled CUDA wheel is used."
import sys
import torch
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available(), "| devices", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(1)
free, total = torch.cuda.mem_get_info()
print("device", torch.cuda.get_device_name(0), "| VRAM", round(total / 2**30, 1), "GiB")
PY

cd "$REPO_DIR" 2>/dev/null || die "repo not found at $REPO_DIR (clone it first, or set REPO_DIR)"

say "3/6 inference packages (torch left untouched)"
if [ "$INSTALL" = "1" ]; then
  "$PYTHON" -m pip install -q --upgrade \
    "chronos-forecasting==2.3.1" \
    "git+https://github.com/mistralai/timestral" \
    "accelerate>=1.1,<2" "einops>=0.7,<1" "transformers>=4.41,<6"
  if [ "$INSTALL_TIMESFM" = "1" ]; then
    "$PYTHON" -m pip install -q "timesfm[torch]==3.0.2" "safetensors>=0.5.3" "huggingface-hub[cli]>=0.30"
  else
    echo "INSTALL_TIMESFM=0 -> TimesFM3 omitted; re-run with INSTALL_TIMESFM=1 to include it"
  fi
else
  echo "INSTALL=0 -> skipping pip"
fi

say "4/6 imports"
"$PYTHON" - <<'PY'
import importlib
for module, symbol in (("chronos", "Chronos2Pipeline"), ("timestral", "TimestralForecaster")):
    try:
        mod = importlib.import_module(module)
        assert hasattr(mod, symbol), f"{module} has no {symbol}"
        print(f"ok {module}.{symbol}")
    except Exception as error:
        print(f"MISSING {module}.{symbol}: {error}")
for module in ("pandas", "numpy"):
    print("ok", module, importlib.import_module(module).__version__)
try:
    import timesfm3  # noqa: F401
    print("ok timesfm3 -> TimesFM3 runs available")
except Exception as error:
    print(f"skip timesfm3: {error}")
PY

say "5/6 repo must carry the current explainer"
for flag in --per-covariate --covariates --max-players; do
  "$PYTHON" scripts/xai_timeshap.py --help 2>/dev/null | grep -q -- "$flag" \
    || die "scripts/xai_timeshap.py has no $flag: push the local changes first, then re-clone (the GitHub copy is behind)"
done
grep -q "TIMESHAP_COLUMNS" utils/covariates.py || die "utils/covariates.py has no TIMESHAP_COLUMNS: stale checkout"
echo "flags present: --per-covariate --covariates --max-players"

say "6/6 data present + result root"
test -d data || die "no data/ directory: the clone is incomplete (data/ is tracked)"
echo "data files: $(find data -type f | wc -l), size: $(du -sh data | cut -f1)"
mkdir -p "$RESULTS"
# Defaults, not assignments: an explicit RESULTS=/PYTHON=... passed by the caller must
# survive sourcing this file (that is how the sweep and ad-hoc runs override paths).
{
  printf 'export REPO_DIR="${REPO_DIR:-%s}"\n' "$REPO_DIR"
  printf 'export RESULTS="${RESULTS:-%s}"\n' "$RESULTS"
  printf 'export PYTHON="${PYTHON:-%s}"\n' "$PYTHON"
  printf 'export ENV_FILE="${ENV_FILE:-%s}"\n' "$ENV_FILE"
} > "$ENV_FILE"
echo "wrote $ENV_FILE - source it in every later cell (shell state does not survive between cells)"

say "dry run: the player set of the default Belgium PV game"
DATASET=belgium SERIES=pv_house_1 MODELS=Chronos2 \
  COVARIATES="hour dayofweek humidity temperature cloud_cover day_ahead_price soc" \
  ENV_FILE="$ENV_FILE" bash "$(dirname "$0")/02_run.sh" --dry-run || true

cat <<EOF

Setup done. Next:
  source $ENV_FILE
  bash $(dirname "$0")/02_run.sh                    # one (series, models) per call
  bash $(dirname "$0")/03_sweep.sh --dry-run        # print the whole roster
  bash $(dirname "$0")/03_sweep.sh                  # the roster, one model per process, logs kept
EOF
