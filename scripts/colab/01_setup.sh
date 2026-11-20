#!/usr/bin/env bash
# Colab / remote-GPU setup for the TimeSHAP per-variable run.
#
# Colab's own python already ships a CUDA build of torch, so this installs the
# forecasting packages only - never torch.  Overriding torch on a hosted runtime
# is how a CUDA-enabled notebook silently becomes a CPU one.
#
#   COLAB:      bash scripts/colab/01_setup.sh
#   local test: PYTHON=python INSTALL=0 bash scripts/colab/01_setup.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"
_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${REPO_DIR:-$_SCRIPT_ROOT}"
RESULTS="${RESULTS:-/content/drive/MyDrive/timeshap_results}"
INSTALL="${INSTALL:-1}"
INSTALL_TIMESFM="${INSTALL_TIMESFM:-0}"
ENV_FILE="${ENV_FILE:-/content/timeshap_env.sh}"

say() { printf '\n=== %s\n' "$*"; }
die() { printf '\n[FATAL] %s\n' "$*" >&2; exit 1; }

say "1/6 accelerator"
command -v nvidia-smi >/dev/null || die "no nvidia-smi: pick a GPU runtime (Runtime > Change runtime type > GPU)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

say "2/6 torch must already be a CUDA build"
"$PYTHON" - <<'PY' || die "torch cannot see the GPU. Do NOT pip install torch here; on Colab run 'Runtime > Disconnect and delete runtime' and start again on a GPU runtime, so the preinstalled CUDA wheel is used."
import sys
import torch
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("cuda available", torch.cuda.is_available(), "| devices", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(1)
free, total = torch.cuda.mem_get_info()
print("device", torch.cuda.get_device_name(0), "| VRAM total", round(total / 2**30, 1), "GiB")
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
  fi
else
  echo "INSTALL=0 -> skipping pip"
fi

say "4/6 imports"
"$PYTHON" - <<'PY'
import importlib
for module, symbol in (("chronos", "Chronos2Pipeline"), ("timestral", "TimestralForecaster")):
    mod = importlib.import_module(module)
    assert hasattr(mod, symbol), f"{module} has no {symbol}"
    print(f"ok {module}.{symbol}")
for module in ("pandas", "numpy"):
    print("ok", module, importlib.import_module(module).__version__)
PY

say "5/6 repo must carry the per-variable TimeSHAP code"
if ! "$PYTHON" scripts/xai_timeshap.py --help 2>/dev/null | grep -q -- "--per-variable"; then
  die "scripts/xai_timeshap.py has no --per-variable. Push the local changes first (git push), then re-clone; the GitHub copy is behind."
fi
"$PYTHON" scripts/xai_timeshap.py --help 2>/dev/null | grep -E -- "--per-variable|--max-players|--covariate-only|--gpu" | sed 's/^ */  /'

say "6/6 data present + result root"
test -d data || die "no data/ directory: the clone is incomplete (data/ is tracked, 41 files)"
echo "data files: $(find data -type f | wc -l), size: $(du -sh data | cut -f1)"
mkdir -p "$RESULTS"
printf 'export REPO_DIR=%q\nexport RESULTS=%q\nexport PYTHON=%q\n' "$REPO_DIR" "$RESULTS" "$PYTHON" > "$ENV_FILE"
echo "wrote $ENV_FILE (source it in every later cell: shell state does not survive between cells)"

say "dry run: which players this game will use"
bash "$(dirname "$0")/02_run.sh" --dry-run

cat <<EOF

Setup done. Next:
  source $ENV_FILE
  bash scripts/colab/02_run.sh            # real run, one model per invocation
EOF
