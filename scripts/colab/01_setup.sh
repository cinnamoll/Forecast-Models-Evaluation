#!/usr/bin/env bash
# Colab setup for the TimeSHAP comparison: one conda env per session.
#
#   bash scripts/colab/01_setup.sh neuralforecast     # TimeXer, TCN, DeepNPTS
#   bash scripts/colab/01_setup.sh gluonts            # TFT (python 3.7 + mxnet cu112)
#
# Installs Miniconda under /content (Colab wipes it when the runtime ends), builds
# only the env you name, checks the torch build in it, writes a source-able env
# file, and finishes with a dry run of the matching XAI script.
#
# Overridables: REPO_DIR RESULTS ENV_FILE CONDA_HOME INSTALL TORCH_PROBE PYTHON
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/Forecast-Models-Evaluation}"
RESULTS="${RESULTS:-/content/drive/MyDrive/forecast-results}"
ENV_FILE="${ENV_FILE:-/content/env.sh}"
CONDA_HOME="${CONDA_HOME:-/content/miniconda3}"
ENVS="${1:-${ENVS:-neuralforecast}}"
INSTALL="${INSTALL:-1}"
TORCH_PROBE="${TORCH_PROBE:-1}"
PYTHON="${PYTHON:-python}"

if [ ! -d "$REPO_DIR/requirements" ]; then
  echo "FAIL: no repo at REPO_DIR=$REPO_DIR" >&2
  echo "  !git clone https://github.com/<user>/Forecast-Models-Evaluation $REPO_DIR" >&2
  exit 1
fi

if [ "$INSTALL" = "1" ]; then
  # Colab exports PYTHONPATH; a conda env create then picks up base packages by
  # accident and conda warns about it.  Nothing in this repo needs it.
  unset PYTHONPATH
  if [ ! -x "$CONDA_HOME/bin/conda" ]; then
    # Miniforge, not Miniconda: repo.anaconda.com now refuses `conda env create`
    # until its Terms of Service are accepted ("CondaToSNonInteractiveError"), and
    # Miniforge ships conda-forge only, which has no such gate.  Every requirement
    # file here lists conda-forge/pytorch/nvidia explicitly.
    echo "== installing miniforge into $CONDA_HOME (~2 min)"
    wget -q -O /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_HOME"
  fi
  export PATH="$CONDA_HOME/bin:$PATH"
  # Belt and braces: if a marker or a stale config still reaches repo.anaconda.com,
  # accept the two channels non-interactively (no-op when they are not configured).
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    >/dev/null 2>&1 || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    >/dev/null 2>&1 || true
  # An older Miniconda left over from a previous cell has no `conda tos`; say so
  # instead of letting `conda env create` fail with the ToS error again.
  if ! conda tos list >/dev/null 2>&1 && command -v conda | grep -q miniconda; then
    echo "NOTE: this conda predates 'conda tos'. If env creation fails on Terms of Service:"
    echo "      rm -rf $CONDA_HOME   # then re-run this script, it now installs miniforge"
  fi
  for env in $ENVS; do
    if conda env list | awk 'NF {print $1}' | grep -qx "$env"; then
      echo "== env $env already present"
      continue
    fi
    echo "== creating env $env (~5-20 min; the gluonts one is the slowest)"
    conda env create -y -n "$env" -f "$REPO_DIR/requirements/env_$env.yml"
  done
  conda clean -afy >/dev/null 2>&1 || true
else
  export PATH="$CONDA_HOME/bin:$PATH"
fi

# A CPU-only torch in neuralforecast means every nf game dies at the first cuda:0.
if [ "$TORCH_PROBE" = "1" ] && command -v conda >/dev/null 2>&1 \
   && conda env list | awk 'NF {print $1}' | grep -qx neuralforecast; then
  NFPY="$(conda run -n neuralforecast python -c 'import sys; print(sys.executable)')"
  echo "== torch in neuralforecast:"
  "$NFPY" - <<'PY'
import torch
print(" ", torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.version.cuda is None:
    raise SystemExit(
        "FAIL: CPU-only torch. Fix with the installed version, not conda:\n"
        "  pip install --index-url https://download.pytorch.org/whl/cu121 "
        f"torch=={torch.__version__.split('+')[0]}+cu121"
    )
PY
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
mkdir -p "$RESULTS"

# Defaults, not assignments: a caller exporting RESULTS=/content/results keeps it.
cat > "$ENV_FILE" <<EOF
export CONDA_HOME="\${CONDA_HOME:-$CONDA_HOME}"
export REPO_DIR="\${REPO_DIR:-$REPO_DIR}"
export RESULTS="\${RESULTS:-$RESULTS}"
export PATH="$CONDA_HOME/bin:\$PATH"
EOF
echo "== wrote $ENV_FILE (source it in every later cell)"

cd "$REPO_DIR"
case "$ENVS" in
  *gluonts*) script=scripts/colab/05_xai_gluonts.sh ;;
  *) script=scripts/colab/04_xai_neuralforecast.sh ;;
esac
if command -v conda >/dev/null 2>&1; then
  echo "== dry run: $script"
  DRY=1 PYTHON="$PYTHON" REPO_DIR="$REPO_DIR" RESULTS="$RESULTS" bash "$script"
else
  echo "== conda absent: skipping the dry run"
fi
