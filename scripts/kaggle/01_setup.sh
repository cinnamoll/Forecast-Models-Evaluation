#!/usr/bin/env bash
# One-time setup for the COV benchmark on a hosted GPU (Kaggle, Colab).
#
# Installs Miniforge under the working dir, builds the two conda envs the four
# models need, checks the neuralforecast env really has a CUDA torch, writes a
# sourceable env file, and finishes with a dry run of the runner.
#
#   bash scripts/kaggle/01_setup.sh                 # install everything, then dry-run
#   INSTALL=0 bash scripts/kaggle/01_setup.sh       # skip installs (dev-box check)
#
# Overridables: REPO_DIR RESULTS ENV_FILE ENVS INSTALL CONDA_HOME
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/Forecast-Models-Evaluation}"
# Empty by default: each script then picks the root of its own --level
# (COV -> results_covariate, L0 -> results).  Set RESULTS to pin one root for both.
RESULTS="${RESULTS:-}"
ENV_FILE="${ENV_FILE:-/kaggle/working/env.sh}"
ENVS="${ENVS:-statsml chronos timesfm3}"
INSTALL="${INSTALL:-1}"
PYTHON="${PYTHON:-python}"
# Ephemeral by default: a conda tree in /kaggle/working is 5-8 GB against the 20 GB
# output cap and every Save Version.  Set CONDA_HOME=/kaggle/working/miniforge3 only
# if you would rather pay that than rebuild the envs each session.
CONDA_HOME="${CONDA_HOME:-/tmp/miniforge3}"

if [ ! -d "$REPO_DIR/requirements" ]; then
  echo "FAIL: no repo at REPO_DIR=$REPO_DIR" >&2
  echo "  git clone <your repo> $REPO_DIR   (Internet on), or attach it as a dataset" >&2
  echo "  and point REPO_DIR at /kaggle/input/<dataset>/Forecast-Models-Evaluation" >&2
  exit 1
fi

if [ "$INSTALL" = "1" ]; then
  if ! command -v conda >/dev/null 2>&1 && [ ! -x "$CONDA_HOME/bin/conda" ]; then
    echo "== installing miniforge into $CONDA_HOME (~2 min)"
    wget -q -O /tmp/miniforge.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_HOME"
  fi
  export PATH="$CONDA_HOME/bin:$PATH"
  for env in $ENVS; do
    if conda env list | awk 'NF {print $1}' | grep -qx "$env"; then
      echo "== env $env already present"
      continue
    fi
    echo "== creating env $env from requirements/env_$env.yml (~5-20 min)"
    conda env create -y -n "$env" -f "$REPO_DIR/requirements/env_$env.yml"
  done
  conda clean -afy >/dev/null 2>&1 || true
else
  export PATH="$CONDA_HOME/bin:$PATH"
fi

# The one failure that costs a whole session: a CPU-only torch in the env that runs
# 3 of the 4 models.  Do not reinstall torch from conda; it ships CUDA already.
TORCH_PROBE="${TORCH_PROBE:-1}"
if [ "$TORCH_PROBE" = "1" ] && command -v conda >/dev/null 2>&1 \
   && conda env list | awk 'NF {print $1}' | grep -qx neuralforecast; then
  NFPY="$(conda run -n neuralforecast python -c 'import sys; print(sys.executable)')"
  echo "== torch in neuralforecast:"
  "$NFPY" - <<'PY'
import torch
print(" ", torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.version.cuda is None:
    raise SystemExit(
        "FAIL: CPU-only torch. Fix with the *installed* version:\n"
        "  pip install --index-url https://download.pytorch.org/whl/cu121 "
        f"torch=={torch.__version__.split('+')[0]}+cu121"
    )
PY
else
  echo "== skipping torch probe: neuralforecast env not present"
fi

# 02_run.py imports utils/protocol.py (pandas) before it shells out to the model
# envs, and env.sh puts the conda *base* on PATH -- a base built by the installer has
# no pandas, only the model envs do.  Without this the first run dies at import.
if command -v conda >/dev/null 2>&1; then
  BASE_PY="$(conda info --base)/bin/python"
  if ! "$BASE_PY" -c "import pandas" >/dev/null 2>&1; then
    echo "== installing pandas into the conda base ($(conda info --base)) for scripts/kaggle/02_run.py"
    "$BASE_PY" -m pip install -q "pandas>=2,<4"
  fi
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

# Defaults, not assignments: a caller that exports RESULTS=/tmp/x keeps it.  RESULTS
# itself stays empty unless given, so each script picks its own level's root.
cat > "$ENV_FILE" <<EOF
export CONDA_HOME="\${CONDA_HOME:-$CONDA_HOME}"
export REPO_DIR="\${REPO_DIR:-$REPO_DIR}"
export RESULTS="\${RESULTS:-$RESULTS}"
export PATH="$CONDA_HOME/bin:\$PATH"
EOF
echo "== wrote $ENV_FILE (source it in every later cell)"

cd "$REPO_DIR"
if command -v conda >/dev/null 2>&1; then
  echo "== dry run: the commands the sweep would execute"
  "$PYTHON" scripts/kaggle/02_run.py --dry-run \
    --models LightGBM --level COV \
    --datasets "${DRY_DATASETS:-germany}" --samplings "${DRY_SAMPLINGS:-25}"
else
  echo "== conda absent: skipping the dry run"
fi
