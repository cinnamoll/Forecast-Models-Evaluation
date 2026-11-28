#!/usr/bin/env bash
# Put conda on PATH for the kaggle/colab wrappers.
#
# A fresh Kaggle/Colab shell has no conda, and `bash scripts/kaggle/03_tune.sh` is
# normally run without sourcing env.sh first, so every `conda run -n ...` died with
# "conda: command not found".  Source this at the top of a wrapper: it is a no-op
# when conda is already reachable (env.sh sourced, or the image ships /opt/conda).
#
# Overridables: CONDA_HOME.  Sourced with `set -euo pipefail` active, so no unset vars.
CONDA_HOME="${CONDA_HOME:-/tmp/miniforge3}"
if ! command -v conda >/dev/null 2>&1; then
  for cand in "$CONDA_HOME" /opt/conda "$HOME/miniforge3" "$HOME/anaconda3" \
              "$HOME/miniconda3" /kaggle/working/miniforge3; do
    if [ -x "$cand/bin/conda" ]; then
      CONDA_HOME="$cand"
      PATH="$cand/bin:$PATH"
      export CONDA_HOME PATH
      break
    fi
  done
fi
