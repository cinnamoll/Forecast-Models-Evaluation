#!/usr/bin/env bash
# Shared bootstrap for the kaggle/colab wrappers.  Source it after the caller's defaults
# (`REPO_DIR`) are set: it puts conda on PATH and defines the results root to fall back to.
#
# Why it exists: a fresh Kaggle/Colab shell has no conda, and `bash scripts/kaggle/0X_*.sh`
# is run without sourcing env.sh first -- so every `conda run -n ...` died with
# "conda: command not found".  And `export RESULTS=...` typed in a notebook cell is a
# SyntaxError, so the results root is hardcoded here instead of exported by hand.
#
# Overridables: CONDA_HOME, RESULTS.  Sourced with `set -euo pipefail` active.

# 1) conda on PATH.  No-op when it is already reachable (env.sh sourced, /opt/conda image).
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

# 2) Results root: one folder for the whole run, no export needed.  On Kaggle it sits in
#    /kaggle/working (outside the repo, so the old results_covariate tree stays untouched);
#    everywhere else it sits next to the repo.  COV and L0 do not collide: the level is a
#    path component of the benchmark tree and part of the tuning JSON name.
if [ -d /kaggle/working ]; then
  KAGGLE_RESULTS=/kaggle/working/results_cov_new
else
  KAGGLE_RESULTS="${REPO_DIR:-/kaggle/working/Forecast-Models-Evaluation}/results_cov_new"
fi
