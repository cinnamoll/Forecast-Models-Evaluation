#!/usr/bin/env bash
# Kaggle wrapper around the Colab setup.  Same explainer, different paths.
#
# Kaggle-specific things this handles:
#   * the code is not on the image: either attach the repo as a Kaggle Dataset
#     (Settings > Add data) or enable Internet (Settings > Internet on) so the
#     clone below works;
#   * /kaggle/input is read-only, so the working copy lives in /kaggle/working;
#   * results written under /kaggle/working are saved with "Save version";
#     /kaggle/temp is scratch and is discarded;
#   * the GPU is usually a T4 (bf16 emulated, but supported) or a P100 (sm_60, no
#     bf16 kernel) - prefer T4 for the Chronos2/Timestral bf16 path.
#
# Usage
#   !bash /kaggle/working/Forecast-Models-Evaluation/scripts/kaggle/01_setup.sh
#   INSTALL_TIMESFM=1 !bash ... (to also install TimesFM3)
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/cinnamoll/Forecast-Models-Evaluation.git}"
export REPO_DIR="${REPO_DIR:-/kaggle/working/Forecast-Models-Evaluation}"
export RESULTS="${RESULTS:-/kaggle/working/results}"
export ENV_FILE="${ENV_FILE:-/kaggle/working/timeshap_env.sh}"
export PYTHON="${PYTHON:-python}"

say() { printf '\n=== %s\n' "$*"; }
die() { printf '\n[FATAL] %s\n' "$*" >&2; exit 1; }

say "0/2 Kaggle preflight"
echo "pwd: $(pwd)"
[ -d /kaggle ] || die "this is not a Kaggle runtime; use scripts/colab/01_setup.sh instead"
[ -w /kaggle/working ] || die "/kaggle/working is not writable"
if [ -d /kaggle/input ]; then
  echo "attached datasets:"; ls /kaggle/input | sed 's/^/  /'
fi

say "repo"
if [ ! -d "$REPO_DIR/data" ]; then
  echo "no checkout with data/ at $REPO_DIR -> cloning $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR" || die "clone failed. Enable Internet (Settings > Internet on) and re-run, or attach the repo as a Kaggle Dataset and set REPO_DIR=/kaggle/input/<dataset>/Forecast-Models-Evaluation"
fi
cd "$REPO_DIR"
echo "repo: $REPO_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"

# The current explainer only exists after it has been pushed; the clone is checked
# for --covariates / TIMESHAP_COLUMNS by the shared setup below.

exec bash "$(dirname "$0")/../colab/01_setup.sh"
