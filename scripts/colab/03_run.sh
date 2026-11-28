#!/usr/bin/env bash
# Transfer-learning step 2 on Colab: the COV benchmark cells, with --gpu appended.
#
#   bash scripts/colab/05_run.sh                                   # TimeXer, TCN, DeepNPTS
#   MODELS=TFT bash scripts/colab/05_run.sh                        # in the gluonts session
#   DATASETS="germany" SAMPLINGS="25" bash scripts/colab/05_run.sh  # a cheap cell first
#   DRY=1 bash scripts/colab/05_run.sh                             # print the commands
#
# "Transfer learning" is the repo's own flow: every cell runs on the parameters found
# by scripts/tune_optuna.py on three tuning series (utils/tuning.py maps the rest),
# and a cell with no tuned file falls back to the paper defaults with a printed note.
# Results land in $RESULTS, so Drive keeps them across Colab sessions.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-/content/Forecast-Models-Evaluation}"
export RESULTS="${RESULTS:-/content/drive/MyDrive/forecast-results}"
export MODELS="${MODEL:-${MODELS:-TimeXer TCN DeepNPTS}}"
DATASETS="${DATASETS:-}"
SAMPLINGS="${SAMPLINGS:-}"
LEVEL="${LEVEL:-COV}"
GPU="${GPU:-0}"
DRY="${DRY:-0}"
FORCE="${FORCE:-0}"
PYTHON="${PYTHON:-python}"

env_for() {
  case "$1" in
    TFT) echo gluonts ;;
    LightGBM) echo statsml ;;
    *) echo neuralforecast ;;
  esac
}

for model in $MODELS; do
  env="$(env_for "$model")"
  if ! conda env list | awk 'NF {print $1}' | grep -qx "$env"; then
    echo "FAIL: env '$env' missing for $model" >&2
    echo "  bash scripts/colab/01_setup.sh $env" >&2
    exit 2
  fi
done

arguments=(--models $MODELS --level "$LEVEL" --results-dir "$RESULTS")
[ -n "$DATASETS" ] && arguments+=(--datasets $DATASETS)
[ -n "$SAMPLINGS" ] && arguments+=(--samplings $SAMPLINGS)
[ "$FORCE" = "1" ] && arguments+=(--force)
[ "$DRY" = "1" ] && arguments+=(--dry-run)

echo "== run COV: $MODELS | datasets: ${DATASETS:-all} | samplings: ${SAMPLINGS:-25 33 50 100} | out: $RESULTS"
cd "$REPO_DIR"
exec "$PYTHON" scripts/kaggle/02_run.py "${arguments[@]}"
