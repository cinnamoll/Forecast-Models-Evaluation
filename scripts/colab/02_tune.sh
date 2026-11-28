#!/usr/bin/env bash
# Transfer-learning step 1 on Colab: the Optuna search that a COV cell borrows from.
#
#   bash scripts/colab/02_tune.sh                     # TimeXer, TCN, DeepNPTS on 3 tuning sets
#   MODELS=TFT bash scripts/colab/02_tune.sh          # in the gluonts session
#   TRIALS=50 HOURS=3 bash scripts/colab/02_tune.sh
#   DRY=1 bash scripts/colab/02_tune.sh               # print the commands
#
# Writes $RESULTS/tuning/<Model>_<set>_COV_100_v2.json, which is exactly the file
# scripts/run_benchmark.py passes as --params_json to a COV cell (utils/tuning.py:
# germany->germany, london->zonnedael9, zonnedael->zonnedael9, belgium->belgium_pv1).
# Studies resume from the same sqlite file, so a second session adds trials.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Canonical repo root + a launcher python that can import pandas (see _common.sh).
# shellcheck source=/dev/null
source "$HERE/_common.sh"

export RESULTS="${RESULTS:-/content/drive/MyDrive/forecast-results}"
export MODELS="${MODEL:-${MODELS:-TimeXer TCN DeepNPTS}}"
export TUNE_SETS="${TUNE_SETS:-germany zonnedael9 belgium_pv1}"
export LEVEL="${LEVEL:-COV}"
export SAMPLING="${SAMPLING:-100}"
export TRIALS="${TRIALS:-30}"
export HOURS="${HOURS:-1.5}"
export MAX_STEPS="${MAX_STEPS:-100}"
export GPU="${GPU:-0}"
export DRY="${DRY:-0}"

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

echo "== tune: $MODELS | sets: $TUNE_SETS | $TRIALS trials, ${HOURS}h/study | out: $RESULTS/tuning"
exec bash "$HERE/../kaggle/03_tune.sh"
