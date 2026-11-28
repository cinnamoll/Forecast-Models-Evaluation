#!/usr/bin/env bash
# TimeSHAP on Colab: the NeuralForecast models (TimeXer, TCN, DeepNPTS), one env.
#
#   bash scripts/colab/04_xai_neuralforecast.sh              # all three, one dataset
#   MODEL=TimeXer bash scripts/colab/04_xai_neuralforecast.sh   # a single model
#   DATASETS="london" SERIES_LIST="london_load" DRY=1 bash scripts/colab/04_xai_neuralforecast.sh
#
# The roster (dataset -> series, per-model resume, --per-variable players, one game
# per model) lives in scripts/kaggle/04_xai.sh; this wrapper only pins the Colab
# paths, the env and the model list, so there is one implementation of the loop.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROSTER="$HERE/../kaggle/04_xai.sh"

# Canonical repo root (a second clone must not run the first one's copy).
# shellcheck source=/dev/null
source "$HERE/_common.sh"
export RESULTS="${RESULTS:-/content/drive/MyDrive/forecast-results}"
export MODELS="${MODEL:-${MODELS:-TimeXer TCN DeepNPTS}}"
export DATASETS="${DATASETS:-germany london}"

for model in $MODELS; do
  case "$model" in
    TimeXer|TCN|DeepNPTS) ;;
    *) echo "FAIL: $model is not a neuralforecast model of this comparison" >&2; exit 2 ;;
  esac
done

echo "== neuralforecast XAI: $MODELS | datasets: $DATASETS | out: $RESULTS/xai_timeshap"
exec bash "$ROSTER"
