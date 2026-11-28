#!/usr/bin/env bash
# TimeSHAP on Colab: TFT, in its own session and its own conda env.
#
#   bash scripts/colab/05_xai_gluonts.sh
#   DATASETS="london" SERIES_LIST="london_load" DRY=1 bash scripts/colab/05_xai_gluonts.sh
#
# Why alone: env ``gluonts`` is python 3.7 + mxnet-cu112 (the slowest, most fragile
# build in the repo), and the TFT predictor trains once per cell before the game —
# it does not share a session comfortably with three NeuralForecast fits.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROSTER="$HERE/../kaggle/04_xai.sh"

# Canonical repo root (a second clone must not run the first one's copy).
# shellcheck source=/dev/null
source "$HERE/_common.sh"
export RESULTS="${RESULTS:-/content/drive/MyDrive/forecast-results}"
export MODELS="${MODEL:-TFT}"
export DATASETS="${DATASETS:-germany london}"

if [ "$MODELS" != "TFT" ]; then
  echo "FAIL: this wrapper only queues TFT (got '$MODELS'); use 02_xai_neuralforecast.sh" >&2
  exit 2
fi

echo "== gluonts XAI: TFT | datasets: $DATASETS | out: $RESULTS/xai_timeshap"
exec bash "$ROSTER"
