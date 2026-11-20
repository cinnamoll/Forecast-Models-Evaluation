#!/usr/bin/env bash
# Kaggle wrapper: same driver as Colab, Kaggle defaults for RESULTS / ENV_FILE.
#
#   !cd /kaggle/working/Forecast-Models-Evaluation && \
#     DATASET=belgium SERIES=pv_house_2 MODELS=Chronos2 \
#     COVARIATES="hour dayofweek humidity temperature cloud_cover day_ahead_price soc" \
#     bash scripts/kaggle/02_run.sh
#
# Results land in /kaggle/working/results and are saved with "Save version".
# One model per invocation; a model whose coalitions.json exists is skipped.
set -euo pipefail

export RESULTS="${RESULTS:-/kaggle/working/results}"
export ENV_FILE="${ENV_FILE:-/kaggle/working/timeshap_env.sh}"
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

exec bash "$(dirname "$0")/../colab/02_run.sh" "$@"
