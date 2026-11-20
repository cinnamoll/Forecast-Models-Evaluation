#!/usr/bin/env bash
# Kaggle wrapper: the whole roster, one (series, model) per process, logs kept.
#
#   !cd /kaggle/working/Forecast-Models-Evaluation && bash scripts/kaggle/03_sweep.sh --dry-run
#   !cd /kaggle/working/Forecast-Models-Evaluation && bash scripts/kaggle/03_sweep.sh
#
# Kaggle sessions are capped (12 h) and do not survive a disconnect, so prefer the
# sequence: setup -> one sweep at SAMPLING=25 -> "Save version" (that commits
# /kaggle/working/results) -> re-run to collect the remaining models.  Everything
# already finished is skipped.
set -euo pipefail

export RESULTS="${RESULTS:-/kaggle/working/results}"
export ENV_FILE="${ENV_FILE:-/kaggle/working/timeshap_env.sh}"
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

# LOG_DIR is left to 03_sweep.sh, which derives it from $RESULTS; override it here
# only when the logs should sit somewhere else (/kaggle/working keeps them).

exec bash "$(dirname "$0")/../colab/03_sweep.sh" "$@"
