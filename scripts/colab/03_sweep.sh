#!/usr/bin/env bash
# The whole TimeSHAP roster, one (series, model) per process, each with its own log.
#
# Sequential on purpose: a hosted runtime has one GPU, and running two games at once
# only halves both.  Every invocation writes log first and result last, so if the
# session dies the finished models are on disk and the next run skips them.
#
# Knobs
#   SAMPLING=25            one sampling rate per sweep
#   MODELS="Chronos2 Timestral TimesFM3"
#   TEMPERATURE=1           include `temperature` (Belgium -> 8 columns -> 12 players / 4096)
#   SERIES_LIST="london:london_load belgium:pv_house_1 ..."   override the roster
#   LOG_DIR=$RESULTS/xai_timeshap/logs
#
# Usage
#   bash scripts/colab/03_sweep.sh --dry-run      # print every command, run nothing
#   bash scripts/colab/03_sweep.sh
set -euo pipefail

[ -n "${ENV_FILE:-}" ] && [ -f "$ENV_FILE" ] && . "$ENV_FILE"
[ -f /content/timeshap_env.sh ] && . /content/timeshap_env.sh

_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS="${RESULTS:-/content/drive/MyDrive/timeshap_results}"
SAMPLING="${SAMPLING:-25}"
MODELS="${MODELS:-Chronos2 Timestral TimesFM3}"
TEMPERATURE="${TEMPERATURE:-0}"
LOG_DIR="${LOG_DIR:-$RESULTS/xai_timeshap/logs}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# dataset:series — the four benchmark datasets, every series the reports use.
DEFAULT_SERIES="london:london_load
germany:germany_load
zonnedael:zonnedael_customer_8
zonnedael:zonnedael_customer_9
zonnedael:zonnedael_customer_43
belgium:pv_house_1
belgium:pv_house_2
belgium:pv_house_3
belgium:pv_house_4
belgium:bess_house_1
belgium:bess_house_2
belgium:bess_house_3
belgium:bess_house_4"
SERIES_LIST="${SERIES_LIST:-$DEFAULT_SERIES}"

BELGIUM_COVARIATES="hour dayofweek humidity temperature cloud_cover day_ahead_price soc"
[ "$TEMPERATURE" = "1" ] && BELGIUM_COVARIATES="$BELGIUM_COVARIATES temperature"

mkdir -p "$LOG_DIR"
failed=0
skipped=0
done_count=0

echo "sampling : $SAMPLING"
echo "models   : $MODELS"
echo "logs     : $LOG_DIR"
echo "belgium covariates: $BELGIUM_COVARIATES"
echo

# The roster is word-split on purpose: newlines or spaces both work.
# shellcheck disable=SC2086
for entry in $SERIES_LIST; do
  [ -z "$entry" ] && continue
  dataset="${entry%%:*}"
  series="${entry##*:}"
  covariates=""
  [ "$dataset" = "belgium" ] && covariates="$BELGIUM_COVARIATES"
  for model in $MODELS; do
    tag=$(printf '%.0f' "$SAMPLING")
    out="$RESULTS/xai_timeshap/$dataset/${series}_COV_${tag}_pervar"
    if [ -f "$out/$model/coalitions.json" ]; then
      echo "SKIP  $dataset/$series $model (already done)"
      skipped=$((skipped + 1))
      continue
    fi
    log="$LOG_DIR/${dataset}_${series}_${model}.log"
    echo "RUN   $dataset/$series $model -> $log"
    if [ "$DRY_RUN" = "1" ]; then
      DATASET="$dataset" SERIES="$series" MODELS="$model" SAMPLING="$SAMPLING" \
      COVARIATES="$covariates" ENV_FILE="${ENV_FILE:-}" REPO_DIR="$_SCRIPT_ROOT" \
        bash "$(dirname "$0")/02_run.sh" --dry-run
      continue
    fi
    if DATASET="$dataset" SERIES="$series" MODELS="$model" SAMPLING="$SAMPLING" \
       COVARIATES="$covariates" ENV_FILE="${ENV_FILE:-}" REPO_DIR="$_SCRIPT_ROOT" \
       bash "$(dirname "$0")/02_run.sh" > "$log" 2>&1; then
      done_count=$((done_count + 1))
      tail -4 "$log" | sed 's/^/      /'
    else
      failed=$((failed + 1))
      echo "      FAILED - tail of $log:"
      tail -6 "$log" | sed 's/^/      /'
    fi
  done
done

echo
echo "done=$done_count skipped=$skipped failed=$failed"
echo "results: $RESULTS/xai_timeshap"
[ "$failed" -gt 0 ] && exit 1
exit 0
