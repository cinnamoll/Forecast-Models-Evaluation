#!/usr/bin/env bash
# Drive scripts/xai_timeshap.py one model at a time, so a killed session costs one
# model instead of the whole sweep.  The explainer has no checkpoint: it writes its
# csv/json files only after the last coalition, so the only resume granularity is
# "this model already wrote coalitions.json -> skip it".
#
# Usage
#   bash scripts/colab/02_run.sh --dry-run
#   bash scripts/colab/02_run.sh
#
# Knobs (env vars, all optional)
#   DATASET=belgium SERIES=pv_house_1 LEVEL=COV SAMPLING=25
#   MODELS="Chronos2 Timestral"        GPU=0 (a hosted runtime has one GPU)
#   PER_COVARIATE=1                    one player per covariate column, plus hist_d1..hist_d4
#                                      (PER_VARIABLE is accepted as the old name)
#   COVARIATE_ONLY=0                   1 = hold the target history fixed (2^C)
#   COVARIATES=""                      empty = the script default (TIMESHAP_COLUMNS, 5 columns);
#                                      Belgium wants the price and the state of charge:
#                                      COVARIATES="hour dayofweek humidity temperature cloud_cover day_ahead_price soc"
#                                      add temperature for 8 columns -> 12 players / 4096 coalitions
#   MAX_PLAYERS=""                     empty = the script default (12)
#
# Never call this through `conda run` on a hosted runtime: `conda run` keeps the
# child's stdout until the child exits, so a 9-minute game shows nothing at all and
# looks frozen.  This script uses "$PYTHON -u" and streams.
set -euo pipefail

[ -n "${ENV_FILE:-}" ] && [ -f "$ENV_FILE" ] && . "$ENV_FILE"
[ -f /content/timeshap_env.sh ] && . /content/timeshap_env.sh

PYTHON="${PYTHON:-python}"
_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${REPO_DIR:-$_SCRIPT_ROOT}"
RESULTS="${RESULTS:-/content/drive/MyDrive/timeshap_results}"
DATASET="${DATASET:-belgium}"
SERIES="${SERIES:-pv_house_1}"
LEVEL="${LEVEL:-COV}"
SAMPLING="${SAMPLING:-25}"
MODELS="${MODELS:-Chronos2 Timestral}"
GPU="${GPU:-0}"
PER_COVARIATE="${PER_COVARIATE:-${PER_VARIABLE:-1}}"
COVARIATE_ONLY="${COVARIATE_ONLY:-0}"
COVARIATES="${COVARIATES:-}"
MAX_PLAYERS="${MAX_PLAYERS:-}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$REPO_DIR"
TAG=$(printf '%.0f' "$SAMPLING")
SUFFIX=""
[ "$PER_COVARIATE" = "1" ] && SUFFIX="${SUFFIX}_pervar"
[ "$COVARIATE_ONLY" = "1" ] && SUFFIX="${SUFFIX}_covonly"
OUT="${RESULTS}/xai_timeshap/${DATASET}/${SERIES}_${LEVEL}_${TAG}${SUFFIX}"

flags=(--dataset "$DATASET" --series "$SERIES" --level "$LEVEL"
       --sampling "$SAMPLING" --gpu "$GPU" --out "$OUT")
[ "$PER_COVARIATE" = "1" ] && flags+=(--per-covariate)
[ "$COVARIATE_ONLY" = "1" ] && flags+=(--covariate-only)
[ -n "$MAX_PLAYERS" ] && flags+=(--max-players "$MAX_PLAYERS")
# Unquoted on purpose: --covariates takes a list of column names.
# shellcheck disable=SC2206
[ -n "$COVARIATES" ] && flags+=(--covariates $COVARIATES)

echo "python      : $PYTHON"
echo "result root : $OUT"
echo "flags       : ${flags[*]}"

for model in $MODELS; do
  echo
  echo "--------------------------------------------------------------- $model"
  if [ -f "$OUT/$model/coalitions.json" ]; then
    echo "skip: $OUT/$model/coalitions.json already exists"
    continue
  fi
  cmd=("$PYTHON" -u scripts/xai_timeshap.py --models "$model" "${flags[@]}")
  if [ "$DRY_RUN" = "1" ]; then
    cmd+=(--dry-run)
    echo "cmd: ${cmd[*]}"
    "${cmd[@]}"
    continue
  fi
  echo "cmd: ${cmd[*]}"
  echo "the game logs one line per coalition only at the end; the summary below is the progress marker"
  started=$SECONDS
  if ! "${cmd[@]}"; then
    echo "[FAILED] $model - see the traceback above."
    echo "         'N players -> 2^N coalitions' means the guard refused the game: raise MAX_PLAYERS,"
    echo "         set COVARIATE_ONLY=1 (16x fewer coalitions, history held fixed), or trim COVARIATES."
    echo "         'soc is not available' means that dataset has no soc file; only Belgium has it."
    exit 1
  fi
  "$PYTHON" - "$OUT/$model/coalitions.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
print("players       :", meta["n_players"], "->", meta["coalitions"], "coalitions")
print("player names  :", ", ".join(meta["players"]))
print("estimator     :", meta["estimator"], "| per_variable", meta.get("per_variable"),
      "| temporal", meta.get("temporal_players"))
print("background rot:", meta.get("background_rotation"), "| no-op dropped", meta.get("noop_dropped"))
print("seconds       :", meta["seconds"], "| additivity residual", meta["additivity_residual"])
PY
  echo "elapsed for $model: $((SECONDS - started)) s"
done

echo
echo "results under: $OUT"
