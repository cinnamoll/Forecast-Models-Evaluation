#!/usr/bin/env bash
# Drive scripts/xai_timeshap.py one model at a time, so a killed session costs one
# model instead of the whole sweep.  The explainer itself has no checkpoint: it
# writes its csv/json files only after the last coalition, so the only resume
# granularity available is "this model already wrote coalitions.json -> skip it".
#
# Usage
#   bash scripts/colab/02_run.sh --dry-run     # print the player set, run nothing
#   bash scripts/colab/02_run.sh               # real run
#
# Knobs (env vars, all optional)
#   DATASET=london SERIES=london_load LEVEL=COV SAMPLING=25
#   MODELS="Chronos2 Timestral"       GPU=0
#   PER_VARIABLE=1                    one player per column (vs one per block)
#   COVARIATE_ONLY=0                  1 = hold the target history fixed (2^C instead of 2^(C+4))
#   MAX_PLAYERS=16                    guard: refuse a game bigger than 2^MAX_PLAYERS
set -euo pipefail

[ -f /content/timeshap_env.sh ] && . /content/timeshap_env.sh

PYTHON="${PYTHON:-python}"
# Repo root: the caller's REPO_DIR if it exists, else the checkout this script lives in.
_SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${REPO_DIR:-$_SCRIPT_ROOT}"
RESULTS="${RESULTS:-/content/drive/MyDrive/timeshap_results}"
DATASET="${DATASET:-london}"
SERIES="${SERIES:-london_load}"
LEVEL="${LEVEL:-COV}"
SAMPLING="${SAMPLING:-25}"
MODELS="${MODELS:-Chronos2 Timestral}"
GPU="${GPU:-0}"
PER_VARIABLE="${PER_VARIABLE:-1}"
COVARIATE_ONLY="${COVARIATE_ONLY:-0}"
MAX_PLAYERS="${MAX_PLAYERS:-16}"

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
[ "$PER_VARIABLE" = "1" ] && SUFFIX="${SUFFIX}_pervar"
[ "$COVARIATE_ONLY" = "1" ] && SUFFIX="${SUFFIX}_covonly"
OUT="${RESULTS}/xai_timeshap/${DATASET}/${SERIES}_${LEVEL}_${TAG}${SUFFIX}"

flags=(--dataset "$DATASET" --series "$SERIES" --level "$LEVEL"
       --sampling "$SAMPLING" --gpu "$GPU"
       --max-players "$MAX_PLAYERS" --out "$OUT")
[ "$PER_VARIABLE" = "1" ] && flags+=(--per-variable)
[ "$COVARIATE_ONLY" = "1" ] && flags+=(--covariate-only)

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
  cmd=("$PYTHON" scripts/xai_timeshap.py --models "$model" "${flags[@]}")
  if [ "$DRY_RUN" = "1" ]; then
    cmd+=(--dry-run)
    echo "cmd: ${cmd[*]}"
    "${cmd[@]}"
    continue
  fi
  echo "cmd: ${cmd[*]}"
  started=$SECONDS
  if ! "${cmd[@]}"; then
    echo "[FAILED] $model - see the traceback above."
    echo "         A guard error ('N players -> 2^N coalitions') means this dataset has more columns"
    echo "         than --max-players allows: raise MAX_PLAYERS, or set COVARIATE_ONLY=1 (drops the"
    echo "         four history players, 16x fewer coalitions, but target_history is held fixed),"
    echo "         or drop the static_* columns from the frame before the game."
    exit 1
  fi
  "$PYTHON" - "$OUT/$model/coalitions.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
print("players       :", meta["n_players"], "->", meta["coalitions"], "coalitions")
print("player names  :", ", ".join(meta["players"]))
print("estimator     :", meta["estimator"], "| per_variable", meta.get("per_variable"),
      "| temporal", meta.get("temporal_players"))
print("seconds       :", meta["seconds"], "| additivity residual", meta["additivity_residual"])
PY
  echo "elapsed for $model: $((SECONDS - started)) s"
done

echo
echo "results under: $OUT"
