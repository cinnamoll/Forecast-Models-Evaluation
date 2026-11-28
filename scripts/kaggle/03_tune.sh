#!/usr/bin/env bash
# Optuna search for the four COV models on a hosted GPU, one study at a time.
#
#   bash scripts/kaggle/03_tune.sh                      # 4 models x 3 tuning sets @ 100%
#   MODELS="LightGBM" bash scripts/kaggle/03_tune.sh
#   TRIALS=50 HOURS=6 bash scripts/kaggle/03_tune.sh    # a longer budget
#   DRY=1 bash scripts/kaggle/03_tune.sh                # print the commands
#
# Writes <RESULTS>/tuning/<Model>_<set>_<LEVEL>_<tag>_v<SPACE_VERSION>.json, which is
# exactly where scripts/run_benchmark.py looks them up (utils/tuning.py).  Studies
# resume from the same sqlite file, so a second session adds trials instead of
# starting over; a study that already holds TRIALS trials is skipped.
set -euo pipefail

# Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline and the conda envs
# do not have matplotlib_inline, so any child that imports pyplot dies with
# "ValueError: Key backend: ... is not a valid value for backend".  Force the headless
# backend: every figure here is written with savefig anyway.
export MPLBACKEND=Agg

REPO_DIR="${REPO_DIR:-/kaggle/working/Forecast-Models-Evaluation}"
LEVEL="${LEVEL:-COV}"
# conda on PATH plus the hardcoded results root: a fresh cell has neither, and
# `export RESULTS=...` in a notebook cell is a SyntaxError.
. "$(dirname "$0")/_bootstrap.sh"
RESULTS="${RESULTS:-$KAGGLE_RESULTS}"
SAMPLING="${SAMPLING:-100}"
MODELS="${MODELS:-LightGBM}"
TUNE_SETS="${TUNE_SETS:-germany zonnedael9 belgium_pv1}"
TRIALS="${TRIALS:-30}"
HOURS="${HOURS:-1.5}"
MAX_STEPS="${MAX_STEPS:-100}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
DRY="${DRY:-0}"
PYTHON="${PYTHON:-python}"

TUNING_DIR="$RESULTS/tuning"
LOG_DIR="$TUNING_DIR/logs"
mkdir -p "$LOG_DIR"

env_for() {  # model -> conda env, the same map scripts/run_benchmark.py uses
  case "$1" in
    LightGBM) echo statsml ;;
    *) echo "unknown model $1 (tunable: LightGBM)" >&2; return 2 ;;
  esac
}

have_trials() {  # <json> <wanted> -> 0 when the file already holds enough trials
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
path, wanted = Path(sys.argv[1]), int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
try:
    used = json.loads(path.read_text(encoding="utf-8"))["used"]["trials"]
except (KeyError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if used >= wanted else 1)
PY
}

done_count=0
skipped=0
failed=0
for model in $MODELS; do
  for set in $TUNE_SETS; do
    json="$TUNING_DIR/${model}_${set}_${LEVEL}_${SAMPLING}_v2.json"
    log="$LOG_DIR/${model}_${set}_${LEVEL}_${SAMPLING}.log"
    if have_trials "$json" "$TRIALS"; then
      echo "SKIP  $model/$set: $TRIALS trials already in $(basename "$json")"
      skipped=$((skipped + 1))
      continue
    fi
    command=(conda run --no-capture-output -n "$(env_for "$model")" python
             scripts/tune_optuna.py --model "$model" --dataset "$set" --level "$LEVEL"
             --sampling "$SAMPLING" --trials "$TRIALS" --hours "$HOURS"
             --max-steps "$MAX_STEPS" --seed "$SEED" --gpu "$GPU" --out-dir "$TUNING_DIR")
    echo "TUNE  $model/$set -> $json"
    echo "      ${command[*]}"
    if [ "$DRY" = "1" ]; then
      done_count=$((done_count + 1))
      continue
    fi
    cd "$REPO_DIR"
    if "${command[@]}" >"$log" 2>&1; then
      tail -n 3 "$log" | sed 's/^/      /'
      done_count=$((done_count + 1))
    else
      echo "FAIL  $model/$set: see $log"
      failed=$((failed + 1))
    fi
  done
done
echo "tuning: $done_count started/finished, $skipped skipped, $failed failed"
[ "$failed" -eq 0 ]
