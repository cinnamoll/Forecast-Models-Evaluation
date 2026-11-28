#!/usr/bin/env bash
# TimeSHAP explanations via scripts/xai_timeshap.py, one (dataset, series, model) at a time.
#
#   bash scripts/kaggle/04_xai.sh                      # 2 datasets x their series x 3 checkpoints
#   DRY=1 bash scripts/kaggle/04_xai.sh                # print the commands, run nothing
#   DATASETS="london" SERIES_LIST="london_load" MODELS="Chronos2" bash scripts/kaggle/04_xai.sh
#
# Only these have a predictor: the three checkpoints plus the five trainable models
# (utils/xai_predictors.py re-scores a fitted model per coalition, no refit).
# Keep MODELS to one conda env per invocation if a cell must be reproducible alone.
#
# Players are one per COVARIATE BLOCK by default (BLOCKS=1): the benchmark's own
# covariate set (hour, dayofweek, 5 weather columns, plus price/state on
# Belgium) forms 5-7 blocks, so the exact game is 2^5..2^7 = 32..128 calls per
# (dataset, series, model).  PER_VARIABLE=1 switches to one player per column,
# which is 2^7..2^9 = 128..512 on the current 7-9 column COV frame.
# XAI_DRY_CHECK=1 prints the real counts without running anything.
set -euo pipefail

# Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline and the conda envs
# do not have matplotlib_inline, so any child that imports pyplot dies with
# "ValueError: Key backend: ... is not a valid value for backend".  Force the headless
# backend: every figure here is written with savefig anyway.
export MPLBACKEND=Agg

REPO_DIR="${REPO_DIR:-/kaggle/working/Forecast-Models-Evaluation}"
LEVEL="${LEVEL:-COV}"
# conda on PATH plus the same results root 02_run.py / 03_tune.sh use, so the tuned JSONs
# handed to --params-dir are the ones the tuner wrote.
. "$(dirname "$0")/_bootstrap.sh"
RESULTS="${RESULTS:-$KAGGLE_RESULTS}"
XAI_OUT="${XAI_OUT:-$RESULTS/xai_timeshap}"
SAMPLING="${SAMPLING:-25}"
MODE="${MODE:-background}"
BLOCKS="${BLOCKS:-1}"
if [ "$BLOCKS" = "1" ]; then
  # Block players over the full COV set: the same covariates the benchmark cells run.
  ALL_COVARIATES="${ALL_COVARIATES:-1}"
  PER_VARIABLE="${PER_VARIABLE:-0}"
else
  ALL_COVARIATES="${ALL_COVARIATES:-0}"
  PER_VARIABLE="${PER_VARIABLE:-1}"
fi
COVARIATES="${COVARIATES:-hour dayofweek humidity temperature cloud_cover}"
COVARIATE_ONLY="${COVARIATE_ONLY:-1}"
MAX_PLAYERS="${MAX_PLAYERS:-12}"
EPOCHS="${EPOCHS:-100}"
GPU="${GPU:-0}"
DRY="${DRY:-0}"
XAI_DRY_CHECK="${XAI_DRY_CHECK:-0}"
PYTHON="${PYTHON:-python}"

MODELS="${MODELS:-LightGBM Chronos2 Timestral TimesFM3}"
DATASETS="${DATASETS:-germany london zonnedael belgium}"
LOG_DIR="$XAI_OUT/logs"
mkdir -p "$LOG_DIR"

env_for() {  # model -> conda env, matching the docstring examples of xai_timeshap.py
  case "$1" in
    Chronos2|Timestral) echo chronos ;;
    TimesFM3) echo timesfm3 ;;
    LightGBM) echo statsml ;;
    *) echo "unknown model $1 (explainable: LightGBM, Chronos2, Timestral, TimesFM3)" >&2; return 2 ;;
  esac
}

series_for() {  # dataset -> the series ids load_long_frame accepts
  case "$1" in
    belgium) echo "pv_house_1 pv_house_2 pv_house_3 pv_house_4 bess_house_1 bess_house_2 bess_house_3 bess_house_4" ;;
    germany) echo germany_load ;;
    london) echo london_load ;;
    zonnedael) echo "zonnedael_customer_8 zonnedael_customer_9 zonnedael_customer_43" ;;
    *) echo "" ;;
  esac
}

tag_of() {  # dataset grid tag used in the framework's own output name
  echo "$(awk -v s="$SAMPLING" 'BEGIN{printf "%d", s}')"
}

started=0
skipped=0
failed=0
for dataset in $DATASETS; do
  versions="$(series_for "$dataset")"
  if [ -z "$versions" ]; then
    echo "SKIP  $dataset: unknown dataset (known: belgium germany london zonnedael)"
    skipped=$((skipped + 1))
    continue
  fi
  series_list="${SERIES_LIST:-$versions}"
  for series in $series_list; do
    for model in $MODELS; do
      out="$XAI_OUT/$dataset/${series}_${LEVEL}_$(tag_of)"
      marker="$out/$model/coalitions.json"
      log="$LOG_DIR/${dataset}_${series}_${model}.log"
      if [ -f "$marker" ]; then
        echo "SKIP  $dataset/$series/$model: $(basename "$marker") on disk"
        skipped=$((skipped + 1))
        continue
      fi
      command=(conda run --no-capture-output -n "$(env_for "$model")" python scripts/xai_timeshap.py
               --dataset "$dataset" --series "$series" --models "$model" --level "$LEVEL"
               --sampling "$SAMPLING" --mode "$MODE" --epochs "$EPOCHS" --gpu "$GPU"
               --max-players "$MAX_PLAYERS" --out "$out" --params-dir "$RESULTS/tuning")
      [ "$ALL_COVARIATES" = "1" ] && command+=(--all-covariates) || command+=(--covariates $COVARIATES)
      [ "$PER_VARIABLE" = "1" ] && command+=(--per-variable)
      [ "$COVARIATE_ONLY" = "1" ] && command+=(--covariate-only)
      [ "$XAI_DRY_CHECK" = "1" ] && command+=(--dry-run)
      echo "XAI   $dataset/$series/$model -> $out/$model"
      echo "      ${command[*]}"
      if [ "$DRY" = "1" ]; then
        started=$((started + 1))
        continue
      fi
      cd "$REPO_DIR"
      if "${command[@]}" >"$log" 2>&1; then
        grep -E "players, .* coalitions|residual" "$log" | tail -n 2 | sed 's/^/      /' || true
        started=$((started + 1))
      else
        echo "FAIL  $dataset/$series/$model: see $log"
        failed=$((failed + 1))
      fi
    done
  done
done
echo "xai: $started run, $skipped skipped, $failed failed"
[ "$failed" -eq 0 ]
