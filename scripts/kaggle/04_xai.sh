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
# Cost: the game is enumerated exactly, so one run is 2^players model calls.  With
# --per-variable and the five covariate players that is 2^5 = 32 per model, which is
# what makes the five-model comparison cheap; adding temporal players multiplies it
# by 2^4.  Pass XAI_DRY_CHECK=1 first to print the real count.
set -euo pipefail

# Colab exports MPLBACKEND=module://matplotlib_inline.backend_inline and the conda envs
# do not have matplotlib_inline, so any child that imports pyplot dies with
# "ValueError: Key backend: ... is not a valid value for backend".  Force the headless
# backend: every figure here is written with savefig anyway.
export MPLBACKEND=Agg

REPO_DIR="${REPO_DIR:-/kaggle/working/Forecast-Models-Evaluation}"
RESULTS="${RESULTS:-/kaggle/working/results}"
XAI_OUT="${XAI_OUT:-$RESULTS/xai_timeshap}"
LEVEL="${LEVEL:-COV}"
SAMPLING="${SAMPLING:-100}"
MODE="${MODE:-background}"
COVARIATES="${COVARIATES:-hour dayofweek humidity temperature cloud_cover}"
PER_VARIABLE="${PER_VARIABLE:-1}"
COVARIATE_ONLY="${COVARIATE_ONLY:-1}"
MAX_PLAYERS="${MAX_PLAYERS:-12}"
EPOCHS="${EPOCHS:-100}"
GPU="${GPU:-0}"
DRY="${DRY:-0}"
XAI_DRY_CHECK="${XAI_DRY_CHECK:-0}"
PYTHON="${PYTHON:-python}"

MODELS="${MODELS:-LightGBM TCN DeepNPTS TimeXer TFT}"
DATASETS="${DATASETS:-germany london}"
LOG_DIR="$XAI_OUT/logs"
mkdir -p "$LOG_DIR"

env_for() {  # model -> conda env, matching the docstring examples of xai_timeshap.py
  case "$1" in
    Chronos2|Timestral) echo chronos ;;
    TimesFM3) echo timesfm3 ;;
    LightGBM) echo statsml ;;
    TFT) echo gluonts ;;
    *) echo neuralforecast ;;
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
               --covariates $COVARIATES --max-players "$MAX_PLAYERS" --out "$out")
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
