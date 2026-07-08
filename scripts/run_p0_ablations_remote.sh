#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/retroperitoneal_tumor_diagnosis}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
IMAGE_ROOT="${IMAGE_ROOT:-$PROJECT/dataset_standard_v0}"
CHAMPION_MASK_DIR="${CHAMPION_MASK_DIR:-/root/autodl-tmp/flare23_champion_outputs}"
LABELS_CSV="${LABELS_CSV:-$PROJECT/data/labels_5class_holdout/all.csv}"
CHAMPION_STATS="${CHAMPION_STATS:-$PROJECT/models/flare23_champion_summary/champion_label14_stats.csv}"
FORCE_REBUILD_CACHE="${FORCE_REBUILD_CACHE:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

cd "$PROJECT"
mkdir -p logs data/labels models/ablations reports/ablations

prepare_cache() {
  local minvox="$1"
  local label_csv="data/labels/champion_minvox${minvox}.csv"
  local cache_root="data/champion_flare23_25d_cache_15x224_minvox${minvox}"
  if [[ "$FORCE_REBUILD_CACHE" != "1" && -f "$cache_root/all.csv" ]]; then
    echo "[cache] reuse $cache_root"
    return 0
  fi
  if [[ ! -f "$cache_root/all.csv" ]]; then
    rm -rf "$cache_root"
  fi
  echo "[cache] build minvox=$minvox -> $cache_root"
  "$PYTHON_BIN" scripts/prepare_champion_minvox_labels.py \
    --labels-csv "$LABELS_CSV" \
    --champion-stats "$CHAMPION_STATS" \
    --out "$label_csv" \
    --min-tumor-voxels "$minvox" \
    2>&1 | tee "logs/prepare_champion_minvox${minvox}.log"

  local reuse_args=()
  if [[ "$minvox" != "5000" && -f data/champion_flare23_25d_cache_15x224_minvox5000/all.csv ]]; then
    reuse_args+=(--reuse-cache-root data/champion_flare23_25d_cache_15x224_minvox5000)
  fi
  if [[ "$minvox" == "0" && -f data/champion_flare23_25d_cache_15x224_minvox1000/all.csv ]]; then
    reuse_args+=(--reuse-cache-root data/champion_flare23_25d_cache_15x224_minvox1000)
  fi
  "$PYTHON_BIN" scripts/build_flare23_25d_cache.py \
    --labels-csv "$label_csv" \
    --image-root "$IMAGE_ROOT" \
    --mask-dir "$CHAMPION_MASK_DIR" \
    --out-root "$cache_root" \
    --image-size 224 \
    --num-slices 15 \
    "${reuse_args[@]}" \
    2>&1 | tee "logs/build_champion_flare23_25d_cache_minvox${minvox}.log"
}

copy_report() {
  local out_dir="$1"
  local report_dir="$2"
  mkdir -p "$report_dir"
  cp "$out_dir"/summary.json "$report_dir"/
  cp "$out_dir"/oof_predictions.csv "$report_dir"/
  cp "$out_dir"/oof_predictions_derived_binary.csv "$report_dir"/
  cp "$out_dir"/oof_predictions_binary_head.csv "$report_dir"/
  cp "$out_dir"/*confusion_matrix.png "$report_dir"/
}

run_resnet() {
  local name="$1"
  local minvox="$2"
  shift 2
  local cache_root="data/champion_flare23_25d_cache_15x224_minvox${minvox}"
  local out_dir="models/ablations/${name}"
  local report_dir="reports/ablations/${name}"
  if [[ "$FORCE_TRAIN" != "1" && -f "$report_dir/summary.json" ]]; then
    echo "[train] reuse $name"
    return 0
  fi
  echo "[train] $name"
  rm -rf "$out_dir" "$report_dir"
  "$PYTHON_BIN" scripts/train_resnet25d_clinical4_cv.py \
    --cache-root "$cache_root" \
    --out-dir "$out_dir" \
    --weights imagenet \
    --epochs 10 \
    --batch-size 4 \
    --num-workers 2 \
    --device cuda \
    --amp \
    "$@" \
    2>&1 | tee "logs/train_p0_${name}.log"
  copy_report "$out_dir" "$report_dir"
}

run_aux() {
  local name="$1"
  local minvox="$2"
  local cache_root="data/champion_flare23_25d_cache_15x224_minvox${minvox}"
  local out_dir="models/ablations/${name}"
  local report_dir="reports/ablations/${name}"
  if [[ "$FORCE_TRAIN" != "1" && -f "$report_dir/summary.json" ]]; then
    echo "[train] reuse $name"
    return 0
  fi
  echo "[train] $name"
  rm -rf "$out_dir" "$report_dir"
  "$PYTHON_BIN" scripts/train_aux_clinical4_cv.py \
    --cache-root "$cache_root" \
    --out-dir "$out_dir" \
    --folds 5 \
    2>&1 | tee "logs/train_p0_${name}.log"
  copy_report "$out_dir" "$report_dir"
}

prepare_cache 5000
prepare_cache 1000
prepare_cache 0

run_aux aux_only_minvox5000 5000
run_resnet no_aux_minvox5000 5000 --channel-set all --mask-channel-init zero --no-aux
run_resnet ct_only_noaux_minvox5000 5000 --channel-set ct_only --mask-channel-init zero --no-aux
run_resnet ct_tumor_noaux_minvox5000 5000 --channel-set ct_tumor --mask-channel-init zero --no-aux
run_resnet ct_tumor_shell_noaux_minvox5000 5000 --channel-set ct_tumor_shell --mask-channel-init zero --no-aux
run_resnet maskinit_small_minvox5000 5000 --channel-set all --mask-channel-init small
run_resnet maskinit_mean_minvox5000 5000 --channel-set all --mask-channel-init mean
run_resnet full_aux_minvox1000 1000 --channel-set all --mask-channel-init zero
run_resnet full_aux_minvox0 0 --channel-set all --mask-channel-init zero

echo "[done] P0 ablations written to reports/ablations"
