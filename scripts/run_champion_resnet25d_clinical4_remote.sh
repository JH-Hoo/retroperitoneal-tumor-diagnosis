#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/retroperitoneal_tumor_diagnosis}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
IMAGE_ROOT="${IMAGE_ROOT:-$PROJECT/dataset_standard_v0}"
CHAMPION_MASK_DIR="${CHAMPION_MASK_DIR:-/root/autodl-tmp/flare23_champion_outputs}"
LABELS_CSV="${LABELS_CSV:-$PROJECT/data/labels_5class_holdout/all.csv}"
CHAMPION_STATS="${CHAMPION_STATS:-$PROJECT/models/flare23_champion_summary/champion_label14_stats.csv}"
MIN_TUMOR_VOXELS="${MIN_TUMOR_VOXELS:-5000}"
OUT_DIR="${OUT_DIR:-$PROJECT/models/champion_resnet25d_clinical4_multitask_minvox5000_cv5}"
REPORT_DIR="${REPORT_DIR:-$PROJECT/reports/champion_resnet25d_clinical4_minvox5000}"

cd "$PROJECT"
mkdir -p logs data/labels data/champion_flare23_25d_cache_15x224_minvox5000 models "$REPORT_DIR"

"$PYTHON_BIN" scripts/prepare_champion_minvox_labels.py \
  --labels-csv "$LABELS_CSV" \
  --champion-stats "$CHAMPION_STATS" \
  --out data/labels/champion_minvox5000.csv \
  --min-tumor-voxels "$MIN_TUMOR_VOXELS" \
  2>&1 | tee logs/prepare_champion_minvox_labels.log

"$PYTHON_BIN" scripts/build_flare23_25d_cache.py \
  --labels-csv data/labels/champion_minvox5000.csv \
  --image-root "$IMAGE_ROOT" \
  --mask-dir "$CHAMPION_MASK_DIR" \
  --out-root data/champion_flare23_25d_cache_15x224_minvox5000 \
  --image-size 224 \
  --num-slices 15 \
  --overwrite \
  2>&1 | tee logs/build_champion_flare23_25d_cache_minvox5000.log

"$PYTHON_BIN" scripts/train_resnet25d_clinical4_cv.py \
  --cache-root data/champion_flare23_25d_cache_15x224_minvox5000 \
  --out-dir "$OUT_DIR" \
  --weights imagenet \
  --mask-channel-init zero \
  --epochs 10 \
  --batch-size 4 \
  --num-workers 2 \
  --device cuda \
  --amp \
  2>&1 | tee logs/train_champion_resnet25d_clinical4_multitask_minvox5000_cv5.log

cp "$OUT_DIR"/summary.json "$REPORT_DIR"/
cp "$OUT_DIR"/oof_predictions.csv "$REPORT_DIR"/
cp "$OUT_DIR"/oof_predictions_derived_binary.csv "$REPORT_DIR"/
cp "$OUT_DIR"/oof_predictions_binary_head.csv "$REPORT_DIR"/
cp "$OUT_DIR"/resnet25d_clinical4_oof_confusion_matrix.png "$REPORT_DIR"/
cp "$OUT_DIR"/resnet25d_derived_binary_oof_confusion_matrix.png "$REPORT_DIR"/
cp "$OUT_DIR"/resnet25d_binary_head_oof_confusion_matrix.png "$REPORT_DIR"/
