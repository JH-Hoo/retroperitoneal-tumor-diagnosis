#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/retroperitoneal_tumor_diagnosis}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT/data/champion_flare23_25d_cache_15x224_minvox5000}"
OUT_DIR="${OUT_DIR:-$PROJECT/reports/pseudo_seg25d_clinical4_minvox5000}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

cd "$PROJECT"
mkdir -p logs reports

if [[ "$FORCE_TRAIN" != "1" && -f "$OUT_DIR/summary.json" ]]; then
  echo "[pseudo-seg] reuse $OUT_DIR"
else
  rm -rf "$OUT_DIR"
  "$PYTHON_BIN" scripts/train_pseudo_seg25d_clinical4_cv.py \
    --cache-root "$CACHE_ROOT" \
    --out-dir "$OUT_DIR" \
    --folds 5 \
    --epochs 10 \
    --batch-size 16 \
    --eval-batch-size 8 \
    --num-workers 2 \
    --device cuda \
    --amp \
    2>&1 | tee logs/train_pseudo_seg25d_clinical4_minvox5000.log
fi

echo "[done] pseudo-seg report written to $OUT_DIR"
