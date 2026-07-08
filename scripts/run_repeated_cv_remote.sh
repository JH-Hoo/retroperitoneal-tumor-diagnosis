#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/retroperitoneal_tumor_diagnosis}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT/data/champion_flare23_25d_cache_15x224_minvox5000}"
SEEDS="${SEEDS:-20260708 20260709 20260710 20260711 20260712}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
PRIMARY_REPORT_DIR="${PRIMARY_REPORT_DIR:-$PROJECT/reports/champion_resnet25d_clinical4_minvox5000}"

cd "$PROJECT"
mkdir -p logs models/repeated_cv reports/repeated_cv/champion_resnet25d_clinical4_minvox5000

for seed in $SEEDS; do
  name="seed_${seed}"
  out_dir="models/repeated_cv/champion_resnet25d_clinical4_minvox5000/${name}"
  report_dir="reports/repeated_cv/champion_resnet25d_clinical4_minvox5000/${name}"
  if [[ "$FORCE_TRAIN" != "1" && -f "$report_dir/summary.json" ]]; then
    echo "[repeat] reuse $name"
    continue
  fi
  if [[ "$FORCE_TRAIN" != "1" && "$seed" == "20260708" && -f "$PRIMARY_REPORT_DIR/summary.json" ]]; then
    echo "[repeat] copy primary report for $name"
    rm -rf "$report_dir"
    mkdir -p "$report_dir"
    cp "$PRIMARY_REPORT_DIR"/summary.json "$report_dir"/
    cp "$PRIMARY_REPORT_DIR"/oof_predictions.csv "$report_dir"/
    cp "$PRIMARY_REPORT_DIR"/oof_predictions_derived_binary.csv "$report_dir"/
    cp "$PRIMARY_REPORT_DIR"/oof_predictions_binary_head.csv "$report_dir"/
    cp "$PRIMARY_REPORT_DIR"/*confusion_matrix.png "$report_dir"/
    continue
  fi
  echo "[repeat] train $name"
  rm -rf "$out_dir" "$report_dir"
  "$PYTHON_BIN" scripts/train_resnet25d_clinical4_cv.py \
    --cache-root "$CACHE_ROOT" \
    --out-dir "$out_dir" \
    --weights imagenet \
    --mask-channel-init zero \
    --channel-set all \
    --epochs 10 \
    --batch-size 4 \
    --num-workers 2 \
    --device cuda \
    --amp \
    --seed "$seed" \
    2>&1 | tee "logs/train_repeated_cv_${name}.log"
  mkdir -p "$report_dir"
  cp "$out_dir"/summary.json "$report_dir"/
  cp "$out_dir"/oof_predictions.csv "$report_dir"/
  cp "$out_dir"/oof_predictions_derived_binary.csv "$report_dir"/
  cp "$out_dir"/oof_predictions_binary_head.csv "$report_dir"/
  cp "$out_dir"/*confusion_matrix.png "$report_dir"/
done

"$PYTHON_BIN" scripts/summarize_repeated_cv.py \
  --runs-root reports/repeated_cv/champion_resnet25d_clinical4_minvox5000 \
  --out-dir reports/repeated_cv/champion_resnet25d_clinical4_minvox5000

echo "[done] repeated CV summary written to reports/repeated_cv/champion_resnet25d_clinical4_minvox5000"
