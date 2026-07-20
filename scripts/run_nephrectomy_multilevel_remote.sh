#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/retroperitoneal_tumor_diagnosis}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
LABELS_JSON="${LABELS_JSON:-${PROJECT_ROOT}/data_private/nephrectomy_cohort_labels.json}"
METADATA_CSV="${METADATA_CSV:-${PROJECT_ROOT}/data/cache_96slice/all.csv}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_ROOT}/dataset_standard_v0/images}"
MASK_DIR="${MASK_DIR:-/root/autodl-tmp/flare23_outputs}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/nephrectomy_multilevel}"
FEATURE_ROOT="${RUN_ROOT}/features"
DEEP_ROOT="${RUN_ROOT}/deep"
RESULT_ROOT="${RUN_ROOT}/results"

mkdir -p "${FEATURE_ROOT}" "${DEEP_ROOT}" "${RESULT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_nephrectomy_multilevel_features.py" \
  --labels-json "${LABELS_JSON}" \
  --metadata-csv "${METADATA_CSV}" \
  --image-dir "${IMAGE_DIR}" \
  --mask-dir "${MASK_DIR}" \
  --out-dir "${FEATURE_ROOT}" \
  --spacing-xyz 1.5,1.5,2.0 \
  --crop-size-zyx 64,64,64 \
  --context-mm 5 \
  --radiomics \
  --jobs "${FEATURE_JOBS:-4}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_nephrectomy_task_features.py" \
  --feature-root "${FEATURE_ROOT}" \
  --out-dir "${DEEP_ROOT}" \
  --folds 5 \
  --epochs "${DEEP_EPOCHS:-20}" \
  --patience "${DEEP_PATIENCE:-5}" \
  --batch-size "${DEEP_BATCH_SIZE:-4}" \
  --workers "${DEEP_WORKERS:-2}" \
  --amp

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_nephrectomy_multilevel_cv.py" \
  --feature-root "${FEATURE_ROOT}" \
  --deep-root "${DEEP_ROOT}" \
  --out-dir "${RESULT_ROOT}" \
  --inner-folds 3 \
  --bootstrap "${BOOTSTRAP_ITERATIONS:-2000}"
