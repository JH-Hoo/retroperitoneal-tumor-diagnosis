#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/root/autodl-tmp}"
PROJECT="${PROJECT:-$BASE/retroperitoneal_tumor_diagnosis}"
CHAMPION_ROOT="${CHAMPION_ROOT:-$BASE/flare23_champion}"
RUN_ROOT="${RUN_ROOT:-$BASE/flare23_champion_run}"
INPUT_SRC="${INPUT_SRC:-$PROJECT/dataset_standard_v0/images}"
INPUT_DIR="${INPUT_DIR:-$RUN_ROOT/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/flare23_champion_outputs}"
LOG_DIR="${LOG_DIR:-$PROJECT/logs}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$LOG_DIR" "$RUN_ROOT" "$OUTPUT_DIR" "$CHAMPION_ROOT/models"
MONITOR_LOG="$LOG_DIR/flare23_champion_monitor.log"
RUN_LOG="$LOG_DIR/flare23_champion_inference.log"
PID_FILE="$LOG_DIR/flare23_champion_inference.pid"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MONITOR_LOG"
}

find_weight() {
  local kind="$1"
  local pattern="$2"
  local found=""
  for dir in \
    "$CHAMPION_ROOT/models" \
    "$BASE/download" \
    "$BASE/downloads" \
    "$BASE/flare23_champion_weights" \
    "$BASE" \
    "/root/download" \
    "/root/Downloads"; do
    [ -d "$dir" ] || continue
    found="$(find "$dir" -maxdepth 4 -type f -iname "$pattern" ! -name '*.part' ! -name '*.aria2' ! -name '*.download' | sort | head -n 1 || true)"
    if [ -n "$found" ]; then
      echo "$found"
      return 0
    fi
  done
  return 1
}

weights_ready() {
  ROI_WEIGHT="$(find_weight roi '*roi*.pt' || true)"
  FINE2_WEIGHT="$(find_weight fine2 '*fine2*.pt' || true)"
  FINE_WEIGHT="$(find_weight fine '*fine*.pt' || true)"
  if [ -n "$FINE_WEIGHT" ] && [ "$FINE_WEIGHT" = "$FINE2_WEIGHT" ]; then
    FINE_WEIGHT=""
  fi
  [ -n "$ROI_WEIGHT" ] && [ -n "$FINE_WEIGHT" ] && [ -n "$FINE2_WEIGHT" ]
}

install_runtime() {
  log "Installing runtime dependencies if needed"
  "$PYTHON_BIN" -m pip install -q connected-components-3d fastremap
  "$PYTHON_BIN" -m pip install -q -e "$CHAMPION_ROOT/Inference" --no-deps
  "$PYTHON_BIN" - <<'PY'
import cc3d, fastremap, SimpleITK, skimage, torch
print("runtime_ok", torch.__version__)
PY
}

stage_weights() {
  link_weight "$ROI_WEIGHT" "$CHAMPION_ROOT/models/model_roi.pt"
  link_weight "$FINE_WEIGHT" "$CHAMPION_ROOT/models/model_fine.pt"
  link_weight "$FINE2_WEIGHT" "$CHAMPION_ROOT/models/model_fine2.pt"
  log "Using ROI weight: $ROI_WEIGHT"
  log "Using fine weight: $FINE_WEIGHT"
  log "Using fine2 weight: $FINE2_WEIGHT"
}

link_weight() {
  local src="$1"
  local dst="$2"
  if [ "$(readlink -f "$src")" != "$(readlink -f "$dst" 2>/dev/null || true)" ]; then
    ln -sf "$src" "$dst"
  fi
}

prepare_inputs() {
  local expected=0
  local existing_linked=0
  local existing_skipped=0
  expected="$(find "$INPUT_SRC" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)"
  existing_linked="$(find "$INPUT_DIR" -maxdepth 1 -type l -name '*_0000.nii.gz' 2>/dev/null | wc -l || true)"
  existing_skipped="$(wc -l < "$RUN_ROOT/skipped_corrupt_inputs.txt" 2>/dev/null || echo 0)"
  if [ "$expected" -gt 0 ] && [ $((existing_linked + existing_skipped)) -eq "$expected" ]; then
    log "Reusing prepared inputs: expected=$expected linked=$existing_linked skipped_corrupt=$existing_skipped"
    return 0
  fi

  log "Preparing all readable input images from $INPUT_SRC"
  rm -rf "$INPUT_DIR"
  mkdir -p "$INPUT_DIR"
  : > "$RUN_ROOT/skipped_corrupt_inputs.txt"
  : > "$RUN_ROOT/input_manifest.tsv"
  local total=0
  local linked=0
  local skipped=0
  for src in "$INPUT_SRC"/*.nii.gz; do
    [ -e "$src" ] || continue
    total=$((total + 1))
    case_id="$(basename "$src" .nii.gz)"
    if gzip -t "$src" >/dev/null 2>&1; then
      ln -sf "$src" "$INPUT_DIR/${case_id}_0000.nii.gz"
      printf "%s\t%s\t%s\n" "$case_id" "$src" "$INPUT_DIR/${case_id}_0000.nii.gz" >> "$RUN_ROOT/input_manifest.tsv"
      linked=$((linked + 1))
    else
      printf "%s\t%s\n" "$case_id" "$src" >> "$RUN_ROOT/skipped_corrupt_inputs.txt"
      skipped=$((skipped + 1))
    fi
  done
  log "Input prep done: total=$total linked=$linked skipped_corrupt=$skipped"
}

already_running() {
  [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1
}

start_inference() {
  if already_running; then
    log "Inference already running with PID $(cat "$PID_FILE")"
    return 0
  fi
  log "Starting champion FLARE23 inference: output=$OUTPUT_DIR"
  (
    cd "$CHAMPION_ROOT"
    "$PYTHON_BIN" Inference/nnunet/run_inference.py -i "$INPUT_DIR" -o "$OUTPUT_DIR"
  ) >> "$RUN_LOG" 2>&1 &
  echo $! > "$PID_FILE"
  log "Started inference PID $(cat "$PID_FILE"), log=$RUN_LOG"
}

while true; do
  if weights_ready; then
    log "All three champion weights detected"
    stage_weights
    install_runtime
    prepare_inputs
    start_inference
    exit 0
  fi
  log "Waiting for weights. Need model_roi.pt, model_fine.pt, model_fine2.pt under download/models; sleeping ${POLL_SECONDS}s"
  find "$BASE" /root -maxdepth 4 -type f \( -iname '*roi*.pt' -o -iname '*fine*.pt' -o -iname '*.part' -o -iname '*.aria2' -o -iname '*.download' \) 2>/dev/null | sort | sed 's/^/[candidate] /' | tee -a "$MONITOR_LOG" || true
  sleep "$POLL_SECONDS"
done
