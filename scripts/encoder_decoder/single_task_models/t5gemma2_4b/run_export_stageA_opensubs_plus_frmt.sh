#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt}"
OPENSUBS_DIR="${OPENSUBS_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only}"
FRMT_DIR="${FRMT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

echo "Exporting Stage A data (OpenSubs + FRMT)"
echo "  opensubs dir: $OPENSUBS_DIR"
echo "  frmt dir: $FRMT_DIR"
echo "  out dir: $OUT_DIR"
"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_translation_stageA_opensubs_frmt.py" \
  --opensubs-train "$OPENSUBS_DIR/translation_train.jsonl" \
  --opensubs-valid "$OPENSUBS_DIR/translation_valid.jsonl" \
  --frmt-train "$FRMT_DIR/translation_train.jsonl" \
  --frmt-valid "$FRMT_DIR/translation_valid.jsonl" \
  --opensubs-cls-train "$OPENSUBS_DIR/classification_train.jsonl" \
  --opensubs-cls-valid "$OPENSUBS_DIR/classification_valid.jsonl" \
  --frmt-cls-train "$FRMT_DIR/classification_train.jsonl" \
  --frmt-cls-valid "$FRMT_DIR/classification_valid.jsonl" \
  --out-dir "$OUT_DIR"
