#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

IN_DIR="${IN_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_label_first_noequal}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

echo "Exporting Stage A translation data (OpenSubs-only, label-first)"
echo "  in dir: $IN_DIR"
echo "  out dir: $OUT_DIR"

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_translation_stageA_opensubs_frmt_label_first.py" \
  --translation-train "$IN_DIR/translation_train.jsonl" \
  --translation-valid "$IN_DIR/translation_valid.jsonl" \
  --out-dir "$OUT_DIR"

echo
echo "Line counts:"
wc -l \
  "$OUT_DIR/train.jsonl" \
  "$OUT_DIR/valid.jsonl"
