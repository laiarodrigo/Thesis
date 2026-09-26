#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

FRMT_DIR="${FRMT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only}"
WIKI_DIR="${WIKI_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki}"
PTBRVID_DIR="${PTBRVID_DIR:-}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

echo "Exporting Stage B data (Wikipedia GPT + FRMT)"
echo "  frmt dir: $FRMT_DIR"
echo "  wiki dir: $WIKI_DIR"
if [[ -n "$PTBRVID_DIR" ]]; then
  echo "  ptbrvid dir: $PTBRVID_DIR"
fi
echo "  out dir: $OUT_DIR"

EXTRA_ARGS=()
if [[ -n "$PTBRVID_DIR" ]]; then
  EXTRA_ARGS+=(
    --ptbrvarid-train "$PTBRVID_DIR/translation_train.jsonl"
    --ptbrvarid-valid "$PTBRVID_DIR/translation_valid.jsonl"
  )
fi

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_translation_gpt_wiki_frmt_mix.py" \
  --frmt-train "$FRMT_DIR/translation_train.jsonl" \
  --frmt-valid "$FRMT_DIR/translation_valid.jsonl" \
  --wiki-train "$WIKI_DIR/translation_train.jsonl" \
  --wiki-valid "$WIKI_DIR/translation_valid.jsonl" \
  --frmt-cls-train "$FRMT_DIR/classification_train.jsonl" \
  --frmt-cls-valid "$FRMT_DIR/classification_valid.jsonl" \
  --wiki-cls-train "$WIKI_DIR/classification_train.jsonl" \
  --wiki-cls-valid "$WIKI_DIR/classification_valid.jsonl" \
  --out-dir "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"

echo
echo "Line counts:"
wc -l \
  "$OUT_DIR/translation_train.jsonl" \
  "$OUT_DIR/translation_valid.jsonl" \
  "$OUT_DIR/classification_train.jsonl" \
  "$OUT_DIR/classification_valid.jsonl"
