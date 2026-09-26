#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MIX_DIR="${MIX_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_translation_plus_cls_noequal}"
CLS_PREFIX="${CLS_PREFIX:-<id>}"
KEEP_EQUAL="${KEEP_EQUAL:-0}"

echo "Building Stage B base mix before adding classification rows."
OUT_DIR="$MIX_DIR" bash "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageB_gpt_wikipedia_plus_frmt.sh"

ARGS=(
  --translation-train "$MIX_DIR/translation_train.jsonl"
  --translation-valid "$MIX_DIR/translation_valid.jsonl"
  --classification-train "$MIX_DIR/classification_train.jsonl"
  --classification-valid "$MIX_DIR/classification_valid.jsonl"
  --out-dir "$OUT_DIR"
  --cls-prefix "$CLS_PREFIX"
)

if [[ "$KEEP_EQUAL" == "1" ]]; then
  ARGS+=(--keep-equal)
fi

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_stageB_gpt_wiki_frmt_translation_plus_cls.py" \
  "${ARGS[@]}" \
  "$@"

echo
echo "Line counts:"
wc -l "$OUT_DIR/train.jsonl" "$OUT_DIR/valid.jsonl"
