#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MIX_DIR="${MIX_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_label_first_with_cls_noequal}"
BR_TOKEN="${BR_TOKEN:-BR}"
PT_TOKEN="${PT_TOKEN:-PT}"
KEEP_EQUAL="${KEEP_EQUAL:-0}"

ARGS=(
  --translation-train "$MIX_DIR/translation_train.jsonl"
  --translation-valid "$MIX_DIR/translation_valid.jsonl"
  --classification-train "$MIX_DIR/classification_train.jsonl"
  --classification-valid "$MIX_DIR/classification_valid.jsonl"
  --out-dir "$OUT_DIR"
  --br-token "$BR_TOKEN"
  --pt-token "$PT_TOKEN"
)

if [[ "$KEEP_EQUAL" == "1" ]]; then
  ARGS+=(--keep-equal)
fi

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_stageB_gpt_wiki_frmt_label_first_with_cls.py" \
  "${ARGS[@]}" \
  "$@"

echo
echo "Line counts:"
wc -l "$OUT_DIR/train.jsonl" "$OUT_DIR/valid.jsonl"
