#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/data/encoder_decoder/t5gemma2/control_string_protocol}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-$REPO_ROOT/data/encoder_decoder/t5gemma2/control_string_eval}"
BR_CONTROL="${BR_CONTROL:-<pt-br>}"
PT_CONTROL="${PT_CONTROL:-<pt-pt>}"
CLASSIFICATION_CONTROL="${CLASSIFICATION_CONTROL:-<cls>}"

build_stage () {
  local source_dir=$1
  local output_name=$2
  local forbidden_dataset=${3:-}

  local args=(
    "$PYTHON_BIN"
    "$REPO_ROOT/scripts/encoder_decoder/control_strings/build_control_string_supervised.py"
    --mixed-train "$source_dir/train.jsonl"
    --mixed-valid "$source_dir/valid.jsonl"
    --out-root "$OUT_ROOT/$output_name"
    --br-control "$BR_CONTROL"
    --pt-control "$PT_CONTROL"
    --classification-control "$CLASSIFICATION_CONTROL"
  )
  if [[ -n "$forbidden_dataset" ]]; then
    args+=(--forbid-dataset-substring "$forbidden_dataset")
  fi
  "${args[@]}"
}

build_stage "$SOURCE_ROOT/stageA_opensubs_only_with_cls" stageA_opensubs_only
build_stage \
  "$SOURCE_ROOT/stageB_gpt_wiki_translation_plus_cls_noequal" \
  stageB_gpt_wiki \
  frmt
build_stage \
  "$SOURCE_ROOT/stageB_gpt_wiki_frmt_translation_plus_cls_noequal" \
  stageB_gpt_wiki_frmt

for stage in stageA_opensubs_only stageB_gpt_wiki stageB_gpt_wiki_frmt; do
  "$PYTHON_BIN" \
    "$REPO_ROOT/scripts/encoder_decoder/control_strings/audit_control_string_data.py" \
    --root "$OUT_ROOT/$stage" \
    --br-control "$BR_CONTROL" \
    --pt-control "$PT_CONTROL" \
    --classification-control "$CLASSIFICATION_CONTROL" \
    --allow-train-valid-overlap
done

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/control_strings/compare_dataset_counts.py" \
  --data-root "$SOURCE_ROOT" \
  --control-root "$OUT_ROOT" \
  --report "$OUT_ROOT/dataset_count_comparison.json"

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/control_strings/build_control_string_eval.py" \
  --source-root "$REPO_ROOT/data/encoder_decoder/t5gemma2/final_eval" \
  --out-root "$EVAL_OUT_ROOT" \
  --br-control "$BR_CONTROL" \
  --pt-control "$PT_CONTROL" \
  --classification-control "$CLASSIFICATION_CONTROL"

echo
echo "Control-string datasets built under: $OUT_ROOT"
find "$OUT_ROOT" -type f -maxdepth 3 -print | sort
