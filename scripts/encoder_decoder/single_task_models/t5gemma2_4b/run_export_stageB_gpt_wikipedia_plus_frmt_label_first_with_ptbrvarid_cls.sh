#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

FRMT_DIR="${FRMT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only}"
WIKI_DIR="${WIKI_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki}"
TRANSLATION_MIX_DIR="${TRANSLATION_MIX_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix}"
PTBRVID_DB="${PTBRVID_DB:-$REPO_ROOT/data/duckdb/subs_ptbr_filtered.duckdb}"
PTBRVID_DATASET="${PTBRVID_DATASET:-PtBrVId}"
PTBRVID_SPLITS="${PTBRVID_SPLITS:-train,valid}"
PTBRVID_EXCLUDE_DOMAINS="${PTBRVID_EXCLUDE_DOMAINS:-}"
PTBRVID_CLASSIFICATION_TOKEN="${PTBRVID_CLASSIFICATION_TOKEN:-<id>}"
PTBRVID_CLS_DIR="${PTBRVID_CLS_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/ptbrvarid_classification}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_with_ptbrvarid_cls_noequal}"
BR_TOKEN="${BR_TOKEN:-BR}"
PT_TOKEN="${PT_TOKEN:-PT}"
EQUAL_TOKEN="${EQUAL_TOKEN:-igual}"
KEEP_EQUAL_TRANSLATION="${KEEP_EQUAL_TRANSLATION:-0}"
KEEP_EQUAL_CLASSIFICATION="${KEEP_EQUAL_CLASSIFICATION:-0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ "$KEEP_EQUAL_CLASSIFICATION" == "1" && "$KEEP_EQUAL_TRANSLATION" != "1" ]]; then
  echo "ERROR: KEEP_EQUAL_CLASSIFICATION=1 requires KEEP_EQUAL_TRANSLATION=1" >&2
  exit 2
fi

echo "Building GPT-Wiki + filtered FRMT translation mix"
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
  --out-dir "$TRANSLATION_MIX_DIR"

echo "Exporting PtBrVId classification"
"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/export_ptbrvarid_classification.py" \
  --ptbrvarid-db "$PTBRVID_DB" \
  --ptbrvarid-dataset "$PTBRVID_DATASET" \
  --splits "$PTBRVID_SPLITS" \
  --classification-token "$PTBRVID_CLASSIFICATION_TOKEN" \
  --exclude-domains "$PTBRVID_EXCLUDE_DOMAINS" \
  --out-dir "$PTBRVID_CLS_DIR"

echo "Building label-first Stage B mix with PtBrVId classification"
LABEL_ARGS=(
  --translation-train "$TRANSLATION_MIX_DIR/translation_train.jsonl"
  --translation-valid "$TRANSLATION_MIX_DIR/translation_valid.jsonl"
  --classification-train "$PTBRVID_CLS_DIR/classification_train.jsonl"
  --classification-valid "$PTBRVID_CLS_DIR/classification_valid.jsonl"
  --out-dir "$OUT_DIR"
  --br-token "$BR_TOKEN"
  --pt-token "$PT_TOKEN"
)

if [[ "$KEEP_EQUAL_TRANSLATION" == "1" ]]; then
  LABEL_ARGS+=(--keep-equal)
fi

if [[ "$KEEP_EQUAL_CLASSIFICATION" == "1" ]]; then
  LABEL_ARGS+=(--keep-equal-classification --equal-token "$EQUAL_TOKEN")
fi

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_stageB_gpt_wiki_frmt_label_first_with_cls.py" \
  "${LABEL_ARGS[@]}"

echo
echo "Line counts:"
wc -l \
  "$OUT_DIR/train.jsonl" \
  "$OUT_DIR/valid.jsonl"
