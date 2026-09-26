#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

FRMT_DIR="${FRMT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only}"
WIKI_DIR="${WIKI_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki}"
PTBRVID_INPUT_CSV="${PTBRVID_INPUT_CSV:-$REPO_ROOT/data/ptbrvarid/translated_stageb_pairs/translated_pairs.csv}"
PTBRVID_EXPORT_DIR="${PTBRVID_EXPORT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/ptbrvarid_translated_stageB}"
MIX_OUT_DIR="${MIX_OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix_plus_ptbrvarid}"
LABEL_OUT_DIR="${LABEL_OUT_DIR:-}"
BR_TOKEN="${BR_TOKEN:-BR}"
PT_TOKEN="${PT_TOKEN:-PT}"
EQUAL_TOKEN="${EQUAL_TOKEN:-igual}"
KEEP_EQUAL="${KEEP_EQUAL:-1}"
KEEP_EQUAL_CLASSIFICATION="${KEEP_EQUAL_CLASSIFICATION:-0}"
FORCE_PTBRVID_EXPORT="${FORCE_PTBRVID_EXPORT:-0}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VALID_RATIO="${VALID_RATIO:-0.1}"
TEST_PER_GROUP="${TEST_PER_GROUP:-0}"
PTBRVID_EXCLUDE_DOMAINS="${PTBRVID_EXCLUDE_DOMAINS:-}"
SEED="${SEED:-42}"

if [[ ! -f "$PTBRVID_INPUT_CSV" ]]; then
  CANDIDATE="$REPO_ROOT/data/ptbrvarid/translated_stageb_pairs_r48_500_t50/translated_pairs.csv"
  if [[ -f "$CANDIDATE" ]]; then
    PTBRVID_INPUT_CSV="$CANDIDATE"
  fi
fi

if [[ -z "$LABEL_OUT_DIR" ]]; then
  if [[ "$KEEP_EQUAL_CLASSIFICATION" == "1" ]]; then
    LABEL_OUT_DIR="$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_plus_ptbrvarid_equal_all"
  elif [[ "$KEEP_EQUAL" == "1" ]]; then
    LABEL_OUT_DIR="$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_plus_ptbrvarid_equal"
  else
    LABEL_OUT_DIR="$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_plus_ptbrvarid_noequal"
  fi
fi

if [[ "$KEEP_EQUAL_CLASSIFICATION" == "1" && "$KEEP_EQUAL" != "1" ]]; then
  echo "ERROR: KEEP_EQUAL_CLASSIFICATION=1 requires KEEP_EQUAL=1" >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ "$FORCE_PTBRVID_EXPORT" == "1" || ! -f "$PTBRVID_EXPORT_DIR/translation_train.jsonl" || ! -f "$PTBRVID_EXPORT_DIR/translation_valid.jsonl" ]]; then
  if [[ ! -f "$PTBRVID_INPUT_CSV" ]]; then
    echo "ERROR: missing PtBrVId translated pairs CSV: $PTBRVID_INPUT_CSV" >&2
    exit 2
  fi

  echo "Exporting PtBrVId translated pairs"
  echo "  input csv: $PTBRVID_INPUT_CSV"
  echo "  out dir: $PTBRVID_EXPORT_DIR"

  "$PYTHON_BIN" \
    "$REPO_ROOT/scripts/ptbrvarid/export_translated_pairs_to_encdec.py" \
    --input-csv "$PTBRVID_INPUT_CSV" \
    --out-dir "$PTBRVID_EXPORT_DIR" \
    --train-ratio "$TRAIN_RATIO" \
    --valid-ratio "$VALID_RATIO" \
    --test-per-group "$TEST_PER_GROUP" \
    --exclude-domains "$PTBRVID_EXCLUDE_DOMAINS" \
    --seed "$SEED"
fi

echo "Exporting Stage B data (GPT-Wiki + FRMT + PtBrVId translation)"
echo "  frmt dir: $FRMT_DIR"
echo "  wiki dir: $WIKI_DIR"
echo "  ptbrvid dir: $PTBRVID_EXPORT_DIR"
echo "  mix out dir: $MIX_OUT_DIR"
echo "  label-first out dir: $LABEL_OUT_DIR"
echo "  keep equal: $KEEP_EQUAL"
echo "  keep equal classification: $KEEP_EQUAL_CLASSIFICATION"
echo "  equal token: $EQUAL_TOKEN"

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_translation_gpt_wiki_frmt_mix.py" \
  --frmt-train "$FRMT_DIR/translation_train.jsonl" \
  --frmt-valid "$FRMT_DIR/translation_valid.jsonl" \
  --wiki-train "$WIKI_DIR/translation_train.jsonl" \
  --wiki-valid "$WIKI_DIR/translation_valid.jsonl" \
  --ptbrvarid-train "$PTBRVID_EXPORT_DIR/translation_train.jsonl" \
  --ptbrvarid-valid "$PTBRVID_EXPORT_DIR/translation_valid.jsonl" \
  --frmt-cls-train "$FRMT_DIR/classification_train.jsonl" \
  --frmt-cls-valid "$FRMT_DIR/classification_valid.jsonl" \
  --wiki-cls-train "$WIKI_DIR/classification_train.jsonl" \
  --wiki-cls-valid "$WIKI_DIR/classification_valid.jsonl" \
  --out-dir "$MIX_OUT_DIR"

LABEL_ARGS=(
  --translation-train "$MIX_OUT_DIR/translation_train.jsonl"
  --translation-valid "$MIX_OUT_DIR/translation_valid.jsonl"
  --classification-train "$MIX_OUT_DIR/classification_train.jsonl"
  --classification-valid "$MIX_OUT_DIR/classification_valid.jsonl"
  --out-dir "$LABEL_OUT_DIR"
  --br-token "$BR_TOKEN"
  --pt-token "$PT_TOKEN"
)

if [[ "$KEEP_EQUAL" == "1" ]]; then
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
  "$LABEL_OUT_DIR/train.jsonl" \
  "$LABEL_OUT_DIR/valid.jsonl"
