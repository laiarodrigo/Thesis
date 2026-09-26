#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OPENSUBS_DIR="${OPENSUBS_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_clean}"
PTBRVID_DB="${PTBRVID_DB:-$REPO_ROOT/data/duckdb/subs_ptbr_filtered.duckdb}"
PTBRVID_SAMPLED_ROWS_CSV="${PTBRVID_SAMPLED_ROWS_CSV:-$REPO_ROOT/data/ptbrvarid/translated_stageb_pairs/sampled_rows.csv}"
MIX_OUT_DIR="${MIX_OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid}"
LABEL_OUT_DIR="${LABEL_OUT_DIR:-}"
BR_TOKEN="${BR_TOKEN:-BR}"
PT_TOKEN="${PT_TOKEN:-PT}"
EQUAL_TOKEN="${EQUAL_TOKEN:-igual}"
KEEP_EQUAL="${KEEP_EQUAL:-1}"
KEEP_EQUAL_CLASSIFICATION="${KEEP_EQUAL_CLASSIFICATION:-0}"
CLASSIFICATION_BALANCE_MODE="${CLASSIFICATION_BALANCE_MODE:-none}"
PTBRVID_EXCLUDE_DOMAINS="${PTBRVID_EXCLUDE_DOMAINS:-}"
TRAIN_PTBRVID_TARGET_SHARE="${TRAIN_PTBRVID_TARGET_SHARE:-0.0}"
VALID_PTBRVID_TARGET_SHARE="${VALID_PTBRVID_TARGET_SHARE:-0.0}"
SEED="${SEED:-42}"
PROGRESS_EVERY="${PROGRESS_EVERY:-200000}"

if [[ ! -f "$PTBRVID_SAMPLED_ROWS_CSV" ]]; then
  CANDIDATE="$REPO_ROOT/data/ptbrvarid/translated_stageb_pairs_r48_500_t50/sampled_rows_resampled.csv"
  if [[ -f "$CANDIDATE" ]]; then
    PTBRVID_SAMPLED_ROWS_CSV="$CANDIDATE"
  fi
fi

if [[ -z "$LABEL_OUT_DIR" ]]; then
  if [[ "$KEEP_EQUAL_CLASSIFICATION" == "1" ]]; then
    LABEL_OUT_DIR="$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_label_first_with_cls_equal_all"
  elif [[ "$KEEP_EQUAL" == "1" ]]; then
    LABEL_OUT_DIR="$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_label_first_with_cls_equal"
  else
    LABEL_OUT_DIR="$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_label_first_with_cls_noequal"
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

echo "Exporting Stage A data (OpenSubs translation + leftover PtBrVId classification)"
echo "  opensubs dir: $OPENSUBS_DIR"
echo "  ptbrvid db: $PTBRVID_DB"
echo "  ptbrvid sampled rows: $PTBRVID_SAMPLED_ROWS_CSV"
echo "  mix out dir: $MIX_OUT_DIR"
echo "  label-first out dir: $LABEL_OUT_DIR"
echo "  keep equal: $KEEP_EQUAL"
echo "  keep equal classification: $KEEP_EQUAL_CLASSIFICATION"
echo "  equal token: $EQUAL_TOKEN"

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_stageA_opensubs_ptbrvarid_cls.py" \
  --opensubs-translation-train "$OPENSUBS_DIR/translation_train.jsonl" \
  --opensubs-translation-valid "$OPENSUBS_DIR/translation_valid.jsonl" \
  --opensubs-classification-train "$OPENSUBS_DIR/classification_train.jsonl" \
  --opensubs-classification-valid "$OPENSUBS_DIR/classification_valid.jsonl" \
  --ptbrvarid-db "$PTBRVID_DB" \
  --ptbrvarid-sampled-csv "$PTBRVID_SAMPLED_ROWS_CSV" \
  --ptbrvarid-exclude-domains "$PTBRVID_EXCLUDE_DOMAINS" \
  --classification-balance-mode "$CLASSIFICATION_BALANCE_MODE" \
  --train-ptbrvarid-target-share "$TRAIN_PTBRVID_TARGET_SHARE" \
  --valid-ptbrvarid-target-share "$VALID_PTBRVID_TARGET_SHARE" \
  --seed "$SEED" \
  --progress-every "$PROGRESS_EVERY" \
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
