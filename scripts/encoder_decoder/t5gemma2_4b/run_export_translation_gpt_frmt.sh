#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

PROJECT_DB_PATH="${PROJECT_DB_PATH:-$REPO_ROOT/data/duckdb/subs_project.duckdb}"
SOURCE_DB_PATH="${SOURCE_DB_PATH:-$REPO_ROOT/data/duckdb/subs.duckdb}"

# Table in subs.duckdb that contains your GPT+FRMT rows.
SOURCE_TABLE="${SOURCE_TABLE:-}"
if [[ -z "$SOURCE_TABLE" ]]; then
  echo "ERROR: set SOURCE_TABLE to your table name in subs.duckdb (example: SOURCE_TABLE=train_data)." >&2
  exit 2
fi

TRAIN_VIEW="${TRAIN_VIEW:-src.$SOURCE_TABLE}"
VALID_VIEW="${VALID_VIEW:-src.$SOURCE_TABLE}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VALID_SPLIT="${VALID_SPLIT:-valid}"

# Comma-separated list.
DATASET_INCLUDE="${DATASET_INCLUDE:-GPT,FRMT}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/gpt_frmt_mix}"

TRANSLATION_TRAIN_FILE="${TRANSLATION_TRAIN_FILE:-translation_train.jsonl}"
TRANSLATION_VALID_FILE="${TRANSLATION_VALID_FILE:-translation_valid.jsonl}"
CLASSIFICATION_TRAIN_FILE="${CLASSIFICATION_TRAIN_FILE:-classification_train.jsonl}"
CLASSIFICATION_VALID_FILE="${CLASSIFICATION_VALID_FILE:-classification_valid.jsonl}"

BATCH_SIZE="${BATCH_SIZE:-10000}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VALID_LIMIT="${VALID_LIMIT:-100000}"

mkdir -p "$OUT_DIR"

DATASET_ARGS=()
IFS=',' read -r -a DATASETS <<< "$DATASET_INCLUDE"
for ds in "${DATASETS[@]}"; do
  ds_trimmed="$(echo "$ds" | xargs)"
  if [[ -n "$ds_trimmed" ]]; then
    DATASET_ARGS+=(--dataset-include "$ds_trimmed")
  fi
done

EXTRA_ARGS=()
if [[ -n "$TRAIN_LIMIT" ]]; then
  EXTRA_ARGS+=(--train-limit "$TRAIN_LIMIT")
fi
if [[ -n "$VALID_LIMIT" ]]; then
  EXTRA_ARGS+=(--valid-limit "$VALID_LIMIT")
fi

echo "Exporting encoder-decoder data from DuckDB table"
echo "  source table: $SOURCE_TABLE"
echo "  train view/split: $TRAIN_VIEW / $TRAIN_SPLIT"
echo "  valid view/split: $VALID_VIEW / $VALID_SPLIT"
echo "  datasets: $DATASET_INCLUDE"
echo "  out dir: $OUT_DIR"

python "$REPO_ROOT/scripts/encoder_decoder/export_encdec_data.py" \
  --project-db "$PROJECT_DB_PATH" \
  --source-db "$SOURCE_DB_PATH" \
  --train-view "$TRAIN_VIEW" \
  --train-split "$TRAIN_SPLIT" \
  --valid-view "$VALID_VIEW" \
  --valid-split "$VALID_SPLIT" \
  --batch-size "$BATCH_SIZE" \
  --out-dir "$OUT_DIR" \
  --translation-train-file "$TRANSLATION_TRAIN_FILE" \
  --translation-valid-file "$TRANSLATION_VALID_FILE" \
  --classification-train-file "$CLASSIFICATION_TRAIN_FILE" \
  --classification-valid-file "$CLASSIFICATION_VALID_FILE" \
  "${DATASET_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
