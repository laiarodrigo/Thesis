#!/usr/bin/env bash
set -euo pipefail

# Evaluate the latest small encoder-decoder full fine-tuned models
# on the golden collection test splits.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

EXPORT_GOLDEN_FROM_DUCKDB="${EXPORT_GOLDEN_FROM_DUCKDB:-1}"
PROJECT_DB_PATH="${PROJECT_DB_PATH:-$REPO_ROOT/data/duckdb/subs_project.duckdb}"
SOURCE_DB_PATH="${SOURCE_DB_PATH:-$REPO_ROOT/data/duckdb/subs.duckdb}"
GOLDEN_VIEW="${GOLDEN_VIEW:-test_data}"
GOLDEN_DATASET="${GOLDEN_DATASET:-Gold}"
GOLDEN_SPLIT="${GOLDEN_SPLIT:-test}"
GOLDEN_EXPORT_DIR="${GOLDEN_EXPORT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection}"

TRANSLATION_DATASET_PATH="${TRANSLATION_DATASET_PATH:-$GOLDEN_EXPORT_DIR/translation_test.jsonl}"
CLASSIFICATION_DATASET_PATH="${CLASSIFICATION_DATASET_PATH:-$GOLDEN_EXPORT_DIR/classification_test.jsonl}"

TRANSLATION_MODEL_ID="${TRANSLATION_MODEL_ID:-$REPO_ROOT/outputs/encoder_decoder/t5gemma2_translation_fullft}"
CLASSIFICATION_MODEL_DIR="${CLASSIFICATION_MODEL_DIR:-$REPO_ROOT/outputs/encoder_decoder/t5gemma2_classification_head_fullft}"
CLASSIFICATION_CONFIG_PATH="${CLASSIFICATION_CONFIG_PATH:-$REPO_ROOT/configs/encoder_decoder/t5gemma2/classification_head_boilerplate.yaml}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_results/encoder_decoder/golden collection}"
TRANSLATION_OUTPUT_DIR="${TRANSLATION_OUTPUT_DIR:-$OUTPUT_DIR/translation}"
CLASSIFICATION_OUTPUT_DIR="${CLASSIFICATION_OUTPUT_DIR:-$OUTPUT_DIR/classification_head}"
TRANSLATION_BATCH_SIZE="${TRANSLATION_BATCH_SIZE:-8}"
CLASSIFICATION_BATCH_SIZE="${CLASSIFICATION_BATCH_SIZE:-16}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

if [[ "$EXPORT_GOLDEN_FROM_DUCKDB" == "1" ]]; then
  python "$REPO_ROOT/scripts/encoder_decoder/eval/export_golden_from_duckdb.py" \
    --project-db "$PROJECT_DB_PATH" \
    --source-db "$SOURCE_DB_PATH" \
    --view "$GOLDEN_VIEW" \
    --dataset "$GOLDEN_DATASET" \
    --split "$GOLDEN_SPLIT" \
    --out-dir "$GOLDEN_EXPORT_DIR"
fi

mkdir -p "$TRANSLATION_OUTPUT_DIR" "$CLASSIFICATION_OUTPUT_DIR"

echo "Golden collection eval (small full-FT)"
echo "  translation model: $TRANSLATION_MODEL_ID"
echo "  classification model: $CLASSIFICATION_MODEL_DIR"
echo "  translation dataset: $TRANSLATION_DATASET_PATH"
echo "  classification dataset: $CLASSIFICATION_DATASET_PATH"
echo "  translation output dir: $TRANSLATION_OUTPUT_DIR"
echo "  classification output dir: $CLASSIFICATION_OUTPUT_DIR"

python "$REPO_ROOT/scripts/encoder_decoder/eval/evaluate_encdec.py" \
  --task translation \
  --dataset-path "$TRANSLATION_DATASET_PATH" \
  --model-id "$TRANSLATION_MODEL_ID" \
  --batch-size "$TRANSLATION_BATCH_SIZE" \
  --max-source-length "$MAX_SOURCE_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output-dir "$TRANSLATION_OUTPUT_DIR"

python "$REPO_ROOT/scripts/encoder_decoder/eval/evaluate_classification_head.py" \
  --config "$CLASSIFICATION_CONFIG_PATH" \
  --dataset-path "$CLASSIFICATION_DATASET_PATH" \
  --model-dir "$CLASSIFICATION_MODEL_DIR" \
  --batch-size "$CLASSIFICATION_BATCH_SIZE" \
  --max-source-length "$MAX_SOURCE_LENGTH" \
  --output-dir "$CLASSIFICATION_OUTPUT_DIR"
