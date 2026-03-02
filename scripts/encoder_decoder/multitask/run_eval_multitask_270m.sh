#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL_ID="${MODEL_ID:-$REPO_ROOT/outputs/encoder_decoder/multitask_compare/t5gemma2_270m_stageB_gpt_adapt}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
TRANSLATION_DATASET="${TRANSLATION_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl}"
CLASSIFICATION_DATASET="${CLASSIFICATION_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/classification_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_results/encoder_decoder/multitask_compare/t5gemma2_270m_stageB_golden}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-512}"
MAX_NEW_TOKENS_TRANSLATION="${MAX_NEW_TOKENS_TRANSLATION:-256}"
MAX_NEW_TOKENS_CLASSIFICATION="${MAX_NEW_TOKENS_CLASSIFICATION:-6}"

mkdir -p "$OUTPUT_DIR"

CMD=(
  python3
  "$REPO_ROOT/scripts/encoder_decoder/multitask/eval_multitask_seq2seq_skeleton.py"
  --model-id "$MODEL_ID"
  --translation-dataset "$TRANSLATION_DATASET"
  --classification-dataset "$CLASSIFICATION_DATASET"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --max-source-length "$MAX_SOURCE_LENGTH"
  --max-new-tokens-translation "$MAX_NEW_TOKENS_TRANSLATION"
  --max-new-tokens-classification "$MAX_NEW_TOKENS_CLASSIFICATION"
)

if [[ -n "$ADAPTER_DIR" ]]; then
  CMD+=(--adapter-dir "$ADAPTER_DIR")
fi

echo "Multitask eval runner"
echo "  model: $MODEL_ID"
echo "  translation dataset: $TRANSLATION_DATASET"
echo "  classification dataset: $CLASSIFICATION_DATASET"
echo "  output dir: $OUTPUT_DIR"

"${CMD[@]}"

