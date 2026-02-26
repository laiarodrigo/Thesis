#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Dataset: golden translation test split by default.
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl}"

# 4B translation setup: base model + LoRA adapter from the r24 run.
MODEL_ID="${MODEL_ID:-google/t5gemma-2-4b-4b}"
ADAPTER_DIR="${ADAPTER_DIR:-$REPO_ROOT/outputs/encoder_decoder/t5gemma2_4b_translation_lora_r24}"

# Separate output tree for easy side-by-side comparisons.
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_results/encoder_decoder/gemma4b/golden_collection/translation_lora_r24_adaptive_beam4}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-3}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"

NUM_BEAMS="${NUM_BEAMS:-4}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.9}"
EARLY_STOPPING="${EARLY_STOPPING:-1}"

ADAPTIVE_MAX_NEW_TOKENS="${ADAPTIVE_MAX_NEW_TOKENS:-1}"
ADAPTIVE_RATIO="${ADAPTIVE_RATIO:-1.3}"
ADAPTIVE_MARGIN="${ADAPTIVE_MARGIN:-10}"
ADAPTIVE_MIN_NEW_TOKENS="${ADAPTIVE_MIN_NEW_TOKENS:-8}"
ADAPTIVE_MAX_NEW_TOKENS_CEILING="${ADAPTIVE_MAX_NEW_TOKENS_CEILING:-512}"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "ERROR: dataset path not found: $DATASET_PATH" >&2
  exit 2
fi

if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "ERROR: adapter directory not found: $ADAPTER_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ "$EARLY_STOPPING" == "1" ]]; then
  EXTRA_ARGS+=(--early-stopping)
fi
if [[ "$ADAPTIVE_MAX_NEW_TOKENS" == "1" ]]; then
  EXTRA_ARGS+=(
    --adaptive-max-new-tokens
    --adaptive-ratio "$ADAPTIVE_RATIO"
    --adaptive-margin "$ADAPTIVE_MARGIN"
    --adaptive-min-new-tokens "$ADAPTIVE_MIN_NEW_TOKENS"
    --adaptive-max-new-tokens-ceiling "$ADAPTIVE_MAX_NEW_TOKENS_CEILING"
  )
fi

echo "Gemma 4B translation eval (LoRA r24 + adaptive length + beam)"
echo "  dataset: $DATASET_PATH"
echo "  model-id: $MODEL_ID"
echo "  adapter: $ADAPTER_DIR"
echo "  output dir: $OUTPUT_DIR"

python "$REPO_ROOT/scripts/encoder_decoder/eval/evaluate_encdec.py" \
  --task translation \
  --dataset-path "$DATASET_PATH" \
  --model-id "$MODEL_ID" \
  --adapter-dir "$ADAPTER_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-source-length "$MAX_SOURCE_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-beams "$NUM_BEAMS" \
  --length-penalty "$LENGTH_PENALTY" \
  --no-repeat-ngram-size "$NO_REPEAT_NGRAM_SIZE" \
  --repetition-penalty "$REPETITION_PENALTY" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
