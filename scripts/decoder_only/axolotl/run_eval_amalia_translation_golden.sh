#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL_ID="${MODEL_ID:-amalia-llm/AMALIA-9B-0626-SFT}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_results/decoder_only/amalia_9b_0626_sft/golden_collection/translation}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

mkdir -p "$OUTPUT_DIR"

MODEL_ID="$MODEL_ID" \
DATASET_PATH="$DATASET_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
BATCH_SIZE="$BATCH_SIZE" \
MAX_INPUT_LENGTH="$MAX_INPUT_LENGTH" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
"$REPO_ROOT/scripts/decoder_only/axolotl/run_eval_translation_chat_model.sh" "$@"
