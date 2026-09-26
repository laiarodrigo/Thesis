#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL_ID="${MODEL_ID:-amalia-llm/AMALIA-9B-0626-SFT}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/classification_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_results/decoder_only/amalia_9b_0626_sft/golden_collection/classification}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
CLASSIFICATION_MODE="${CLASSIFICATION_MODE:-score-sequences}"
CLASSIFICATION_CANDIDATES="${CLASSIFICATION_CANDIDATES:-pt-br pt-pt}"

mkdir -p "$OUTPUT_DIR"

MODEL_ID="$MODEL_ID" \
DATASET_PATH="$DATASET_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
BATCH_SIZE="$BATCH_SIZE" \
MAX_INPUT_LENGTH="$MAX_INPUT_LENGTH" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
CLASSIFICATION_MODE="$CLASSIFICATION_MODE" \
CLASSIFICATION_CANDIDATES="$CLASSIFICATION_CANDIDATES" \
"$REPO_ROOT/scripts/decoder_only/axolotl/run_eval_classification_chat_model.sh" "$@"
