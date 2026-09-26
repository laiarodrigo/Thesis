#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARGS=(
  --model-id "$MODEL_ID"
  --dataset-path "$DATASET_PATH"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --max-input-length "$MAX_INPUT_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --trust-remote-code
)

if [[ -n "${ADAPTER_DIR:-}" ]]; then
  ARGS+=(--adapter-dir "$ADAPTER_DIR")
fi

if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  ARGS+=(--tokenizer-path "$TOKENIZER_PATH")
fi

"$PYTHON_BIN" scripts/decoder_only/axolotl/evaluate_translation_chat_model.py "${ARGS[@]}" "$@"
