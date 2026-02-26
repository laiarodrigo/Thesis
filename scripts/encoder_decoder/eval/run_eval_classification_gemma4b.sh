#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Dataset: golden classification test split by default.
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/classification_test.jsonl}"

# 4B classification head + LoRA setup.
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/encoder_decoder/t5gemma2_4b/classification_head_boilerplate.yaml}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/outputs/encoder_decoder/t5gemma2_4b_classification_head_lora}"

# Keep output grouped under gemma4b.
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_results/encoder_decoder/gemma4b/golden_collection/classification_head_lora}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-512}"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "ERROR: dataset path not found: $DATASET_PATH" >&2
  exit 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config path not found: $CONFIG_PATH" >&2
  exit 2
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model directory not found: $MODEL_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

echo "Gemma 4B classification eval (classification head + LoRA)"
echo "  dataset: $DATASET_PATH"
echo "  config: $CONFIG_PATH"
echo "  model-dir: $MODEL_DIR"
echo "  output dir: $OUTPUT_DIR"

python "$REPO_ROOT/scripts/encoder_decoder/eval/evaluate_classification_head.py" \
  --config "$CONFIG_PATH" \
  --dataset-path "$DATASET_PATH" \
  --model-dir "$MODEL_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-source-length "$MAX_SOURCE_LENGTH" \
  --output-dir "$OUTPUT_DIR"
