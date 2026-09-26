#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl}"
MODEL_ID="${MODEL_ID:-google/t5gemma-2-270m-270m}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_results/encoder_decoder/threshold_eval}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-512}"
BR_TOKEN="${BR_TOKEN:-BR}"
PT_TOKEN="${PT_TOKEN:-PT}"
THRESHOLD="${THRESHOLD:-}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.00,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50}"
PREVIEW_COUNT="${PREVIEW_COUNT:-15}"

ARGS=(
  --dataset-path "$DATASET_PATH"
  --model-id "$MODEL_ID"
  --batch-size "$BATCH_SIZE"
  --max-source-length "$MAX_SOURCE_LENGTH"
  --br-token "$BR_TOKEN"
  --pt-token "$PT_TOKEN"
  --threshold-grid "$THRESHOLD_GRID"
  --preview-count "$PREVIEW_COUNT"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$ADAPTER_DIR" ]]; then
  ARGS+=(--adapter-dir "$ADAPTER_DIR")
fi
if [[ -n "$TOKENIZER_PATH" ]]; then
  ARGS+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "$THRESHOLD" ]]; then
  ARGS+=(--threshold "$THRESHOLD")
fi

python3 "$REPO_ROOT/scripts/encoder_decoder/eval/evaluate_label_first_threshold.py" "${ARGS[@]}"
