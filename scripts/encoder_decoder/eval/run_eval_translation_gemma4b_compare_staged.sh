#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <stageB|stageA> <r16|r24> [DATASET_PATH optional env]" >&2
  exit 2
fi

STAGE="$1"
RANK="$2"

case "$STAGE" in
  stageA|stageB) ;;
  *) echo "Invalid stage: $STAGE" >&2; exit 2 ;;
esac
case "$RANK" in
  r16|r24) ;;
  *) echo "Invalid rank: $RANK" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_ID="${MODEL_ID:-google/t5gemma-2-4b-4b}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl}"
ADAPTER_DIR="${ADAPTER_DIR:-$REPO_ROOT/outputs/encoder_decoder/compare_staged/t5gemma2_4b_translation_${RANK}_${STAGE}_$([[ "$STAGE" == "stageA" ]] && echo opensubs_frmt || echo gpt_adapt)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_results/encoder_decoder/gemma4b/comparison_staged/${STAGE}_${RANK}_adaptive_beam4}"

NUM_BEAMS="${NUM_BEAMS:-4}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-4}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.2}"
ADAPTIVE_RATIO="${ADAPTIVE_RATIO:-1.15}"
ADAPTIVE_MARGIN="${ADAPTIVE_MARGIN:-6}"
ADAPTIVE_MIN_NEW_TOKENS="${ADAPTIVE_MIN_NEW_TOKENS:-8}"
ADAPTIVE_MAX_NEW_TOKENS_CEILING="${ADAPTIVE_MAX_NEW_TOKENS_CEILING:-384}"

mkdir -p "$OUTPUT_DIR"

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
  --early-stopping \
  --adaptive-max-new-tokens \
  --adaptive-ratio "$ADAPTIVE_RATIO" \
  --adaptive-margin "$ADAPTIVE_MARGIN" \
  --adaptive-min-new-tokens "$ADAPTIVE_MIN_NEW_TOKENS" \
  --adaptive-max-new-tokens-ceiling "$ADAPTIVE_MAX_NEW_TOKENS_CEILING" \
  --no-repeat-ngram-size "$NO_REPEAT_NGRAM_SIZE" \
  --repetition-penalty "$REPETITION_PENALTY" \
  --output-dir "$OUTPUT_DIR"
