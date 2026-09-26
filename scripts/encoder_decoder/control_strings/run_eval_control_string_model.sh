#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 NAME ADAPTER_DIR DATA_KIND OUT_ROOT [NUM_BEAMS]" >&2
  exit 2
fi

NAME="$1"
ADAPTER_DIR="$2"
DATA_KIND="$3"
OUT_ROOT="$4"
NUM_BEAMS="${5:-1}"
MODEL_ID="google/t5gemma-2-4b-4b"
CONTROL_EVAL_ROOT="${CONTROL_EVAL_ROOT:-data/encoder_decoder/t5gemma2/control_string_eval}"
CLASSIFICATION_CANDIDATE_BR="${CLASSIFICATION_CANDIDATE_BR:-<pt-br>}"
CLASSIFICATION_CANDIDATE_PT="${CLASSIFICATION_CANDIDATE_PT:-<pt-pt>}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

case "$DATA_KIND" in
  encoder_unified|decoder_unified) ;;
  *) echo "DATA_KIND must be encoder_unified or decoder_unified" >&2; exit 2 ;;
esac

case "$ADAPTER_DIR" in
  outputs/encoder_decoder/control_strings/*|outputs/encoder_decoder/control_strings_ptbr/*) ;;
  *) echo "Expected a control-string adapter: $ADAPTER_DIR" >&2; exit 2 ;;
esac

translation_args=(
  --task translation
  --model-id "$MODEL_ID"
  --adapter-dir "$ADAPTER_DIR"
  --tokenizer-path "$ADAPTER_DIR"
  --batch-size 4
  --max-source-length 512
  --max-new-tokens 256
  --adaptive-max-new-tokens
)

if [[ "$NUM_BEAMS" -gt 1 ]]; then
  translation_args+=(
    --num-beams "$NUM_BEAMS"
    --length-penalty 0.8
    --early-stopping
    --adaptive-ratio 1.15
    --adaptive-margin 6
    --adaptive-min-new-tokens 8
    --adaptive-max-new-tokens-ceiling 384
    --no-repeat-ngram-size 4
    --repetition-penalty 1.2
  )
else
  translation_args+=(
    --adaptive-ratio 1.3
    --adaptive-margin 10
    --adaptive-min-new-tokens 16
    --adaptive-max-new-tokens-ceiling 256
  )
fi

for dataset in frmt golden; do
  python scripts/encoder_decoder/eval/evaluate_encdec.py \
    "${translation_args[@]}" \
    --dataset-path "${CONTROL_EVAL_ROOT}/${dataset}/${DATA_KIND}/translation_test.jsonl" \
    --output-dir "${OUT_ROOT}/${NAME}/${dataset}_translation"

  python scripts/encoder_decoder/eval/evaluate_encdec.py \
    --task classification \
    --dataset-path "${CONTROL_EVAL_ROOT}/${dataset}/${DATA_KIND}/classification_test.jsonl" \
    --model-id "$MODEL_ID" \
    --adapter-dir "$ADAPTER_DIR" \
    --tokenizer-path "$ADAPTER_DIR" \
    --batch-size 16 \
    --max-source-length 512 \
    --classification-mode score-sequences \
    --classification-candidates "$CLASSIFICATION_CANDIDATE_BR" "$CLASSIFICATION_CANDIDATE_PT" \
    --output-dir "${OUT_ROOT}/${NAME}/${dataset}_classification"
done
