#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  cat >&2 <<'USAGE'
Usage: run_eval_4b_model.sh NAME ADAPTER_DIR DATA_KIND OUT_ROOT [NUM_BEAMS]

DATA_KIND is one of:
  translation_only
  encoder_unified
  decoder_unified

Examples:
  CUDA_VISIBLE_DEVICES=1 run_eval_4b_model.sh B4-E-G outputs/... encoder_unified eval_results/... 1
  CUDA_VISIBLE_DEVICES=1 run_eval_4b_model.sh B4-E-G outputs/... encoder_unified eval_results/... 4
USAGE
  exit 2
fi

NAME="$1"
ADAPTER_DIR="$2"
DATA_KIND="$3"
OUT_ROOT="$4"
NUM_BEAMS="${5:-1}"

case "$ADAPTER_DIR" in
  *compare_staged*|*comparison_staged*)
    if [[ "${ALLOW_NON_FINAL_ADAPTER:-0}" != "1" ]]; then
      echo "Refusing recovered/legacy adapter in final-ablation eval wrapper: $ADAPTER_DIR" >&2
      echo "Use ALLOW_NON_FINAL_ADAPTER=1 only for an intentional recovered-run diagnostic." >&2
      exit 2
    fi
    ;;
esac

case "$ADAPTER_DIR" in
  outputs/encoder_decoder/final/*) ;;
  *)
    if [[ "${ALLOW_NON_FINAL_ADAPTER:-0}" != "1" ]]; then
      echo "Expected a final-protocol adapter under outputs/encoder_decoder/final/: $ADAPTER_DIR" >&2
      echo "Use ALLOW_NON_FINAL_ADAPTER=1 only for an intentional diagnostic." >&2
      exit 2
    fi
    ;;
esac

MODEL_ID="google/t5gemma-2-4b-4b"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

translation_args=(
  --task translation
  --model-id "$MODEL_ID"
  --adapter-dir "$ADAPTER_DIR"
  --tokenizer-path "$ADAPTER_DIR"
  --batch-size 4
  --max-source-length 512
  --max-new-tokens 256
)

if [[ "$NUM_BEAMS" -gt 1 ]]; then
  translation_args+=(
    --num-beams "$NUM_BEAMS"
    --length-penalty 0.8
    --early-stopping
    --adaptive-max-new-tokens
    --adaptive-ratio 1.15
    --adaptive-margin 6
    --adaptive-min-new-tokens 8
    --adaptive-max-new-tokens-ceiling 384
    --no-repeat-ngram-size 4
    --repetition-penalty 1.2
  )
else
  translation_args+=(
    --adaptive-max-new-tokens
    --adaptive-ratio 1.3
    --adaptive-margin 10
    --adaptive-min-new-tokens 16
    --adaptive-max-new-tokens-ceiling 256
  )
fi

for dataset in frmt golden; do
  python scripts/encoder_decoder/eval/evaluate_encdec.py \
    "${translation_args[@]}" \
    --dataset-path "data/encoder_decoder/t5gemma2/final_eval/${dataset}/${DATA_KIND}/translation_test.jsonl" \
    --output-dir "${OUT_ROOT}/${NAME}/${dataset}_translation"
done

if [[ "$DATA_KIND" != "translation_only" ]]; then
  for dataset in frmt golden; do
    python scripts/encoder_decoder/eval/evaluate_encdec.py \
      --task classification \
      --dataset-path "data/encoder_decoder/t5gemma2/final_eval/${dataset}/${DATA_KIND}/classification_noequal_test.jsonl" \
      --model-id "$MODEL_ID" \
      --adapter-dir "$ADAPTER_DIR" \
      --tokenizer-path "$ADAPTER_DIR" \
      --batch-size 16 \
      --max-source-length 512 \
      --classification-mode score-first-token \
      --classification-candidates "<pt-br>" "<pt-pt>" \
      --output-dir "${OUT_ROOT}/${NAME}/${dataset}_classification"
  done
fi
