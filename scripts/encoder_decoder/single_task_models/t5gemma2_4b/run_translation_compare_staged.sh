#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <stageA|stageB> <r16|r24> [extra args for step4a launcher]" >&2
  exit 2
fi

STAGE="$1"
RANK="$2"
shift 2

case "$STAGE" in
  stageA)
    case "$RANK" in
      r16) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r16_stageA_opensubs_frmt.yaml" ;;
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_frmt.yaml" ;;
      *) echo "Invalid rank for stageA: $RANK (use r16 or r24)" >&2; exit 2 ;;
    esac
    ;;
  stageB)
    case "$RANK" in
      r16) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r16_stageB_gpt_adapt.yaml" ;;
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageB_gpt_adapt.yaml" ;;
      *) echo "Invalid rank for stageB: $RANK (use r16 or r24)" >&2; exit 2 ;;
    esac
    ;;
  *)
    echo "Invalid stage: $STAGE (use stageA or stageB)" >&2
    exit 2
    ;;
esac

python3 scripts/encoder_decoder/single_task_models/t5gemma2/step4a_translation_train_boilerplate.py --config "$CONFIG" "$@"
