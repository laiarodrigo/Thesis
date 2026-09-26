#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <stageA|stageB> [extra args for step4b launcher]" >&2
  exit 2
fi

STAGE="$1"
shift

case "$STAGE" in
  stageA)
    CONFIG="configs/encoder_decoder/t5gemma2/comparison_staged/classification_head_fullft_stageA_opensubs_frmt.yaml"
    ;;
  stageB)
    CONFIG="configs/encoder_decoder/t5gemma2/comparison_staged/classification_head_fullft_stageB_gpt_adapt.yaml"
    ;;
  *)
    echo "Invalid stage: $STAGE (use stageA or stageB)" >&2
    exit 2
    ;;
esac

python3 scripts/encoder_decoder/single_task_models/t5gemma2/step4b_classification_head_boilerplate.py --config "$CONFIG" "$@"
