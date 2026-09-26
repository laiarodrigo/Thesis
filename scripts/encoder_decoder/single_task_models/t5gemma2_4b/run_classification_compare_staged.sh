#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <stageA|stageAplus|stageB|stageBwiki> [extra args for step4b launcher]" >&2
  exit 2
fi

STAGE="$1"
shift
NOTE=""

case "$STAGE" in
  stageA)
    CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/classification_head_r24_stageA_opensubs_frmt.yaml"
    NOTE="stageA uses the historical OpenSubs-only classification Stage A config. For OpenSubs+FRMT Stage A, use: $0 stageAplus"
    ;;
  stageAplus)
    CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/classification_head_r24_stageA_opensubs_plus_frmt.yaml"
    ;;
  stageB)
    CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/classification_head_r24_stageB_gpt_adapt.yaml"
    NOTE="stageB uses the GPT+FRMT classification Stage B mix. For GPT-only Wikipedia Stage B, use: $0 stageBwiki"
    ;;
  stageBwiki)
    CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/classification_head_r24_stageB_gpt_wikipedia_only.yaml"
    ;;
  *)
    echo "Invalid stage: $STAGE (use stageA, stageAplus, stageB or stageBwiki)" >&2
    exit 2
    ;;
esac

if [[ -n "$NOTE" ]]; then
  echo "NOTE: $NOTE" >&2
fi

python3 scripts/encoder_decoder/single_task_models/t5gemma2/step4b_classification_head_boilerplate.py --config "$CONFIG" "$@"
