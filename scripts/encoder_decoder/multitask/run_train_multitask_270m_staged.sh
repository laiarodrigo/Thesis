#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <stageA|stageB> [--execute]" >&2
  exit 2
fi

STAGE="$1"
shift || true

case "$STAGE" in
  stageA)
    CONFIG_PATH="configs/encoder_decoder/multitask_270m/stageA_opensubs_frmt.yaml"
    ;;
  stageB)
    CONFIG_PATH="configs/encoder_decoder/multitask_270m/stageB_gpt_adapt.yaml"
    ;;
  *)
    echo "Invalid stage: $STAGE (use stageA or stageB)." >&2
    exit 2
    ;;
esac

python3 scripts/encoder_decoder/multitask/train_multitask_seq2seq.py \
  --config "$CONFIG_PATH" \
  "$@"

