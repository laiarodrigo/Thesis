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
    CONFIG_PATH="configs/encoder_decoder/multitask_4b/stageA_compare_staged_v2.yaml"
    ;;
  stageB)
    CONFIG_PATH="configs/encoder_decoder/multitask_4b/stageB_gpt_refresh2_frmt.yaml"
    ;;
  *)
    echo "Invalid stage: $STAGE (use stageA or stageB)." >&2
    exit 2
    ;;
esac

python3 scripts/encoder_decoder/multitask/train_multitask_seq2seq.py \
  --config "$CONFIG_PATH" \
  "$@"
