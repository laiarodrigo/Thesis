#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/encoder_decoder/multitask_4b/stageB_gpt_wiki_frmt_from_translation_stageA_r48_noequal.yaml}"
shift || true

python3 scripts/encoder_decoder/multitask/train_multitask_seq2seq.py \
  --config "$CONFIG_PATH" \
  "$@"
