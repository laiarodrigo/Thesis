#!/usr/bin/env bash
set -euo pipefail

python3 scripts/decoder_only/axolotl/export_qwen_chat_data.py \
  --task-mode translation \
  --out-dir data/decoder_only/qwen3_translation \
  --train-file train.jsonl \
  --valid-file valid.jsonl \
  "$@"
