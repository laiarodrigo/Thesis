#!/usr/bin/env bash
set -euo pipefail

python3 scripts/test_qwen_lora.py \
  --no-adapter \
  --model-id outputs/decoder_only/qwen3_translation/fullft \
  --eval-tasks translation \
  --output-dir eval_results/decoder_only/qwen3_translation/fullft \
  "$@"
