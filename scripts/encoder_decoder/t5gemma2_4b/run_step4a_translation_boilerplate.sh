#!/usr/bin/env bash
set -euo pipefail

python3 scripts/encoder_decoder/t5gemma2/step4a_translation_train_boilerplate.py \
  --config configs/encoder_decoder/t5gemma2_4b/translation_lora_boilerplate.yaml \
  "$@"
