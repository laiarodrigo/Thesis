#!/usr/bin/env bash
set -euo pipefail

python3 scripts/encoder_decoder/t5gemma2/step4b_classification_head_boilerplate.py \
  --config configs/encoder_decoder/t5gemma2_4b/classification_head_boilerplate.yaml \
  "$@"
