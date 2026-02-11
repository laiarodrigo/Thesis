#!/usr/bin/env bash
set -euo pipefail

python scripts/encoder_decoder/eval/evaluate_encdec.py \
  --task classification \
  "$@"
