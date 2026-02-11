#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/encoder_decoder/translation_lora.yaml}"
python scripts/encoder_decoder/train_encdec_lora.py --config "$CONFIG_PATH"
