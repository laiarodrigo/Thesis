#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/decoder_only/axolotl/qwen3_lora.yaml}"

axolotl preprocess "$CONFIG_PATH" --debug
axolotl train "$CONFIG_PATH"
