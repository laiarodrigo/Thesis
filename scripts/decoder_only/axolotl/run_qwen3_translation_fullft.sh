#!/usr/bin/env bash
set -euo pipefail

# Work around missing telemetry whitelist in some axolotl wheels.
export AXOLOTL_DO_NOT_TRACK="${AXOLOTL_DO_NOT_TRACK:-1}"

CONFIG_PATH="${1:-configs/decoder_only/axolotl/qwen3_translation_fullft.yaml}"

axolotl preprocess "$CONFIG_PATH" --debug
axolotl train "$CONFIG_PATH"
