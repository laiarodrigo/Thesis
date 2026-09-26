#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CONFIG_PATH LOG_PATH" >&2
  exit 2
fi

CONFIG_PATH="$1"
LOG_PATH="$2"

case "$CONFIG_PATH" in
  *comparison_staged*|*compare_staged_v2*)
    if [[ "${ALLOW_NON_FINAL_CONFIG:-0}" != "1" ]]; then
      echo "Refusing non-final config/data path in final-ablation wrapper: $CONFIG_PATH" >&2
      echo "Set ALLOW_NON_FINAL_CONFIG=1 only for an intentional recovered-run diagnostic." >&2
      exit 2
    fi
    ;;
esac

case "$CONFIG_PATH" in
  configs/encoder_decoder/t5gemma2_4b/final/*|configs/encoder_decoder/t5gemma2_4b/control_strings/*) ;;
  *)
    if [[ "${ALLOW_NON_FINAL_CONFIG:-0}" != "1" ]]; then
      echo "Expected a final or control-string protocol config: $CONFIG_PATH" >&2
      echo "Set ALLOW_NON_FINAL_CONFIG=1 only for an intentional diagnostic." >&2
      exit 2
    fi
    ;;
esac

mkdir -p "$(dirname "$LOG_PATH")"

cd "$(git rev-parse --show-toplevel)"
PYTHONUNBUFFERED=1 python scripts/encoder_decoder/train_encdec_lora.py \
  --config "$CONFIG_PATH" \
  2>&1 | tee "$LOG_PATH"
