#!/usr/bin/env bash
set -euo pipefail

python scripts/encoder_decoder/eval/evaluate_classification_head.py "$@"
