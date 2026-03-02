#!/usr/bin/env bash
set -euo pipefail

python scripts/encoder_decoder/single_task_models/export_encdec_data.py "$@"
