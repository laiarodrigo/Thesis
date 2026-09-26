#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

"$REPO_ROOT/scripts/decoder_only/axolotl/run_eval_amalia_translation_golden.sh" "$@"
"$REPO_ROOT/scripts/decoder_only/axolotl/run_eval_amalia_translation_frmt.sh" "$@"
"$REPO_ROOT/scripts/decoder_only/axolotl/run_eval_amalia_classification_golden.sh" "$@"
"$REPO_ROOT/scripts/decoder_only/axolotl/run_eval_amalia_classification_frmt.sh" "$@"
