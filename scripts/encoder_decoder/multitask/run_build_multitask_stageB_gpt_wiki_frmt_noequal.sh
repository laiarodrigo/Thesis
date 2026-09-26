#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TRANSLATION_TRAIN_IN="${TRANSLATION_TRAIN_IN:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/translation_train.jsonl}"
TRANSLATION_VALID_IN="${TRANSLATION_VALID_IN:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/translation_valid.jsonl}"
CLASSIFICATION_TRAIN_IN="${CLASSIFICATION_TRAIN_IN:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/classification_train.jsonl}"
CLASSIFICATION_VALID_IN="${CLASSIFICATION_VALID_IN:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/classification_valid.jsonl}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/multitask_stageB_gpt_wiki_frmt_noequal}"
EQUAL_MAX_RATIO="${EQUAL_MAX_RATIO:-0.0}"
EQUAL_MAX_RATIO_BY_DATASET="${EQUAL_MAX_RATIO_BY_DATASET:-}"
EQUAL_SEED="${EQUAL_SEED:-42}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" "$REPO_ROOT/scripts/encoder_decoder/multitask/build_multitask_jsonl.py" \
  --translation-in "$TRANSLATION_TRAIN_IN" \
  --classification-in "$CLASSIFICATION_TRAIN_IN" \
  --out "$OUT_DIR/train.jsonl" \
  --equal-max-ratio "$EQUAL_MAX_RATIO" \
  --equal-max-ratio-by-dataset "$EQUAL_MAX_RATIO_BY_DATASET" \
  --equal-seed "$EQUAL_SEED"

"$PYTHON_BIN" "$REPO_ROOT/scripts/encoder_decoder/multitask/build_multitask_jsonl.py" \
  --translation-in "$TRANSLATION_VALID_IN" \
  --classification-in "$CLASSIFICATION_VALID_IN" \
  --out "$OUT_DIR/valid.jsonl" \
  --equal-max-ratio "$EQUAL_MAX_RATIO" \
  --equal-max-ratio-by-dataset "$EQUAL_MAX_RATIO_BY_DATASET" \
  --equal-seed "$EQUAL_SEED"

echo "Wrote multitask Stage B data -> $OUT_DIR"
wc -l "$OUT_DIR/train.jsonl" "$OUT_DIR/valid.jsonl"
