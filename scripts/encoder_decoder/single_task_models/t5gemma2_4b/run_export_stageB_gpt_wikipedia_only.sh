#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INPUT_CSV="${INPUT_CSV:-$REPO_ROOT/data/wikipedia_pt_variant_csv/pt_variant_prompts_wikipedia_merged.csv}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VALID_RATIO="${VALID_RATIO:-0.1}"
SEED="${SEED:-42}"

PAIRS_ALL_FILE="${PAIRS_ALL_FILE:-pairs_all.jsonl}"
TRANSLATION_ALL_FILE="${TRANSLATION_ALL_FILE:-translation_all.jsonl}"
CLASSIFICATION_ALL_FILE="${CLASSIFICATION_ALL_FILE:-classification_all.jsonl}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "ERROR: missing input CSV: $INPUT_CSV" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

echo "Exporting Stage B data (GPT-only from merged Wikipedia CSV)"
echo "  input csv: $INPUT_CSV"
echo "  out dir: $OUT_DIR"
echo "  train ratio: $TRAIN_RATIO"
echo "  valid ratio: $VALID_RATIO"
echo "  seed: $SEED"
"$PYTHON_BIN" "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2/step2_build_all_tasks_from_csv.py" \
  --input-csv "$INPUT_CSV" \
  --out-dir "$OUT_DIR" \
  --pairs-file "$PAIRS_ALL_FILE" \
  --translation-file "$TRANSLATION_ALL_FILE" \
  --classification-file "$CLASSIFICATION_ALL_FILE"

"$PYTHON_BIN" "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2/step3_split_tasks.py" \
  --pairs-file "$OUT_DIR/$PAIRS_ALL_FILE" \
  --out-dir "$OUT_DIR" \
  --train-ratio "$TRAIN_RATIO" \
  --valid-ratio "$VALID_RATIO" \
  --seed "$SEED"

echo
echo "Line counts:"
wc -l \
  "$OUT_DIR/translation_train.jsonl" \
  "$OUT_DIR/translation_valid.jsonl" \
  "$OUT_DIR/translation_test.jsonl" \
  "$OUT_DIR/classification_train.jsonl" \
  "$OUT_DIR/classification_valid.jsonl" \
  "$OUT_DIR/classification_test.jsonl"
