#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INPUT_CSV="${INPUT_CSV:-$REPO_ROOT/data/ptbrvarid/translated_stageb_pairs/translated_pairs.csv}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/ptbrvarid_translated_stageB}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VALID_RATIO="${VALID_RATIO:-0.1}"
TEST_PER_GROUP="${TEST_PER_GROUP:-0}"
SEED="${SEED:-42}"
EXCLUDE_DOMAINS="${EXCLUDE_DOMAINS:-}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "ERROR: missing input CSV: $INPUT_CSV" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

echo "Exporting PtBrVId translated pairs to encdec JSONL"
echo "  input csv: $INPUT_CSV"
echo "  out dir: $OUT_DIR"
echo "  train ratio: $TRAIN_RATIO"
echo "  valid ratio: $VALID_RATIO"
echo "  test per group: $TEST_PER_GROUP"
echo "  exclude domains: ${EXCLUDE_DOMAINS:-<none>}"
echo "  seed: $SEED"

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/ptbrvarid/export_translated_pairs_to_encdec.py" \
  --input-csv "$INPUT_CSV" \
  --out-dir "$OUT_DIR" \
  --train-ratio "$TRAIN_RATIO" \
  --valid-ratio "$VALID_RATIO" \
  --test-per-group "$TEST_PER_GROUP" \
  --exclude-domains "$EXCLUDE_DOMAINS" \
  --seed "$SEED"

echo
echo "Line counts:"
wc -l \
  "$OUT_DIR/pairs_train.jsonl" \
  "$OUT_DIR/pairs_valid.jsonl" \
  "$OUT_DIR/pairs_test.jsonl" \
  "$OUT_DIR/translation_train.jsonl" \
  "$OUT_DIR/translation_valid.jsonl" \
  "$OUT_DIR/classification_source_test.jsonl"
