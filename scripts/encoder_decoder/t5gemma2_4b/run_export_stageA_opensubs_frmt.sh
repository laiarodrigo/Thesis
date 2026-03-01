#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

PROJECT_DB_PATH="${PROJECT_DB_PATH:-$REPO_ROOT/data/duckdb/subs_project.duckdb}"
SOURCE_DB_PATH="${SOURCE_DB_PATH:-$REPO_ROOT/data/duckdb/subs.duckdb}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SOURCE_TABLE="${SOURCE_TABLE:-train_data}"
TRAIN_VIEW="${TRAIN_VIEW:-$SOURCE_TABLE}"
VALID_VIEW="${VALID_VIEW:-$SOURCE_TABLE}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VALID_SPLIT="${VALID_SPLIT:-valid}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged/stageA_opensubs_frmt}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

echo "== Pre-export dataset counts =="
"$PYTHON_BIN" "$REPO_ROOT/scripts/encoder_decoder/report_duckdb_dataset_counts.py" \
  --project-db "$PROJECT_DB_PATH" \
  --source-db "$SOURCE_DB_PATH" \
  --view "$TRAIN_VIEW" \
  --split "$TRAIN_SPLIT"

"$PYTHON_BIN" "$REPO_ROOT/scripts/encoder_decoder/report_duckdb_dataset_counts.py" \
  --project-db "$PROJECT_DB_PATH" \
  --source-db "$SOURCE_DB_PATH" \
  --view "$VALID_VIEW" \
  --split "$VALID_SPLIT"

echo "== Exporting Stage A data (OpenSubs + FRMT) =="
"$PYTHON_BIN" "$REPO_ROOT/scripts/encoder_decoder/export_encdec_data.py" \
  --project-db "$PROJECT_DB_PATH" \
  --source-db "$SOURCE_DB_PATH" \
  --train-view "$TRAIN_VIEW" \
  --train-split "$TRAIN_SPLIT" \
  --valid-view "$VALID_VIEW" \
  --valid-split "$VALID_SPLIT" \
  --dataset-include OpenSubs \
  --dataset-include FRMT \
  --out-dir "$OUT_DIR" \
  --translation-train-file translation_train.jsonl \
  --translation-valid-file translation_valid.jsonl \
  --classification-train-file classification_train.jsonl \
  --classification-valid-file classification_valid.jsonl
