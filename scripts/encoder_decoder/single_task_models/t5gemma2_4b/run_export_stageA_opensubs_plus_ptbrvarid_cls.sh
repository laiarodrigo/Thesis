#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OPENSUBS_DIR="${OPENSUBS_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_clean}"
PTBRVID_DB="${PTBRVID_DB:-$REPO_ROOT/data/duckdb/subs_ptbr_filtered.duckdb}"
PTBRVID_SAMPLED_ROWS_CSV="${PTBRVID_SAMPLED_ROWS_CSV:-$REPO_ROOT/data/ptbrvarid/translated_stageb_pairs/sampled_rows.csv}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid}"
CLASSIFICATION_BALANCE_MODE="${CLASSIFICATION_BALANCE_MODE:-none}"
PTBRVID_EXCLUDE_DOMAINS="${PTBRVID_EXCLUDE_DOMAINS:-}"
TRAIN_PTBRVID_TARGET_SHARE="${TRAIN_PTBRVID_TARGET_SHARE:-0.0}"
VALID_PTBRVID_TARGET_SHARE="${VALID_PTBRVID_TARGET_SHARE:-0.0}"
SEED="${SEED:-42}"
PROGRESS_EVERY="${PROGRESS_EVERY:-200000}"

if [[ ! -f "$PTBRVID_SAMPLED_ROWS_CSV" ]]; then
  CANDIDATE="$REPO_ROOT/data/ptbrvarid/translated_stageb_pairs_r48_500_t50/sampled_rows_resampled.csv"
  if [[ -f "$CANDIDATE" ]]; then
    PTBRVID_SAMPLED_ROWS_CSV="$CANDIDATE"
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ ! -f "$PTBRVID_SAMPLED_ROWS_CSV" ]]; then
  echo "ERROR: missing sampled rows CSV: $PTBRVID_SAMPLED_ROWS_CSV" >&2
  exit 2
fi

OPENSUBS_DIR_REAL="$("$PYTHON_BIN" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$OPENSUBS_DIR")"
OUT_DIR_REAL="$("$PYTHON_BIN" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$OUT_DIR")"
if [[ "$OPENSUBS_DIR_REAL" == "$OUT_DIR_REAL" ]]; then
  echo "ERROR: OPENSUBS_DIR and OUT_DIR resolve to the same directory." >&2
  echo "This would read an already-augmented classification file and write PtBrVId rows back into it." >&2
  echo "Use a clean OpenSubs-only source directory and a separate PtBrVId output directory." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

echo "Exporting Stage A data (OpenSubs translation + leftover PtBrVId classification)"
echo "  opensubs dir: $OPENSUBS_DIR"
echo "  ptbrvid db: $PTBRVID_DB"
echo "  ptbrvid sampled rows: $PTBRVID_SAMPLED_ROWS_CSV"
echo "  out dir: $OUT_DIR"
echo "  classification balance mode: $CLASSIFICATION_BALANCE_MODE"
echo "  ptbrvid exclude domains: ${PTBRVID_EXCLUDE_DOMAINS:-<none>}"
echo "  train ptbrvid target share: $TRAIN_PTBRVID_TARGET_SHARE"
echo "  valid ptbrvid target share: $VALID_PTBRVID_TARGET_SHARE"
echo "  seed: $SEED"
echo "  progress every: $PROGRESS_EVERY"

"$PYTHON_BIN" \
  "$REPO_ROOT/scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_stageA_opensubs_ptbrvarid_cls.py" \
  --opensubs-translation-train "$OPENSUBS_DIR/translation_train.jsonl" \
  --opensubs-translation-valid "$OPENSUBS_DIR/translation_valid.jsonl" \
  --opensubs-classification-train "$OPENSUBS_DIR/classification_train.jsonl" \
  --opensubs-classification-valid "$OPENSUBS_DIR/classification_valid.jsonl" \
  --ptbrvarid-db "$PTBRVID_DB" \
  --ptbrvarid-sampled-csv "$PTBRVID_SAMPLED_ROWS_CSV" \
  --ptbrvarid-exclude-domains "$PTBRVID_EXCLUDE_DOMAINS" \
  --classification-balance-mode "$CLASSIFICATION_BALANCE_MODE" \
  --train-ptbrvarid-target-share "$TRAIN_PTBRVID_TARGET_SHARE" \
  --valid-ptbrvarid-target-share "$VALID_PTBRVID_TARGET_SHARE" \
  --seed "$SEED" \
  --progress-every "$PROGRESS_EVERY" \
  --out-dir "$OUT_DIR"

echo
echo "Line counts:"
wc -l \
  "$OUT_DIR/translation_train.jsonl" \
  "$OUT_DIR/translation_valid.jsonl" \
  "$OUT_DIR/classification_train.jsonl" \
  "$OUT_DIR/classification_valid.jsonl"
