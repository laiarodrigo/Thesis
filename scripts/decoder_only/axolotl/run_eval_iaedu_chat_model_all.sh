#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL_ID="${MODEL_ID:-iaedu-gpt-latest}"
MODEL_SLUG="${MODEL_SLUG:-}"
DATA_PROTOCOL="${DATA_PROTOCOL:-control_strings}"
DATA_KIND="${DATA_KIND:-decoder_unified}"
DATASETS="${DATASETS:-golden frmt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/eval_results/decoder_only/iaedu_chat_models}"
CLASSIFICATION_BATCH_SIZE="${CLASSIFICATION_BATCH_SIZE:-32}"
CONCURRENCY="${CONCURRENCY:-4}"
CLASSIFICATION_CANDIDATES="${CLASSIFICATION_CANDIDATES:-}"
TRANSLATION_IAEDU_ENV_FILE="${TRANSLATION_IAEDU_ENV_FILE:-${IAEDU_ENV_FILE:-bla.env}}"
CLASSIFICATION_IAEDU_ENV_FILE="${CLASSIFICATION_IAEDU_ENV_FILE:-${IAEDU_ENV_FILE:-bla.env}}"

normalize_dataset_name() {
  local raw_name="$1"
  case "$raw_name" in
    golden|golden_collection) printf 'golden' ;;
    frmt|frmt_only) printf 'frmt' ;;
    *) return 1 ;;
  esac
}

slugify_model_id() {
  printf '%s' "$1" | tr '/:@ ' '_' | tr -cs '[:alnum:]_.-' '_'
}

resolve_translation_dataset_path() {
  local protocol="$1"
  local dataset_name="$2"
  local data_kind="$3"
  case "$protocol" in
    legacy)
      case "$dataset_name" in
        golden) printf '%s/data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl' "$REPO_ROOT" ;;
        frmt) printf '%s/data/encoder_decoder/t5gemma2/frmt_only/translation_test.jsonl' "$REPO_ROOT" ;;
        *) return 1 ;;
      esac
      ;;
    final)
      printf '%s/data/encoder_decoder/t5gemma2/final_eval/%s/%s/translation_test.jsonl' "$REPO_ROOT" "$dataset_name" "$data_kind"
      ;;
    control_strings)
      printf '%s/data/encoder_decoder/t5gemma2/control_string_eval/%s/%s/translation_test.jsonl' "$REPO_ROOT" "$dataset_name" "$data_kind"
      ;;
    control_strings_ptbr)
      printf '%s/data/encoder_decoder/t5gemma2/control_string_ptbr_eval/%s/%s/translation_test.jsonl' "$REPO_ROOT" "$dataset_name" "$data_kind"
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_classification_dataset_path() {
  local protocol="$1"
  local dataset_name="$2"
  local data_kind="$3"
  case "$protocol" in
    legacy)
      case "$dataset_name" in
        golden) printf '%s/data/encoder_decoder/t5gemma2/golden_collection/classification_test.jsonl' "$REPO_ROOT" ;;
        frmt) printf '%s/data/encoder_decoder/t5gemma2/frmt_only/classification_test.jsonl' "$REPO_ROOT" ;;
        *) return 1 ;;
      esac
      ;;
    final)
      [[ "$data_kind" != "translation_only" ]] || return 1
      printf '%s/data/encoder_decoder/t5gemma2/final_eval/%s/%s/classification_noequal_test.jsonl' "$REPO_ROOT" "$dataset_name" "$data_kind"
      ;;
    control_strings)
      printf '%s/data/encoder_decoder/t5gemma2/control_string_eval/%s/%s/classification_test.jsonl' "$REPO_ROOT" "$dataset_name" "$data_kind"
      ;;
    control_strings_ptbr)
      printf '%s/data/encoder_decoder/t5gemma2/control_string_ptbr_eval/%s/%s/classification_test.jsonl' "$REPO_ROOT" "$dataset_name" "$data_kind"
      ;;
    *)
      return 1
      ;;
  esac
}

case "$DATA_PROTOCOL" in
  legacy)
    OUTPUT_VIEW="legacy"
    if [[ -z "$CLASSIFICATION_CANDIDATES" ]]; then
      CLASSIFICATION_CANDIDATES="pt-br pt-pt"
    fi
    ;;
  final)
    case "$DATA_KIND" in
      translation_only|encoder_unified|decoder_unified) ;;
      *) echo "ERROR: invalid DATA_KIND=$DATA_KIND" >&2; exit 2 ;;
    esac
    OUTPUT_VIEW="$DATA_KIND"
    if [[ -z "$CLASSIFICATION_CANDIDATES" ]]; then
      CLASSIFICATION_CANDIDATES="pt-br pt-pt"
    fi
    ;;
  control_strings)
    case "$DATA_KIND" in
      encoder_unified|decoder_unified) ;;
      *) echo "ERROR: invalid DATA_KIND=$DATA_KIND" >&2; exit 2 ;;
    esac
    OUTPUT_VIEW="$DATA_KIND"
    if [[ -z "$CLASSIFICATION_CANDIDATES" ]]; then
      CLASSIFICATION_CANDIDATES="<pt-br> <pt-pt>"
    fi
    ;;
  control_strings_ptbr)
    case "$DATA_KIND" in
      encoder_unified|decoder_unified) ;;
      *) echo "ERROR: invalid DATA_KIND=$DATA_KIND" >&2; exit 2 ;;
    esac
    OUTPUT_VIEW="$DATA_KIND"
    if [[ -z "$CLASSIFICATION_CANDIDATES" ]]; then
      CLASSIFICATION_CANDIDATES="BR PT"
    fi
    ;;
  *)
    echo "ERROR: DATA_PROTOCOL must be one of legacy|final|control_strings|control_strings_ptbr" >&2
    exit 2
    ;;
esac

if [[ -z "$MODEL_SLUG" ]]; then
  MODEL_SLUG="$(slugify_model_id "$MODEL_ID")"
fi

read -r -a DATASET_ARRAY <<< "$DATASETS"

for raw_dataset in "${DATASET_ARRAY[@]}"; do
  dataset_name="$(normalize_dataset_name "$raw_dataset")" || { echo "ERROR: unsupported dataset=$raw_dataset" >&2; exit 2; }
  translation_dataset="$(resolve_translation_dataset_path "$DATA_PROTOCOL" "$dataset_name" "$DATA_KIND")" || { echo "ERROR: bad translation dataset combo" >&2; exit 2; }
  classification_dataset="$(resolve_classification_dataset_path "$DATA_PROTOCOL" "$dataset_name" "$DATA_KIND")" || { echo "ERROR: bad classification dataset combo" >&2; exit 2; }

  translation_output_dir="$OUTPUT_ROOT/$MODEL_SLUG/$DATA_PROTOCOL/$OUTPUT_VIEW/$dataset_name/translation"
  classification_output_dir="$OUTPUT_ROOT/$MODEL_SLUG/$DATA_PROTOCOL/$OUTPUT_VIEW/$dataset_name/classification"
  mkdir -p "$translation_output_dir" "$classification_output_dir"

  echo "[$(date)] model=$MODEL_ID dataset=$dataset_name task=translation"
  MODEL_ID="$MODEL_ID" \
  DATASET_PATH="$translation_dataset" \
  OUTPUT_DIR="$translation_output_dir" \
  CONCURRENCY="$CONCURRENCY" \
  IAEDU_ENV_FILE="$TRANSLATION_IAEDU_ENV_FILE" \
  "${REPO_ROOT}/scripts/decoder_only/axolotl/run_eval_translation_iaedu_chat_model.sh" "$@"

  echo "[$(date)] model=$MODEL_ID dataset=$dataset_name task=classification"
  MODEL_ID="$MODEL_ID" \
  DATASET_PATH="$classification_dataset" \
  OUTPUT_DIR="$classification_output_dir" \
  BATCH_SIZE="$CLASSIFICATION_BATCH_SIZE" \
  CONCURRENCY="$CONCURRENCY" \
  CLASSIFICATION_CANDIDATES="$CLASSIFICATION_CANDIDATES" \
  IAEDU_ENV_FILE="$CLASSIFICATION_IAEDU_ENV_FILE" \
  "${REPO_ROOT}/scripts/decoder_only/axolotl/run_eval_classification_iaedu_chat_model.sh" "$@"
done
