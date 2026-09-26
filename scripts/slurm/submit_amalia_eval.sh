#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/scripts/slurm/eval_amalia_chat_model.sbatch"

PARTITION="${PARTITION:-hlt_msc}"
GPU_GRES="${GPU_GRES:-gpu:1}"
TARGET_NODE="${TARGET_NODE:-}"
MODEL_ID="${MODEL_ID:-amalia-llm/AMALIA-9B-0626-SFT}"
VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
DATA_PROTOCOL="${DATA_PROTOCOL:-control_strings}"
DATA_KIND="${DATA_KIND:-decoder_unified}"

normalize_dataset_name() {
  local raw_name="$1"
  case "$raw_name" in
    golden|golden_collection)
      printf 'golden'
      ;;
    frmt|frmt_only)
      printf 'frmt'
      ;;
    *)
      return 1
      ;;
  esac
}

slugify_model_id() {
  printf '%s' "$1" | tr '/:@ ' '_' | tr -cs '[:alnum:]_.-' '_'
}

MODEL_SLUG="${MODEL_SLUG:-$(slugify_model_id "$MODEL_ID")}"

submit_job() {
  local job_name="$1"
  local task="$2"
  local dataset_name="$3"
  local -a sbatch_args=(
    --parsable
    --partition="$PARTITION"
    --gres="$GPU_GRES"
    --job-name="$job_name"
    --export=ALL
  )

  if [[ -n "$TARGET_NODE" ]]; then
    sbatch_args+=(--nodelist="$TARGET_NODE")
  fi

  MODEL_ID="$MODEL_ID" \
  MODEL_SLUG="$MODEL_SLUG" \
  TASK="$task" \
  DATASET_NAME="$dataset_name" \
  DATA_PROTOCOL="$DATA_PROTOCOL" \
  DATA_KIND="$DATA_KIND" \
  VENV_PATH="$VENV_PATH" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results/decoder_only/external_chat_models}" \
  CLASSIFICATION_MODE="${CLASSIFICATION_MODE:-}" \
  CLASSIFICATION_CANDIDATES="${CLASSIFICATION_CANDIDATES:-}" \
  TOKENIZER_PATH="${TOKENIZER_PATH:-}" \
  ADAPTER_DIR="${ADAPTER_DIR:-}" \
  PYTHON_BIN="${PYTHON_BIN:-}" \
  sbatch "${sbatch_args[@]}" "$SBATCH_SCRIPT"
}

echo "Submitting AMALIA evaluation jobs with:"
echo "  partition=$PARTITION"
echo "  gres=$GPU_GRES"
if [[ -n "$TARGET_NODE" ]]; then
  echo "  target_node=$TARGET_NODE"
fi
echo "  model_id=$MODEL_ID"
echo "  data_protocol=$DATA_PROTOCOL"
echo "  data_kind=$DATA_KIND"
echo "  venv_path=$VENV_PATH"

submit_job "amalia_gld_tr" "translation" "$(normalize_dataset_name golden)"
submit_job "amalia_frmt_tr" "translation" "$(normalize_dataset_name frmt)"
submit_job "amalia_gld_cls" "classification" "$(normalize_dataset_name golden)"
submit_job "amalia_frmt_cls" "classification" "$(normalize_dataset_name frmt)"
