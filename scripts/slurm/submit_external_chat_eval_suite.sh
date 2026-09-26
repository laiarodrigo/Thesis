#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/scripts/slurm/eval_amalia_chat_model.sbatch"

PARTITION="${PARTITION:-hlt_msc}"
GPU_GRES="${GPU_GRES:-gpu:1}"
TARGET_NODE="${TARGET_NODE:-}"
VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
DATA_PROTOCOL="${DATA_PROTOCOL:-control_strings}"
DATA_KIND="${DATA_KIND:-decoder_unified}"
DATASETS="${DATASETS:-golden frmt}"
TASKS="${TASKS:-translation classification}"
MODEL_IDS="${MODEL_IDS:-amalia-llm/AMALIA-9B-0626-SFT;Qwen/Qwen3-4B;microsoft/Phi-4-mini-instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results/decoder_only/external_chat_models}"
SKIP_DATA_PREFLIGHT="${SKIP_DATA_PREFLIGHT:-0}"

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

resolve_dataset_path() {
  local protocol="$1"
  local dataset_name="$2"
  local data_kind="$3"
  local task="$4"

  case "$protocol" in
    legacy)
      case "$dataset_name" in
        golden) dataset_root="data/encoder_decoder/t5gemma2/golden_collection" ;;
        frmt) dataset_root="data/encoder_decoder/t5gemma2/frmt_only" ;;
        *) return 1 ;;
      esac
      case "$task" in
        translation) printf '%s/translation_test.jsonl' "$dataset_root" ;;
        classification) printf '%s/classification_test.jsonl' "$dataset_root" ;;
        *) return 1 ;;
      esac
      ;;
    final)
      case "$data_kind" in
        translation_only|encoder_unified|decoder_unified) ;;
        *) return 1 ;;
      esac
      dataset_root="data/encoder_decoder/t5gemma2/final_eval/${dataset_name}/${data_kind}"
      case "$task" in
        translation) printf '%s/translation_test.jsonl' "$dataset_root" ;;
        classification)
          [[ "$data_kind" != "translation_only" ]] || return 1
          printf '%s/classification_noequal_test.jsonl' "$dataset_root"
          ;;
        *) return 1 ;;
      esac
      ;;
    control_strings)
      case "$data_kind" in
        encoder_unified|decoder_unified) ;;
        *) return 1 ;;
      esac
      dataset_root="data/encoder_decoder/t5gemma2/control_string_eval/${dataset_name}/${data_kind}"
      case "$task" in
        translation) printf '%s/translation_test.jsonl' "$dataset_root" ;;
        classification) printf '%s/classification_test.jsonl' "$dataset_root" ;;
        *) return 1 ;;
      esac
      ;;
    control_strings_ptbr)
      case "$data_kind" in
        encoder_unified|decoder_unified) ;;
        *) return 1 ;;
      esac
      dataset_root="data/encoder_decoder/t5gemma2/control_string_ptbr_eval/${dataset_name}/${data_kind}"
      case "$task" in
        translation) printf '%s/translation_test.jsonl' "$dataset_root" ;;
        classification) printf '%s/classification_test.jsonl' "$dataset_root" ;;
        *) return 1 ;;
      esac
      ;;
    *)
      return 1
      ;;
  esac
}

submit_job() {
  local model_id="$1"
  local model_slug="$2"
  local task="$3"
  local dataset_name="$4"
  local job_name="$5"
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

  MODEL_ID="$model_id" \
  MODEL_SLUG="$model_slug" \
  TASK="$task" \
  DATASET_NAME="$dataset_name" \
  DATA_PROTOCOL="$DATA_PROTOCOL" \
  DATA_KIND="$DATA_KIND" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  VENV_PATH="$VENV_PATH" \
  CLASSIFICATION_MODE="${CLASSIFICATION_MODE:-}" \
  CLASSIFICATION_CANDIDATES="${CLASSIFICATION_CANDIDATES:-}" \
  TOKENIZER_PATH="${TOKENIZER_PATH:-}" \
  ADAPTER_DIR="${ADAPTER_DIR:-}" \
  PYTHON_BIN="${PYTHON_BIN:-}" \
  sbatch "${sbatch_args[@]}" "$SBATCH_SCRIPT"
}

IFS=';' read -r -a MODEL_ID_ARRAY <<< "$MODEL_IDS"
read -r -a DATASET_ARRAY <<< "$DATASETS"
read -r -a TASK_ARRAY <<< "$TASKS"

if [[ "${#MODEL_ID_ARRAY[@]}" -eq 0 ]]; then
  echo "ERROR: MODEL_IDS is empty" >&2
  exit 2
fi

if [[ "$SKIP_DATA_PREFLIGHT" != "1" ]]; then
  for raw_dataset in "${DATASET_ARRAY[@]}"; do
    dataset_name="$(normalize_dataset_name "$raw_dataset")" || {
      echo "ERROR: unsupported dataset name: $raw_dataset" >&2
      exit 2
    }
    for task in "${TASK_ARRAY[@]}"; do
      dataset_path="$(resolve_dataset_path "$DATA_PROTOCOL" "$dataset_name" "$DATA_KIND" "$task")" || {
        echo "ERROR: unsupported dataset/task/protocol combination: protocol=$DATA_PROTOCOL data_kind=$DATA_KIND dataset=$dataset_name task=$task" >&2
        exit 2
      }
      if [[ ! -f "$REPO_ROOT/$dataset_path" ]]; then
        echo "ERROR: missing dataset path: $REPO_ROOT/$dataset_path" >&2
        echo "Set SKIP_DATA_PREFLIGHT=1 only if these files exist on the SLURM side but not in the current workspace." >&2
        exit 2
      fi
    done
  done
fi

mkdir -p "$REPO_ROOT/$OUTPUT_ROOT"
MANIFEST_PATH="$REPO_ROOT/$OUTPUT_ROOT/submitted_suite_$(date +%Y%m%d_%H%M%S).tsv"
printf 'job_id\tmodel_id\tmodel_slug\tdata_protocol\tdata_kind\tdataset\ttask\toutput_dir\n' > "$MANIFEST_PATH"

echo "Submitting external chat-model evaluation jobs with:"
echo "  partition=$PARTITION"
echo "  gres=$GPU_GRES"
if [[ -n "$TARGET_NODE" ]]; then
  echo "  target_node=$TARGET_NODE"
fi
echo "  data_protocol=$DATA_PROTOCOL"
echo "  data_kind=$DATA_KIND"
echo "  venv_path=$VENV_PATH"
echo "  output_root=$OUTPUT_ROOT"

for model_id in "${MODEL_ID_ARRAY[@]}"; do
  [[ -n "$model_id" ]] || continue
  model_slug="$(slugify_model_id "$model_id")"
  for raw_dataset in "${DATASET_ARRAY[@]}"; do
    dataset_name="$(normalize_dataset_name "$raw_dataset")" || {
      echo "ERROR: unsupported dataset name: $raw_dataset" >&2
      exit 2
    }
    for task in "${TASK_ARRAY[@]}"; do
      job_name="$(printf '%s_%s_%s' "$model_slug" "$dataset_name" "$task" | cut -c1-80)"
      job_id="$(submit_job "$model_id" "$model_slug" "$task" "$dataset_name" "$job_name")"
      output_dir="$OUTPUT_ROOT/$model_slug/$DATA_PROTOCOL/$DATA_KIND/$dataset_name/$task"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$job_id" \
        "$model_id" \
        "$model_slug" \
        "$DATA_PROTOCOL" \
        "$DATA_KIND" \
        "$dataset_name" \
        "$task" \
        "$output_dir" >> "$MANIFEST_PATH"
      echo "  submitted job_id=$job_id model=$model_id dataset=$dataset_name task=$task"
    done
  done
done

echo
echo "Manifest: $MANIFEST_PATH"
