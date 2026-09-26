#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPU="${GPU:-${1:-}}"
VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
TRANSLATION_BATCH_SIZE="${TRANSLATION_BATCH_SIZE:-4}"
CLASSIFICATION_BATCH_SIZE="${CLASSIFICATION_BATCH_SIZE:-16}"

if [[ -z "$GPU" ]]; then
  echo "ERROR: set GPU to a confirmed free g07 GPU index." >&2
  exit 2
fi

cd "$REPO_ROOT"
source "$VENV_PATH/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

STAGE_A_CONFIG="configs/encoder_decoder/t5gemma2_4b/final/translation_r48_stageA_opensubs_only_encoder_unified_noequal_cls_unrestricted_cls_loss_final.yaml"
STAGE_B_CONFIG="configs/encoder_decoder/t5gemma2_4b/final/translation_r48_stageB_gpt_wikipedia_only_encoder_unified_noequal_cls_unrestricted_cls_loss_full_pipeline_final.yaml"
STAGE_A_OUTPUT="outputs/encoder_decoder/final/t5gemma2_4b_translation_r48_stageA_opensubs_only_encoder_unified_noequal_cls_unrestricted_cls_loss_final"
STAGE_B_OUTPUT="outputs/encoder_decoder/final/t5gemma2_4b_translation_r48_stageB_gpt_wiki_encoder_unified_noequal_cls_unrestricted_cls_loss_full_pipeline_final"
EVAL_ROOT="eval_results/encoder_decoder/final_4b_ablation_beam4/B4-E-G-noeq-unrestricted-full-pipeline"
LOG_ROOT="logs/final_4b_noeq_unrestricted_full_pipeline"

mkdir -p "$LOG_ROOT"

latest_complete_checkpoint() {
  local output_dir=$1
  local candidate
  local latest=""
  local latest_step=-1
  local step

  for candidate in "$output_dir"/checkpoint-*; do
    [[ -d "$candidate" ]] || continue
    step=${candidate##*-}
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if [[ -s "$candidate/adapter_config.json" \
      && -s "$candidate/adapter_model.safetensors" \
      && -s "$candidate/trainer_state.json" \
      && -s "$candidate/optimizer.pt" \
      && -s "$candidate/scheduler.pt" ]]; then
      if (( step > latest_step )); then
        latest=$candidate
        latest_step=$step
      fi
    fi
  done
  printf '%s' "$latest"
}

run_training_stage() {
  local label=$1
  local config=$2
  local output_dir=$3
  local log_path=$4
  local resume_checkpoint

  if [[ -s "$output_dir/adapter_config.json" \
    && -s "$output_dir/adapter_model.safetensors" \
    && -s "$output_dir/tokenizer.json" ]]; then
    echo "[$(date)] $label final output already exists; skipping training"
    return
  fi

  resume_checkpoint=$(latest_complete_checkpoint "$output_dir")
  if [[ -n "$resume_checkpoint" ]]; then
    echo "[$(date)] $label resuming from $resume_checkpoint"
    THESIS_RESUME_FROM_CHECKPOINT="$resume_checkpoint" \
      python scripts/encoder_decoder/train_encdec_lora.py \
        --config "$config" \
        2>&1 | tee -a "$log_path"
  else
    echo "[$(date)] $label starting from the configured initialization"
    env -u THESIS_RESUME_FROM_CHECKPOINT -u RESUME_FROM_CHECKPOINT \
      python scripts/encoder_decoder/train_encdec_lora.py \
        --config "$config" \
        2>&1 | tee -a "$log_path"
  fi
}

write_output_checksums() {
  local output_dir=$1
  (
    cd "$output_dir"
    sha256sum adapter_config.json adapter_model.safetensors tokenizer.json \
      > sanity_checksums.sha256
  )
}

for path in \
  "$STAGE_A_CONFIG" \
  "$STAGE_B_CONFIG" \
  data/encoder_decoder/t5gemma2/final_protocol/stageA_opensubs_only/encoder_unified_noequal_cls/train.jsonl \
  data/encoder_decoder/t5gemma2/final_protocol/stageA_opensubs_only/encoder_unified_noequal_cls/valid.jsonl \
  data/encoder_decoder/t5gemma2/final_protocol/stageB_gpt_wiki/encoder_unified_noequal_cls/train.jsonl \
  data/encoder_decoder/t5gemma2/final_protocol/stageB_gpt_wiki/encoder_unified_noequal_cls/valid.jsonl; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing or empty required file: $path" >&2
    exit 2
  fi
done

echo "[$(date)] host=$(hostname) physical_gpu=$GPU"
nvidia-smi -i "$GPU" \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv

run_training_stage \
  "Stage A" \
  "$STAGE_A_CONFIG" \
  "$STAGE_A_OUTPUT" \
  "$LOG_ROOT/A4-E-noeq-unrestricted-full_g07_gpu${GPU}.log"

test -s "$STAGE_A_OUTPUT/adapter_config.json"
test -s "$STAGE_A_OUTPUT/adapter_model.safetensors"
write_output_checksums "$STAGE_A_OUTPUT"

run_training_stage \
  "Stage B" \
  "$STAGE_B_CONFIG" \
  "$STAGE_B_OUTPUT" \
  "$LOG_ROOT/B4-E-G-noeq-unrestricted-full_g07_gpu${GPU}.log"

test -s "$STAGE_B_OUTPUT/adapter_config.json"
test -s "$STAGE_B_OUTPUT/adapter_model.safetensors"
write_output_checksums "$STAGE_B_OUTPUT"

for dataset in frmt golden; do
  data_root="data/encoder_decoder/t5gemma2/final_eval/${dataset}/encoder_unified"

  python scripts/encoder_decoder/eval/evaluate_encdec.py \
    --task translation \
    --dataset-path "$data_root/translation_test.jsonl" \
    --model-id google/t5gemma-2-4b-4b \
    --adapter-dir "$STAGE_B_OUTPUT" \
    --tokenizer-path "$STAGE_B_OUTPUT" \
    --batch-size "$TRANSLATION_BATCH_SIZE" \
    --max-source-length 512 \
    --max-new-tokens 256 \
    --num-beams 4 \
    --length-penalty 0.8 \
    --early-stopping \
    --adaptive-max-new-tokens \
    --adaptive-ratio 1.15 \
    --adaptive-margin 6 \
    --adaptive-min-new-tokens 8 \
    --adaptive-max-new-tokens-ceiling 384 \
    --no-repeat-ngram-size 4 \
    --repetition-penalty 1.2 \
    --output-dir "$EVAL_ROOT/${dataset}_translation" \
    2>&1 | tee -a "$LOG_ROOT/eval_${dataset}_translation_g07_gpu${GPU}.log"

  python scripts/encoder_decoder/eval/evaluate_encdec.py \
    --task classification \
    --dataset-path "$data_root/classification_noequal_test.jsonl" \
    --model-id google/t5gemma-2-4b-4b \
    --adapter-dir "$STAGE_B_OUTPUT" \
    --tokenizer-path "$STAGE_B_OUTPUT" \
    --batch-size "$CLASSIFICATION_BATCH_SIZE" \
    --max-source-length 512 \
    --classification-mode score-first-token \
    --classification-candidates '<pt-br>' '<pt-pt>' \
    --output-dir "$EVAL_ROOT/${dataset}_classification" \
    2>&1 | tee -a "$LOG_ROOT/eval_${dataset}_classification_g07_gpu${GPU}.log"
done

echo "[$(date)] full pipeline completed"
