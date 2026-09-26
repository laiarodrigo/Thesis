#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_SCRIPT="$REPO_ROOT/scripts/decoder_only/axolotl/run_eval_external_chat_model_all.sh"

VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
DATA_PROTOCOL="${DATA_PROTOCOL:-control_strings}"
DATA_KIND="${DATA_KIND:-decoder_unified}"
DATASETS="${DATASETS:-golden frmt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/eval_results/decoder_only/external_chat_models}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/decoder_only/external_chat_models}"
GPU_MODEL_SPECS="${GPU_MODEL_SPECS:-0|amalia-llm/AMALIA-9B-0626-SFT;1|Qwen/Qwen3-4B;4|microsoft/Phi-4-mini-instruct}"

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
fi

mkdir -p "$LOG_ROOT"
timestamp="$(date +%Y%m%d_%H%M%S)"

if command -v rg >/dev/null 2>&1; then
  PROCESS_MONITOR_CMD="ps -fu $USER | rg 'run_eval_external_chat_model_all|evaluate_(translation|classification)_chat_model'"
else
  PROCESS_MONITOR_CMD="ps -fu $USER | grep -E 'run_eval_external_chat_model_all|evaluate_(translation|classification)_chat_model' | grep -v grep"
fi

IFS=';' read -r -a SPEC_ARRAY <<< "$GPU_MODEL_SPECS"

echo "Launching external chat-model evals on host=$(hostname)"
echo "  data_protocol=$DATA_PROTOCOL"
echo "  data_kind=$DATA_KIND"
echo "  datasets=$DATASETS"
echo "  output_root=$OUTPUT_ROOT"
echo "  log_root=$LOG_ROOT"

for spec in "${SPEC_ARRAY[@]}"; do
  [[ -n "$spec" ]] || continue
  gpu_index="${spec%%|*}"
  model_id="${spec#*|}"
  model_slug="$(printf '%s' "$model_id" | tr '/:@ ' '_' | tr -cs '[:alnum:]_.-' '_')"
  log_path="$LOG_ROOT/${timestamp}_${gpu_index}_${model_slug}.log"

  echo "  gpu=$gpu_index model=$model_id log=$log_path"

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu_index" \
    MODEL_ID="$model_id" \
    MODEL_SLUG="$model_slug" \
    DATA_PROTOCOL="$DATA_PROTOCOL" \
    DATA_KIND="$DATA_KIND" \
    DATASETS="$DATASETS" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    bash "$RUN_SCRIPT" \
    >"$log_path" 2>&1 &

  echo "    pid=$!"
done

echo
echo "Use these to monitor:"
echo "  $PROCESS_MONITOR_CMD"
echo "  tail -f $LOG_ROOT/${timestamp}_*.log"
