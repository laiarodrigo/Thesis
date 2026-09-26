#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_SCRIPT="$REPO_ROOT/scripts/decoder_only/axolotl/run_eval_iaedu_chat_model_all.sh"

VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
MODEL_ID="${MODEL_ID:-iaedu-gpt-latest}"
MODEL_SLUG="${MODEL_SLUG:-}"
DATA_PROTOCOL="${DATA_PROTOCOL:-control_strings}"
DATA_KIND="${DATA_KIND:-decoder_unified}"
DATASETS="${DATASETS:-golden frmt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/eval_results/decoder_only/iaedu_chat_models}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/decoder_only/iaedu_chat_models}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_RETRIES="${MAX_RETRIES:-4}"
RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
IAEDU_ENV_FILE="${IAEDU_ENV_FILE:-bla.env}"

slugify_model_id() {
  printf '%s' "$1" | tr '/:@ ' '_' | tr -cs '[:alnum:]_.-' '_'
}

if [[ -z "$MODEL_SLUG" ]]; then
  MODEL_SLUG="$(slugify_model_id "$MODEL_ID")"
fi

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
fi

mkdir -p "$LOG_ROOT"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$LOG_ROOT/${timestamp}_${MODEL_SLUG}.log"

if command -v rg >/dev/null 2>&1; then
  PROCESS_MONITOR_CMD="ps -fu $USER | rg 'run_eval_iaedu_chat_model_all|evaluate_(translation|classification)_iaedu_chat_model'"
else
  PROCESS_MONITOR_CMD="ps -fu $USER | grep -E 'run_eval_iaedu_chat_model_all|evaluate_(translation|classification)_iaedu_chat_model' | grep -v grep"
fi

echo "Launching IAEDU chat-model eval on host=$(hostname)"
echo "  model_id=$MODEL_ID"
echo "  data_protocol=$DATA_PROTOCOL"
echo "  data_kind=$DATA_KIND"
echo "  datasets=$DATASETS"
echo "  concurrency=$CONCURRENCY"
echo "  output_root=$OUTPUT_ROOT"
echo "  log_path=$log_path"

nohup env \
  MODEL_ID="$MODEL_ID" \
  MODEL_SLUG="$MODEL_SLUG" \
  DATA_PROTOCOL="$DATA_PROTOCOL" \
  DATA_KIND="$DATA_KIND" \
  DATASETS="$DATASETS" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  CONCURRENCY="$CONCURRENCY" \
  MAX_RETRIES="$MAX_RETRIES" \
  RETRY_BACKOFF_SECONDS="$RETRY_BACKOFF_SECONDS" \
  REQUEST_TIMEOUT="$REQUEST_TIMEOUT" \
  IAEDU_ENV_FILE="$IAEDU_ENV_FILE" \
  bash "$RUN_SCRIPT" \
  >"$log_path" 2>&1 < /dev/null &

echo "  pid=$!"
echo
echo "Use these to monitor:"
echo "  $PROCESS_MONITOR_CMD"
echo "  tail -f $log_path"
