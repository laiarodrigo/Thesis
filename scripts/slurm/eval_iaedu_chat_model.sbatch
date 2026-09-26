#!/usr/bin/env bash
#SBATCH --job-name=iaedu_eval
#SBATCH --partition=cpu-cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -f "$HOME/thesis_t5/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/thesis_t5/bin/activate"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"

MODEL_ID="${MODEL_ID:-iaedu-gpt55}"
DATA_PROTOCOL="${DATA_PROTOCOL:-control_strings}"
DATA_KIND="${DATA_KIND:-decoder_unified}"
DATASETS="${DATASETS:-golden frmt}"
CONCURRENCY="${CONCURRENCY:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/eval_results/decoder_only/iaedu_chat_models}"
IAEDU_ENV_FILE="${IAEDU_ENV_FILE:-iaedu_gpt55.env}"
CLASSIFICATION_BATCH_SIZE="${CLASSIFICATION_BATCH_SIZE:-32}"
MAX_RETRIES="${MAX_RETRIES:-4}"
RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
CLASSIFICATION_CANDIDATES="${CLASSIFICATION_CANDIDATES:-}"

MODEL_ID="$MODEL_ID" \
DATA_PROTOCOL="$DATA_PROTOCOL" \
DATA_KIND="$DATA_KIND" \
DATASETS="$DATASETS" \
CONCURRENCY="$CONCURRENCY" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
IAEDU_ENV_FILE="$IAEDU_ENV_FILE" \
CLASSIFICATION_BATCH_SIZE="$CLASSIFICATION_BATCH_SIZE" \
MAX_RETRIES="$MAX_RETRIES" \
RETRY_BACKOFF_SECONDS="$RETRY_BACKOFF_SECONDS" \
REQUEST_TIMEOUT="$REQUEST_TIMEOUT" \
CLASSIFICATION_CANDIDATES="$CLASSIFICATION_CANDIDATES" \
bash scripts/decoder_only/axolotl/run_eval_iaedu_chat_model_all.sh
