#!/usr/bin/env bash
#SBATCH --job-name=t5g4b_eval_tr
#SBATCH --partition=hlt_msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Optional: activate conda env.
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV_NAME"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi || true

EVAL_SCRIPT="${EVAL_SCRIPT:-scripts/encoder_decoder/eval/run_eval_translation_gemma4b_r24_adaptive.sh}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/encoder_decoder/t5gemma2_4b_translation_lora_r24}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/encoder_decoder/gemma4b/golden_collection/translation_lora_r24_adaptive_beam4}"
DATASET_PATH="${DATASET_PATH:-}"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "ERROR: missing eval script: $EVAL_SCRIPT" >&2
  exit 2
fi
if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "ERROR: missing adapter dir: $ADAPTER_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

RUN_ENV=(
  "ADAPTER_DIR=$ADAPTER_DIR"
  "OUTPUT_DIR=$OUTPUT_DIR"
)
if [[ -n "$DATASET_PATH" ]]; then
  RUN_ENV+=("DATASET_PATH=$DATASET_PATH")
fi

echo "Running eval:"
echo "  script=$EVAL_SCRIPT"
echo "  adapter=$ADAPTER_DIR"
echo "  output=$OUTPUT_DIR"
if [[ -n "$DATASET_PATH" ]]; then
  echo "  dataset=$DATASET_PATH"
fi

env "${RUN_ENV[@]}" bash "$EVAL_SCRIPT"
