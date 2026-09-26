#!/usr/bin/env bash
#SBATCH --job-name=t5g4b_eval_gw
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

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV_NAME"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi || true

EVAL_SCRIPT="${EVAL_SCRIPT:-scripts/encoder_decoder/eval/run_eval_translation_gemma4b_r24_adaptive.sh}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/encoder_decoder/compare_staged/t5gemma2_4b_translation_r24_stageB_gpt_wiki}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageB_gpt_wiki}"
DATASET_PATH="${DATASET_PATH:-data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl}"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "ERROR: missing eval script: $EVAL_SCRIPT" >&2
  exit 2
fi
if [[ ! -d "$ADAPTER_DIR" ]]; then
  echo "ERROR: missing adapter dir: $ADAPTER_DIR" >&2
  exit 2
fi
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "ERROR: missing dataset path: $DATASET_PATH" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

RUN_ENV=(
  "ADAPTER_DIR=$ADAPTER_DIR"
  "OUTPUT_DIR=$OUTPUT_DIR"
  "DATASET_PATH=$DATASET_PATH"
)

echo "Running eval:"
echo "  script=$EVAL_SCRIPT"
echo "  adapter=$ADAPTER_DIR"
echo "  output=$OUTPUT_DIR"
echo "  dataset=$DATASET_PATH"

env "${RUN_ENV[@]}" bash "$EVAL_SCRIPT"
