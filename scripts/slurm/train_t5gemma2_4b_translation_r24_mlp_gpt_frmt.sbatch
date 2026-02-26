#!/usr/bin/env bash
#SBATCH --job-name=t5g4b_r24_gf
#SBATCH --partition=hlt_msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG_PATH="configs/encoder_decoder/t5gemma2_4b/translation_lora_r24_gpt_frmt_boilerplate.yaml"
TRAIN_FILE="data/encoder_decoder/t5gemma2/gpt_frmt_mix/translation_train.jsonl"
VALID_FILE="data/encoder_decoder/t5gemma2/gpt_frmt_mix/translation_valid.jsonl"

# Optional: activate your conda env before running.
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV_NAME"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi || true

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing config: $CONFIG_PATH" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "ERROR: missing train file: $TRAIN_FILE" >&2
  echo "Run export first with scripts/encoder_decoder/t5gemma2_4b/run_export_translation_gpt_frmt.sh" >&2
  exit 2
fi
if [[ ! -f "$VALID_FILE" ]]; then
  echo "ERROR: missing valid file: $VALID_FILE" >&2
  echo "Run export first with scripts/encoder_decoder/t5gemma2_4b/run_export_translation_gpt_frmt.sh" >&2
  exit 2
fi

bash scripts/encoder_decoder/t5gemma2_4b/run_step4a_translation_lora_r24_gpt_frmt.sh --execute
