#!/usr/bin/env bash
#SBATCH --job-name=t5g4b_trans_r24mlp
#SBATCH --partition=hlt_phd
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

CONFIG_PATH="configs/encoder_decoder/t5gemma2_4b/translation_lora_r24_boilerplate.yaml"

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

if [[ "${ALLOW_NON_GPT_DATA:-0}" != "1" ]]; then
  if grep -Eiq "golden[ _]collection" "$CONFIG_PATH"; then
    echo "ERROR: config references golden collection; training must use GPT train/valid only." >&2
    exit 2
  fi
  if ! grep -Eq '^[[:space:]]*train_path:[[:space:]]*data/encoder_decoder/t5gemma2/translation_train\.jsonl' "$CONFIG_PATH"; then
    echo "ERROR: unexpected translation train_path. Expected GPT train file." >&2
    exit 2
  fi
  if ! grep -Eq '^[[:space:]]*valid_path:[[:space:]]*data/encoder_decoder/t5gemma2/translation_valid\.jsonl' "$CONFIG_PATH"; then
    echo "ERROR: unexpected translation valid_path. Expected GPT valid file." >&2
    exit 2
  fi
fi

bash scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_step4a_translation_lora_r24.sh --execute
