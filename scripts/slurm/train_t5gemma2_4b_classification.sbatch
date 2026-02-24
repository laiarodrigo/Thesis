#!/usr/bin/env bash
#SBATCH --job-name=t5g4b_class_head
#SBATCH --partition=hlt_phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Optional: activate your conda env before running.
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV_NAME"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi || true

SMOKE_FLAG=()
if [[ "${SMOKE_RUN:-0}" == "1" ]]; then
  SMOKE_FLAG+=(--smoke-run)
fi

bash scripts/encoder_decoder/t5gemma2_4b/run_step4b_classification_head_boilerplate.sh --execute "${SMOKE_FLAG[@]}"

