#!/usr/bin/env bash
#SBATCH --job-name=t5g4b_cmp
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

# Required: full config path under repo.
CONFIG_PATH="${CONFIG_PATH:-}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "ERROR: set CONFIG_PATH, e.g. CONFIG_PATH=configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_frmt.yaml" >&2
  exit 2
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  # Example: VENV_PATH=$HOME/venvs/thesis_t5
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV_NAME"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi || true

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing config: $CONFIG_PATH" >&2
  exit 2
fi

python3 scripts/encoder_decoder/single_task_models/t5gemma2/step4a_translation_train_boilerplate.py --config "$CONFIG_PATH" --execute
