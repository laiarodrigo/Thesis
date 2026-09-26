#!/usr/bin/env bash
#SBATCH --job-name=t5g_cls_cmp
#SBATCH --partition=hlt_msc
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

CONFIG_PATH="${CONFIG_PATH:-}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "ERROR: set CONFIG_PATH, e.g. CONFIG_PATH=configs/encoder_decoder/t5gemma2_4b/comparison_staged/classification_head_r24_stageA_opensubs_frmt.yaml" >&2
  exit 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing config: $CONFIG_PATH" >&2
  exit 2
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "ERROR: could not initialize conda; set CONDA_SH or VENV_PATH." >&2
    exit 2
  fi
  conda activate "$CONDA_ENV_NAME"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi || true

python3 scripts/encoder_decoder/single_task_models/t5gemma2/step4b_classification_head_boilerplate.py --config "$CONFIG_PATH" --execute
