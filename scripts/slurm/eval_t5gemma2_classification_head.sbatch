#!/usr/bin/env bash
#SBATCH --job-name=t5g_eval_cls
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

CONFIG_PATH="${CONFIG_PATH:-}"
MODEL_DIR="${MODEL_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DATASET_PATH="${DATASET_PATH:-data/encoder_decoder/t5gemma2/golden_collection/classification_test.jsonl}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "ERROR: set CONFIG_PATH to a classification config yaml." >&2
  exit 2
fi
if [[ -z "$MODEL_DIR" ]]; then
  echo "ERROR: set MODEL_DIR to the trained classification output dir." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: set OUTPUT_DIR for evaluation outputs." >&2
  exit 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing config: $CONFIG_PATH" >&2
  exit 2
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: missing model dir: $MODEL_DIR" >&2
  exit 2
fi
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "ERROR: missing dataset path: $DATASET_PATH" >&2
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

mkdir -p "$OUTPUT_DIR"

python3 scripts/encoder_decoder/eval/evaluate_classification_head.py \
  --config "$CONFIG_PATH" \
  --model-dir "$MODEL_DIR" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR"
