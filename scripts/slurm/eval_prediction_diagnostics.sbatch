#!/usr/bin/env bash
#SBATCH --job-name=pred_diag
#SBATCH --partition=hlt_msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
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

PREDICTION_RUNS="${PREDICTION_RUNS:-}"
BASELINE_NAME="${BASELINE_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
BLEU_WEIGHT="${BLEU_WEIGHT:-0.5}"
WER_WEIGHT="${WER_WEIGHT:-0.5}"
TOP_SUSPECTS="${TOP_SUSPECTS:-50}"

if [[ -z "$PREDICTION_RUNS" ]]; then
  echo "ERROR: set PREDICTION_RUNS as name=path;name=path;..." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: set OUTPUT_DIR" >&2
  exit 2
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
echo "[$(date)] output_dir=${OUTPUT_DIR}"

declare -a CMD
CMD=(
  python
  scripts/encoder_decoder/eval/diagnose_prediction_rewards.py
  --output-dir "$OUTPUT_DIR"
  --bleu-weight "$BLEU_WEIGHT"
  --wer-weight "$WER_WEIGHT"
  --top-suspects "$TOP_SUSPECTS"
)

if [[ -n "$BASELINE_NAME" ]]; then
  CMD+=(--baseline "$BASELINE_NAME")
fi

IFS=';' read -r -a RUN_ARRAY <<< "$PREDICTION_RUNS"
for spec in "${RUN_ARRAY[@]}"; do
  trimmed="${spec// }"
  [[ -n "$trimmed" ]] || continue
  CMD+=(--run "$trimmed")
done

printf '[%s] command:' "$(date)"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
