#!/usr/bin/env bash
#SBATCH --job-name=stagec_subset
#SBATCH --partition=hlt_msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

INPUT_PATHS="${INPUT_PATHS:-}"
CSV_SPECS="${CSV_SPECS:-}"
OUT_DIR="${OUT_DIR:-}"
MAX_TOTAL="${MAX_TOTAL:-20000}"
PREVIEW_ROWS="${PREVIEW_ROWS:-50}"

if [[ -z "$OUT_DIR" ]]; then
  echo "ERROR: set OUT_DIR to the desired output directory." >&2
  exit 2
fi

if [[ -z "$INPUT_PATHS" && -z "$CSV_SPECS" ]]; then
  echo "ERROR: set INPUT_PATHS and/or CSV_SPECS." >&2
  exit 2
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV_NAME"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
echo "[$(date)] out_dir=${OUT_DIR}"

declare -a CMD
CMD=(
  python
  scripts/encoder_decoder/stage_c/build_stage_c_subset.py
  --out-dir "$OUT_DIR"
  --max-total "$MAX_TOTAL"
  --sample-per-bucket-preview "$PREVIEW_ROWS"
)

IFS=',' read -r -a INPUT_ARRAY <<< "$INPUT_PATHS"
for path in "${INPUT_ARRAY[@]}"; do
  if [[ -n "${path// }" ]]; then
    CMD+=(--translation-jsonl "$path")
  fi
done

IFS=',' read -r -a CSV_ARRAY <<< "$CSV_SPECS"
for spec in "${CSV_ARRAY[@]}"; do
  if [[ -n "${spec// }" ]]; then
    CMD+=(--csv-spec "$spec")
  fi
done

printf '[%s] command:' "$(date)"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
