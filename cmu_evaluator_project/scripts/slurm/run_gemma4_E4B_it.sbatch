#!/usr/bin/env bash
#SBATCH --job-name=cmu_gemma4_E4B_it
#SBATCH --partition=hlt_msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=cmu_evaluator_project/logs/slurm/%x-%j.out
#SBATCH --error=cmu_evaluator_project/logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/cfs/home/u036584/repos/Thesis}}"
VENV_PATH="${GEMMA_VENV_PATH:-$HOME/gemma4_cmu}"
SOURCE_PATH="${SOURCE_PATH:-$REPO_ROOT/cmu_evaluator_project/data/prepared/pilot_01/source/source.json}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/cmu_evaluator_project/data/prepared/pilot_01/candidates/gemma4_E4B_it.json}"

cd "$REPO_ROOT"
source "$VENV_PATH/bin/activate"
export TOKENIZERS_PARALLELISM=false

[[ -f "$SOURCE_PATH" ]] || { echo "ERROR: missing source: $SOURCE_PATH" >&2; exit 2; }
[[ ! -e "$OUTPUT_PATH" ]] || { echo "ERROR: refusing to overwrite $OUTPUT_PATH" >&2; exit 2; }

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi
python - <<'PY'
import sys
import torch
import transformers

print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA unavailable; refusing CPU inference")
total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print("cuda_device_name=", torch.cuda.get_device_name(0))
print("cuda_total_memory_gib=", round(total_gib, 2))
if total_gib < 22:
    sys.exit("ERROR: Gemma 4 E4B bf16 weights need a GPU with at least 22 GiB for this job")
PY

python -u cmu_evaluator_project/scripts/run_external_candidate.py \
  --source "$SOURCE_PATH" \
  --output "$OUTPUT_PATH" \
  --model-id google/gemma-4-E4B-it \
  --revision ee0ef6023621cff504d758262d4e04895a5af4a2

python -m json.tool "$OUTPUT_PATH" >/dev/null
echo "OUTPUT_JSON=$OUTPUT_PATH"
