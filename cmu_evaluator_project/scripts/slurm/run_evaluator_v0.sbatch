#!/usr/bin/env bash
#SBATCH --job-name=cmu_eval_v0
#SBATCH --partition=hlt_msc
#SBATCH --nodelist=g06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:RTX_6000_24GB:1
#SBATCH --output=cmu_evaluator_project/logs/slurm/%x-%j.out
#SBATCH --error=cmu_evaluator_project/logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/cfs/home/u036584/repos/Thesis}}"
VENV_PATH="${GEMMA_VENV_PATH:-$HOME/gemma4_cmu}"
INPUT_PATH="${INPUT_PATH:-$REPO_ROOT/cmu_evaluator_project/data/benchmark/dev_01_v0_input.jsonl}"
INSTRUCTIONS_PATH="${INSTRUCTIONS_PATH:-$REPO_ROOT/cmu_evaluator_project/guidelines/CMU_BP_EP_evaluator_v0_instructions.md}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/cmu_evaluator_project/runtime/evaluator_dev/v0_predictions.jsonl}"

cd "$REPO_ROOT"
source "$VENV_PATH/bin/activate"
export TOKENIZERS_PARALLELISM=false

[[ -f "$INPUT_PATH" ]] || { echo "ERROR: missing V0 input: $INPUT_PATH" >&2; exit 2; }
[[ -f "$INSTRUCTIONS_PATH" ]] || { echo "ERROR: missing V0 instructions: $INSTRUCTIONS_PATH" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT_PATH")" cmu_evaluator_project/logs/slurm

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
echo "INPUT_PATH=$INPUT_PATH"
echo "INSTRUCTIONS_PATH=$INSTRUCTIONS_PATH"
echo "OUTPUT_PATH=$OUTPUT_PATH"
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
    sys.exit("ERROR: Gemma 4 E4B requires a GPU with at least 22 GiB for this job")
PY

python -u cmu_evaluator_project/scripts/run_evaluator_v0.py \
  --input "$INPUT_PATH" \
  --instructions "$INSTRUCTIONS_PATH" \
  --output "$OUTPUT_PATH" \
  --model-id google/gemma-4-E4B-it \
  --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --max-new-tokens 512

python - "$INPUT_PATH" "$OUTPUT_PATH" <<'PY'
import json
import sys
from pathlib import Path

input_path, output_path = map(Path, sys.argv[1:])
inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
outputs = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
input_ids = {row.get("evaluation_id") for row in inputs}
output_ids = {row.get("evaluation_id") for row in outputs}
if len(input_ids) != len(inputs) or len(output_ids) != len(outputs) or output_ids != input_ids:
    raise SystemExit(
        f"ERROR: V0 input/output identity mismatch: "
        f"input_rows={len(inputs)} input_ids={len(input_ids)} "
        f"output_rows={len(outputs)} output_ids={len(output_ids)}"
    )
valid = sum(row.get("parse_valid") is True for row in outputs)
print(f"V0_OUTPUT rows={len(outputs)} parse_valid={valid} parse_invalid={len(outputs)-valid} path={output_path}")
PY
