#!/usr/bin/env bash
#SBATCH --job-name=cmu_eval_v1
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
VARIANT="${VARIANT:?Set VARIANT to A, B, C, or E}"
INPUT_PATH="${INPUT_PATH:-$REPO_ROOT/cmu_evaluator_project/data/benchmark/v1_37_input.jsonl}"
PROMPT_DIR="${PROMPT_DIR:-$REPO_ROOT/cmu_evaluator_project/guidelines/v1}"
EXPECTED_ROWS="${EXPECTED_ROWS:-37}"

case "$VARIANT" in
  A) DEFAULT_OUTPUT="v1_a_37_predictions.jsonl" ;;
  B) DEFAULT_OUTPUT="v1_b_37_predictions.jsonl" ;;
  C) DEFAULT_OUTPUT="v1_c_37_predictions.jsonl" ;;
  E) DEFAULT_OUTPUT="v1_e_37_predictions.jsonl" ;;
  *) echo "ERROR: VARIANT must be A, B, C, or E" >&2; exit 2 ;;
esac

OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/cmu_evaluator_project/runtime/evaluator_v1/$DEFAULT_OUTPUT}"

cd "$REPO_ROOT"
source "$VENV_PATH/bin/activate"
export TOKENIZERS_PARALLELISM=false

[[ -f "$INPUT_PATH" ]] || { echo "ERROR: missing V1 input: $INPUT_PATH" >&2; exit 2; }
[[ -d "$PROMPT_DIR" ]] || { echo "ERROR: missing V1 prompt directory: $PROMPT_DIR" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT_PATH")" cmu_evaluator_project/logs/slurm

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na} variant=$VARIANT"
echo "INPUT_PATH=$INPUT_PATH"
echo "PROMPT_DIR=$PROMPT_DIR"
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

python - "$INPUT_PATH" "$EXPECTED_ROWS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
ids = [row.get("evaluation_id") for row in rows]
if len(rows) != expected or len(set(ids)) != expected:
    raise SystemExit(
        f"ERROR: expected {expected} unique V1 inputs, got rows={len(rows)} ids={len(set(ids))}"
    )
print(f"V1_INPUT rows={len(rows)} unique_ids={len(set(ids))}")
PY

python -u cmu_evaluator_project/scripts/run_evaluator_v1.py \
  --variant "$VARIANT" \
  --input "$INPUT_PATH" \
  --prompt-dir "$PROMPT_DIR" \
  --output "$OUTPUT_PATH" \
  --model-id google/gemma-4-E4B-it \
  --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --max-new-tokens 512

python - "$VARIANT" "$INPUT_PATH" "$OUTPUT_PATH" <<'PY'
import json
import sys
from pathlib import Path

variant = sys.argv[1]
input_path, output_path = map(Path, sys.argv[2:])
inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
outputs = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
input_ids = {row.get("evaluation_id") for row in inputs}
output_ids = {row.get("evaluation_id") for row in outputs}
if len(input_ids) != len(inputs) or len(output_ids) != len(outputs) or output_ids != input_ids:
    raise SystemExit(
        f"ERROR: V1 input/output identity mismatch: input_rows={len(inputs)} "
        f"input_ids={len(input_ids)} output_rows={len(outputs)} output_ids={len(output_ids)}"
    )
if any(row.get("variant") != variant for row in outputs):
    raise SystemExit("ERROR: V1 output contains the wrong variant")
protocols = {row.get("protocol_sha256") for row in outputs}
if len(protocols) != 1:
    raise SystemExit(f"ERROR: V1 output contains inconsistent protocols: {protocols}")
valid = sum(row.get("parse_valid") is True for row in outputs)
print(
    f"V1_OUTPUT variant={variant} rows={len(outputs)} parse_valid={valid} "
    f"parse_invalid={len(outputs)-valid} protocol_sha256={next(iter(protocols))} "
    f"path={output_path}"
)
PY
