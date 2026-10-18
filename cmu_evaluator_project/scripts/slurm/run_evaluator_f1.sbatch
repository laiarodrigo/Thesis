#!/usr/bin/env bash
#SBATCH --job-name=f1_e4b_thinking
#SBATCH --partition=hlt_msc
#SBATCH --nodelist=g06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:RTX_6000_24GB:1
#SBATCH --output=cmu_evaluator_project/logs/model_reasoning/%x-%j.out
#SBATCH --error=cmu_evaluator_project/logs/model_reasoning/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/cfs/home/u036584/repos/Thesis}}"
VENV_PATH="${GEMMA_VENV_PATH:-$HOME/gemma4_cmu}"
RUN_KIND="${RUN_KIND:-official}"
INPUT_PATH="$REPO_ROOT/cmu_evaluator_project/data/benchmark/v1_37_input.jsonl"
PROMPT_DIR="$REPO_ROOT/cmu_evaluator_project/guidelines/v1"
OUTPUT_DIR="$REPO_ROOT/cmu_evaluator_project/runtime/evaluator_model_reasoning"

case "$RUN_KIND" in
  smoke)
    OUTPUT_PATH="$OUTPUT_DIR/f1_e4b_thinking_smoke2_predictions.jsonl"
    EXPECTED_ROWS=2
    SELECTION_ARGS=(
      --evaluation-id diag-v0-s01-correct
      --evaluation-id diag-v0-s01-meaning-change
    )
    ;;
  official)
    OUTPUT_PATH="$OUTPUT_DIR/f1_e4b_thinking_37_predictions.jsonl"
    EXPECTED_ROWS=37
    SELECTION_ARGS=()
    ;;
  *) echo "ERROR: RUN_KIND must be smoke or official" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"
source "$VENV_PATH/bin/activate"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR" cmu_evaluator_project/logs/model_reasoning
test -s "$INPUT_PATH"
test -s "$PROMPT_DIR/stage1_A.md"
test -s "$PROMPT_DIR/stage2_score.md"

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na} run_kind=$RUN_KIND"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python - <<'PY'
import sys
import torch
import transformers

print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("cuda_available=", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA unavailable")
total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print("cuda_device_name=", torch.cuda.get_device_name(0))
print("cuda_total_memory_gib=", round(total_gib, 2))
if total_gib < 22:
    sys.exit("ERROR: F1 requires an allocated GPU with at least 22 GiB")
PY

python -u cmu_evaluator_project/scripts/run_evaluator_v1.py \
  --experiment-id F1 \
  --variant A \
  --input "$INPUT_PATH" \
  --prompt-dir "$PROMPT_DIR" \
  --output "$OUTPUT_PATH" \
  --model-id google/gemma-4-E4B-it \
  --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --dtype auto \
  --enable-thinking \
  --stage1-max-new-tokens 4096 \
  --stage2-max-new-tokens 2048 \
  "${SELECTION_ARGS[@]}"

python - "$OUTPUT_PATH" "$EXPECTED_ROWS" "$RUN_KIND" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
run_kind = sys.argv[3]
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
ids = [row.get("evaluation_id") for row in rows]
if len(rows) != expected or len(set(ids)) != expected:
    raise SystemExit(
        f"ERROR: expected {expected} unique F1 rows, got rows={len(rows)} ids={len(set(ids))}"
    )
if any(row.get("experiment_id") != "F1" for row in rows):
    raise SystemExit("ERROR: non-F1 record in F1 output")
valid = sum(row.get("parse_valid") is True for row in rows)
if run_kind == "smoke" and valid != expected:
    raise SystemExit(
        f"ERROR: F1 smoke requires {expected}/{expected} structurally valid rows, got {valid}"
    )
print(f"F1_OUTPUT rows={len(rows)} valid={valid} invalid={len(rows)-valid} path={path}")
PY
