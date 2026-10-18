#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/cfs/home/u036584/repos/Thesis}"
VENV_PATH="${GEMMA_VENV_PATH:-$HOME/gemma4_cmu}"
RUN_KIND="${RUN_KIND:-official}"
F2_GPU_INDICES="${F2_GPU_INDICES:-0,1}"
INPUT_PATH="$REPO_ROOT/cmu_evaluator_project/data/benchmark/v1_37_input.jsonl"
PROMPT_DIR="$REPO_ROOT/cmu_evaluator_project/guidelines/v1"
OUTPUT_DIR="$REPO_ROOT/cmu_evaluator_project/runtime/evaluator_model_reasoning"

if [[ "$(hostname -s)" != "g07" ]]; then
  echo "ERROR: the authorized F2 placement is g07" >&2
  exit 2
fi
if [[ "$F2_GPU_INDICES" != "0,1" ]]; then
  echo "ERROR: F2 must use the authorized g07 GPU pair 0,1" >&2
  exit 2
fi

case "$RUN_KIND" in
  smoke)
    OUTPUT_PATH="$OUTPUT_DIR/f2_31b_no_thinking_smoke2_predictions.jsonl"
    EXPECTED_ROWS=2
    SELECTION_ARGS=(
      --evaluation-id diag-v0-s01-correct
      --evaluation-id diag-v0-s01-meaning-change
    )
    ;;
  official)
    OUTPUT_PATH="$OUTPUT_DIR/f2_31b_no_thinking_37_predictions.jsonl"
    EXPECTED_ROWS=37
    SELECTION_ARGS=()
    ;;
  *) echo "ERROR: RUN_KIND must be smoke or official" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"
source "$VENV_PATH/bin/activate"
export CUDA_VISIBLE_DEVICES="$F2_GPU_INDICES"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR" cmu_evaluator_project/logs/model_reasoning
test -s "$INPUT_PATH"
test -s "$PROMPT_DIR/stage1_A.md"
test -s "$PROMPT_DIR/stage2_score.md"

echo "[$(date)] host=$(hostname) run_kind=$RUN_KIND physical_gpus=$F2_GPU_INDICES"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv

python - <<'PY'
import sys
import torch
import transformers
import accelerate

print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("accelerate=", accelerate.__version__)
print("cuda_device_count=", torch.cuda.device_count())
if torch.cuda.device_count() != 2:
    sys.exit("ERROR: F2 requires exactly two visible authorized GPUs")
for index in range(2):
    with torch.cuda.device(index):
        total_gib = torch.cuda.get_device_properties(index).total_memory / 1024**3
        free_gib = torch.cuda.mem_get_info(index)[0] / 1024**3
        print(
            index,
            torch.cuda.get_device_name(index),
            "total_gib=", round(total_gib, 2),
            "free_gib=", round(free_gib, 2),
            "bf16=", torch.cuda.is_bf16_supported(),
        )
        if total_gib < 45 or free_gib < 45 or not torch.cuda.is_bf16_supported():
            sys.exit(
                "ERROR: F2 requires two currently free BF16-capable GPUs "
                "with at least 45 GiB total and free on each"
            )
PY

python -u cmu_evaluator_project/scripts/run_evaluator_v1.py \
  --experiment-id F2 \
  --variant A \
  --input "$INPUT_PATH" \
  --prompt-dir "$PROMPT_DIR" \
  --output "$OUTPUT_PATH" \
  --model-id google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --dtype bfloat16 \
  --no-enable-thinking \
  --stage1-max-new-tokens 512 \
  --stage2-max-new-tokens 512 \
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
        f"ERROR: expected {expected} unique F2 rows, got rows={len(rows)} ids={len(set(ids))}"
    )
if any(row.get("experiment_id") != "F2" for row in rows):
    raise SystemExit("ERROR: non-F2 record in F2 output")
valid = sum(row.get("parse_valid") is True for row in rows)
if run_kind == "smoke" and valid != expected:
    raise SystemExit(
        f"ERROR: F2 smoke requires {expected}/{expected} structurally valid rows, got {valid}"
    )
print(f"F2_OUTPUT rows={len(rows)} valid={valid} invalid={len(rows)-valid} path={path}")
PY
