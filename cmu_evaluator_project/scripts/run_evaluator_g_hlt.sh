#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/cfs/home/u036584/repos/Thesis}"
VENV_PATH="${GEMMA_VENV_PATH:-$HOME/gemma4_cmu}"
RUN_EXPERIMENT="${RUN_EXPERIMENT:?set RUN_EXPERIMENT to G0 or G1}"
INPUT_PATH="$REPO_ROOT/cmu_evaluator_project/data/benchmark/gemma31b_g0_g1_eval_96.jsonl"
DEMONSTRATIONS_PATH="$REPO_ROOT/cmu_evaluator_project/data/benchmark/gemma31b_g1_eight_shot_demonstrations.json"
MANIFEST_PATH="$REPO_ROOT/cmu_evaluator_project/data/benchmark/gemma31b_g0_g1_manifest.json"
PROMPT_DIR="$REPO_ROOT/cmu_evaluator_project/guidelines/v1"
OUTPUT_DIR="$REPO_ROOT/cmu_evaluator_project/runtime/evaluator_gemma31b"

HOST_SHORT="$(hostname -s)"
case "$HOST_SHORT" in
  g07)
    DEFAULT_GPU_INDICES="0,1"
    AUTHORIZED_GPU_INDICES_PATTERN="^(0,1|2,3)$"
    AUTHORIZED_GPU_INDICES_DESCRIPTION="0,1 or 2,3"
    ;;
  g09)
    DEFAULT_GPU_INDICES="3,7"
    AUTHORIZED_GPU_INDICES_PATTERN="^3,7$"
    AUTHORIZED_GPU_INDICES_DESCRIPTION="3,7"
    ;;
  *)
    echo "ERROR: authorized Gemma 31B placements are g07 or g09" >&2
    exit 2
    ;;
esac
GPU_INDICES="${G_GPU_INDICES:-$DEFAULT_GPU_INDICES}"
if [[ ! "$GPU_INDICES" =~ $AUTHORIZED_GPU_INDICES_PATTERN ]]; then
  echo "ERROR: authorized GPU pair on $HOST_SHORT is $AUTHORIZED_GPU_INDICES_DESCRIPTION" >&2
  exit 2
fi

case "$RUN_EXPERIMENT" in
  G0)
    OUTPUT_PATH="$OUTPUT_DIR/g0_31b_zero_shot_96_predictions.jsonl"
    DEMONSTRATION_ARGS=()
    ;;
  G1)
    OUTPUT_PATH="$OUTPUT_DIR/g1_31b_eight_shot_96_predictions.jsonl"
    DEMONSTRATION_ARGS=(--demonstrations "$DEMONSTRATIONS_PATH")
    ;;
  *) echo "ERROR: RUN_EXPERIMENT must be G0 or G1" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"
source "$VENV_PATH/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR" cmu_evaluator_project/logs/gemma31b
python cmu_evaluator_project/scripts/build_gemma31b_g0_g1.py \
  --audit-frozen \
  --evaluation-input "$INPUT_PATH" \
  --demonstrations "$DEMONSTRATIONS_PATH" \
  --manifest "$MANIFEST_PATH"

echo "[$(date)] experiment=$RUN_EXPERIMENT host=$(hostname) physical_gpus=$GPU_INDICES"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv

python - <<'PY'
import sys
import torch

print("torch=", torch.__version__)
print("cuda_device_count=", torch.cuda.device_count())
if torch.cuda.device_count() != 2:
    sys.exit("ERROR: G0/G1 requires exactly two visible authorized GPUs")
for index in range(2):
    with torch.cuda.device(index):
        total_gib = torch.cuda.get_device_properties(index).total_memory / 1024**3
        free_gib = torch.cuda.mem_get_info(index)[0] / 1024**3
        print(index, torch.cuda.get_device_name(index), "total_gib=", round(total_gib, 2),
              "free_gib=", round(free_gib, 2), "bf16=", torch.cuda.is_bf16_supported())
        if total_gib < 45 or free_gib < 45 or not torch.cuda.is_bf16_supported():
            sys.exit("ERROR: both visible authorized GPUs must have at least 45 GiB free and support BF16")
PY

python -u cmu_evaluator_project/scripts/run_evaluator_v1.py \
  --experiment-id "$RUN_EXPERIMENT" \
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
  "${DEMONSTRATION_ARGS[@]}"

python - "$OUTPUT_PATH" "$RUN_EXPERIMENT" "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
experiment = sys.argv[2]
manifest = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
ids = [row.get("evaluation_id") for row in rows]
expected = manifest["evaluation_ids"]
if len(rows) != 96 or len(set(ids)) != 96 or ids != expected:
    raise SystemExit(f"ERROR: {experiment} output IDs/order differ from frozen 96-row manifest")
if any(row.get("experiment_id") != experiment for row in rows):
    raise SystemExit(f"ERROR: non-{experiment} record in {experiment} output")
valid = sum(row.get("parse_valid") is True for row in rows)
print(f"{experiment}_OUTPUT rows=96 valid={valid} invalid={96-valid} path={output_path}")
PY
