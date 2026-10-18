#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/cfs/home/u036584/repos/Thesis}"
VENV_PATH="${GEMMA_VENV_PATH:-$HOME/gemma4_cmu}"
RUN_EXPERIMENT="${RUN_EXPERIMENT:?set RUN_EXPERIMENT to G24S1, G2, G3, or G4}"
INPUT_PATH="${G_INPUT_PATH:-$REPO_ROOT/cmu_evaluator_project/data/benchmark/gemma31b_g2_g4_eval_72.jsonl}"
DEMONSTRATIONS_PATH="${G_DEMONSTRATIONS_PATH:-$REPO_ROOT/cmu_evaluator_project/data/benchmark/gemma31b_g2_g4_demonstrations.json}"
MANIFEST_PATH="${G_MANIFEST_PATH:-$REPO_ROOT/cmu_evaluator_project/data/benchmark/gemma31b_g2_g4_manifest.json}"
OUTPUT_DIR="${G_OUTPUT_DIR:-$REPO_ROOT/cmu_evaluator_project/runtime/evaluator_gemma31b}"
PROMPT_DIR="${G_PROMPT_DIR:-$REPO_ROOT/cmu_evaluator_project/guidelines/v1}"

HOST_SHORT="$(hostname -s)"
case "$HOST_SHORT" in
  g07) AUTHORIZED_GPU_INDICES_PATTERN="^(0,1|2,3|4,5)$" ;;
  g09) AUTHORIZED_GPU_INDICES_PATTERN="^(3,7)$" ;;
  *) echo "ERROR: verified Gemma 31B HLT hosts are g07 and g09" >&2; exit 2 ;;
esac
GPU_INDICES="${G_GPU_INDICES:?set G_GPU_INDICES to a verified free physical GPU pair}"
if [[ ! "$GPU_INDICES" =~ $AUTHORIZED_GPU_INDICES_PATTERN ]]; then
  echo "ERROR: unverified GPU pair $GPU_INDICES on $HOST_SHORT" >&2
  exit 2
fi

case "$RUN_EXPERIMENT" in
  G24S1)
    DEFAULT_OUTPUT="$OUTPUT_DIR/g24_improved_stage1_72_predictions.jsonl"
    EXTRA_ARGS=(--stage1-only)
    ;;
  G2)
    DEFAULT_OUTPUT="$OUTPUT_DIR/g2_31b_improved_stage1_72_predictions.jsonl"
    : "${G_STAGE1_SOURCE:?set G_STAGE1_SOURCE to the completed G24S1 artifact}"
    EXTRA_ARGS=(--stage1-source "$G_STAGE1_SOURCE")
    ;;
  G3)
    DEFAULT_OUTPUT="$OUTPUT_DIR/g3_31b_improved_stage2_72_predictions.jsonl"
    : "${G_STAGE1_SOURCE:?set G_STAGE1_SOURCE to the frozen official G0 predictions}"
    EXTRA_ARGS=(--stage1-source "$G_STAGE1_SOURCE")
    ;;
  G4)
    DEFAULT_OUTPUT="$OUTPUT_DIR/g4_31b_improved_both_72_predictions.jsonl"
    : "${G_STAGE1_SOURCE:?set G_STAGE1_SOURCE to the completed G24S1 artifact}"
    EXTRA_ARGS=(--stage1-source "$G_STAGE1_SOURCE")
    ;;
  *) echo "ERROR: RUN_EXPERIMENT must be G24S1, G2, G3, or G4" >&2; exit 2 ;;
esac
OUTPUT_PATH="${G_OUTPUT_PATH:-$DEFAULT_OUTPUT}"

cd "$REPO_ROOT"
source "$VENV_PATH/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
export TOKENIZERS_PARALLELISM=false
if [[ -n "${G_MODEL_CACHE:-}" ]]; then
  export HF_HOME="$G_MODEL_CACHE"
fi

mkdir -p "$(dirname "$OUTPUT_PATH")" cmu_evaluator_project/logs/gemma31b
python cmu_evaluator_project/scripts/build_gemma31b_g2_g4.py \
  --audit-frozen \
  --evaluation-input "$INPUT_PATH" \
  --demonstrations "$DEMONSTRATIONS_PATH" \
  --manifest "$MANIFEST_PATH"

echo "[$(date)] experiment=$RUN_EXPERIMENT host=$(hostname) physical_gpus=$GPU_INDICES output=$OUTPUT_PATH"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv

python - <<'PY'
import sys
import torch
if torch.cuda.device_count() != 2:
    sys.exit("ERROR: exactly two visible GPUs are required")
for index in range(2):
    with torch.cuda.device(index):
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        free = torch.cuda.mem_get_info(index)[0] / 1024**3
        print(index, torch.cuda.get_device_name(index), "total_gib=", round(total, 2),
              "free_gib=", round(free, 2), "bf16=", torch.cuda.is_bf16_supported())
        if total < 45 or free < 45 or not torch.cuda.is_bf16_supported():
            sys.exit("ERROR: both visible GPUs must have >=45 GiB free and BF16 support")
PY

python -u cmu_evaluator_project/scripts/run_evaluator_v1.py \
  --experiment-id "$RUN_EXPERIMENT" \
  --variant A \
  --input "$INPUT_PATH" \
  --prompt-dir "$PROMPT_DIR" \
  --demonstrations "$DEMONSTRATIONS_PATH" \
  --output "$OUTPUT_PATH" \
  --model-id google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --dtype bfloat16 \
  --no-enable-thinking \
  --stage1-max-new-tokens 512 \
  --stage2-max-new-tokens 512 \
  "${EXTRA_ARGS[@]}"

python - "$OUTPUT_PATH" "$RUN_EXPERIMENT" "$MANIFEST_PATH" <<'PY'
import json, sys
from pathlib import Path
path, experiment, manifest_path = map(Path, (sys.argv[1], sys.argv[2], sys.argv[3]))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
ids = [row.get("evaluation_id") for row in rows]
if ids != manifest["evaluation_ids"] or len(set(ids)) != 72:
    raise SystemExit("ERROR: output IDs/order differ from frozen 72-row manifest")
if any(row.get("experiment_id") != str(experiment) for row in rows):
    raise SystemExit("ERROR: output contains a different experiment identity")
valid = sum(row.get("parse_valid") is True for row in rows)
print(f"{experiment}_OUTPUT rows=72 valid={valid} invalid={72-valid} path={path}")
PY
