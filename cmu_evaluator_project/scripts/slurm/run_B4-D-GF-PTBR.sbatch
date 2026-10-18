#!/usr/bin/env bash
#SBATCH --job-name=cmu_B4_D_GF_PTBR
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
VENV_PATH="${THESIS_VENV_PATH:-$HOME/thesis_t5}"
ADAPTER_DIR="$REPO_ROOT/outputs/encoder_decoder/control_strings_ptbr/t5gemma2_4b_stageB_gpt_wiki_frmt_decoder_ptbr"
CONFIG_PATH="$REPO_ROOT/configs/encoder_decoder/t5gemma2_4b/control_strings_ptbr/stageB_gpt_wiki_frmt_decoder_ptbr.yaml"
SOURCE_PATH="${SOURCE_PATH:-$REPO_ROOT/cmu_evaluator_project/data/prepared/pilot_01/source/source.json}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/cmu_evaluator_project/data/prepared/pilot_01/candidates/B4-D-GF-PTBR.json}"

cd "$REPO_ROOT"
source "$VENV_PATH/bin/activate"
export TOKENIZERS_PARALLELISM=false

for path in "$ADAPTER_DIR/adapter_config.json" "$ADAPTER_DIR/adapter_model.safetensors" "$ADAPTER_DIR/tokenizer.json" "$CONFIG_PATH" "$SOURCE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_PATH" ]] || { echo "ERROR: refusing to overwrite $OUTPUT_PATH" >&2; exit 2; }

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi
python - <<'PY'
import sys
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA unavailable; refusing CPU inference")
print("cuda_device_name=", torch.cuda.get_device_name(0))
total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print("cuda_total_memory_gib=", round(total_gib, 2))
if total_gib < 22:
    sys.exit("ERROR: thesis 4B-4B inference requires a GPU with at least 22 GiB")
PY

python -u cmu_evaluator_project/scripts/run_thesis_candidate.py \
  --source "$SOURCE_PATH" \
  --output "$OUTPUT_PATH" \
  --system-id B4-D-GF-PTBR \
  --adapter-dir "$ADAPTER_DIR" \
  --training-config "$CONFIG_PATH" \
  --tokenizer-path "$ADAPTER_DIR"

python -m json.tool "$OUTPUT_PATH" >/dev/null
echo "OUTPUT_JSON=$OUTPUT_PATH"
