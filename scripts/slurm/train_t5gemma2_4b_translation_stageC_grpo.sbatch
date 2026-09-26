#!/usr/bin/env bash
#SBATCH --job-name=t5g4b_c
#SBATCH --partition=hlt_msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

resolve_output_dir() {
  python3 - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import os
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(os.path.abspath(str(cfg["training"]["output_dir"])))
PY
}

sync_output_dir_back() {
  local src="${THESIS_OUTPUT_DIR_OVERRIDE:-}"
  local dst="${THESIS_FINAL_OUTPUT_DIR:-}"
  if [[ -z "$src" || -z "$dst" || ! -d "$src" ]]; then
    return 0
  fi

  local dst_parent tmp_dst backup_dst had_existing=0
  dst_parent="$(dirname "$dst")"
  tmp_dst="${dst}.sync_tmp_${SLURM_JOB_ID:-manual}"
  backup_dst="${dst}.sync_prev_${SLURM_JOB_ID:-manual}"

  mkdir -p "$dst_parent"
  rm -rf "$tmp_dst" "$backup_dst"
  mkdir -p "$tmp_dst"
  echo "[$(date)] syncing output dir back to $dst"
  if [[ -d "$dst" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$dst"/ "$tmp_dst"/
    else
      cp -a "$dst"/. "$tmp_dst"/
    fi
  fi
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src"/ "$tmp_dst"/
  else
    cp -a "$src"/. "$tmp_dst"/
  fi

  if [[ -e "$dst" ]]; then
    mv "$dst" "$backup_dst"
    had_existing=1
  fi

  if mv "$tmp_dst" "$dst"; then
    if [[ "$had_existing" == "1" ]]; then
      rm -rf "$backup_dst"
    fi
  else
    if [[ "$had_existing" == "1" && -e "$backup_dst" ]]; then
      mv "$backup_dst" "$dst" || true
    fi
    return 1
  fi

  echo "[$(date)] sync complete"
}

setup_hf_auth() {
  if [[ -n "${HF_TOKEN:-}" || -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    echo "Using Hugging Face token from the current environment."
    return 0
  fi

  local env_file="${HF_ENV_FILE:-$PWD/bla.env}"
  if [[ ! -f "$env_file" ]]; then
    echo "No HF env file found at $env_file; relying on existing HF auth/cache."
    return 0
  fi

  local token
  token="$(
    python3 - "$env_file" <<'PY'
from pathlib import Path
import re
import sys

env_path = Path(sys.argv[1])
patterns = ["HF_TOKEN_SLURM", "HF_TOKEN"]
values = {}
for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
    if not match:
        continue
    key, value = match.groups()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    values[key] = value
for key in patterns:
    value = values.get(key, "").strip()
    if value:
        print(value)
        break
PY
  )"

  if [[ -z "$token" ]]; then
    echo "HF env file found but no HF token variable was loaded from $env_file."
    return 0
  fi

  export HF_TOKEN="$token"
  export HUGGINGFACE_HUB_TOKEN="$token"
  echo "Loaded Hugging Face token from $env_file."
}

setup_local_cache_dirs() {
  if [[ "${USE_LOCAL_HF_CACHE:-1}" != "1" ]]; then
    echo "USE_LOCAL_HF_CACHE=0; leaving HF caches on shared storage."
    return 0
  fi
  local base="${SCRATCH_BASE:-}"
  if [[ -z "$base" ]]; then
    echo "No local scratch base found; leaving HF caches on shared storage."
    return 0
  fi

  local cache_root="$base/thesis_cache/${SLURM_JOB_ID:-manual}"
  export TMPDIR="$cache_root/tmp"
  export HF_HOME="$cache_root/hf_home"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export TRANSFORMERS_CACHE="$HF_HOME/transformers"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export XDG_CACHE_HOME="$cache_root/xdg"
  export TORCH_HOME="$cache_root/torch"

  mkdir -p \
    "$TMPDIR" \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$HF_HUB_CACHE" \
    "$XDG_CACHE_HOME" \
    "$TORCH_HOME"

  echo "Using node-local caches:"
  echo "  TMPDIR=$TMPDIR"
  echo "  HF_HOME=$HF_HOME"
  echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
  echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
  echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
  echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
  echo "  TORCH_HOME=$TORCH_HOME"
}

CONFIG_PATH="${CONFIG_PATH:-}"
if [[ -z "$CONFIG_PATH" ]]; then
  echo "ERROR: set CONFIG_PATH, e.g. CONFIG_PATH=configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gptrefresh2.yaml" >&2
  exit 2
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV_NAME"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
nvidia-smi || true

python3 - <<'PY'
import sys
import torch

print(
    "Torch CUDA check:"
    f" cuda_available={torch.cuda.is_available()}"
    f" device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
)
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception as exc:
            name = f"<error: {exc!r}>"
        print(f"  cuda:{idx} name={name}")
else:
    print("ERROR: torch.cuda.is_available() is False inside the SLURM job. Refusing to run Stage C on CPU.", file=sys.stderr)
    sys.exit(3)
PY

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing config: $CONFIG_PATH" >&2
  exit 2
fi

THESIS_FINAL_OUTPUT_DIR="$(resolve_output_dir)"
export THESIS_FINAL_OUTPUT_DIR
echo "Configured output dir: $THESIS_FINAL_OUTPUT_DIR"

USE_SLURM_TMPDIR_OUTPUT="${USE_SLURM_TMPDIR_OUTPUT:-1}"
SCRATCH_BASE="${SLURM_TMPDIR:-${TMPDIR:-}}"
setup_hf_auth
setup_local_cache_dirs
if [[ "$USE_SLURM_TMPDIR_OUTPUT" == "1" && -n "$SCRATCH_BASE" ]]; then
  THESIS_OUTPUT_DIR_OVERRIDE="$SCRATCH_BASE/thesis_outputs/${SLURM_JOB_ID:-manual}/$(basename "$THESIS_FINAL_OUTPUT_DIR")"
  export THESIS_OUTPUT_DIR_OVERRIDE
  mkdir -p "$THESIS_OUTPUT_DIR_OVERRIDE"
  echo "Using node-local scratch output dir: $THESIS_OUTPUT_DIR_OVERRIDE"
  trap sync_output_dir_back EXIT INT TERM
else
  echo "Using shared filesystem output dir directly."
fi

python3 scripts/encoder_decoder/stage_c/train_stage_c_seq2seq_grpo.py --config "$CONFIG_PATH" --execute
