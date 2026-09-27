#!/usr/bin/env bash
#SBATCH --job-name=thesis_stats
#SBATCH --partition=cpu-cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

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

  local base="${SLURM_TMPDIR:-}"
  if [[ -z "$base" ]]; then
    echo "No SLURM_TMPDIR found; leaving HF caches on shared storage."
    return 0
  fi

  local cache_root="$base/thesis_stats_cache/${SLURM_JOB_ID:-manual}"
  export TMPDIR="$cache_root/tmp"
  export HF_HOME="$cache_root/hf_home"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export TRANSFORMERS_CACHE="$HF_HOME/transformers"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  export XDG_CACHE_HOME="$cache_root/xdg"

  mkdir -p \
    "$TMPDIR" \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$HF_HUB_CACHE" \
    "$XDG_CACHE_HOME"

  echo "Using node-local caches:"
  echo "  TMPDIR=$TMPDIR"
  echo "  HF_HOME=$HF_HOME"
  echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
  echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
  echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
  echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
}

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV_NAME"
fi

REPO_ROOT="${REPO_ROOT:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/thesis_dataset_statistics/job_${SLURM_JOB_ID:-manual}}"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "ERROR: repo root not found: $REPO_ROOT" >&2
  exit 2
fi

if [[ ! -f "$REPO_ROOT/scripts/analysis/thesis_dataset_statistics.py" ]]; then
  echo "ERROR: missing script: $REPO_ROOT/scripts/analysis/thesis_dataset_statistics.py" >&2
  exit 2
fi

if [[ ! -f "$REPO_ROOT/scripts/analysis/run_thesis_dataset_statistics.sh" ]]; then
  echo "ERROR: missing wrapper: $REPO_ROOT/scripts/analysis/run_thesis_dataset_statistics.sh" >&2
  exit 2
fi

cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    export PYTHON_BIN
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN
    PYTHON_BIN="$(command -v python3)"
  fi
fi

setup_hf_auth
setup_local_cache_dirs

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
echo "[$(date)] repo_root=$REPO_ROOT"
echo "[$(date)] output_dir=$OUTPUT_DIR"
echo "[$(date)] python_bin=${PYTHON_BIN:-<auto>}"

CMD=(
  bash
  "$REPO_ROOT/scripts/analysis/run_thesis_dataset_statistics.sh"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "${REFERENCE_CONFIG:-}" ]]; then
  CMD+=(--reference-config "$REFERENCE_CONFIG")
fi
if [[ -n "${TOKENIZER_SOURCE:-}" ]]; then
  CMD+=(--tokenizer-source "$TOKENIZER_SOURCE")
fi
if [[ "${TOKENIZER_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  CMD+=(--tokenizer-local-files-only)
fi
if [[ -n "${PROJECT_DB:-}" ]]; then
  CMD+=(--project-db "$PROJECT_DB")
fi
if [[ -n "${SOURCE_DB:-}" ]]; then
  CMD+=(--source-db "$SOURCE_DB")
fi
if [[ -n "${PTBRVARID_DB:-}" ]]; then
  CMD+=(--ptbrvarid-db "$PTBRVARID_DB")
fi
if [[ -n "${PTBRVARID_SAMPLED_CSV:-}" ]]; then
  CMD+=(--ptbrvarid-sampled-csv "$PTBRVARID_SAMPLED_CSV")
fi
if [[ -n "${PTBRVARID_EXCLUDE_DOMAINS:-}" ]]; then
  CMD+=(--ptbrvarid-exclude-domains "$PTBRVARID_EXCLUDE_DOMAINS")
fi
if [[ -n "${PTBRVARID_TRANSLATED_DIR:-}" ]]; then
  CMD+=(--ptbrvarid-translated-dir "$PTBRVARID_TRANSLATED_DIR")
fi
if [[ -n "${TOKEN_BATCH_SIZE:-}" ]]; then
  CMD+=(--token-batch-size "$TOKEN_BATCH_SIZE")
fi
if [[ -n "${DUCKDB_BATCH_SIZE:-}" ]]; then
  CMD+=(--duckdb-batch-size "$DUCKDB_BATCH_SIZE")
fi
if [[ -n "${TOKEN_LOG_EVERY:-}" ]]; then
  CMD+=(--token-log-every "$TOKEN_LOG_EVERY")
fi
if [[ "${SKIP_TOKEN_COUNTS:-0}" == "1" ]]; then
  CMD+=(--skip-token-counts)
fi

printf '[%s] command:' "$(date)"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "[$(date)] generated files:"
find "$OUTPUT_DIR" -maxdepth 1 -type f | sort
echo "[$(date)] done"
