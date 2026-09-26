#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PARTITION="${PARTITION:-cpu-cluster}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
JOB_NAME="${JOB_NAME:-thesis_stats}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/thesis_dataset_statistics/manual_$(date +%Y%m%d_%H%M%S)}"

for req in \
  "thesis_dataset_statistics.py" \
  "run_thesis_dataset_statistics.sh" \
  "scripts/slurm/run_thesis_dataset_statistics.sbatch"
do
  [[ -f "$req" ]] || { echo "ERROR: missing file $req" >&2; exit 2; }
done

SBATCH_ENV=(
  "REPO_ROOT=$REPO_ROOT"
  "OUTPUT_DIR=$OUTPUT_DIR"
)

for name in \
  VENV_PATH \
  CONDA_ENV_NAME \
  CONDA_SH \
  PYTHON_BIN \
  HF_ENV_FILE \
  HF_TOKEN \
  HUGGINGFACE_HUB_TOKEN \
  USE_LOCAL_HF_CACHE \
  REFERENCE_CONFIG \
  TOKENIZER_SOURCE \
  TOKENIZER_LOCAL_FILES_ONLY \
  PROJECT_DB \
  SOURCE_DB \
  PTBRVARID_DB \
  PTBRVARID_SAMPLED_CSV \
  PTBRVARID_TRANSLATED_DIR \
  TOKEN_BATCH_SIZE \
  DUCKDB_BATCH_SIZE \
  TOKEN_LOG_EVERY \
  SKIP_TOKEN_COUNTS
do
  if [[ -n "${!name:-}" ]]; then
    SBATCH_ENV+=("$name=${!name}")
  fi
done

JID="$(
  env "${SBATCH_ENV[@]}" \
  sbatch --parsable \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME_LIMIT" \
    scripts/slurm/run_thesis_dataset_statistics.sbatch
)"

cat <<EOF
Submitted thesis dataset statistics job.

Job:
  $JID

Output dir:
  $OUTPUT_DIR

Monitor:
  squeue --job $JID
  sacct -j $JID
  tail -f slurm-${JOB_NAME}-${JID}.out
EOF
