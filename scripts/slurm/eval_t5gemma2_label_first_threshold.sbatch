#!/usr/bin/env bash
#SBATCH --job-name=t5g_lfthr
#SBATCH --partition=hlt_msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL_ID="${MODEL_ID:-}"
DATASET_PATH="${DATASET_PATH:-data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-384}"
BR_TOKEN="${BR_TOKEN:-BR}"
PT_TOKEN="${PT_TOKEN:-PT}"
THRESHOLD="${THRESHOLD:-}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.00,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50}"
PREVIEW_COUNT="${PREVIEW_COUNT:-15}"

if [[ -z "$MODEL_ID" ]]; then
  echo "ERROR: set MODEL_ID to the seq2seq model/checkpoint directory." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: set OUTPUT_DIR for threshold evaluation outputs." >&2
  exit 2
fi

if [[ ! -d "$MODEL_ID" ]]; then
  echo "ERROR: missing model dir: $MODEL_ID" >&2
  exit 2
fi
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "ERROR: missing dataset path: $DATASET_PATH" >&2
  exit 2
fi
if [[ -n "$TOKENIZER_PATH" && ! -d "$TOKENIZER_PATH" ]]; then
  echo "ERROR: missing tokenizer path: $TOKENIZER_PATH" >&2
  exit 2
fi
if [[ -n "$ADAPTER_DIR" && ! -d "$ADAPTER_DIR" ]]; then
  echo "ERROR: missing adapter dir: $ADAPTER_DIR" >&2
  exit 2
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "ERROR: could not initialize conda; set CONDA_SH or VENV_PATH." >&2
    exit 2
  fi
  conda activate "$CONDA_ENV_NAME"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi || true

mkdir -p "$OUTPUT_DIR"

ARGS=(
  --dataset-path "$DATASET_PATH"
  --model-id "$MODEL_ID"
  --batch-size "$BATCH_SIZE"
  --max-source-length "$MAX_SOURCE_LENGTH"
  --br-token "$BR_TOKEN"
  --pt-token "$PT_TOKEN"
  --threshold-grid "$THRESHOLD_GRID"
  --preview-count "$PREVIEW_COUNT"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$TOKENIZER_PATH" ]]; then
  ARGS+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "$ADAPTER_DIR" ]]; then
  ARGS+=(--adapter-dir "$ADAPTER_DIR")
fi
if [[ -n "$THRESHOLD" ]]; then
  ARGS+=(--threshold "$THRESHOLD")
fi

echo "Running label-first threshold eval:"
echo "  model=$MODEL_ID"
echo "  dataset=$DATASET_PATH"
echo "  output=$OUTPUT_DIR"
echo "  batch_size=$BATCH_SIZE"
echo "  max_source_length=$MAX_SOURCE_LENGTH"
echo "  br_token=$BR_TOKEN"
echo "  pt_token=$PT_TOKEN"
echo "  threshold=${THRESHOLD:-<sweep only>}"
echo "  threshold_grid=$THRESHOLD_GRID"
echo "  preview_count=$PREVIEW_COUNT"

python3 scripts/encoder_decoder/eval/evaluate_label_first_threshold.py "${ARGS[@]}"
