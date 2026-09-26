#!/usr/bin/env bash
#SBATCH --job-name=t5g_seqcls
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
DATASET_PATH="${DATASET_PATH:-data/encoder_decoder/t5gemma2/golden_collection/classification_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-384}"
CLASSIFICATION_MODE="${CLASSIFICATION_MODE:-score-sequences}"
CLASSIFICATION_CANDIDATES="${CLASSIFICATION_CANDIDATES:-BR PT igual}"

if [[ -z "$MODEL_ID" ]]; then
  echo "ERROR: set MODEL_ID to a seq2seq model id or local checkpoint directory." >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: set OUTPUT_DIR for evaluation outputs." >&2
  exit 2
fi

if [[ -e "$MODEL_ID" && ! -d "$MODEL_ID" ]]; then
  echo "ERROR: MODEL_ID exists but is not a directory: $MODEL_ID" >&2
  exit 2
fi
if [[ ! -e "$MODEL_ID" ]]; then
  echo "MODEL_ID is not a local directory; treating it as a remote model id: $MODEL_ID"
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
  --task classification
  --dataset-path "$DATASET_PATH"
  --model-id "$MODEL_ID"
  --classification-mode "$CLASSIFICATION_MODE"
  --batch-size "$BATCH_SIZE"
  --max-source-length "$MAX_SOURCE_LENGTH"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$TOKENIZER_PATH" ]]; then
  ARGS+=(--tokenizer-path "$TOKENIZER_PATH")
fi
if [[ -n "$ADAPTER_DIR" ]]; then
  ARGS+=(--adapter-dir "$ADAPTER_DIR")
fi

read -r -a CLASS_CAND_ARRAY <<< "$CLASSIFICATION_CANDIDATES"
if [[ "${#CLASS_CAND_ARRAY[@]}" -gt 0 ]]; then
  ARGS+=(--classification-candidates "${CLASS_CAND_ARRAY[@]}")
fi

echo "Running seq2seq classification eval:"
echo "  model=$MODEL_ID"
echo "  dataset=$DATASET_PATH"
echo "  output=$OUTPUT_DIR"
echo "  mode=$CLASSIFICATION_MODE"
echo "  batch_size=$BATCH_SIZE"
echo "  max_source_length=$MAX_SOURCE_LENGTH"
echo "  candidates=$CLASSIFICATION_CANDIDATES"

python3 scripts/encoder_decoder/eval/evaluate_encdec.py "${ARGS[@]}"
