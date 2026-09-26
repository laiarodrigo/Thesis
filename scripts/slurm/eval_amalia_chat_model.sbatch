#!/usr/bin/env bash
#SBATCH --job-name=chat_eval
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

TASK="${TASK:-translation}"
DATA_PROTOCOL="${DATA_PROTOCOL:-control_strings}"
DATASET_NAME="${DATASET_NAME:-golden}"
DATA_KIND="${DATA_KIND:-decoder_unified}"
MODEL_ID="${MODEL_ID:-amalia-llm/AMALIA-9B-0626-SFT}"
MODEL_SLUG="${MODEL_SLUG:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_results/decoder_only/external_chat_models}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
CLASSIFICATION_MODE="${CLASSIFICATION_MODE:-score-sequences}"
CLASSIFICATION_CANDIDATES="${CLASSIFICATION_CANDIDATES:-}"

normalize_dataset_name() {
  local raw_name="$1"
  case "$raw_name" in
    golden|golden_collection)
      printf 'golden'
      ;;
    frmt|frmt_only)
      printf 'frmt'
      ;;
    *)
      return 1
      ;;
  esac
}

slugify_model_id() {
  printf '%s' "$1" | tr '/:@ ' '_' | tr -cs '[:alnum:]_.-' '_'
}

DATASET_KEY="$(normalize_dataset_name "$DATASET_NAME")" || {
  echo "ERROR: DATASET_NAME must be one of: golden, golden_collection, frmt, frmt_only" >&2
  exit 2
}

if [[ -z "$MODEL_SLUG" ]]; then
  MODEL_SLUG="$(slugify_model_id "$MODEL_ID")"
fi

case "$DATA_PROTOCOL" in
  legacy)
    OUTPUT_VIEW="legacy"
    case "$DATASET_KEY" in
      golden)
        DATA_DIR="data/encoder_decoder/t5gemma2/golden_collection"
        OUTPUT_DATASET_DIR="golden_collection"
        ;;
      frmt)
        DATA_DIR="data/encoder_decoder/t5gemma2/frmt_only"
        OUTPUT_DATASET_DIR="frmt"
        ;;
    esac
    if [[ -z "$CLASSIFICATION_CANDIDATES" ]]; then
      CLASSIFICATION_CANDIDATES="pt-br pt-pt"
    fi
    ;;
  final)
    OUTPUT_VIEW="$DATA_KIND"
    case "$DATA_KIND" in
      translation_only|encoder_unified|decoder_unified) ;;
      *)
        echo "ERROR: DATA_KIND must be one of: translation_only, encoder_unified, decoder_unified" >&2
        exit 2
        ;;
    esac
    DATA_DIR="data/encoder_decoder/t5gemma2/final_eval/${DATASET_KEY}/${DATA_KIND}"
    OUTPUT_DATASET_DIR="$DATASET_KEY"
    if [[ -z "$CLASSIFICATION_CANDIDATES" ]]; then
      CLASSIFICATION_CANDIDATES="pt-br pt-pt"
    fi
    ;;
  control_strings)
    OUTPUT_VIEW="$DATA_KIND"
    case "$DATA_KIND" in
      encoder_unified|decoder_unified) ;;
      *)
        echo "ERROR: DATA_KIND must be one of: encoder_unified, decoder_unified" >&2
        exit 2
        ;;
    esac
    DATA_DIR="data/encoder_decoder/t5gemma2/control_string_eval/${DATASET_KEY}/${DATA_KIND}"
    OUTPUT_DATASET_DIR="$DATASET_KEY"
    if [[ -z "$CLASSIFICATION_CANDIDATES" ]]; then
      CLASSIFICATION_CANDIDATES="<pt-br> <pt-pt>"
    fi
    ;;
  control_strings_ptbr)
    OUTPUT_VIEW="$DATA_KIND"
    case "$DATA_KIND" in
      encoder_unified|decoder_unified) ;;
      *)
        echo "ERROR: DATA_KIND must be one of: encoder_unified, decoder_unified" >&2
        exit 2
        ;;
    esac
    DATA_DIR="data/encoder_decoder/t5gemma2/control_string_ptbr_eval/${DATASET_KEY}/${DATA_KIND}"
    OUTPUT_DATASET_DIR="$DATASET_KEY"
    if [[ -z "$CLASSIFICATION_CANDIDATES" ]]; then
      CLASSIFICATION_CANDIDATES="BR PT"
    fi
    ;;
  *)
    echo "ERROR: DATA_PROTOCOL must be one of: legacy, final, control_strings, control_strings_ptbr" >&2
    exit 2
    ;;
esac

case "$TASK" in
  translation)
    EVAL_SCRIPT="scripts/decoder_only/axolotl/run_eval_translation_chat_model.sh"
    DATASET_PATH="${DATASET_PATH:-$DATA_DIR/translation_test.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$MODEL_SLUG/$DATA_PROTOCOL/$OUTPUT_VIEW/$OUTPUT_DATASET_DIR/translation}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
    ;;
  classification)
    EVAL_SCRIPT="scripts/decoder_only/axolotl/run_eval_classification_chat_model.sh"
    case "$DATA_PROTOCOL" in
      legacy|control_strings|control_strings_ptbr)
        DATASET_PATH="${DATASET_PATH:-$DATA_DIR/classification_test.jsonl}"
        ;;
      final)
        if [[ "$DATA_KIND" == "translation_only" ]]; then
          echo "ERROR: classification is unavailable for DATA_PROTOCOL=final with DATA_KIND=translation_only" >&2
          exit 2
        fi
        DATASET_PATH="${DATASET_PATH:-$DATA_DIR/classification_noequal_test.jsonl}"
        ;;
    esac
    OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$MODEL_SLUG/$DATA_PROTOCOL/$OUTPUT_VIEW/$OUTPUT_DATASET_DIR/classification}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
    ;;
  *)
    echo "ERROR: TASK must be one of: translation, classification" >&2
    exit 2
    ;;
esac

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "ERROR: missing eval script: $EVAL_SCRIPT" >&2
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

if [[ -n "${ENV_SETUP_CMD:-}" ]]; then
  eval "$ENV_SETUP_CMD"
elif [[ -n "${VENV_PATH:-}" ]]; then
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
elif [[ -f "$HOME/thesis_t5/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/thesis_t5/bin/activate"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi || true

mkdir -p "$OUTPUT_DIR"

RUN_ENV=(
  "MODEL_ID=$MODEL_ID"
  "DATASET_PATH=$DATASET_PATH"
  "OUTPUT_DIR=$OUTPUT_DIR"
  "BATCH_SIZE=$BATCH_SIZE"
  "MAX_INPUT_LENGTH=$MAX_INPUT_LENGTH"
  "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
)

if [[ -n "${PYTHON_BIN:-}" ]]; then
  RUN_ENV+=("PYTHON_BIN=$PYTHON_BIN")
fi
if [[ -n "$TOKENIZER_PATH" ]]; then
  RUN_ENV+=("TOKENIZER_PATH=$TOKENIZER_PATH")
fi
if [[ -n "$ADAPTER_DIR" ]]; then
  RUN_ENV+=("ADAPTER_DIR=$ADAPTER_DIR")
fi
if [[ "$TASK" == "classification" ]]; then
  RUN_ENV+=(
    "CLASSIFICATION_MODE=$CLASSIFICATION_MODE"
    "CLASSIFICATION_CANDIDATES=$CLASSIFICATION_CANDIDATES"
  )
fi

echo "Running external chat-model eval:"
echo "  task=$TASK"
echo "  data_protocol=$DATA_PROTOCOL"
echo "  data_kind=$DATA_KIND"
echo "  dataset_name=$OUTPUT_DATASET_DIR"
echo "  model=$MODEL_ID"
echo "  model_slug=$MODEL_SLUG"
echo "  dataset=$DATASET_PATH"
echo "  output=$OUTPUT_DIR"
echo "  batch_size=$BATCH_SIZE"
echo "  max_input_length=$MAX_INPUT_LENGTH"
echo "  max_new_tokens=$MAX_NEW_TOKENS"
if [[ "$TASK" == "classification" ]]; then
  echo "  classification_mode=$CLASSIFICATION_MODE"
  echo "  classification_candidates=$CLASSIFICATION_CANDIDATES"
fi

env "${RUN_ENV[@]}" bash "$EVAL_SCRIPT"
