#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PARTITION="${PARTITION:-hlt_msc}"
TARGET_NODE="${TARGET_NODE:-}"
GPU_GRES="${GPU_GRES:-gpu:1}"
BUILD_GPU_GRES="${BUILD_GPU_GRES:-}"
TRAIN_TIME="${TRAIN_TIME:-7-00:00:00}"
EVAL_TIME="${EVAL_TIME:-2-00:00:00}"
EVAL_THROTTLE="${EVAL_THROTTLE:-3}"
BUILD_ONLY="${BUILD_ONLY:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
CONFIRM_DATASET_COUNTS="${CONFIRM_DATASET_COUNTS:-0}"

if [[ "$BUILD_ONLY" == "1" && "$SKIP_BUILD" == "1" ]]; then
  echo "BUILD_ONLY=1 and SKIP_BUILD=1 are mutually exclusive" >&2
  exit 2
fi
if [[ "$BUILD_ONLY" != "1" && "$CONFIRM_DATASET_COUNTS" != "1" ]]; then
  echo "Refusing to train before dataset-count review. Run BUILD_ONLY=1, inspect" >&2
  echo "dataset_count_comparison.json, then set CONFIRM_DATASET_COUNTS=1." >&2
  exit 2
fi

mkdir -p \
  logs/control_strings/build \
  logs/control_strings/train \
  logs/control_strings/eval

node_args=()
if [[ -n "$TARGET_NODE" ]]; then
  node_args+=(--nodelist="$TARGET_NODE")
fi
build_gpu_args=()
if [[ -n "$BUILD_GPU_GRES" ]]; then
  build_gpu_args+=(--gres="$BUILD_GPU_GRES")
fi

build_dependency=()
if [[ "$SKIP_BUILD" == "1" ]]; then
  for report in \
    data/encoder_decoder/t5gemma2/control_string_protocol/stageA_opensubs_only/audit_report.json \
    data/encoder_decoder/t5gemma2/control_string_protocol/stageB_gpt_wiki/audit_report.json \
    data/encoder_decoder/t5gemma2/control_string_protocol/stageB_gpt_wiki_frmt/audit_report.json \
    data/encoder_decoder/t5gemma2/control_string_protocol/dataset_count_comparison.json \
    data/encoder_decoder/t5gemma2/control_string_eval/build_report.json
  do
    [[ -s "$report" ]] || { echo "Missing completed build report: $report" >&2; exit 2; }
  done
  BUILD_JOB_ID=SKIPPED
else
  BUILD_JOB_ID=$(sbatch --parsable \
    --partition="$PARTITION" \
    "${node_args[@]}" \
    "${build_gpu_args[@]}" \
    --job-name=build_4b_ctrlstr \
    --cpus-per-task=2 \
    --mem=64G \
    --time=12:00:00 \
    --output=logs/control_strings/build/%x_%j.out <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
cd ~/repos/Thesis
source ~/thesis_t5/bin/activate
PYTHONUNBUFFERED=1 bash scripts/encoder_decoder/control_strings/build_all_control_string_data.sh
SBATCH
  )
  build_dependency=(--dependency="afterok:${BUILD_JOB_ID}")
fi

printf 'BUILD_JOB_ID=%s\n' "$BUILD_JOB_ID"
if [[ "$BUILD_ONLY" == "1" ]]; then
  exit 0
fi

STAGE_A_JOB_ID=$(sbatch --parsable \
  "${build_dependency[@]}" \
  --partition="$PARTITION" \
  "${node_args[@]}" \
  --gres="$GPU_GRES" \
  --job-name=A4_ctrlstr \
  --array=0-1%2 \
  --cpus-per-task=8 \
  --mem=96G \
  --time="$TRAIN_TIME" \
  --export=ALL,THESIS_PRECISION_OVERRIDE=bf16 \
  --output=logs/control_strings/train/%x_%A_%a.out <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
cd ~/repos/Thesis
source ~/thesis_t5/bin/activate

CONFIGS=(
  configs/encoder_decoder/t5gemma2_4b/control_strings/stageA_opensubs_encoder_control_strings.yaml
  configs/encoder_decoder/t5gemma2_4b/control_strings/stageA_opensubs_decoder_control_strings.yaml
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "CONFIG=$CONFIG HOST=$(hostname) PRECISION=$THESIS_PRECISION_OVERRIDE"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
PYTHONUNBUFFERED=1 python scripts/encoder_decoder/train_encdec_lora.py --config "$CONFIG"
SBATCH
)

STAGE_B_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${STAGE_A_JOB_ID}" \
  --partition="$PARTITION" \
  "${node_args[@]}" \
  --gres="$GPU_GRES" \
  --job-name=B4_ctrlstr \
  --array=0-3%4 \
  --cpus-per-task=8 \
  --mem=96G \
  --time="$TRAIN_TIME" \
  --export=ALL \
  --output=logs/control_strings/train/%x_%A_%a.out <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
cd ~/repos/Thesis
source ~/thesis_t5/bin/activate

CONFIGS=(
  configs/encoder_decoder/t5gemma2_4b/control_strings/stageB_gpt_wiki_encoder_control_strings.yaml
  configs/encoder_decoder/t5gemma2_4b/control_strings/stageB_gpt_wiki_decoder_control_strings.yaml
  configs/encoder_decoder/t5gemma2_4b/control_strings/stageB_gpt_wiki_frmt_encoder_control_strings.yaml
  configs/encoder_decoder/t5gemma2_4b/control_strings/stageB_gpt_wiki_frmt_decoder_control_strings.yaml
)
PRECISIONS=(bf16 bf16 bf16 bf16)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
export THESIS_PRECISION_OVERRIDE="${PRECISIONS[$SLURM_ARRAY_TASK_ID]}"
echo "CONFIG=$CONFIG HOST=$(hostname) PRECISION=$THESIS_PRECISION_OVERRIDE"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true
PYTHONUNBUFFERED=1 python scripts/encoder_decoder/train_encdec_lora.py --config "$CONFIG"
SBATCH
)

EVAL_JOB_ID=$(sbatch --parsable \
  --dependency="afterok:${STAGE_B_JOB_ID}" \
  --partition="$PARTITION" \
  "${node_args[@]}" \
  --gres="$GPU_GRES" \
  --job-name=eval_4b_ctrlstr \
  --array="0-23%${EVAL_THROTTLE}" \
  --cpus-per-task=4 \
  --mem=64G \
  --time="$EVAL_TIME" \
  --output=logs/control_strings/eval/%x_%A_%a.out <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail
cd ~/repos/Thesis
source ~/thesis_t5/bin/activate

MODELS=(
  "B4-E-G-control-strings|encoder_unified|outputs/encoder_decoder/control_strings/t5gemma2_4b_stageB_gpt_wiki_encoder_control_strings"
  "B4-D-G-control-strings|decoder_unified|outputs/encoder_decoder/control_strings/t5gemma2_4b_stageB_gpt_wiki_decoder_control_strings"
  "B4-E-GF-control-strings|encoder_unified|outputs/encoder_decoder/control_strings/t5gemma2_4b_stageB_gpt_wiki_frmt_encoder_control_strings"
  "B4-D-GF-control-strings|decoder_unified|outputs/encoder_decoder/control_strings/t5gemma2_4b_stageB_gpt_wiki_frmt_decoder_control_strings"
)
DATASETS=(frmt golden)
TASKS=(translation_greedy translation_beam4 classification)

model_idx=$((SLURM_ARRAY_TASK_ID / 6))
remainder=$((SLURM_ARRAY_TASK_ID % 6))
dataset_idx=$((remainder / 3))
task_idx=$((remainder % 3))

IFS='|' read -r NAME DATA_KIND ADAPTER <<< "${MODELS[$model_idx]}"
DATASET="${DATASETS[$dataset_idx]}"
TASK="${TASKS[$task_idx]}"
DATA_ROOT="data/encoder_decoder/t5gemma2/control_string_eval/${DATASET}/${DATA_KIND}"
OUT_ROOT="eval_results/encoder_decoder/control_strings/${NAME}"

echo "NAME=$NAME DATA_KIND=$DATA_KIND DATASET=$DATASET TASK=$TASK HOST=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true

if [[ "$TASK" == "classification" ]]; then
  python scripts/encoder_decoder/eval/evaluate_encdec.py \
    --task classification \
    --dataset-path "${DATA_ROOT}/classification_test.jsonl" \
    --model-id google/t5gemma-2-4b-4b \
    --adapter-dir "$ADAPTER" \
    --tokenizer-path "$ADAPTER" \
    --batch-size 16 \
    --max-source-length 512 \
    --classification-mode score-sequences \
    --classification-candidates '<pt-br>' '<pt-pt>' \
    --output-dir "${OUT_ROOT}/${DATASET}_classification"
elif [[ "$TASK" == "translation_beam4" ]]; then
  python scripts/encoder_decoder/eval/evaluate_encdec.py \
    --task translation \
    --dataset-path "${DATA_ROOT}/translation_test.jsonl" \
    --model-id google/t5gemma-2-4b-4b \
    --adapter-dir "$ADAPTER" \
    --tokenizer-path "$ADAPTER" \
    --batch-size 4 \
    --max-source-length 512 \
    --max-new-tokens 256 \
    --num-beams 4 \
    --length-penalty 0.8 \
    --early-stopping \
    --adaptive-max-new-tokens \
    --adaptive-ratio 1.15 \
    --adaptive-margin 6 \
    --adaptive-min-new-tokens 8 \
    --adaptive-max-new-tokens-ceiling 384 \
    --no-repeat-ngram-size 4 \
    --repetition-penalty 1.2 \
    --output-dir "${OUT_ROOT}/${DATASET}_translation_beam4"
else
  python scripts/encoder_decoder/eval/evaluate_encdec.py \
    --task translation \
    --dataset-path "${DATA_ROOT}/translation_test.jsonl" \
    --model-id google/t5gemma-2-4b-4b \
    --adapter-dir "$ADAPTER" \
    --tokenizer-path "$ADAPTER" \
    --batch-size 4 \
    --max-source-length 512 \
    --max-new-tokens 256 \
    --adaptive-max-new-tokens \
    --adaptive-ratio 1.3 \
    --adaptive-margin 10 \
    --adaptive-min-new-tokens 16 \
    --adaptive-max-new-tokens-ceiling 256 \
    --output-dir "${OUT_ROOT}/${DATASET}_translation_greedy"
fi
SBATCH
)

printf 'STAGE_A_JOB_ID=%s\n' "$STAGE_A_JOB_ID"
printf 'STAGE_B_JOB_ID=%s\n' "$STAGE_B_JOB_ID"
printf 'EVAL_JOB_ID=%s\n' "$EVAL_JOB_ID"
