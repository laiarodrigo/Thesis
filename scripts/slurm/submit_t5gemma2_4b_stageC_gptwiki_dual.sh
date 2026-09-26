#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
PARTITION="${PARTITION:-hlt_msc}"
NODELIST="${NODELIST:-g06}"
GPU_GRES="${GPU_GRES:-gpu:RTX_6000_24GB:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
BUILD_MEM="${BUILD_MEM:-64G}"
TRAIN_MEM="${TRAIN_MEM:-64G}"
BUILD_TIME="${BUILD_TIME:-24:00:00}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"

STAGEB_JOB_ID="${STAGEB_JOB_ID:-}"
BUILD_SUBSET="${BUILD_SUBSET:-0}"
STAGEC_SUBSET_DIR="${STAGEC_SUBSET_DIR:-$REPO_ROOT/data/encoder_decoder/stage_c_subset/frmt_gpt_wiki_data}"
STAGEC_SUBSET_JSONL="${STAGEC_SUBSET_JSONL:-$STAGEC_SUBSET_DIR/stage_c_subset.jsonl}"
STAGEC_INPUT_PATHS="${STAGEC_INPUT_PATHS:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only/translation_train.jsonl,$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki/translation_train.jsonl}"
STAGEC_MAX_TOTAL="${STAGEC_MAX_TOTAL:-20000}"
STAGEC_PREVIEW_ROWS="${STAGEC_PREVIEW_ROWS:-50}"

BLEU_CONFIG="${BLEU_CONFIG:-configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gpt_wiki.yaml}"
WER_CONFIG="${WER_CONFIG:-configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gpt_wiki_wer.yaml}"

for req in \
  "$BLEU_CONFIG" \
  "$WER_CONFIG" \
  "scripts/slurm/build_stage_c_subset.sbatch" \
  "scripts/slurm/train_t5gemma2_4b_translation_stageC_grpo.sbatch" \
  "scripts/encoder_decoder/stage_c/build_stage_c_subset.py" \
  "scripts/encoder_decoder/stage_c/train_stage_c_seq2seq_grpo.py"
do
  [[ -e "$req" ]] || { echo "ERROR: missing file $req" >&2; exit 2; }
done

IFS=',' read -r -a INPUT_ARRAY <<< "$STAGEC_INPUT_PATHS"
for path in "${INPUT_ARRAY[@]}"; do
  trimmed="${path// }"
  [[ -n "$trimmed" ]] || continue
  [[ -f "$trimmed" ]] || { echo "ERROR: missing Stage C input path $trimmed" >&2; exit 2; }
done

J_SUBSET=""
if [[ "$BUILD_SUBSET" == "1" ]]; then
  J_SUBSET="$(
    INPUT_PATHS="$STAGEC_INPUT_PATHS" \
    OUT_DIR="$STAGEC_SUBSET_DIR" \
    MAX_TOTAL="$STAGEC_MAX_TOTAL" \
    PREVIEW_ROWS="$STAGEC_PREVIEW_ROWS" \
    VENV_PATH="$VENV_PATH" \
    sbatch --parsable \
      --job-name=stagec_sub \
      --partition="$PARTITION" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$BUILD_MEM" \
      --time="$BUILD_TIME" \
      scripts/slurm/build_stage_c_subset.sbatch
  )"
elif [[ ! -f "$STAGEC_SUBSET_JSONL" ]]; then
  echo "ERROR: missing Stage C subset: $STAGEC_SUBSET_JSONL" >&2
  echo "Set BUILD_SUBSET=1 to submit the subset build first." >&2
  exit 2
fi

deps=()
if [[ -n "$STAGEB_JOB_ID" ]]; then
  deps+=("$STAGEB_JOB_ID")
fi
if [[ -n "$J_SUBSET" ]]; then
  deps+=("$J_SUBSET")
fi

dependency_args=()
if (( ${#deps[@]} > 0 )); then
  dep_expr="$(IFS=:; echo "${deps[*]}")"
  dependency_args+=(--dependency="afterok:${dep_expr}")
fi

J_BLEU="$(
  CONFIG_PATH="$BLEU_CONFIG" \
  VENV_PATH="$VENV_PATH" \
  sbatch --parsable \
    "${dependency_args[@]}" \
    --job-name=t5g4b_c_bleu \
    --partition="$PARTITION" \
    --nodelist="$NODELIST" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$TRAIN_MEM" \
    --time="$TRAIN_TIME" \
    scripts/slurm/train_t5gemma2_4b_translation_stageC_grpo.sbatch
)"

J_WER="$(
  CONFIG_PATH="$WER_CONFIG" \
  VENV_PATH="$VENV_PATH" \
  sbatch --parsable \
    "${dependency_args[@]}" \
    --job-name=t5g4b_c_wer \
    --partition="$PARTITION" \
    --nodelist="$NODELIST" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$TRAIN_MEM" \
    --time="$TRAIN_TIME" \
    scripts/slurm/train_t5gemma2_4b_translation_stageC_grpo.sbatch
)"

cat <<EOF
Submitted Stage C jobs for the gpt_wiki Stage B model.

Stage B dependency: ${STAGEB_JOB_ID:-none}
Stage C subset job: ${J_SUBSET:-not submitted}
BLEU+copy penalty: $J_BLEU
WER reward:        $J_WER

Subset dir:
  $STAGEC_SUBSET_DIR
EOF
