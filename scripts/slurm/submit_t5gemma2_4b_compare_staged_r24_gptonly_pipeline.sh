#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
PARTITION="${PARTITION:-hlt_msc}"
GPU_GRES="${GPU_GRES:-gpu:RTX_3090_24GB:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
EXPORT_MEM="${EXPORT_MEM:-64G}"
TRAIN_MEM="${TRAIN_MEM:-30G}"
EVAL_MEM="${EVAL_MEM:-30G}"
EXPORT_TIME="${EXPORT_TIME:-24:00:00}"
TRAIN_TIME="${TRAIN_TIME:-14-00:00:00}"
EVAL_TIME="${EVAL_TIME:-24:00:00}"

STAGEA_DATA_DIR="${STAGEA_DATA_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt}"
STAGEB_DATA_DIR="${STAGEB_DATA_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki}"

GPT_INPUT_CSV="${GPT_INPUT_CSV:-$REPO_ROOT/data/wikipedia_pt_variant_csv/pt_variant_prompts_wikipedia_merged.csv}"
STAGEB_TRAIN_RATIO="${STAGEB_TRAIN_RATIO:-0.8}"
STAGEB_VALID_RATIO="${STAGEB_VALID_RATIO:-0.1}"
STAGEB_SEED="${STAGEB_SEED:-42}"

STAGEA_CONFIG="${STAGEA_CONFIG:-configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_plus_frmt.yaml}"
STAGEB_CONFIG="${STAGEB_CONFIG:-configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageB_gpt_wikipedia_only.yaml}"

STAGEA_MODEL_DIR="${STAGEA_MODEL_DIR:-outputs/encoder_decoder/compare_staged/t5gemma2_4b_translation_r24_stageA_opensubs_frmt}"
STAGEB_MODEL_DIR="${STAGEB_MODEL_DIR:-outputs/encoder_decoder/compare_staged/t5gemma2_4b_translation_r24_stageB_gpt_wiki}"

GOLDEN_DATASET="${GOLDEN_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl}"
FRMT_DATASET="${FRMT_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only/translation_test.jsonl}"
EVAL_SCRIPT="${EVAL_SCRIPT:-scripts/encoder_decoder/eval/run_eval_translation_gemma4b_r24_adaptive.sh}"
GOLDEN_EVAL_OUT="${GOLDEN_EVAL_OUT:-eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageB_gpt_wiki}"
FRMT_EVAL_OUT="${FRMT_EVAL_OUT:-eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r24_stageB_gpt_wiki}"

for req in \
  "$STAGEA_CONFIG" \
  "$STAGEB_CONFIG" \
  "$GPT_INPUT_CSV" \
  "$GOLDEN_DATASET" \
  "$FRMT_DATASET" \
  "scripts/slurm/train_t5gemma2_4b_translation_compare_staged.sbatch" \
  "scripts/slurm/eval_t5gemma2_4b_translation_r24_stageB_gpt_wiki.sbatch" \
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageA_opensubs_plus_frmt.sh" \
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageB_gpt_wikipedia_only.sh" \
  "$EVAL_SCRIPT"
do
  [[ -e "$req" ]] || { echo "ERROR: missing file $req" >&2; exit 2; }
done

export_stagea_wrap() {
  printf "set -euxo pipefail; cd '%s'; source '%s/bin/activate'; OUT_DIR='%s' bash scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageA_opensubs_plus_frmt.sh" \
    "$REPO_ROOT" \
    "$VENV_PATH" \
    "$STAGEA_DATA_DIR"
}

export_stageb_wrap() {
  printf "set -euxo pipefail; cd '%s'; source '%s/bin/activate'; INPUT_CSV='%s' OUT_DIR='%s' TRAIN_RATIO='%s' VALID_RATIO='%s' SEED='%s' bash scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageB_gpt_wikipedia_only.sh" \
    "$REPO_ROOT" \
    "$VENV_PATH" \
    "$GPT_INPUT_CSV" \
    "$STAGEB_DATA_DIR" \
    "$STAGEB_TRAIN_RATIO" \
    "$STAGEB_VALID_RATIO" \
    "$STAGEB_SEED"
}

J_EXPORT_A="$(
  sbatch --parsable \
    --job-name=t5g4b_gpta \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$EXPORT_MEM" \
    --time="$EXPORT_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="$(export_stagea_wrap)"
)"

J_EXPORT_B="$(
  sbatch --parsable \
    --job-name=t5g4b_gptb \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$EXPORT_MEM" \
    --time="$EXPORT_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="$(export_stageb_wrap)"
)"

J_STAGEA="$(
  CONFIG_PATH="$STAGEA_CONFIG" \
  VENV_PATH="$VENV_PATH" \
  sbatch --parsable \
    --dependency=afterok:"$J_EXPORT_A" \
    --job-name=t5g4b_s1 \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$TRAIN_MEM" \
    --time="$TRAIN_TIME" \
    scripts/slurm/train_t5gemma2_4b_translation_compare_staged.sbatch
)"

J_STAGEB="$(
  CONFIG_PATH="$STAGEB_CONFIG" \
  VENV_PATH="$VENV_PATH" \
  sbatch --parsable \
    --dependency=afterok:"$J_STAGEA":"$J_EXPORT_B" \
    --job-name=t5g4b_s2 \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$TRAIN_MEM" \
    --time="$TRAIN_TIME" \
    scripts/slurm/train_t5gemma2_4b_translation_compare_staged.sbatch
)"

J_EVAL_GOLDEN="$(
  EVAL_SCRIPT="$EVAL_SCRIPT" \
  ADAPTER_DIR="$STAGEB_MODEL_DIR" \
  OUTPUT_DIR="$GOLDEN_EVAL_OUT" \
  DATASET_PATH="$GOLDEN_DATASET" \
  VENV_PATH="$VENV_PATH" \
  sbatch --parsable \
    --dependency=afterok:"$J_STAGEB" \
    --job-name=t5g4b_eval_g \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$EVAL_MEM" \
    --time="$EVAL_TIME" \
    scripts/slurm/eval_t5gemma2_4b_translation_r24_stageB_gpt_wiki.sbatch
)"

J_EVAL_FRMT="$(
  EVAL_SCRIPT="$EVAL_SCRIPT" \
  ADAPTER_DIR="$STAGEB_MODEL_DIR" \
  OUTPUT_DIR="$FRMT_EVAL_OUT" \
  DATASET_PATH="$FRMT_DATASET" \
  VENV_PATH="$VENV_PATH" \
  sbatch --parsable \
    --dependency=afterok:"$J_STAGEB" \
    --job-name=t5g4b_eval_f \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$EVAL_MEM" \
    --time="$EVAL_TIME" \
    scripts/slurm/eval_t5gemma2_4b_translation_r24_stageB_gpt_wiki.sbatch
)"

cat <<EOF
Submitted compare-staged r24 pipeline.

Data prep:
  Stage A export:         $J_EXPORT_A
  Stage B GPT export:     $J_EXPORT_B

Training:
  Stage A train:          $J_STAGEA
  Stage B train:          $J_STAGEB

Evaluation:
  Golden Collection:      $J_EVAL_GOLDEN
  FRMT:                   $J_EVAL_FRMT

Model dirs:
  Stage A:                $STAGEA_MODEL_DIR
  Stage B:                $STAGEB_MODEL_DIR
EOF
