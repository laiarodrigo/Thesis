#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
PARTITION="${PARTITION:-hlt_msc}"
GPU_GRES="${GPU_GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-48G}"
TRAIN_TIME="${TRAIN_TIME:-08:00:00}"
EVAL_TIME="${EVAL_TIME:-04:00:00}"

STAGEA_CONFIG="${STAGEA_CONFIG:-configs/encoder_decoder/multitask_270m/stageA_opensubs_frmt.yaml}"
STAGEB_CONFIG="${STAGEB_CONFIG:-configs/encoder_decoder/multitask_270m/stageB_gpt_adapt.yaml}"
STAGEA_OUT="${STAGEA_OUT:-outputs/encoder_decoder/multitask_compare/t5gemma2_270m_stageA_opensubs_frmt}"
STAGEB_OUT="${STAGEB_OUT:-outputs/encoder_decoder/multitask_compare/t5gemma2_270m_stageB_gpt_adapt}"

GOLDEN_TRANSLATION_DATASET="${GOLDEN_TRANSLATION_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl}"
GOLDEN_CLASSIFICATION_DATASET="${GOLDEN_CLASSIFICATION_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/classification_test.jsonl}"
GOLDEN_EVAL_OUT="${GOLDEN_EVAL_OUT:-eval_results/encoder_decoder/multitask_compare/t5gemma2_270m_stageB_golden}"

FRMT_TRANSLATION_DATASET="${FRMT_TRANSLATION_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/translation_test.jsonl}"
FRMT_CLASSIFICATION_DATASET="${FRMT_CLASSIFICATION_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/classification_test.jsonl}"
FRMT_EVAL_OUT="${FRMT_EVAL_OUT:-eval_results/encoder_decoder/multitask_compare/t5gemma2_270m_stageB_frmt}"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-384}"
MAX_NEW_TOKENS_TRANSLATION="${MAX_NEW_TOKENS_TRANSLATION:-192}"
MAX_NEW_TOKENS_CLASSIFICATION="${MAX_NEW_TOKENS_CLASSIFICATION:-2}"
ADAPTIVE_MAX_NEW_TOKENS="${ADAPTIVE_MAX_NEW_TOKENS:-1}"
ADAPTIVE_RATIO="${ADAPTIVE_RATIO:-1.15}"
ADAPTIVE_MARGIN="${ADAPTIVE_MARGIN:-6}"
ADAPTIVE_MIN_NEW_TOKENS="${ADAPTIVE_MIN_NEW_TOKENS:-6}"
ADAPTIVE_MAX_NEW_TOKENS_CEILING="${ADAPTIVE_MAX_NEW_TOKENS_CEILING:-192}"

cd "$REPO_ROOT"

for req in \
  "$STAGEA_CONFIG" \
  "$STAGEB_CONFIG" \
  "scripts/encoder_decoder/multitask/train_multitask_seq2seq.py" \
  "scripts/encoder_decoder/multitask/patch_saved_config_vocab.py" \
  "scripts/encoder_decoder/multitask/run_eval_multitask_270m.sh"
do
  [[ -e "$req" ]] || { echo "ERROR: missing file $req" >&2; exit 2; }
done

for dataset in \
  "$GOLDEN_TRANSLATION_DATASET" \
  "$GOLDEN_CLASSIFICATION_DATASET" \
  "$FRMT_TRANSLATION_DATASET" \
  "$FRMT_CLASSIFICATION_DATASET"
do
  [[ -e "$dataset" ]] || { echo "ERROR: missing dataset $dataset" >&2; exit 2; }
done

J_A="$(
  sbatch --parsable \
    --job-name=mt270_a \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TRAIN_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="set -euxo pipefail; cd '$REPO_ROOT'; source '$VENV_PATH/bin/activate'; python -u scripts/encoder_decoder/multitask/train_multitask_seq2seq.py --config '$STAGEA_CONFIG' --execute"
)"

J_PATCH="$(
  sbatch --parsable \
    --dependency=afterok:"$J_A" \
    --job-name=mt270_patch \
    --partition="$PARTITION" \
    --cpus-per-task=1 \
    --mem=4G \
    --time=00:10:00 \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="set -euxo pipefail; cd '$REPO_ROOT'; source '$VENV_PATH/bin/activate'; python -u scripts/encoder_decoder/multitask/patch_saved_config_vocab.py --model-dir '$STAGEA_OUT'"
)"

J_B="$(
  sbatch --parsable \
    --dependency=afterok:"$J_PATCH" \
    --job-name=mt270_b \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TRAIN_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="set -euxo pipefail; cd '$REPO_ROOT'; source '$VENV_PATH/bin/activate'; python -u scripts/encoder_decoder/multitask/train_multitask_seq2seq.py --config '$STAGEB_CONFIG' --execute"
)"

eval_wrap() {
  local out_dir="$1"
  local translation_dataset="$2"
  local classification_dataset="$3"
  local extra_adaptive=""
  if [[ "$ADAPTIVE_MAX_NEW_TOKENS" == "1" ]]; then
    extra_adaptive="ADAPTIVE_MAX_NEW_TOKENS=1"
  else
    extra_adaptive="ADAPTIVE_MAX_NEW_TOKENS=0"
  fi
  printf "set -euxo pipefail; cd '%s'; source '%s/bin/activate'; MODEL_ID='%s/%s' OUTPUT_DIR='%s/%s' TRANSLATION_DATASET='%s' CLASSIFICATION_DATASET='%s' BATCH_SIZE='%s' MAX_SOURCE_LENGTH='%s' MAX_NEW_TOKENS_TRANSLATION='%s' MAX_NEW_TOKENS_CLASSIFICATION='%s' ADAPTIVE_RATIO='%s' ADAPTIVE_MARGIN='%s' ADAPTIVE_MIN_NEW_TOKENS='%s' ADAPTIVE_MAX_NEW_TOKENS_CEILING='%s' %s bash scripts/encoder_decoder/multitask/run_eval_multitask_270m.sh" \
    "$REPO_ROOT" \
    "$VENV_PATH" \
    "$REPO_ROOT" \
    "$STAGEB_OUT" \
    "$REPO_ROOT" \
    "$out_dir" \
    "$translation_dataset" \
    "$classification_dataset" \
    "$BATCH_SIZE" \
    "$MAX_SOURCE_LENGTH" \
    "$MAX_NEW_TOKENS_TRANSLATION" \
    "$MAX_NEW_TOKENS_CLASSIFICATION" \
    "$ADAPTIVE_RATIO" \
    "$ADAPTIVE_MARGIN" \
    "$ADAPTIVE_MIN_NEW_TOKENS" \
    "$ADAPTIVE_MAX_NEW_TOKENS_CEILING" \
    "$extra_adaptive"
}

J_E_GOLD="$(
  sbatch --parsable \
    --dependency=afterok:"$J_B" \
    --job-name=mt270_eval_gold \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$EVAL_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="$(eval_wrap "$GOLDEN_EVAL_OUT" "$GOLDEN_TRANSLATION_DATASET" "$GOLDEN_CLASSIFICATION_DATASET")"
)"

J_E_FRMT="$(
  sbatch --parsable \
    --dependency=afterok:"$J_B" \
    --job-name=mt270_eval_frmt \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$EVAL_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="$(eval_wrap "$FRMT_EVAL_OUT" "$FRMT_TRANSLATION_DATASET" "$FRMT_CLASSIFICATION_DATASET")"
)"

echo "Submitted multitask 270m chain:"
echo "  stageA      = $J_A"
echo "  patch_vocab = $J_PATCH"
echo "  stageB      = $J_B"
echo "  eval_golden = $J_E_GOLD"
echo "  eval_frmt   = $J_E_FRMT"
echo
echo "Monitor:"
echo "  squeue -j $J_A,$J_PATCH,$J_B,$J_E_GOLD,$J_E_FRMT"
