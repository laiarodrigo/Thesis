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

STAGEA_CONFIG="${STAGEA_CONFIG:-configs/encoder_decoder/multitask_270m/stageA_opensubs_frmt_smoke.yaml}"
STAGEB_CONFIG="${STAGEB_CONFIG:-configs/encoder_decoder/multitask_270m/stageB_gpt_adapt_smoke.yaml}"
STAGEA_OUT="${STAGEA_OUT:-outputs/encoder_decoder/multitask_compare/t5gemma2_270m_stageA_opensubs_frmt_smoke}"
STAGEB_OUT="${STAGEB_OUT:-outputs/encoder_decoder/multitask_compare/t5gemma2_270m_stageB_gpt_adapt_smoke}"
EVAL_OUT="${EVAL_OUT:-eval_results/encoder_decoder/multitask_compare/t5gemma2_270m_stageB_gpt_adapt_smoke_golden}"

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

J_A="$(
  sbatch --parsable \
    --job-name=mt270_a_smoke \
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
    --job-name=mt270_patch_smoke \
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
    --job-name=mt270_b_smoke \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TRAIN_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="set -euxo pipefail; cd '$REPO_ROOT'; source '$VENV_PATH/bin/activate'; python -u scripts/encoder_decoder/multitask/train_multitask_seq2seq.py --config '$STAGEB_CONFIG' --execute"
)"

J_E="$(
  sbatch --parsable \
    --dependency=afterok:"$J_B" \
    --job-name=mt270_eval_smoke \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$EVAL_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="set -euxo pipefail; cd '$REPO_ROOT'; source '$VENV_PATH/bin/activate'; MODEL_ID='$REPO_ROOT/$STAGEB_OUT' OUTPUT_DIR='$REPO_ROOT/$EVAL_OUT' bash scripts/encoder_decoder/multitask/run_eval_multitask_270m.sh"
)"

echo "Submitted smoke chain:"
echo "  stageA=$J_A"
echo "  patch =$J_PATCH"
echo "  stageB=$J_B"
echo "  eval  =$J_E"
echo
echo "Monitor:"
echo "  squeue -j $J_A,$J_PATCH,$J_B,$J_E"
