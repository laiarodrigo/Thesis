#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
PARTITION="${PARTITION:-hlt_msc}"
GPU_GRES="${GPU_GRES:-gpu:RTX_3090_24GB:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-30G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
PRECISION="${PRECISION:-fp16}"

case "$PRECISION" in
  fp16)
    OLD_CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_only_smoke_fp16.yaml"
    NEW_CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_frmt_smoke_fp16.yaml"
    ;;
  bf16)
    OLD_CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_only_smoke_bf16.yaml"
    NEW_CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_frmt_smoke_bf16.yaml"
    ;;
  *)
    echo "ERROR: PRECISION must be fp16 or bf16" >&2
    exit 2
    ;;
esac

for req in \
  "$OLD_CONFIG" \
  "$NEW_CONFIG" \
  "scripts/slurm/train_t5gemma2_4b_translation_compare_staged.sbatch"
do
  [[ -f "$req" ]] || { echo "ERROR: missing file $req" >&2; exit 2; }
done

J_OLD="$(
  CONFIG_PATH="$OLD_CONFIG" \
  VENV_PATH="$VENV_PATH" \
  sbatch --parsable \
    --job-name="t5g4b_old_${PRECISION}" \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME_LIMIT" \
    scripts/slurm/train_t5gemma2_4b_translation_compare_staged.sbatch
)"

J_NEW="$(
  CONFIG_PATH="$NEW_CONFIG" \
  VENV_PATH="$VENV_PATH" \
  sbatch --parsable \
    --job-name="t5g4b_new_${PRECISION}" \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME_LIMIT" \
    scripts/slurm/train_t5gemma2_4b_translation_compare_staged.sbatch
)"

cat <<EOF
Submitted Stage A smoke comparison.

Precision:
  $PRECISION

GPU request:
  $GPU_GRES

Configs:
  old/opensubs_only: $OLD_CONFIG
  new/opensubs_plus_frmt: $NEW_CONFIG

Jobs:
  old: $J_OLD
  new: $J_NEW
EOF
