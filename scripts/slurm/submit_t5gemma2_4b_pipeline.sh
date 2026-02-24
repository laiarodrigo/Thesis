#!/usr/bin/env bash
set -euo pipefail

PARTITION="${PARTITION:-hlt_phd}"
GPU_GRES="${GPU_GRES:-gpu:RTX_PRO_6000_96GB:1}"

echo "Submitting translation (partition=$PARTITION, gres=$GPU_GRES)..."
trans_submit="$(
  sbatch \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    scripts/slurm/train_t5gemma2_4b_translation.sbatch
)"
trans_job_id="$(awk '{print $4}' <<<"$trans_submit")"
echo "  $trans_submit"

if [[ "${SMOKE_RUN:-0}" == "1" ]]; then
  echo "Submitting classification (smoke run) after translation succeeds..."
  cls_submit="$(
    sbatch \
      --partition="$PARTITION" \
      --gres="$GPU_GRES" \
      --dependency="afterok:${trans_job_id}" \
      --export=ALL,SMOKE_RUN=1 \
      scripts/slurm/train_t5gemma2_4b_classification.sbatch
  )"
else
  echo "Submitting classification after translation succeeds..."
  cls_submit="$(
    sbatch \
      --partition="$PARTITION" \
      --gres="$GPU_GRES" \
      --dependency="afterok:${trans_job_id}" \
      scripts/slurm/train_t5gemma2_4b_classification.sbatch
  )"
fi

cls_job_id="$(awk '{print $4}' <<<"$cls_submit")"
echo "  $cls_submit"
echo "Done. translation_job_id=$trans_job_id classification_job_id=$cls_job_id"

