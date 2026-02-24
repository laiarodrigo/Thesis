# SLURM launchers for T5Gemma2 4B

These scripts run the existing large-model launchers:
- `scripts/encoder_decoder/t5gemma2_4b/run_step4a_translation_boilerplate.sh --execute`
- `scripts/encoder_decoder/t5gemma2_4b/run_step4b_classification_head_boilerplate.sh --execute`

## Files

- `scripts/slurm/train_t5gemma2_4b_translation.sbatch`
- `scripts/slurm/train_t5gemma2_4b_classification.sbatch`
- `scripts/slurm/submit_t5gemma2_4b_pipeline.sh` (submits translation first, classification with `afterok` dependency)

## Recommended GPU choice

For `google/t5gemma-2-4b-4b`, safest request is a high-VRAM card:
- Prefer `gpu:RTX_PRO_6000_96GB:1` when available.
- If queue is too long, try `gpu:1` (any GPU) and do a smoke run first.

Reason: translation is LoRA (lighter), but classification-head training still keeps a large seq2seq backbone in memory and can exceed 24 GB depending on runtime overhead.

## Submit examples

Single translation job:

```bash
sbatch --partition=hlt_phd --gres=gpu:RTX_PRO_6000_96GB:1 scripts/slurm/train_t5gemma2_4b_translation.sbatch
```

Single classification job:

```bash
sbatch --partition=hlt_phd --gres=gpu:RTX_PRO_6000_96GB:1 scripts/slurm/train_t5gemma2_4b_classification.sbatch
```

Pipeline (translation then classification):

```bash
bash scripts/slurm/submit_t5gemma2_4b_pipeline.sh
```

With overrides:

```bash
PARTITION=hlt_msc GPU_GRES=gpu:RTX_PRO_6000_96GB:1 bash scripts/slurm/submit_t5gemma2_4b_pipeline.sh
```

Run only a quick classification smoke test:

```bash
SMOKE_RUN=1 PARTITION=hlt_msc GPU_GRES=gpu:1 bash scripts/slurm/submit_t5gemma2_4b_pipeline.sh
```

## Monitoring

- `squeue --me`
- `sacct -j <job_id>`
- `tail -f slurm-<job_name>-<job_id>.out`
- `scancel <job_id>`

