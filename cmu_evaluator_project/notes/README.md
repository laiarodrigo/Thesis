# CMU evaluator project notes

Only notes with a confidently known date use the `YYYY-MM-DD_<topic>.md` convention. Older notes have no reliable date in their contents or available Git context, so their existing filenames are retained rather than assigned invented dates.

## Process index

### Early planning

- [`cmu_llm_judge_data_options.md`](cmu_llm_judge_data_options.md) — initial task, data, and evaluator construction options.
- [`CMU_meeting_preparation_v5.md`](CMU_meeting_preparation_v5.md) — meeting preparation and early scope decisions.

### Pilot preparation

- [`cmu_annotation_pilot_source_selection.md`](cmu_annotation_pilot_source_selection.md) — early BP source-selection record.
- [`thesis_candidate_models.md`](thesis_candidate_models.md) — canonical final thesis candidate systems.
- [`external_model_prompt.md`](external_model_prompt.md) — fixed Gemma candidate prompt.
- [`slurm_sync_commands.md`](slurm_sync_commands.md) — pilot cluster synchronization record.

### Pilot annotation and guideline refinement

- [`pearmut_1.1.5_findings.md`](pearmut_1.1.5_findings.md) — Pearmut behavior and reproducibility findings.
- [`pilot_timing_plan.md`](pilot_timing_plan.md) — pilot timing interpretation and procedure.

### Source mining and review

- Source-mining methods, outputs, and the unchanged manual review record live under [`../data/passage_selection/`](../data/passage_selection/).

### Campaign decisions and current plan

- [`2026-10-09_annotation_campaign_decisions.md`](2026-10-09_annotation_campaign_decisions.md) — concise record of the Alon/Rodrigo decisions.
- [`human_annotation_campaign_plan.md`](human_annotation_campaign_plan.md) — single living DEV/TEST campaign plan.
- [`2026-10-15_evaluator_development_plan.md`](2026-10-15_evaluator_development_plan.md) — current V0→V4 evaluator-development ladder and implementation record.

### DEV preparation and TEST collection

- The selected DEV source and generation status are indexed in [`../data/prepared/dev_01/README.md`](../data/prepared/dev_01/README.md).
- The 10 planned DEV Pearmut assignments are indexed in [`../data/pearmut/dev_01/README.md`](../data/pearmut/dev_01/README.md).
- TEST remains held out and the full benchmark remains unfrozen. The selected provisional Pool 1 collection is configured in [`../configs/pool_01_collection.json`](../configs/pool_01_collection.json); generic later-pool mechanics remain documented by [`../configs/annotation_campaign_TEMPLATE.json`](../configs/annotation_campaign_TEMPLATE.json).
