# Schema And Dataflow Notes

## Resolved Files
- Reference config: `configs/encoder_decoder/t5gemma2/comparison_staged/translation_fullft_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls_from_stageA_opensubs_plus_ptbrvarid.yaml`
- Tokenizer source: `/cfs/home/u036584/repos/Thesis/outputs/encoder_decoder/compare_staged/t5gemma2_270m_fullft_stageA_opensubs_plus_ptbrvarid_label_first_with_cls`
- Project DB: `data/duckdb/subs_project.duckdb`
- Source DB: `data/duckdb/subs.duckdb`
- PtBrVId DB: `data/duckdb/subs_ptbr_filtered.duckdb`
- PtBrVId sampled-rows CSV: `data/ptbrvarid/translated_stageb_pairs_r48_500_t50/sampled_rows_resampled.csv`
- PtBrVId Stage A excluded domains: `social_media, web`
- PtBrVId translated-stage dir: `data/encoder_decoder/t5gemma2/ptbrvarid_translated_stageB_r48_500_t50`

## Tokenizer Resolution
The tokenizer is resolved from the reference config by following model.base_model until a non-local model id or a directory with tokenizer files is found.
- configs/encoder_decoder/t5gemma2/comparison_staged/translation_fullft_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls_from_stageA_opensubs_plus_ptbrvarid.yaml :: outputs/encoder_decoder/compare_staged/t5gemma2_270m_fullft_stageA_opensubs_plus_ptbrvarid_label_first_with_cls

## Split Logic
- OpenSubs/FRMT/Gold are assembled in `scripts/project/build_project_db.py`. OpenSubs enters the unified train_data view as train-only, FRMT dev is split into train/valid, and Gold is test-only.
- Stage A OpenSubs-only exports come from `run_export_stageA_opensubs_frmt.sh` with `--dataset-include OpenSubs`; that means the exported valid file is empty because train_data has no OpenSubs valid rows. The thesis statistics use `stageA_opensubs_only_clean` as the clean source name so old `stageA_opensubs_only` rebuilds cannot leak PtBrVId rows into OpenSubs-only counts.
- Stage A OpenSubs+FRMT translation valid top-up is implemented in `scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_translation_stageA_opensubs_frmt.py` (VALID_MIN_ROWS lines: [13]; classification top-up lines: [14]).
- Stage A label-first valid top-up is implemented in `scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_translation_stageA_opensubs_frmt_label_first.py` (VALID_MIN_ROWS lines: [14]).
- For thesis-facing stage summaries, the script also emits an effective-train assumption profile for Stage A OpenSubs+FRMT runs: train exposure is treated as OpenSubs-only because the OpenSubs volume dwarfs FRMT, while valid remains the real built split.
- Stage B GPT+FRMT uses `scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_translation_gpt_wiki_frmt_mix.py` for the filtered mix, `scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_stageB_gpt_wiki_frmt_translation_plus_cls.py` for translation+classification mixing (VALID_MIN_ROWS lines: [14]), and `scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_stageB_gpt_wiki_frmt_label_first_with_cls.py` for the label-first+cls builder (VALID_MIN_ROWS lines: [14]).
- The older non-staged GPT+FRMT translation family is represented separately via `scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_translation_gpt_frmt.sh`. That exporter takes a caller-specified source table and does not apply the Stage B FRMT filter or any PtBrVId augmentation.
- The trainer consumes already-built JSONL files via `scripts/slurm/train_t5gemma2_4b_translation_compare_staged.sbatch` and does not move rows between train and valid during training itself.

## PtBrVId Loading
- Stage A PtBrVId augmentation is loaded directly from DuckDB in `scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_stageA_opensubs_ptbrvarid_cls.py`. The helper resolves `subs_ptbr_filtered.duckdb` first and falls back to `subs_filtered_final.duckdb` when needed.
- Stage A PtBrVId thesis counts use the unique leftover count from DuckDB plus the sampled-rows exclusion CSV. They also apply the Stage A domain exclusion set and do not use the legacy `stageA_opensubs_only/classification_train.jsonl` file, because that file may already contain previously added PtBrVId rows.
- Using PtBrVId DB: /cfs/home/u036584/repos/Thesis/data/duckdb/subs_ptbr_filtered.duckdb

## Golden Collection
- Golden Collection is treated as test-only. Training Slurm wrappers explicitly reject configs that reference Golden Collection for training.

## Stage B Optional PtBrVId
- `scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageB_gpt_wikipedia_plus_frmt.sh` supports optional PtBrVId translation rows via `PTBRVID_DIR`, and the current clean PtBrVId configs point to dedicated `plus_ptbrvarid` Stage B data directories.

## PtBrVId Schema
- ptbrvarid_metrics columns in data/duckdb/subs_ptbr_filtered.duckdb: dataset, domain, split, raw, after_nonempty, after_jusText, after_author, after_clean, after_dedup, after_filters, after_IQR, drop_empty, drop_jusText, drop_author, drop_clean, drop_dedup, drop_filters, drop_IQR, IQR_lo, IQR_hi
- ptbrvarid_jt_dropped_examples columns in data/duckdb/subs_ptbr_filtered.duckdb: id_md5, domain, split, label, reason, raw_len, raw_preview

## Config/Path Anomalies
- configs/encoder_decoder/t5gemma2/comparison_staged/classification_head_fullft_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/classification_train.jsonl.
- configs/encoder_decoder/t5gemma2/comparison_staged/translation_fullft_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/translation_train.jsonl.
- configs/encoder_decoder/t5gemma2/comparison_staged/translation_fullft_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls.yaml is named '*with_cls*' but points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_noequal/train.jsonl, which is the translation-only label-first path.
- configs/encoder_decoder/t5gemma2_4b/comparison_staged/classification_head_r24_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/classification_train.jsonl.
- configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r16_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/translation_train.jsonl.
- configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/translation_train.jsonl.
- configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/translation_train.jsonl.
- configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls.yaml is named '*with_cls*' but points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_noequal/train.jsonl, which is the translation-only label-first path.
