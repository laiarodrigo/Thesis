# Dissertation results

This directory contains machine-readable copies of the result tables reported
in the dissertation. The CSV files are small, reviewable publication artifacts;
large prediction JSONL files and model checkpoints are deliberately excluded.

Files:

- `protocol_selection_270m.csv`: directional protocol-selection results from
  Table `tab:protocol-ablation-270m`. BLEU and TER retain the precision of the
  recovered evaluation summaries.
- `main_4b_classification.csv`: 4B identification results from Table
  `tab:r48-classification-breakdown`.
- `main_4b_rewriting.csv`: directional 4B rewriting results from Table
  `tab:r48-translation-breakdown`.
- `stage_c_270m_ablation.csv`: isolated 270M reward diagnostics from Tables
  `tab:appendix-rl-d` and `tab:appendix-rl-e`.
- `no_equal_ablation.csv`: secondary no-equal diagnostic from Table
  `tab:no-equal-ablation`.
- `golden_copy_comparison.csv`: sentence-level copy-comparison rates from Table
  `tab:golden-copy-comparison-rates`.
- `external_models.csv`: external-model comparison from Table
  `tab:appendix-external-model-comparison`.
- `final_config_manifest.txt`: the 46 configuration files required by the 34
  reported dissertation models and their Stage A and Stage B lineage.

The dissertation source remains the authoritative presentation of these
results. Evaluation summary JSON files and checksums should be added after the
canonical artifacts are restored from the Slurm or OneDrive archive.
