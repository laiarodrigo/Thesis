# Dissertation artifact release

This repository tracks the code, configurations, LaTeX projects, curated
result tables, and release manifests. Large datasets, databases, evaluation
runs, and model weights remain outside Git.

## Current manifests

- `results/dissertation/model_manifest.csv` lists exactly 34 models reported
  in the dissertation: 21 T5Gemma 2 4B LoRA adapters and 13 fully fine-tuned
  T5Gemma 2 270M models.
- `data/manifests/dissertation_data_manifest.csv` lists the 26 JSONL files
  referenced by the 46 final training configurations.
- `results/dissertation/final_config_manifest.txt` lists those 46
  configurations, including the Stage A parents required to reproduce later
  stages.

The model manifest records the weight type, expected weight filename, known
byte size, configuration path, and exact dissertation table row or rows. The
known weights total 34,837,868,824 bytes, approximately 32.45 GiB.

The SHA-256 fields are intentionally empty until they are calculated from the
actual Slurm artifacts. A hash must never be reconstructed from a filename or
an old file listing.

## Complete hashes and sizes on Slurm

First make the current repository revision available on Slurm. From the
repository root there, run:

```bash
python scripts/release/complete_artifact_manifests.py --write --strict
```

The command streams every file through SHA-256, updates both CSV manifests in
place, and exits with a nonzero status if any of the 34 model weights, 34
configuration files, or 26 data files is missing. Review the result before
committing it:

```bash
git diff -- results/dissertation/model_manifest.csv \
  data/manifests/dissertation_data_manifest.csv
```

The script can be run without `--write` for a read-only availability check.

## OneDrive backup boundary

The repository backup made with rclone is independent of Git and
`.gitignore`. Files under `data/` are copied unless the rclone command itself
excludes them. The comprehensive job discussed for this repository excludes
environments and the general `outputs/` tree, then copies the selected
reported model directories separately.

Do not delete databases, raw corpora, or duplicated large data directories
until all of the following are true:

1. the current rclone job has completed successfully;
2. `rclone size` reports the expected destination size;
3. `rclone check` passes with the same filters used by the copy job;
4. the 34 model paths and the 26 data paths in these manifests exist remotely;
5. at least one representative database, JSONL file, and model weight has been
   restored to a temporary directory and opened successfully.

The three removed files were narrow exceptions: an API error payload, a small
intermediate diversity report, and the obsolete
`data/pt_variant_prompts_500.csv`. They were generated or legacy artifacts,
not the final GPT-Wikipedia dataset.

## Public release sequence

GitHub should remain the source repository and contain no model weights or
large databases. After the repository and OneDrive backup are stable:

1. publish the reported model artifacts and model cards on Hugging Face;
2. include the applicable Gemma terms and the required Gemma notice with the
   derivative model release;
3. archive the distributable training and evaluation data on Zenodo;
4. add the Zenodo DOI and Hugging Face identifiers to the manifest
   `external_uri` fields and to both LaTeX projects;
5. create a tagged repository release matching the archived artifacts.

Before publishing OpenSubtitles-derived material, record the exact retrieval
date and confirm that the planned redistribution complies with the source
license. The canonical database archives and final GPT-Wikipedia source table
also need to be selected before the Zenodo upload; this is deliberately left
until after backup verification.
