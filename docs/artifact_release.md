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

The SHA-256 fields are calculated from the actual Slurm artifacts. A hash must
never be reconstructed from a filename or an old file listing.

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

## Publish the reported models on Hugging Face

The public release contains only inference artifacts. It excludes intermediate
checkpoint directories, optimizer and scheduler state, `training_args.bin`,
candidate-debug files, and training-metric logs. Audit the 34 model folders
before creating the public repository:

```bash
python scripts/release/publish_hf_models.py --audit
```

Upload a two-model pilot consisting of one LoRA adapter and one fully fine-tuned
model:

```bash
python scripts/release/publish_hf_models.py \
  --upload \
  --only B4-D-G-PTBR \
  --only F270-D-G-PTBR
```

After verifying that both pilot folders load correctly, publish the full set.
The uploader records every verified model-folder address in the manifest and
can be run again safely after an interrupted transfer.

```bash
python scripts/release/publish_hf_models.py --upload
```

The complete release is indexed at
`laiarodrigo/portuguese-variant-t5gemma2`, with one folder for each dissertation
model ID.

## Publish the training and evaluation data release

The data release is separate from the model repository. It covers
both training and evaluation resources. It hosts the task-formatted
GPT-Wikipedia and FRMT-derived artifacts, while representing OpenSubtitles by
reconstruction metadata and the Golden Collection by an upstream reference.

Audit the exact local files on Slurm:

```bash
python scripts/release/publish_hf_data.py --audit
```

The audit creates
`data/manifests/dissertation_dataset_release.csv`, adds the available FRMT and
Golden Collection split hashes, and reports which artifacts will be hosted,
referenced, or reconstructed. It does not rerun training or evaluation.

After reviewing that inventory, upload the hosted subset and dataset card to a
private Hugging Face dataset repository:

```bash
python scripts/release/publish_hf_data.py --upload
```

The uploader is resumable at the file level and verifies remote paths, sizes,
and available LFS SHA-256 values. A successful run ends with
`HF_DATA_PRIVATE_UPLOAD_COMPLETED`. The target repository is
`laiarodrigo/portuguese-variant-data`.

The reviewed public release retains a compound `license: other` declaration.
Its row-level manifest records the source, applicable license information, and
release mode for every entry. OpenSubtitles-derived text and Golden Collection
text are not republished. Refresh the card and manifests, verify the hosted
files, and change the repository visibility in one guarded operation:

```bash
python scripts/release/publish_hf_data.py --upload --make-public
```

A successful publication ends with `HF_DATA_PUBLICATION_COMPLETED`.

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
3. publish the curated data subset and provenance manifest on Hugging Face;
4. add the Hugging Face identifiers to the manifests and both LaTeX projects;
5. optionally archive a fixed release on Zenodo and add its DOI;
6. create a tagged repository release matching the public artifacts.

Before publishing OpenSubtitles-derived material, record the exact retrieval
date and confirm that the planned redistribution complies with the source
license. The canonical database archives and final GPT-Wikipedia source table
also need to be selected before the Zenodo upload; this is deliberately left
until after backup verification.
