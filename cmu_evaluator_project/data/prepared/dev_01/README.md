# TED2020 DEV 01

This prepared namespace mirrors `pilot_01/`:

```text
dev_01/
├── source/       selected source and provenance (ready)
├── candidates/   one output per generated system (pending cluster jobs)
└── combined/     merged four-system dataset and review (pending generation)
```

The copy baseline is created by the candidate builder; it is not an inference job. The three generated systems retain the completed pilot protocol and canonical IDs `<passage_id>::<sentence_id>::<system_id>`.

## Cluster generation

From `/cfs/home/u036584/repos/Thesis`, run:

```bash
sbatch --time=04:00:00 \
  --export=ALL,SOURCE_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/dev_01/source/source.json,OUTPUT_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/dev_01/candidates/B4-D-G-PTBR.json \
  cmu_evaluator_project/scripts/slurm/run_B4-D-G-PTBR.sbatch

sbatch --time=04:00:00 \
  --export=ALL,SOURCE_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/dev_01/source/source.json,OUTPUT_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/dev_01/candidates/B4-D-GF-PTBR.json \
  cmu_evaluator_project/scripts/slurm/run_B4-D-GF-PTBR.sbatch

sbatch --time=04:00:00 \
  --export=ALL,SOURCE_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/dev_01/source/source.json,OUTPUT_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/dev_01/candidates/gemma4_E4B_it.json \
  cmu_evaluator_project/scripts/slurm/run_gemma4_E4B_it.sbatch
```

The thesis jobs require the canonical final Stage B adapters and configs already checked by the wrappers. The Gemma job pins the same model revision as the pilot. All runners process the canonical source one sentence at a time and refuse to overwrite outputs.

After all three JSON files exist:

```bash
python cmu_evaluator_project/scripts/build_candidate_dataset.py \
  --config cmu_evaluator_project/configs/dev_01.json
```

Then follow `../../pearmut/dev_01/README.md` to build the 10 assignment-specific campaigns.
