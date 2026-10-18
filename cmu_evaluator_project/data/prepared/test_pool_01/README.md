# TED2020 TEST Pool 1

This namespace contains the four selected Strong TEST passages for provisional external-annotator Pool 1 collection. It is not DEV and does not freeze the full final TEST benchmark.

```text
test_pool_01/
├── source/       4 passages / 40 exact TED XML sentence units (ready)
├── candidates/   one output per generated system (pending cluster jobs)
└── combined/     canonical 160-record dataset (blocked on generated outputs)
```

The copy baseline is produced exactly from `source_bp` by the candidate builder. The three generated systems use the frozen pilot protocol and generate one sentence at a time.

## Exact cluster generation commands

First synchronize the current generation inputs from the local repository. This
helper deliberately excludes all Pearmut campaign data and annotations:

```bash
bash cmu_evaluator_project/scripts/slurm/sync_cmu_pilot_to_slurm.sh
```

Then, from `/cfs/home/u036584/repos/Thesis` on
`u036584@slurm.hlt.inesc-id.pt`:

```bash
sbatch --time=04:00:00 \
  --export=ALL,SOURCE_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/test_pool_01/source/source.json,OUTPUT_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/test_pool_01/candidates/B4-D-G-PTBR.json \
  cmu_evaluator_project/scripts/slurm/run_B4-D-G-PTBR.sbatch

sbatch --time=04:00:00 \
  --export=ALL,SOURCE_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/test_pool_01/source/source.json,OUTPUT_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/test_pool_01/candidates/B4-D-GF-PTBR.json \
  cmu_evaluator_project/scripts/slurm/run_B4-D-GF-PTBR.sbatch

sbatch --time=04:00:00 \
  --export=ALL,SOURCE_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/test_pool_01/source/source.json,OUTPUT_PATH=/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/test_pool_01/candidates/gemma4_E4B_it.json \
  cmu_evaluator_project/scripts/slurm/run_gemma4_E4B_it.sbatch
```

The first two jobs enforce the final Stage B decoder-controlled PTBR adapters. The Gemma job pins `google/gemma-4-E4B-it` to the same revision and deterministic conservative prompt/settings as the pilot. All runners refuse to overwrite existing output.

After all three jobs finish, retrieve their exact JSON outputs to the local
repository:

```bash
rsync -avP u036584@slurm.hlt.inesc-id.pt:/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/test_pool_01/candidates/B4-D-G-PTBR.json cmu_evaluator_project/data/prepared/test_pool_01/candidates/
rsync -avP u036584@slurm.hlt.inesc-id.pt:/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/test_pool_01/candidates/B4-D-GF-PTBR.json cmu_evaluator_project/data/prepared/test_pool_01/candidates/
rsync -avP u036584@slurm.hlt.inesc-id.pt:/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/test_pool_01/candidates/gemma4_E4B_it.json cmu_evaluator_project/data/prepared/test_pool_01/candidates/
```

After all three JSON outputs are present locally:

```bash
python cmu_evaluator_project/scripts/build_candidate_dataset.py \
  --config cmu_evaluator_project/configs/test_pool_01.json

python cmu_evaluator_project/scripts/build_annotation_campaigns.py \
  --config cmu_evaluator_project/configs/pool_01_collection.json \
  --overwrite
```

The second command deterministically regenerates the already-built pilot artifact and private mappings while adding the real TEST campaign. It must be run before either Pool 1 campaign is added to a live Pearmut root; `--overwrite` here applies only to reproducible repository artifacts, never to a live Pearmut campaign.
