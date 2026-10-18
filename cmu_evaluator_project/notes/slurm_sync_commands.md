# Slurm log and candidate-output synchronization

Repository evidence identifies:

- SSH target: `u036584@slurm.hlt.inesc-id.pt`
- remote repository: `/cfs/home/u036584/repos/Thesis`
- job stdout/stderr: `cmu_evaluator_project/logs/slurm/%x-%j.out` and `.err`
- original pilot outputs: flat files under
  `cmu_evaluator_project/data/prepared/` on the cluster

The completed jobs wrote model outputs to:

```text
/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/ted2020_pilot_01_B4-D-G-PTBR.json
/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/ted2020_pilot_01_B4-D-GF-PTBR.json
/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/ted2020_pilot_01_gemma4_E4B_it.json
```

Run the following **from the local repository root**. First list without
copying:

```bash
ssh u036584@slurm.hlt.inesc-id.pt \
  "find /cfs/home/u036584/repos/Thesis/cmu_evaluator_project/logs/slurm -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort"

ssh u036584@slurm.hlt.inesc-id.pt \
  "find /cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared -maxdepth 1 -type f -name 'ted2020_pilot_01_*.json' -printf '%f\t%s bytes\n' | sort"
```

Dry-run the log sync:

```bash
rsync -avPn \
  u036584@slurm.hlt.inesc-id.pt:/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/logs/slurm/ \
  cmu_evaluator_project/logs/slurm/
```

Remove `n` only after checking the list:

```bash
rsync -avP \
  u036584@slurm.hlt.inesc-id.pt:/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/logs/slurm/ \
  cmu_evaluator_project/logs/slurm/
```

Dry-run each model-output copy into the reorganized local locations by adding
`-n` to these commands; then run them as shown:

```bash
rsync -avP \
  u036584@slurm.hlt.inesc-id.pt:/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/ted2020_pilot_01_B4-D-G-PTBR.json \
  cmu_evaluator_project/data/prepared/pilot_01/candidates/B4-D-G-PTBR.json

rsync -avP \
  u036584@slurm.hlt.inesc-id.pt:/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/ted2020_pilot_01_B4-D-GF-PTBR.json \
  cmu_evaluator_project/data/prepared/pilot_01/candidates/B4-D-GF-PTBR.json

rsync -avP \
  u036584@slurm.hlt.inesc-id.pt:/cfs/home/u036584/repos/Thesis/cmu_evaluator_project/data/prepared/ted2020_pilot_01_gemma4_E4B_it.json \
  cmu_evaluator_project/data/prepared/pilot_01/candidates/gemma4_E4B_it.json
```

These destinations already contain the completed local pilot. Use dry-run and
checksum comparison before replacing anything. The reorganized Slurm scripts
now target the structured paths for any future rerun.

If `rsync` is unavailable, the equivalent form is:

```bash
scp REMOTE_HOST:REMOTE_FILE LOCAL_FILE
```

`logs/slurm/` is only computational evidence. Human Pearmut exports belong in
`data/annotations/raw/`; benchmark construction never reads Slurm logs.
