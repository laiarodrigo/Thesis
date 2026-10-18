# DEV 01 Pearmut assignments

DEV uses one selected passage per assignment/link and Rodrigo is the sole annotator. The 10 planned assignment IDs are `dev01-p01` through `dev01-p10`, in the selected-passage order recorded in `configs/dev_01_annotation_campaign.json`.

Candidate generation is not available in this local environment, so no campaign JSON or runnable Pearmut link is claimed yet. After the three generated candidate files have been synchronized, run:

```bash
python cmu_evaluator_project/scripts/build_candidate_dataset.py \
  --config cmu_evaluator_project/configs/dev_01.json

python cmu_evaluator_project/scripts/build_annotation_campaigns.py \
  --config cmu_evaluator_project/configs/dev_01_annotation_campaign.json
```

This produces one JSON campaign in `campaigns/dev01-pNN.json` and one researcher-only system mapping in `private/dev01-pNN.json` for each passage, plus `private/assignment_manifest_PRIVATE.json`.

Import and launch the generated campaigns from the Pearmut working directory:

```bash
for campaign in cmu_evaluator_project/data/pearmut/dev_01/campaigns/*.json; do
  pearmut add "$campaign"
done
pearmut run
```

Pearmut prints the dashboard URL; use the dashboard to obtain the 10 annotator URLs. The repository cannot state the final URL pattern before the server is running.

System identities stay hidden in campaign files. Treat `private/` and the assignment manifest as researcher-only.
