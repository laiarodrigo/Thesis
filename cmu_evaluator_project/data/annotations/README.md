# Human annotation storage

```text
annotations/
├── raw/          exact Pearmut exports; immutable source of truth
├── normalized/   canonical, merge-ready individual judgments
└── adjudicated/  optional outputs only if an adjudication policy is adopted
```

Never edit or overwrite a file in `raw/`. Copy each Pearmut JSONL export here
before analysis and retain its original bytes. Computational logs do not
belong here.

`normalize_pearmut_annotations.py` combines raw exports with the private
assignment mappings and canonical candidate data. It produces one record per
candidate, annotator, and submission, with IDs, score, spans, severity,
comments, timing evidence, and provenance. It never averages or overwrites
individual judgments. Exact repeated submissions fail loudly. Non-identical
repeat submissions also fail by default and can only be preserved using the
explicit `--repeat-policy preserve` option after review.

Shared task-based campaign exports are normalized by exact `(campaign_id,
user_id)` assignment. A raw row with a missing or unknown user ID fails rather
than being guessed. Pool manifests retain `pending`, `accepted`, or `rejected`
screening status separately from immutable raw rows. Normalized TEST records
remain present for every status and expose `eligible_for_gold`; only
`accepted` records are eligible.

After a manual screening update, report accepted coverage with:

```bash
python cmu_evaluator_project/scripts/summarize_screening_coverage.py \
  --assignment-manifest cmu_evaluator_project/data/pearmut/pool_01/private/pool_01_manifest_PRIVATE.json
```

The report counts distinct accepted users per unchanged TEST passage and
states whether zero, one, or two replacement assignments are needed.

The completed pilot export has not been moved or rewritten. It can later be
copied into `raw/` and described in an assignment manifest for import through
the same normalizer. Because that export contains repeated submissions, the
normalizer will require an explicit repeat policy; the earlier pilot-specific
choice to inspect the latest submission is not a default benchmark rule.
