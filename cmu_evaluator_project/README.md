# CMU evaluator project

This directory isolates the BP→EP human-evaluation work from the thesis
experiments.

```text
configs/                    campaign intent and reproducible settings
deployment/                 always-on Railway deployment/runbook
data/prepared/              frozen sources and generated candidates
data/pearmut/               Pearmut presentation exports/private mappings
data/annotations/raw/       immutable human annotation exports
data/annotations/normalized merge-ready individual judgments
data/annotations/adjudicated optional later adjudication
data/benchmark/             later benchmark packaging and schema
comparisons/v0/             official V0 qualitative and quantitative reports
diagnostics/                evaluator-only synthetic diagnostics
logs/slurm/                 computational stdout/stderr only
notes/                      research plans and implementation findings
scripts/                    reusable preparation/campaign/analysis tools
tests/                      synthetic and frozen-pilot validation
runtime/                    ignored mutable Pearmut roots and backups
```

The benchmark pipeline depends on prepared data, assignment metadata, private
anonymization mappings, and raw annotations. It does **not** depend on Slurm
logs.

## Current mechanics

- stable candidate IDs and assignment IDs;
- deterministic balanced two-distinct-annotator TEST allocation;
- one passage per Pearmut assignment/link;
- independently seeded assignment-level anonymization;
- campaign/manifest generation;
- immutable raw storage and lossless normalization;
- duplicate detection and descriptive disagreement reporting;
- layered benchmark schema and document-safe split rule.

## Current campaign status

- DEV annotation and canonical normalization are complete: 10 passages, 97 source sentences, and 388 candidate judgments.
- The official zero-shot V0 evaluation is complete: 388 predictions, of which 365 passed strict structural and substring validation.
- The V0 quantitative report and 10 passage-level qualitative reports are under `comparisons/v0/`.
- The separate 25-instance source-sensitivity diagnostic and its report are under `diagnostics/v0_source_sensitivity/`.
- Evaluator development now proceeds to V1 using DEV only. Held-out evaluation remains outside the development loop.

The chronological evaluator record, frozen V0 prompt and protocol, results links, findings, and remaining development ladder are maintained in [`notes/2026-10-15_evaluator_development_plan.md`](notes/2026-10-15_evaluator_development_plan.md).

The disposable `external_smoke_01` campaign exercises the protected public
browser path before Pool 1 deployment. The real service is hosted on Railway,
uses a mounted persistent volume, and remains available with Rodrigo's laptop
off. It uses `SMOKE-FRIEND-01`, a separate campaign namespace, and an
allow-listing proxy; it is never gold eligible. Railway deployment, retrieval,
backup, restore, and the mandatory restart check are documented in
`deployment/railway/README.md`. The older localhost/Quick-Tunnel scripts are
debug-only. Pool 1's public identity remains only `A01`–`A08`; its real-name
roster is researcher-private, Git-ignored, and excluded from the container.

DEV intentionally contains 97 source units across 10 passages. With four
candidate systems, its intended total is exactly 388 candidate judgments.

`configs/annotation_campaign_TEMPLATE.json` is deliberately non-runnable until
the TEST passage list and annotator IDs are supplied. It already fixes coverage
at two distinct annotators per passage and balanced pair-varying allocation.
Generate manifests, private mappings, and campaigns with:

```bash
python cmu_evaluator_project/scripts/build_annotation_campaigns.py \
  --config <FILLED_CAMPAIGN_CONFIG>
```

Normalize backed-up raw exports without collapsing annotators:

```bash
python cmu_evaluator_project/scripts/normalize_pearmut_annotations.py \
  --candidate-dataset <COMBINED_CANDIDATES_JSON> \
  --assignment-manifest <PRIVATE_ASSIGNMENT_MANIFEST> \
  --output cmu_evaluator_project/data/annotations/normalized/<NAME>.json
```

Then produce descriptive—not consensus—comparisons with:

```bash
python cmu_evaluator_project/scripts/analyze_annotation_disagreement.py \
  --input <NORMALIZED_JSON> \
  --output <DESCRIPTIVE_REPORT_JSON> \
  --score-threshold 20
```
