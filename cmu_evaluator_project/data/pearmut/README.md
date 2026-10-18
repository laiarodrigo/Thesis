# Pearmut campaign exports

This directory contains presentation artifacts for the completed pilot.
Canonical source/candidate data lives in `../prepared/pilot_01/`; human export
JSONL belongs in `../annotations/raw/` after backup.

- `ted2020_pilot_01_cesa_template.json` is the historical non-runnable
  pre-generation template.
- `ted2020_pilot_01_cesa.json` is the original whole-passage campaign export.
- `ted2020_pilot_01_cesa_resumable.json` is the completed pilot layout with one
  judged sentence per submitted page and full source-passage context.
- `ted2020_pilot_01_system_mapping_PRIVATE.json` is researcher-only.
- `test/` contains smoke-test material.

Pearmut 1.1.5 task-based `data` can contain one or many ordered documents.
The assignment generator supports both legacy one-assignment campaigns and
shared campaigns with explicit ordered `info.users`, one task per user, and
one or several passage documents per link. It creates an independently seeded
private system mapping for each assignment, hides model names, and disables
Pearmut's unseeded shuffle.

DEV assignment instructions are in `dev_01/README.md`. Pool 1 shared campaign
and private mapping intent is in `../../configs/pool_01_collection.json`;
later pools may derive from that schema without modifying live campaigns.
Coverage (two distinct annotators per passage) and one-passage TEST tasks are
fixed. Do not treat `System-A`-style labels as permanent identities;
normalization resolves them through the private assignment mapping.

`external_smoke_01/` is the isolated disposable public-web check. Its
`SMOKE-FRIEND-01` identity, runtime, manifest, and outputs never enter Pool 1
or gold data. The actual service is Railway-hosted and volume-backed; see
`../../deployment/railway/README.md`. The localhost/Quick-Tunnel helpers in
the smoke README are explicitly debug-only.
