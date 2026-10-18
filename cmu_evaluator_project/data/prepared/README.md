# Prepared CMU evaluator data

Prepared data is canonical and independent of Pearmut. Pearmut campaign JSON
is a presentation/export format, not the source of truth.

## Directory convention

Each prepared dataset or campaign uses:

```text
<dataset_or_campaign_id>/
├── source/       frozen documents, passages, and sentences
├── candidates/   one generation-output file per real system
└── combined/     canonical merged candidates and review material
```

`pilot_01/` is the completed 10-sentence pilot. Its eight moved artifacts are
byte-identical to the former flat files; `pilot_01/checksums.sha256` records
their pre-move hashes. Absolute `source_output_file` values inside the frozen
combined JSON are historical generation provenance and were deliberately not
rewritten.

`dev_01/` contains the selected 10-passage / 97-sentence-unit DEV source.
Candidate and combined directories are reserved but remain empty until the
three cluster generation jobs finish.

`test_pool_01/` contains the four selected Strong TEST passages / 40 exact XML
sentence units for provisional external-annotator collection. Its generated
candidates are pending. The larger passage material under
`../passage_selection/` remains source-selection history, and the full final
TEST benchmark is not frozen.

## Canonical identities

Source items are aligned by `(document_id, passage_id, sentence_id)`. New
combined datasets also store one deterministic candidate ID per real system:

```text
<passage_id>::<sentence_id>::<system_id>
```

Components are URL-escaped before joining. Human-facing labels such as
`System-A` never enter this ID. The builder rejects duplicate source IDs,
duplicate system IDs, candidate-ID collisions, missing outputs, reordered
outputs, source mismatches, and empty candidates. Frozen legacy datasets that
predate the explicit `candidate_ids` field remain readable because the same ID
is derived and validated in memory.

## Reusable preparation pipeline

Campaign-specific paths and systems belong in a config such as
`../../configs/pilot_01.json`:

```text
frozen source + per-system outputs
  -> build_candidate_dataset.py
  -> canonical combined dataset
  -> assignment-specific campaign tooling
```

The copy condition is constructed from the exact source. Generated-system
files preserve both raw model output and cleaned candidate text. Generation
provenance stays in canonical data and is never exposed to annotators.

The assignment tools support multiple documents/passages, any system count,
arbitrary annotator overlap, independent deterministic blinding, legacy
one-assignment campaigns, and ordered shared task-based campaigns with one
explicit stable user per task. Pool 1 is configured in
`../../configs/pool_01_collection.json`; the generic template remains for
later collections.
