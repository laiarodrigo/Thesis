# Benchmark data model

The evaluator data keeps four layers separate.

1. **Source** — TED `document_id`, `passage_id`, ordered `sentence_id`, exact
   BP text, and passage context.
2. **Candidate** — stable `candidate_id`, real `system_id`, candidate text,
   and generation provenance. Presentation labels such as `System-A` are not
   permanent IDs.
3. **Human judgment** — one immutable normalized record per individual
   annotator/candidate/submission, including score, error spans, severity,
   comments, timing evidence, assignment IDs, and raw-export provenance.
4. **Optional derived consensus or adjudication** — aggregates or adjudicated
   decisions, only if a later policy defines them. These records reference the
   contributing judgments and never replace them.

The current DEV evaluator join is `dev_01_evaluator.json`: 10 passages, 97
source sentences, and 388 final candidate judgments. Formal gold consists of
error spans with Minor/Major severity and the 0–100 score. English Pearmut
comments are preserved as non-gold development notes. The lossless normalized
artifact retains every submission; where an item was resubmitted, the
canonical join selects the latest submission and records the superseded
submission IDs.

`dev_01_v0_input.jsonl` is the system-blind inference projection. It contains
opaque evaluation IDs and only the Portuguese passage context, clearly marked
current BP sentence, and one candidate. It contains no candidate IDs, system
metadata, human labels, human comments, or references.

Every passage and sentence from the same TED document must remain in one
partition during any later partitioning to prevent document-level leakage.

The benchmark builder must consume canonical prepared data and normalized
annotations. It must not depend on `logs/slurm/`.
