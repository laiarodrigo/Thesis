# TED2020 DEV 01 — prepared source

This directory is the canonical prepared source for the 10 selected DEV passages. Exact sentence text, IDs, review provenance, and passage-level notes are recorded in [`../../../passage_selection/ted2020_selected_candidates.md`](../../../passage_selection/ted2020_selected_candidates.md); machine-readable data is in `source.json`.

- **Passages:** 10
- **Sentence units:** 97
- **Source:** authoritative TED2020 pt-BR XML files listed per document
- **Generation unit:** one sentence at a time
- **Annotation context:** reconstructed full passage
- **TEST status:** no TEST passage is frozen

## Source preparation

Exactly 2 sentence(s) were normalized: sentences `48` and `49` of `ted2020-946_s43-s51` had only the leading `TD:` label and its immediate whitespace removed. Their raw text is retained in `raw_source_bp` and the transformation is recorded in `source_preparation`.

No other selected sentence received an ad-hoc correction, normalization, or rewrite. The raw XML and historical manual-review file were not modified.
