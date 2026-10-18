# TED2020 larger-pool selection method

## Inputs

- Linguistic inventory: `cmu_evaluator_project/notes/cmu_llm_judge_data_options.md`, specifically **Stage 2 — Linguistic Coverage / Data Requirements**.
- Pilot source: `cmu_evaluator_project/data/prepared/pilot_01/source/source.json`.
- Pilot annotations: `data/outputs/ted2020_pilot_01_cesa.jsonl`, with campaign state/timing checked in `data/progress.json`.
- Corpus: all XML files in `data/external/TED2020/TED2020/raw/pt_br/`.

The JSONL contains 12 submissions for 10 page indices: indices 0 and 1 were submitted twice. Pilot observations used the **last JSONL record for each `item_i`**, never an average. Only anonymous `System-A`–`System-D` judgments were inspected; the private mapping was not read.

## What the pilot changed

The pilot supported keeping overt subject pronouns and address forms weak: annotator comments often described them as preferences rather than errors. It supported more attention to clitic placement and possessive determiners, but these remain construction-dependent and therefore do not drive automatic ranking. It also exposed clearer orthographic cases (`purê/puré`, `içamos/içámos`) and showed that plausible alternatives such as `economizar/poupar` are not automatically required. These observations are supporting evidence, not linguistic ground truth.

## Lexical discovery and verification

Candidate strings came from the Stage-2 inventory, corpus-wide frequency/concordance scans, and recurring BP/EP comparison candidates. Each potentially meaningful signal was checked for the exact TED sense against a Brazilian dictionary, a Portugal-based dictionary with regional labels/equivalents, natural Brazilian usage, and clearly Portugal-context usage. A `.pt` domain hit alone was never accepted as EP evidence. Cultural/institutional references were excluded from automatic adaptation.

The audit and contradictory evidence are in `ted2020_bp_lexical_verification.md`. Only `strongly_bp_marked` items receive strong weight. `bp_preferred_ep_attested` items receive weak weight; `shared_not_useful` and `uncertain` items receive none. Ambiguous strong strings use local context guards.

## Exhaustive scan and transparent scoring

`cmu_evaluator_project/scripts/rank_ted2020_bp_passages.py` parses every `<s>` node, preserves its ID/order/text, and scores every overlapping window of lengths 8, 9, and 10. Components are stored separately in JSON:

- 7 points per distinct strong lexical type;
- 1.5 per strong hit, capped at six;
- 2 per distinct BP-preferred-but-EP-attested type and 0.5 per hit, capped at four;
- 1.5 per distinct verified orthographic signal;
- 2 per `estar + gerund` occurrence;
- 0.25 per weak contextual structural flag;
- a distribution bonus for evidence spread across multiple sentences.

A window qualifies with at least two strong types distributed over at least two sentences, or one strong type plus at least two supporting lexical/orthographic/progressive signals. Repetition alone cannot qualify a passage strongly.

## Diversity adjustment

Raw candidates are sorted by transparent score. Greedy selection adds a novelty bonus for signal types not yet well represented and permits only one window per TED document. This removes overlapping near-duplicates while retaining 50 strong candidates from different talks. Both the raw rank and diversity-selection utility are preserved in JSON.

## Limitations

- Regex/context guards are high-precision heuristics, not full word-sense disambiguation; every candidate still needs human review.
- TED segmentation can contain stage directions, sentence fragments, and translation artifacts; these are flagged where automatically detectable.
- A strong regional marker does not imply one unique correct EP rewrite, and culture-specific content must preserve its referent.
- The shortlist is a source candidate pool only. No PT-PT TED target was opened or used, no model was run, and no candidate is final.
