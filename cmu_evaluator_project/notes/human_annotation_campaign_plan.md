# Human annotation campaign — current plan

DEV and held-out TEST remain separate. Rodrigo annotates DEV; external candidate annotators contribute provisional TEST judgments only after completing independent pilot-screening material. Rodrigo's completed pilot answers are never included in their campaign.

## Candidate collection and screening

Candidate annotators complete two links in the same collection round:

1. the existing pilot material in a pool-specific shared task-based campaign;
2. one assigned TEST passage in a separate pool-specific shared task-based campaign.

Collecting TEST before the manual pilot-screening decision saves time. Every candidate starts with private screening status `pending`. Rodrigo later changes that status to `accepted` or `rejected`; there is no automatic pass/fail threshold. All raw pilot and TEST submissions remain immutable regardless of the decision. A TEST judgment is eligible for gold only when its annotator is accepted.

Each final TEST passage requires two accepted distinct annotators. After screening, coverage is computed per passage:

- two accepted annotators: complete;
- one accepted annotator: one replacement assignment needed;
- zero accepted annotators: two replacement assignments needed.

Replacement annotators judge the same frozen passage. A rejection never causes source replacement or raw-data deletion.

## Pool 1

Pool 1 uses anonymous research IDs `A01`–`A08`. Real names live only in the Git-ignored researcher roster `data/pearmut/pool_01/private/annotator_roster_PRIVATE.json`; public campaigns, URLs, logs, and analysis data use only the research IDs.

The four selected TEST Pool 1 passages are exactly:

- `ted2020-1188_s9.2-s18`;
- `ted2020-2279_s83-s91`;
- `ted2020-1565_s7.2-s14`;
- `ted2020-2568_s127-s135`.

Each has 10 sentence units. Each is provisionally assigned to exactly two distinct Pool 1 users, while every Pool 1 user receives exactly one TEST passage. Generation and annotation remain sentence-specific; the full passage is reconstructed only as context.

Pool 1 uses two shared Pearmut campaigns:

- `ted2020_pilot_screening_pool_01_cesa`;
- `ted2020_test_strong_pool_01_cesa`.

Both campaigns use the same explicit ordered users `A01`–`A08`, one task per user, hidden model identity, and deterministic researcher-side mappings. Each person receives exactly one user-specific URL from each campaign.

TEST Pool 1 used the previous instruction wording. The running campaign was not redistributed or rebuilt after the introductory clarification was added to the current instructions.

## Later recruitment pools

Alon may recruit additional annotators. Later pools use new IDs and new shared campaigns (`pool_02`, `pool_03`, and so on); users are never appended destructively to a live Pearmut campaign. A later pool may receive the pilot, replacement assignments for existing TEST passages, or additional selected TEST passages. Stable passage, sentence, and candidate IDs do not change between pools.

The final number of TEST passages and annotators remains TBD. Selecting Pool 1 does not freeze the full final TEST benchmark.

## DEV

DEV remains the separately selected Rodrigo-annotated development set used to build the fixed-weight evaluator. Its intended repository definition is 10 passages, 97 source sentence units, and four candidate systems: exactly 388 candidate judgments. No sentences are added merely to reach 400.

## External web smoke test

Before Pool 1 URLs are shared, one disposable non-Pool user (`SMOKE-FRIEND-01`) completes the independent pilot material through the protected Railway HTTPS path. Campaign `ted2020_pilot_external_smoke_01_cesa` has its own volume-backed runtime namespace, raw JSONL, progress, local private mapping, and backup. The raw JSONL and progress must survive both a Railway service restart and a redeploy before production links are released. Its records are always `eligible_for_gold: false` and never alter Pool 1 allocation or screening. The laptop/Quick-Tunnel path is debug-only and is not the campaign host.

## Passage and annotation protocol

- coherent passages contain 8–10 TED sentence units, preferably 10 when quality is comparable;
- one passage is assigned per Pearmut TEST link;
- models generate every sentence independently;
- annotators see reconstructed passage context but judge only the current sentence candidates;
- cESA uses Minor/Major spans, a 0–100 score, an optional comment, and hidden system identity;
- deterministic private mappings recover canonical candidate/model identity;
- raw annotations are preserved before any screening-based gold filtering.
