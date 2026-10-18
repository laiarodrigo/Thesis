# CMU Meeting Preparation

## Purpose

This note prepares the discussion with Alon.

It addresses three questions from his email:

1. what the available data realistically allows us to build;
2. how the project flow changes depending on scope;
3. what non-data decisions remain once the data options are clear.

Items that require project decisions remain open.

---

# 1. Data Options

## 1.1 Resource matrix

The table summarizes the relevant resources, their context, and their support for each project scope.

| Resource | Context | EP target/version | Domain | Evaluator-only | Generator + evaluator | Main limitation |
| --- | --- | --- | --- | --- | --- | --- |
| CHAVE | Full ordered documents | Separate, unaligned Público EP collection; no BP→EP target | News | Strong natural BP source with full-document context | Source-only; conservative EP targets require creation/validation | 1994–1995 only; dated and domain-limited |
| TED / TED2020 | Full ordered transcripts | Parallel PT-PT translation; not a conservative BP→EP target | Talks / spoken presentations | Strong contextual BP source and candidate resource | Pairs only after filtering/validation | Independent translations may reformulate; alignment is not always strictly sentence-to-sentence |
| FRMT | Isolated sentences in our local version | PT-PT translation from the same English source; not a conservative adaptation | Wikipedia / encyclopedic | Strong targeted sentence-level source/challenge resource | Selected pairs after filtering/human validation | No document IDs or order; pairs may reformulate |
| DSL-TL | Short, sometimes multi-sentence extracts; no document reconstruction | No | Journalistic | Strong source/challenge resource if short extracts are in scope | Weak directly; requires new conservative EP targets | No document context or BP→EP targets; labels concern source variety, not adaptation quality |
| PtBrVId | Multi-sentence passages; no reliable cross-row document structure | No aligned BP→EP target | Multi-domain | Secondary, mainly for domain diversity | Source-only; targets must be created | No reliable document structure or aligned targets |
| GPT-Wikipedia | Mostly isolated synthetic examples; some self-contained multi-sentence examples | Generated conservative EP version; audited, but not exhaustively pair-validated | Synthetic Wikipedia-inspired | Targeted/secondary synthetic source and candidate resource | Strong conservative training resource | Synthetic; no authentic document context; some residual reformulation |
| Golden Collection | Isolated sentences | Manually curated conservative EP reference | Mixed sentence-level content | Strong targeted resource with conservative and null cases | Secondary; best for evaluation/validation rather than primary training | Small; no document context; a few noisy/outlier references |
| OpenSubtitles | Noisy multi-line dialogue context | Noisy corresponding PT-PT material; not conservative adaptation data | Dialogue / subtitles | Secondary source of dialogue examples | Weak; requires heavy filtering/validation | Noisy correspondence/alignment |

Local/inspected resources include TED2020, FRMT, DSL-TL, PtBrVId, GPT-Wikipedia, and the Golden Collection. CHAVE access has been requested and is pending.

---

## 1.2 What the available data allows

### Context

```text
full document
    ↓
multi-sentence sequence / excerpt
    ↓
isolated sentence
```

CHAVE and TED2020 support full documents. Other resources provide excerpts, local dialogue context, or sentence-level examples for targeted evaluation.

**Question for Alon:** Must every benchmark item be a complete document, or can coherent excerpts and isolated challenge sentences be included?

---

### Domains

The available resources cover news, talks, journalistic and encyclopedic text, dialogue, and mixed-domain material. Full-document data is limited to news and talks.

**Questions for Alon:** Should the benchmark cover one domain, multiple domains, or explicit cross-domain robustness? Should domain be part of the evaluator input?

### Candidate adaptations

Evaluator candidates can come from:

- null / unchanged BP;
- existing conservative T5Gemma model;
- external models;
- existing PT-PT material as candidate or challenge output where appropriate;
- GPT-Wikipedia and Golden Collection references;
- deliberately created challenging candidates;
- a new generator, if generator development is included.

These sources are sufficient to build varied candidates for a human-annotated evaluator benchmark.

---

### If generator development is included

GPT-Wikipedia is the strongest direct conservative training resource. The Golden Collection is conservative but small and better reserved for validation/evaluation. Other resources require new targets or filtering and human validation. The main bottleneck is validated natural, contextual BP→EP supervision.

**Question for Alon:** Is generator development worth including during the visit, or should existing systems provide candidates for an evaluator-focused project?

---

# 2. Full Project Possibilities

```text
0. CORE TASK
   BP → EP only
   conservative adaptation
   fixed sentence-to-sentence mapping
   no splitting / merging / reordering
                         │
                         ↓
1. DATA OPTIONS
   sources
   full documents / context
   domains
   existing BP→EP pairs
   candidate options
                         │
                         ↓
2. PROJECT SCOPE
                         │
              ┌──────────┴───────────┐
              │                      │
       EVALUATOR ONLY        GENERATOR + EVALUATOR
      (no new generator)             │
              │                      │
              │               Need conservative
              │               BP→EP training pairs
              │                      │
              │               train/develop generator
              │                      │
              │               generator outputs
              │                      │
              └──────────────┬───────┘
                             ↓
3. CREATE / COLLECT CANDIDATES
   - null / unchanged BP
   - existing conservative model
   - external models
   - existing dataset outputs
   - challenging candidates
   - new generator output, if that branch is chosen
                             │
                             ↓
4. HUMAN ANNOTATION
                             ↓
5. HUMAN-LABELED EVALUATOR DATA
                             │
                  ┌──────────┴──────────┐
                  │                     │
         NO EVALUATOR SFT         EVALUATOR SFT
                  │                     │
         dev + held-out          train / dev /
           test gold          held-out test gold
                  │                     │
      prompts / instructions      fine-tune on train
      few-shot / RAG              develop on dev
      compare models
                  │                     │
                  └──────────┬──────────┘
                             ↓
6. HELD-OUT HUMAN BENCHMARK
   evaluator prediction
          vs
   unseen human gold
                             │
                             ↓
7. WMT-INSPIRED META-EVALUATION
                             │
                             ↓
8. ROBUSTNESS / FAILURE ANALYSIS
   domains
   linguistic differences
   context
   error types
   candidate/model source
   benchmark limitations
```

# 3. Remaining Design Decisions

Stages 3–8 require decisions about candidate selection, human gold, evaluator development and output, meta-evaluation, and failure analysis.

## 3.1 Stage 3 — Candidate set construction

**Decision:** Candidate composition and balance.

Candidate sources can include:

- null / unchanged BP;
- the existing conservative T5Gemma model;
- PT-PT outputs from parallel resources as candidate/challenge outputs rather than conservative gold;
- GPT-Wikipedia and Golden Collection references where appropriate;
- external systems such as GPT, Gemini, Gemma, and Qwen;
- deliberately constructed problematic candidates;
- a new generator only under the generator + evaluator branch.

The earlier project plan identifies orthographic, lexical/multiword, and morphosyntactic differences, plus shared/common cases. Candidate coverage should also include null cases, under-adaptation, unnecessary rewriting, and meaning-changing outputs.

For one source:

```text
BP source/context
      ↓
candidate A — null
candidate B — conservative model
candidate C — external/model/resource output
...
```

Annotators judge a selected candidate set; every system need not appear for every source.

**Work required:**

- choose the core candidate systems and sources;
- decide how many outputs per BP source are practical;
- balance quality, BP/EP phenomena, and failure types;
- hide or randomize candidate identity for annotators where possible.

**Decision for the meeting**

> Which candidate sources should anchor the benchmark, and how should we balance systems, quality levels, and BP/EP phenomena?

---

## 3.2 Stage 4 — Human annotation

Annotators should receive the context regime chosen in Section 1.

Pearmut is a possible annotation interface. It can present a BP source/context with several candidates, hide or randomize system identities, configure instructions, collect spans, severity, and scores, and export annotations.

The default conceptual item is:

```text
BP source/context
+
several candidate adaptations
```

A conservative EP reference is not part of the default item; whether it should ever be visible is a separate optional decision.

Other formats include different severity scales, ordinal/Likert ratings, pairwise preference or ranking, post-edit effort, binary acceptability, and other score ranges. Alon suggested WMT-style methodology, which provides established formats for error spans, quality scores, and error-free detection.

| Human-gold option | WMT-like formulation | What it gives us | Work / limitation |
| --- | --- | --- | --- |
| Error annotation | Target error spans + minor/major severity | Localized diagnosis | Detailed guidelines; higher annotation cost |
| Direct quality score | 0–100 cESA-style overall-quality rating | Overall quality judgment | BP→EP-specific anchors; limited diagnostic detail |
| Requires correction? | Similar to WMT error-free detection | Acceptability without editing | Precise definition of “no correction” |
| Error category | Optional metadata; not central to WMT scoring | Failure analysis | Requires a small, reliable taxonomy |
| Rich annotation | Spans + severity + score + requires-correction; optional category | Multiple evaluator formulations | Highest annotation effort |

A provisional error-category list is:

- under-adaptation / residual BP;
- unnecessary adaptation / rewriting;
- meaning change, addition, or omission;
- invalid or unnatural EP;
- context inconsistency.

WMT-style spans with minor/major severity and a 0–100 score are options, not requirements. Their compatibility benefits must be weighed against annotation cost. “Requires correction?” directly captures minimum-change acceptability.

> Should we follow the WMT annotation format or simplify it for conservative BP→EP rewriting?

> Should “requires correction?” be collected explicitly as human gold?

> How many annotators per candidate are realistic, and how should disagreements be handled?

---

## 3.3 Stage 5 — Human-labeled gold data and evaluator development

After quality control, Stage-4 annotations become human-labeled evaluator data.

```text
raw human annotations
        ↓
quality control / aggregation
        ↓
human-labeled gold data
```

For multiple annotators, we must decide whether to retain individual judgments, aggregate them, adjudicate important disagreements, or exclude unreliable cases. Candidates from the same BP source/context should remain in one split to prevent source/document leakage.

### No evaluator SFT

```text
human-labeled gold data
        ↓
development + held-out test
```

Development labels guide prompts, instructions, evaluator-model selection, labeled few-shot demonstrations, retrieval-augmented generation (RAG), output format, and final setup. This reuses existing annotations; it does not create more human gold. Evaluator weights remain unchanged, and held-out test labels remain unseen.

### Evaluator SFT

```text
human-labeled gold data
        ↓
train + dev + held-out test
```

Train gold updates evaluator weights, dev gold supports development and model selection, and held-out test gold remains unseen.

Generator training and evaluator supervised fine-tuning (SFT) are separate:

| | Generator training | Evaluator SFT |
| --- | --- | --- |
| Learns to | Rewrite BP→EP | Judge BP→EP candidates |
| Input | BP source | BP source/context + candidate |
| Supervision | Conservative EP target | Human evaluator judgment |
| Project decision | Stage 2 | Stage 5 |

The evaluator-only branch may include evaluator SFT; it excludes only new generator development.

No evaluator SFT needs fewer labels and avoids fine-tuning, making it suitable as the first path for a short project. Evaluator SFT can learn the project’s judgment criterion directly, but requires enough gold for train, dev, and test splits and adds training complexity.

> Should we begin without evaluator SFT—using development data for prompts, few-shot examples, and RAG—and add SFT only if enough human-labeled data is available?

---

## 3.4 Stages 6–7 — Evaluator output and meta-evaluation

Stage 4 defines the human gold. Stage 6 defines the evaluator prediction. Stage 7 compares the prediction with the corresponding human gold.

```text
STAGE 4 — human gold may contain
spans + severity + score + correction-needed
                  ↓
STAGE 6 — evaluator may predict
spans, score, correction-needed, or a joint output
                  ↓
STAGE 7 — compare each prediction
with the corresponding human gold
```

Collecting richer human gold does not require every evaluator to predict every field.

### Option 1 — Error spans + severity

- **Human gold:** error spans with minor/major severity.
- **Evaluator prediction:** error spans with severity.
- **Comparison:** severity-weighted micro-averaged MPP F-score, a severity-aware span-overlap measure.
- **Work:** stabilize span output and matching; define minor/major for conservative adaptation.
- **Value:** localized error detection and severity agreement.

### Option 2 — Direct quality score

- **Human gold:** direct 0–100 cESA-style quality score.
- **Evaluator prediction:** quality score.
- **Comparison:** evaluator–human agreement or correlation; WMT 2026 reports segment and aggregated system levels.
- **Work:** define anchors and select suitable agreement/correlation statistics.
- **Value:** overall quality agreement.

### Option 3 — Requires correction?

- **Human gold:** “requires correction?” yes/no.
- **Evaluator prediction:** yes/no.
- **Comparison:** Matthews correlation coefficient (MCC), a balanced binary measure; precision and recall aid interpretation.
- **Work:** define “requires correction” and decide whether the label is direct or derived.
- **Value:** minimum-change accept/reject decision, similar to WMT error-free detection.

### Option 4 — Joint output

- **Human gold:** multiple fields.
- **Evaluator prediction:** two or more of spans/severity, score, requires-correction, and optional category.
- **Comparison:** the corresponding measures for each field.
- **Work:** more complex prompts, output parsing, and modeling.
- **Value:** diagnosis and overall decision-making.

### Derived versus direct outputs

Severity penalties can convert predicted errors into a score; error presence can produce an error-free label; and score thresholds can produce a binary decision. WMT’s penalties and thresholds were designed for machine-translation evaluation and should not automatically become BP→EP rules. Direct and derived outputs can be compared later.

| Evaluator output | Human gold | WMT-inspired comparison | Main value |
| --- | --- | --- | --- |
| Error spans + severity | Human spans + severity | Severity-weighted MPP F-score | Detailed diagnosis |
| 0–100 score | Human score | Score correlation/agreement | Overall quality |
| Requires correction? | Human binary judgment | MCC; precision/recall | Conservative accept/reject decision |
| Joint output | Multiple gold fields | Corresponding measures above | Richest analysis |

> Which evaluator output should be central: error diagnosis, direct quality score, “requires correction?”, or a joint formulation?

> Should we use WMT meta-measures where applicable and adapt only what conservative rewriting requires?

---

## 3.5 Stage 8 — Robustness and failure analysis

Preserve the metadata needed for the selected analyses:

- candidate-system metadata → compare null, T5Gemma, external-model, and other candidate sources;
- domain metadata → test domain and cross-domain robustness;
- BP/EP phenomenon metadata → test the variety differences identified in the earlier plan;
- error categories → compare under-adaptation, unnecessary rewriting, meaning changes, and related failures;
- context metadata → compare full-document and shorter-context behavior if both are retained;
- null/copy examples → test whether evaluators unnecessarily demand changes;
- evaluator model → compare failure profiles across judge models.

> Which robustness and failure-analysis dimensions should we prioritize within the available project time?

---

# 4. Meeting Notes

## Decisions

- Keep the project **evaluator-only**; do not develop a new generator during
  the visit.
- Use **three to five candidates per source sentence**. The default four are:
  copy/null, the strongest available conservative model, a weaker model, and a
  contrast candidate that adds an otherwise missing behaviour (especially
  over-rewriting or meaning drift).
- Use **two or three domains**, not a single-domain dataset, with approximately
  **five to fifteen documents per domain**.
- Focus human gold on the first two options in Section 3.2: **error spans with
  minor/major severity** and a **0–100 direct quality score**.
- Investigate which BP–EP differences are required changes versus optional or
  context-dependent alternatives before treating them as annotation errors.
- Define an unambiguous, impact-based mapping for minor and major errors and
  prepare annotator-facing guidelines.
- Run a **40-source-sentence self-pilot** before the main annotation campaign.

## Follow-up actions

1. Start from the working annotation guidelines in
   [`CMU_BP_EP_annotation_guidelines_v0.1.md`](../guidelines/CMU_BP_EP_annotation_guidelines_v0.1.md).
2. Build a literature-backed phenomenon decision sheet distinguishing required,
   context-dependent, acceptable-alternative, and unresolved BP–EP differences.
3. Select a balanced 40-sentence pilot across three domains and multiple
   documents, retaining reliable context and document metadata.
4. Produce and blind four candidates per source, then annotate all 160 candidate
   outputs and log difficult decisions.
5. Revise and freeze the guidelines and data-selection rules before selecting
   the main dataset or recruiting annotators.
