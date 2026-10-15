# Evaluator development plan

The fixed development ladder is:

`V0 zero-shot → V1 prompt improvement → V2 few-shot → V3 RAG → V4 optional calibration/output refinement → FREEZE → held-out TEST evaluation`

## Development sequence

### DEV annotation and normalization

DEV annotation completed with 10 passages, 97 source sentences, and four candidates per sentence. The immutable Pearmut exports contain 392 submitted judgments because one sentence was accidentally submitted twice. Lossless normalization preserves every submission and its provenance; the canonical DEV join selects the later submission for that repeated sentence and therefore contains 388 judgments. Formal gold is the human error spans with Minor/Major severity and the 0–100 score. The data layers and repeat policy are described in the [benchmark data model](../data/benchmark/README.md).

### V0: zero-shot Gemma 4

- Use English instructions with Portuguese source, context, and candidate text.
- Use no labeled examples.
- Predict exact candidate substrings with Minor/Major severity and a 0–100 quality score.
- Derive zero-based, end-exclusive candidate offsets deterministically in Python after generation.
- Use deterministic inference with `google/gemma-4-E4B-it` at revision `ee0ef6023621cff504d758262d4e04895a5af4a2`.
- Keep the inference artifact blind: no reference, system identity, human labels, human comments, other candidates, or evaluation results are rendered into the model prompt.

#### Research question

Can zero-shot Gemma 4 reproduce Rodrigo's DEV judgments for conservative BP→EP rewriting without seeing labeled examples?

#### Experimental unit

- One candidate is judged independently.
- Passage context is supplied only for interpretation.
- 97 source sentences × 4 candidates = 388 Gemma judgments in the official full DEV run.
- Candidate identity is hidden.

#### Exact V0 fixed instructions

The contents below are exactly [`guidelines/CMU_BP_EP_evaluator_v0_instructions.md`](../guidelines/CMU_BP_EP_evaluator_v0_instructions.md).

````text
# Task

Evaluate one candidate sentence as a conservative rewrite from Brazilian Portuguese (BP) into standard European Portuguese (EP).

The goal is to preserve the source meaning and wording as closely as possible while making the changes necessary for grammatical, natural European Portuguese.

A change is not automatically correct because it sounds more typical of EP. An unchanged expression is not automatically incorrect because it comes from BP.

## Evaluation criteria

Assess the candidate according to these four criteria:

1. **Meaning preservation:** Does the candidate preserve the source meaning, information, and discourse function? Identify omissions, additions, and meaning changes.

2. **Required variety adaptation:** Does the candidate retain any BP constructions that are clearly unacceptable in standard EP? Identify required adaptations that were not made. Do not penalize BP expressions that are also acceptable in EP.

3. **Conservative rewriting:** Has the candidate unnecessarily changed words, grammar, sentence structure, or register that were already acceptable in EP? A grammatically valid EP expression can still be an error if it represents an unjustified change. Do not penalize necessary adaptations merely because they differ from the source.

4. **EP grammaticality and naturalness:** Is the candidate grammatical, natural, and appropriate in its passage context? Accept different valid EP constructions when they preserve the source and satisfy the conservative rewriting objective.

Use the BP passage as context and the current BP sentence as the source of meaning. Evaluate only the EP candidate.

Portuguese allows omitted subjects, and BP `você` has no single mandatory EP replacement. Accept natural, meaning-preserving constructions that fit the context and register.

## Error annotation

Report each independent error using the shortest exact substring of the EP candidate that identifies the problem.

Use these severity labels:

- `Minor`: A localized problem that does not substantially compromise the rewrite, including small unnecessary substitutions, minor grammatical or punctuation issues, or limited stylistic changes.
- `Major`: A problem that materially compromises the task, including meaning errors, significant omissions, necessary variety adaptations left undone, serious grammatical or contextual errors, or substantial unnecessary rewriting.

Group dependent changes within the same grammatical construction rather than marking multiple overlapping errors.

If content is missing and no existing candidate text can identify the omission, use `[MISSING]`.

Do not use `[MISSING]` when the meaning is preserved through a different grammatical construction.

All error substrings must come from the candidate, never from the BP source.

## Quality score

Assign a whole-candidate quality score from 0 to 100 using these broad bands:

- 85–100: Very good
- 65–80: Good
- 45–60: Acceptable
- 25–40: Borderline
- 0–20: Not acceptable

Use intermediate scores when appropriate.

Assign 100 if and only if no errors are identified.

Scores below 100 must reflect the overall impact and severity of the errors. Major errors should not normally receive very-good scores.

Do not use a mechanical points-per-error formula.

## Output

Return exactly one JSON object with `errors` and `score`.

Each error contains:
- `text`: the exact erroneous candidate substring, or `[MISSING]`.
- `severity`: `Minor` or `Major`.

Do not generate character offsets. They are calculated separately.

If an error substring occurs multiple times in the candidate, identify which occurrence is erroneous using the optional `occurrence` field, a one-based index counting identical substring occurrences from the beginning of the candidate.

For unique substrings, omit `occurrence`.

For missing content, use `[MISSING]` without `occurrence`.

Use an empty `errors` array and score 100 when no errors are identified.

Return only valid JSON. Do not provide reasoning, explanations, Markdown, or additional text.
````

#### Exact V0 runtime template

The verified Gemma 4 runtime uses `Gemma4Processor` with Transformers 5.19.0. Existing GPU output records show the processor rendering the prompt as one user turn. The fixed instructions and evaluation data therefore remain in one user message, separated by `# Evaluation data`; `{context}` is one `[position] source sentence` per line. The candidate is the final data section.

````text
{fixed_instructions}

# Evaluation data

PORTUGUESE PASSAGE CONTEXT (BP — REFERENCE ONLY):
{context}

CURRENT BP SOURCE SENTENCE (REFERENCE ONLY, position {position}):
{source}

EP CANDIDATE TO EVALUATE (ONLY TEXT TO ANNOTATE):
{candidate}
````

The processor wraps that single message in the recorded Gemma chat-template structure:

````text
<bos><|turn>user
{user_message}<turn|>
<|turn>model
````

#### Output schemas

For an ordinary unique error substring, Gemma returns:

```json
{"errors":[{"text":"exact candidate substring","severity":"Minor"}],"score":75}
```

For a repeated exact substring, Gemma may use the optional one-based `occurrence` field:

```json
{"errors":[{"text":"teu","severity":"Major","occurrence":2}],"score":40}
```

For an error-free candidate, Gemma returns:

```json
{"errors":[],"score":100}
```

The strict parser locates exact occurrences and produces the normalized representation used by the viewer and later metrics:

```json
{"error_spans":[{"start":12,"end":15,"text":"teu","severity":"Major","missing":false,"occurrence":2}],"score":40}
```

Normalized Gemma offsets are Python character indexes with exclusive ends. Gemma never generates offsets. `occurrence` is optional, one-based, and retained when supplied. For repeated substrings it must identify an existing exact occurrence; the parser never assigns one from prediction order. `[MISSING]` must not have `occurrence` and becomes a zero-length span at `len(candidate)`. An absent substring, unresolved or out-of-range occurrence, overlap or duplicate, malformed JSON, invalid label, or inconsistent score makes the prediction invalid; no fuzzy matching or repair is performed.

The parser accepts only a plain JSON object, one JSON object enclosed by a single Markdown code fence optionally labeled `json`, or an opening permitted fence followed by one complete JSON object when the recorded generation stopper demonstrably fired at that object. Only the wrapper is removed. Explanations, multiple objects, malformed content, unrelated text, and other fence forms remain invalid. Every output record preserves the exact continuation and separately records raw JSON-only compliance, whether wrapper normalization occurred, the recognized representation, parser validity, and whether the generation stopper fired.

#### Gemma sees / Gemma does not see

Gemma sees:

- the fixed English evaluator instructions above;
- the complete BP passage context, with sentence positions, as reference material;
- the clearly marked current BP source sentence, as reference material;
- one EP candidate sentence as the final and only text to annotate;
- the required JSON format.

Gemma does not see:

- a reference rewrite;
- candidate or system identity;
- any other candidate for the current source sentence;
- human spans, severities, scores, or comments;
- labeled or few-shot examples;
- prior predictions or evaluation results.

#### V0 data flow

`blind DEV input → Gemma predictions → manual sanity inspection → DEV metrics → failure analysis → V1`

#### V0 run status

The preliminary full-DEV attempts and the 36-instance passage pilot were protocol-validation runs and are not official V0 results. The eight-instance substring-protocol smoke produced 7 valid predictions and 1 correctly rejected ambiguous-substring prediction (`dev-v0-0018`), with no trailing-content failures. Optional one-based occurrence disambiguation and the final fixed prompt were then added. A subsequent 12-instance smoke produced 12 complete `json`-fenced objects: no fence was truncated by the old stopper and no response contained other text. Strict local wrapper normalization made 10 judgments valid; `dev-v0-0017` and `dev-v0-0018` remained invalid because repeated substrings were reported without `occurrence`. The follow-up 12-instance wrapper-handling smoke then stopped all 12 fenced generations immediately after the complete JSON object, recorded all 12 as normalized and not raw-format compliant, and again yielded 10 valid judgments plus the same 2 strict ambiguity failures.

#### Official V0 DEV run

The official V0 DEV run completed over all 388 blind candidate rows. The SLURM submission ID is not embedded in the prediction artifact and was not present in the retrieved local logs. The frozen configuration was:

- Input: [`data/benchmark/dev_01_v0_input.jsonl`](../data/benchmark/dev_01_v0_input.jsonl) (388 unique evaluation IDs, 97 source sentences, 10 passages).
- Output: [`runtime/evaluator_dev/v0_official_dev_predictions.jsonl`](../runtime/evaluator_dev/v0_official_dev_predictions.jsonl), SHA-256 `2bc01c6bef4f8b9b5bd69494486638ac6d806364b0f869fde8dc0aa7f028f707`.
- Model: `google/gemma-4-E4B-it` at requested revision `ee0ef6023621cff504d758262d4e04895a5af4a2`.
- Deterministic generation: `do_sample=false`, `max_new_tokens=512`, `use_cache=true`, and `enable_thinking=false`.
- Prompt structure: `2.0-single-user-evaluation-data`.
- Model-output schema: `2.1-substring-occurrence`.
- Output parser: `1.1-strict-markdown-wrapper`.
- Instructions SHA-256: `c9d22177d847f66fbe28c15bfad47bab810b342be7d47998852ae8ebd073a798`.
- Combined prompt/schema SHA-256: `71aa52e88c9e1c37ed3fd7efff9c960c7bf4224dce2e5a60a8ae6f5e418adeb2`.

Validation found exactly 388 unique expected IDs with no missing, duplicate, or unexpected records and no reconstructed input-message, candidate-mapping, model, revision, or generation-configuration mismatches. Of the 388 predictions, 365 passed all format, structural, and exact-substring validation and 23 remained invalid. Fifteen records failed only because at least one reported substring was absent from the candidate, seven failed only because a reported substring was repeated without a valid `occurrence`, and one record contained both failure types. Across invalid records, this corresponds to 16 absent-substring errors and 12 ambiguous-substring errors. No output reached the token limit.

Seventy-four raw continuations were plain JSON. The other 314 began with the permitted `json` Markdown fence, were stopped immediately after their complete JSON object, and were transparently wrapper-normalized; none of those 314 is marked raw-format compliant. The completion stopper fired for all 388 generations.

The ten qualitative comparison reports are under [`comparisons/v0/`](../comparisons/v0/), one `v0_comparison_<passage-id>.html` file per DEV passage. Together they contain 97 sentence rows and 388 candidate cards, preserve four candidate judgments per sentence, and visibly retain all 23 invalid predictions.

#### V0 quantitative evaluation

The complete results and denominators are in the [official V0 DEV analysis](../comparisons/v0/v0_dev_analysis.md). On the 365 structurally valid predictions, score MAE was 49.647, mean Gemma−human score was +45.619, and Spearman correlation was 0.035. Exact span F1 was 0.155 and overlap span F1 was 0.331. Major-error recall was particularly low: 255 of 271 human Major spans were missed. Four-candidate ranking results used the 80 of 97 sentences for which all four predictions were valid. The descriptive residuals ran opposite to self-preference: Gemma-generated candidates had a smaller positive score residual than the pooled other systems. This is not a causal or significance claim.

#### Source-sensitivity diagnostic

After the official analysis, a separate synthetic diagnostic tested whether V0 actually used its BP source. SLURM job `13862` produced exactly 25 predictions: five sentence groups crossed with correct, meaning-change, major-omission, unrelated, and conflicting-source conditions. Twenty-three predictions were structurally valid. The two invalid outputs were omission cases in which Gemma copied absent BP-source text into an error span instead of using `[MISSING]`. Inputs, raw predictions, all 25 results, and the interpretation are preserved under [`diagnostics/v0_source_sensitivity/`](../diagnostics/v0_source_sensitivity/); the detailed account is the [diagnostic results report](../diagnostics/v0_source_sensitivity/v0_source_sensitivity_results.md).

#### Main V0 findings and motivation for V1

V0 clearly consults the BP source: it consistently penalized unrelated candidates and lowered all five fixed candidates when their sources were replaced by conflicting sources. It nevertheless has poor agreement with the formal DEV gold. Scores are strongly inflated, Major errors and localized semantic changes are often missed, omissions are handled inconsistently, and valid required BP→EP adaptations are sometimes penalized as unnecessary changes. V1 should therefore improve the zero-shot instructions using these DEV failure patterns while keeping the model, blind single-candidate setup, strict output validation, and formal task definition fixed.

### V1: prompt improvement

Improve the evaluator instructions based on DEV failure analysis.

### V2: few-shot

Use DEV labels as few-shot examples. During DEV evaluation, exclude the current item and every example from the same passage.

### V3: RAG

Retrieve relevant guidelines and/or DEV examples. During DEV evaluation, apply the same leakage restriction: exclude the current item and every example from the same passage.

### V4: optional calibration/output refinement

Optionally calibrate scores and refine the output format.

After development, freeze the evaluator and evaluate it once on held-out TEST.

## V0 implementation record

The immutable Pearmut exports are preserved under `data/annotations/raw/dev_01/`. Lossless normalization keeps all submissions and provenance in `data/annotations/normalized/dev_01.json`. The canonical DEV gold deterministically selects the later submission for the accidentally repeated sentence, yielding 388 final candidate judgments. These local gold-bearing artifacts include annotation provenance and English development comments and are not part of the versioned blind inference bundle.

The local gold-bearing join is `data/benchmark/dev_01_evaluator.json`. The cluster input is `data/benchmark/dev_01_v0_input.jsonl`; it contains only opaque evaluation IDs, BP passage/source/candidate text, and no gold, comments, reference, candidate IDs, or system metadata. Evaluation IDs are used for record matching but are not rendered into prompts.

V0 uses [`guidelines/CMU_BP_EP_evaluator_v0_instructions.md`](../guidelines/CMU_BP_EP_evaluator_v0_instructions.md), [`scripts/run_evaluator_v0.py`](../scripts/run_evaluator_v0.py), and [`scripts/slurm/run_evaluator_v0.sbatch`](../scripts/slurm/run_evaluator_v0.sbatch). Predictions are append-only and resumable only when their combined prompt/schema hash and output-parser version match. Each output record preserves the messages, actual rendered prompt, raw generation, format status, parser diagnostics, instruction hash, prompt-structure version, output-schema version, output-parser version, and combined prompt/schema hash. The generation stopper recognizes a complete leading object in either plain output or an allowed opening Markdown fence. Strict parsing still rejects malformed or extra content present in the saved continuation.

## Labels and permissible development material

Formal DEV gold consists of error spans, Minor/Major severity, and a 0–100 quality score.

Rodrigo's English Pearmut comments are development notes, not formal gold. They may support failure analysis, prompt design, retrieval, and example selection, but a comment must never be shown for the item being evaluated.

Human annotation guidelines must not be substantively changed mid-campaign. Evaluator prompts may be improved using DEV.

## Immediate next step

Begin V1 prompt improvement from the documented V0 DEV failure analysis. Do not change the human guidelines or use held-out data during evaluator development.
