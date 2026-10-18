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

#### 2026-10-16 four-variant pilot

The first V1 experiment is a controlled 37-item pilot that keeps the frozen V0 model, revision, deterministic generation settings, blind single-candidate input, and strict validation principles fixed. It compares four prompt/output decompositions implemented by one reusable runner:

- **V1-A:** identify all errors first, then assign the whole-candidate score in a separate pass.
- **V1-B:** use the same two-stage design, but require an ordered check of meaning preservation, omissions, required BP→EP adaptation, conservative rewriting, and EP grammaticality/naturalness before emitting errors.
- **V1-C:** identify candidate-side errors and source-side omissions separately. Source-side omission substrings must match the current BP source exactly and use explicit occurrence disambiguation when repeated. Validated omissions are deterministically normalized to separate candidate-end `[MISSING]` annotations while preserving the source-side diagnostics; they are never inferred or repaired.
- **V1-E:** run independent accuracy/variety and fluency/conservatism error passes, combine only non-overlapping, non-duplicate, severity-consistent validated annotations, then score them in a separate pass. Conflicts make the judgment invalid rather than being resolved heuristically.

All variants use [`scripts/run_evaluator_v1.py`](../scripts/run_evaluator_v1.py), the prompt components under [`guidelines/v1/`](../guidelines/v1/), and the shared blind input [`data/benchmark/v1_37_input.jsonl`](../data/benchmark/v1_37_input.jsonl), SHA-256 `4668f3a2ac508cb4df862e34a7aa867163cddea332d65e827d0794e7f690ee56`. The first 25 rows are byte-for-byte the unchanged source-sensitivity diagnostic input. The remaining 12 rows are three complete four-candidate DEV groups from three different passages:

- `ted2020-24360_s25-s33`, sentence position 3 (`dev-v0-0165`–`dev-v0-0168`): a mixed group containing omission annotations, missed Major errors, and error-free candidates that V0 penalized.
- `ted2020-2609_s82-s91`, sentence position 4 (`dev-v0-0205`–`dev-v0-0208`): a difficult group with missed Major errors and an error-free candidate that V0 penalized, without a human omission annotation.
- `ted2020-946_s43-s51`, sentence position 3 (`dev-v0-0361`–`dev-v0-0364`): a moderate group containing an omission, missed Major errors, and an error-free candidate that V0 penalized.

These DEV groups were selected to cover several documented V0 failure types rather than only the largest score disagreements. The inference rows retain the V0 blind schema: selection condition, candidate system, human gold, and development comments are not included or rendered into prompts.

The frozen pilot prompt/protocol hashes are:

| Variant | Stage-1 prompt hash(es) | Shared score prompt hash | Protocol hash |
| --- | --- | --- | --- |
| A | `fde2e0a7224edd694c66268d40d4a5f448e5e617a362b0bdb69e3803650f90ed` | `d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd` | `888db49e814be9dd74b51d9e99289896d2c8d482534ebf2a5a9f0103705d8537` |
| B | `4ffb3461e5882a63f037f547a36a186c0d06f618620926996588df92a468e887` | `d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd` | `58dd41ea02d1eb7b1cef13afd265ddcf5c04500e3e35f8bd887d8da844a33a33` |
| C | `a036fdd0f77064baaef3c6688e210c831f0a8074518aa8e35b5dcd51b06e3911` | `d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd` | `524d754869b2120e9b945aa6e36b83f01f0562e9dacab7a68d765c2888bd734b` |
| E | accuracy `18ed14a24a723d90fe640a0793a0cc94e4de6795d2c9c5d03fe11946c539d9cd`; fluency `81ffed5e64461e47008f4761fa2f08fa4bad6baed0660ffd2a394d1b94ab5131` | `d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd` | `f180feb322b538e75e589aca7c7408acc56327416e4face0021488d2b304a90e` |

The four append-only outputs are `runtime/evaluator_v1/v1_a_37_predictions.jsonl`, `v1_b_37_predictions.jsonl`, `v1_c_37_predictions.jsonl`, and `v1_e_37_predictions.jsonl`. All four runs completed. The [combined exploratory report](../comparisons/v1/v1_37_comparison.md) and [visual comparison](../comparisons/v1/v0_v1_37_comparison.html) separate synthetic behavior from formal DEV agreement and compare each variant with frozen V0. On the 12 DEV judgments, none of A/B/C/E overlapped a human span or recalled a human Major span; B and C did not improve score agreement over A, and E had substantially lower structural validity. C was the only variant to represent all five synthetic omissions correctly, but it produced false positives on four of five correct adaptations. These results motivate testing whether V1-A's failure is due to reasoning mode or model capacity before further prompt redesign. This 12-item DEV subset remains a prompt-direction diagnostic, not a conclusive evaluator comparison.

#### 2026-10-16 thinking and larger-model comparison

This controlled follow-up reuses the existing V1-A predictions as the E4B thinking-disabled control; it does not rerun V1-A. Two new configurations use the identical 37-row blind input and the byte-identical V1-A Stage 1 and Stage 2 prompts:

- **F1:** `google/gemma-4-E4B-it` at revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, native Gemma thinking enabled in both stages, deterministic generation, baseline `dtype=auto` loading, and budgets of 4096 Stage-1 and 2048 Stage-2 tokens. It is assigned through SLURM to one `RTX_6000_24GB` GPU on `g06` in partition `hlt_msc`.
- **F2:** `google/gemma-4-31B-it` at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, thinking disabled, deterministic unquantized BF16 inference, and 512-token budgets in both stages. The 62.6 GB checkpoint is assigned with `device_map="auto"` across the two authorized physical RTX A6000 48 GB GPUs 0 and 1 on direct host `g07`; CPU/disk spill or any different visible-GPU placement is rejected.

The verified HLT software stack is PyTorch `2.6.0+cu124`, Transformers `5.19.0`, and Accelerate `1.15.0`, with BF16 reported as supported. The frozen prompts retain hashes `fde2e0a7224edd694c66268d40d4a5f448e5e617a362b0bdb69e3803650f90ed` (Stage 1) and `d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd` (Stage 2). After smoke validation of the installed Transformers response API, the full-input protocol hashes are `1ea219447286c089423b361fce10c821efaaee95cd9c4dd24931c553d6ea7d71` for F1 and `06fec449668da4ae5ff2b4964dcd0fee63b002a7f4d296b3de93f71a2aed693e` for F2.

Thinking and final-answer channels are saved separately. The completion stopper examines only the post-thinking final-answer segment, and strict JSON and candidate-substring validation is applied only to the parsed final answer. Stage 1 thoughts are not passed into Stage 2. Both runs are append-only and resumable, with separate smoke and official output paths under `runtime/evaluator_model_reasoning/`.

The first F1 smoke, SLURM job `13960`, generated two complete thought channels and complete Stage-1 JSON answers but was correctly stopped by the smoke gate at 0/2 valid: Transformers `5.19.0` requires the rendered prompt prefix when parsing a generated response because the chat template can prefill assistant-message structure. The runner was corrected to pass the exact prompt-token prefix, record parser version `1.2-v1-channel-aware-prefix`, and use new `smoke2` filenames so the failed raw outputs remained untouched. Corrected F1 smoke job `13961` and the corrected F2 direct-host smoke both passed at 2/2 valid.

The official F1 run was SLURM job `13962`; the F2 direct run used tmux session `f2_31b_37` and recorded Python process ID `306417` on `g07`. Both preserved all 37 expected IDs in the shared input order, with no missing, duplicate, unexpected, prompt-hash, protocol-hash, model-revision, thinking-setting, dtype, or input-mapping mismatches. No generation hit its token limit. F1 saved nonempty, separately parsed thinking for both stages of every item; F2 saved no thinking, as configured.

- F1 output: [`runtime/evaluator_model_reasoning/f1_e4b_thinking_37_predictions.jsonl`](../runtime/evaluator_model_reasoning/f1_e4b_thinking_37_predictions.jsonl), SHA-256 `a0cc0fc7a5b693ab7d9ee7429f420162c1bcb5e4066bfbed9715dd2fd8791cb6`. Thirty-five of 37 judgments were valid. Both exclusions were Stage-2 consistency failures: Stage 1 returned no errors but Stage 2 returned score 98 rather than the required 100.
- F2 output: [`runtime/evaluator_model_reasoning/f2_31b_no_thinking_37_predictions.jsonl`](../runtime/evaluator_model_reasoning/f2_31b_no_thinking_37_predictions.jsonl), SHA-256 `e0c68eea8bfd1e164cf290ca0f7d456b39c657cb3d77d4e8909405cc1e80e6a4`. All 37 judgments were valid.

The [combined Markdown report](../comparisons/model_reasoning/v1a_f1_f2_37_results.md) and [unified HTML comparison](../comparisons/model_reasoning/v1a_f1_f2_37_comparison.html) contain the complete results and raw-stage inspection. On the five synthetic groups, both F1 and F2 accepted all five correct adaptations, detected all five meaning changes and all five omissions as Major, used `[MISSING]` for every omission, and rejected all five unrelated candidates. F1 lowered all five fixed candidates under conflicting sources; F2 lowered four of five.

On the 12 formal DEV judgments, F1 did not improve over V1-A: among its 10 valid judgments, score MAE was 51.500, overlap span F1 and Major recall were both zero, and it correctly left only one of four human-error-free candidates error-free. F2 was materially better on this small diagnostic subset: 12/12 valid, score MAE 31.917, Spearman 0.752, exact span F1 0.182, overlap span F1 0.273, Major recall 3/11, three of four human-error-free candidates accepted, exact top-set agreement on one of three sentence groups, and any top-set overlap on all three. Neither new configuration produced candidate-substring violations or copied BP-source spans. These are encouraging capacity results, not general conclusions: the formal comparison contains only 12 judgments from three sentences.

#### 2026-10-16 expanded Gemma 31B zero-shot/eight-shot comparison

G0 and G1 are a frozen paired comparison on a larger, leakage-controlled DEV subset. Both use `google/gemma-4-31B-it` at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, unquantized BF16, thinking disabled, deterministic generation, 512-token Stage-1 and Stage-2 budgets, the existing strict parser, and the unchanged V1-A instruction texts. G0 is the zero-shot F2 protocol on the expanded input. G1 appends eight frozen human-labeled demonstrations to both stages; Stage 1 demonstrations contain human error JSON but no scores, while Stage 2 demonstrations contain the same errors and their human score JSON. Evaluated instances still receive only their own blind text, and G1 Stage 2 receives G1's validated Stage-1 prediction rather than evaluation gold.

The deterministic document split uses seed `20261016` and the original TED document ID. The two demonstration documents are `ted2020-2203` and `ted2020-27951`. The eight evaluation documents are `ted2020-18219`, `ted2020-2433`, `ted2020-24360`, `ted2020-2609`, `ted2020-1663`, `ted2020-2860`, `ted2020-1886`, and `ted2020-946`. The three previously inspected pilot documents were excluded from demonstration eligibility before selection. The frozen audit found no document overlap and no exact or NFKC/casefold/whitespace-normalized source/candidate text duplication between demonstration and evaluation pools.

Eight demonstrations come from eight distinct source sentences in the two reserved documents: two error-free score-100 adaptations, two important BP-retention Major-error cases, two missing-content cases, and two conservative-rewriting cases. The reserved documents did not contain two unambiguous altered-fact or altered-entity cases, so missing-content examples supply the supported meaning-preservation coverage rather than inventing annotations or drawing examples from evaluation documents. The model-visible payload excludes system identity, filenames, evaluation IDs, annotator identity, and English development comments.

The evaluation sample contains three complete sentence groups per evaluation document: 24 sentences × four candidates = 96 judgments. It retains the three previously inspected groups (12 judgments) and adds 21 deterministically selected groups (84 judgments). It is an exploratory subset, not the complete DEV distribution. G0 and G1 share identical evaluation IDs, text, context, and order.

Frozen artifacts:

- [split manifest](../data/benchmark/gemma31b_g0_g1_manifest.json), SHA-256 `371e317b63fa110a31d1693c18e69f3889137c533b3b61dbcf7d2859a23a615f`;
- [blind 96-row input](../data/benchmark/gemma31b_g0_g1_eval_96.jsonl), SHA-256 `e0e3f358a763885afc6c336e1278bbf7affc905b24352acbce1b66fe47024f52`;
- [eight-shot model-visible payload](../data/benchmark/gemma31b_g1_eight_shot_demonstrations.json), SHA-256 `da5d0bf157a902cbc51341c390b228772b83c46e52b800149e0e08a8fea3c2ad`;
- Stage-1 demonstration appendix SHA-256 `7afc2f90ea84087122f7dd360b163dff16a51a0929ab2dc8df8d9ad05dfce15c`;
- Stage-2 demonstration appendix SHA-256 `16ab381cbf54858533af6c8d977f50da2dec4f9bbd0f312262e20520725881fb`.

The unchanged instruction hashes remain `fde2e0a7224edd694c66268d40d4a5f448e5e617a362b0bdb69e3803650f90ed` for Stage 1 and `d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd` for Stage 2. On the complete 96-row selection, the frozen G0 protocol hash is `0ff7b308434330a8e62079093ea2a89eedf5a319628955f256528fb9de4801ed`; the G1 protocol hash, which additionally binds the demonstration payload and appendices, is `630efaf44e923e55df6c98a1c93aced3a6acba7444ab04893c85693103d0e09a`. Re-running F2 on its original 37-row input still reproduces its prior protocol hash, confirming that G0 did not alter the baseline rendering or configuration.

The official append-only outputs are `runtime/evaluator_gemma31b/g0_31b_zero_shot_96_predictions.jsonl` (SHA-256 `417da110bdc09d0231f01eaa325635337f2b7e93b07747cf0a0da730f203b53f`) and `runtime/evaluator_gemma31b/g1_31b_eight_shot_96_predictions.jsonl` (SHA-256 `8de2c5f6ccb174cd29b139ba3554b6a396d421eadfe4cf3ee1096e354f14880a`). G0 ran on g09 physical RTX 6000 Ada GPUs 3 and 7 while G1 ran concurrently on g07 physical RTX A6000 GPUs 2 and 3. Both pairs passed a two-device CUDA/BF16/free-memory probe before use. Model, revision, BF16 precision, prompts, parser, and deterministic decoding settings were identical, but the different GPU architectures are a minor numerical reproducibility confound because cross-architecture kernels are not guaranteed to be bitwise identical. The launcher rejects any other unapproved host or pair, CPU/disk model spill, insufficient free memory, or a failed frozen leakage audit. The runner also rejects any rendered prompt whose input tokens plus output budget exceed the model-declared context window.

Both runs produced 96 unique expected IDs in frozen order with the expected input, prompt, protocol, model, parser, and demonstration hashes. Neither run truncated a generation. Each produced 95 structurally valid judgments and independently rejected `dev-v0-0323` after Stage 1 copied `quase` from the BP source although it was absent from the candidate; Stage 2 was correctly skipped. The context preflight passed against the model's 262,144-token window: G0's maximum Stage-1 input was 1,115 tokens and G1's was 4,769 tokens, with both 512-token output budgets included in the enforced checks.

The [paired Markdown report](../comparisons/gemma31b/g0_g1_96_comparison.md) and [self-contained HTML viewer](../comparisons/gemma31b/g0_g1_96_comparison.html) separate all 96 judgments, the 84 new judgments, and the 12 prior pilot judgments. Metrics use the 95 common-valid judgments overall, including 83 new and all 12 pilot judgments. Official severity-weighted MPP uses `span-mt-metrics-eval` pinned at commit `a6de7050297524a0d96bde1d93adb1203b0d118b`, excludes zero-length `[MISSING]` markers, and reports omission behavior separately.

On all 95 common-valid judgments, G1 reduced score MAE from 43.821 to 40.474 and positive bias from 41.000 to 34.789. It increased MPP F1 from 0.284 to 0.300, Major-error recall from 0.169 to 0.197, and `[MISSING]` recall from 0 to 0.176 with no false-positive missing markers. These gains do not establish an overall winner: G1 reduced Pearson from 0.382 to 0.352, Spearman from 0.449 to 0.346, PDP from 0.504 to 0.469, exact top-set agreement from 9/23 to 4/23, and any top-set overlap from 21/23 to 19/23; exact and ordinary overlap span F1 were nearly unchanged and slightly lower overall. On the 84 newly selected judgments, G1's MPP, overlap F1, Major recall, omission recall, MAE, and bias improved, while its correlations, PDP, and ranking agreement declined. On the 12 previously inspected judgments, G1 lowered MAE and bias but substantially reduced span and Major-error agreement. The few-shot effect is therefore metric- and subset-dependent and must be inspected qualitatively before choosing the next evaluator configuration.

#### 2026-10-16 G2/G3/G4 stage-specific demonstration design

The original G1 demonstrations did not provide sufficiently clear, stage-specific coverage. G2/G3/G4 therefore use separately selected eight-example sets from `ted2020-946` and `ted2020-1886`: Stage 1 prioritizes exact span identification, severity, omissions, factual changes, conservative rewriting, and correct unchanged/adapted forms; Stage 2 prioritizes the relationship between annotations and holistic scores. Five judgments are shared. Stage 1 receives error JSON only and never demonstration scores. Stage 2 receives error and score JSON. Its deliberately exploratory score distribution is `0, 10, 65, 70, 80, 90, 100, 100`; it is not a deterministic mapping from error count or severity to score.

The factorial configurations isolate demonstration effects while preserving the frozen G0/G1 model, revision, prompts, parser, deterministic BF16 generation, disabled thinking, and two-stage target construction: G2 uses improved Stage 1 with zero-shot Stage 2; G3 reuses compatible frozen G0 Stage-1 outputs and uses improved Stage 2; G4 uses both improved stages. The improved Stage-1 predictions are generated once as a portable append-only artifact and reused unchanged by G2 and G4. Invalid Stage-1 outputs remain invalid and cause dependent Stage 2 to be skipped.

The common comparison is restricted to 72 judgments (18 complete four-candidate groups) from `ted2020-18219`, `ted2020-2433`, `ted2020-24360`, `ted2020-2609`, `ted2020-1663`, and `ted2020-2860`. The eight prior-pilot judgments are `dev-v0-0165`–`0168` and `dev-v0-0205`–`0208`; the other 64 are the new slice. The frozen [manifest](../data/benchmark/gemma31b_g2_g4_manifest.json), [blind input](../data/benchmark/gemma31b_g2_g4_eval_72.jsonl), and [stage-specific demonstrations](../data/benchmark/gemma31b_g2_g4_demonstrations.json) record provenance and hashes. Their audit confirms document-level separation, five shared demonstrations, no evaluation gold/comments/system identity in blind targets, no Stage-1 scores, and no held-out data access.

This is DEV experimentation, not final held-out evaluation. Implementation and local audits are complete; no G2/G3/G4 GPU inference has yet been run.

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

Inspect the paired G0/G1 HTML and Markdown results, especially the 84 newly selected judgments, before choosing the next evaluator iteration. Do not select G1 from score MAE alone; weigh its improved MPP/Major/omission behavior against its weaker score correlation and ranking agreement. Do not change the human guidelines or use held-out data during evaluator development.
