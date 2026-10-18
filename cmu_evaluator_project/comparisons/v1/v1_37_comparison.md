# V1 four-variant 37-example comparison

Frozen baseline: V0 at commit `2ff1d575`.

This pilot is exploratory: it contains 25 synthetic diagnostics and only 12 DEV judgments from three complete four-candidate sentence groups. It cannot establish generalizable superiority.

## Coverage and selected DEV groups

- Shared input: **37** unique rows in identical order for all variants.
- Synthetic: **25** rows.
- DEV: **12** rows from **3** passages/sentence groups.

| Passage / sentence | Evaluation IDs |
|---|---|
| ted2020-24360_s25-s33 / 27 | dev-v0-0165, dev-v0-0166, dev-v0-0167, dev-v0-0168 |
| ted2020-2609_s82-s91 / 85 | dev-v0-0205, dev-v0-0206, dev-v0-0207, dev-v0-0208 |
| ted2020-946_s43-s51 / 45 | dev-v0-0361, dev-v0-0362, dev-v0-0363, dev-v0-0364 |

## Prompt configurations

| Variant | Protocol SHA-256 | Prompt SHA-256 values |
|---|---|---|
| V1-A | 888db49e814be9dd74b51d9e99289896d2c8d482534ebf2a5a9f0103705d8537 | stage1=fde2e0a7224edd694c66268d40d4a5f448e5e617a362b0bdb69e3803650f90ed; stage2=d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd |
| V1-B | 58dd41ea02d1eb7b1cef13afd265ddcf5c04500e3e35f8bd887d8da844a33a33 | stage1=4ffb3461e5882a63f037f547a36a186c0d06f618620926996588df92a468e887; stage2=d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd |
| V1-C | 524d754869b2120e9b945aa6e36b83f01f0562e9dacab7a68d765c2888bd734b | stage1=a036fdd0f77064baaef3c6688e210c831f0a8074518aa8e35b5dcd51b06e3911; stage2=d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd |
| V1-E | f180feb322b538e75e589aca7c7408acc56327416e4face0021488d2b304a90e | stage1_accuracy=18ed14a24a723d90fe640a0793a0cc94e4de6795d2c9c5d03fe11946c539d9cd; stage1_fluency=81ffed5e64461e47008f4761fa2f08fa4bad6baed0660ffd2a394d1b94ab5131; stage2=d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd |

## Synthetic source-sensitivity results

| Evaluator | Valid | Correct-adaptation FP | Meaning change | Major omission | Unrelated | Source conflict |
|---|---|---|---|---|---|---|
| V0 | 23/25 | 5/5 | Major 3/5; lower 5/5; mean Δ -29.000 | valid 3/5; [MISSING] 0/3; any Major 1/3 | Major 5/5; mean Δ -89.000 (N=5) | lower 5/5; mean Δ -58.000 |
| A | 23/25 | 0/5 | Major 5/5; lower 5/5; mean Δ -45.000 | valid 3/5; [MISSING] 0/3; any Major 1/3 | Major 3/5; mean Δ -47.000 (N=5) | lower 5/5; mean Δ -53.000 |
| B | 23/25 | 1/5 | Major 3/5; lower 3/5; mean Δ -28.000 | valid 3/5; [MISSING] 0/3; any Major 1/3 | Major 3/5; mean Δ -41.000 (N=5) | lower 4/5; mean Δ -46.000 |
| C | 24/25 | 4/5 | Major 3/4; lower 4/4; mean Δ -31.250 | valid 5/5; [MISSING] 5/5; any Major 5/5 | Major 5/5; mean Δ -77.000 (N=5) | lower 4/5; mean Δ -53.000 |
| E | 17/25 | 0/4 | Major 3/4; lower 3/3; mean Δ -48.333 | valid 1/5; [MISSING] 0/1; any Major 0/1 | Major 5/5; mean Δ -85.000 (N=4) | lower 2/2; mean Δ -60.000 |

Score differences (Δ) are condition score minus the paired correct-candidate score; negative values are the expected direction.

## DEV agreement results

| Evaluator | Valid | MAE | Signed Δ | Pearson | Spearman | Exact F1 (M/P/G) | Overlap F1 (M/P/G) | Major recall | FP on error-free | Top-set agreement |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 12/12 | 46.667 | 42.500 | 0.168 | -0.148 | 0.000 (0/14/14) | 0.071 (1/14/14) | 0.000 (0/11) | 4/4 | 0/3 exact; 2/3 overlap |
| A | 12/12 | 45.000 | 33.000 | -0.254 | -0.331 | 0.000 (0/13/14) | 0.000 (0/13/14) | 0.000 (0/11) | 4/4 | 0/3 exact; 0/3 overlap |
| B | 11/12 | 46.182 | 33.091 | -0.574 | -0.639 | 0.000 (0/15/13) | 0.000 (0/15/13) | 0.000 (0/10) | 4/4 | 0/2 exact; 0/2 overlap |
| C | 12/12 | 46.500 | 34.500 | -0.369 | -0.489 | 0.000 (0/12/14) | 0.000 (0/12/14) | 0.000 (0/11) | 4/4 | 0/3 exact; 0/3 overlap |
| E | 6/12 | 66.667 | 51.667 | -0.613 | -0.850 | 0.000 (0/3/10) | 0.000 (0/3/10) | 0.000 (0/10) | 2/2 | 0/0 exact; 0/0 overlap |

Span cells show matched/predicted/gold counts. Ranking uses only complete four-candidate groups. Correlations and all DEV differences are exploratory because N≤12.

## Intervention differences relative to V1-A

| Variant vs A | Δ valid N | Δ MAE | Δ overlap F1 | Δ Major recall | Δ error-free FP | Δ exact top sets |
|---|---|---|---|---|---|---|
| B | -1 | 1.182 | 0.000 | 0.000 | 0 | 0 |
| C | 0 | 1.500 | 0.000 | 0.000 | 0 | 0 |
| E | -6 | 21.667 | 0.000 | 0.000 | -2 | 0 |

Negative Δ MAE and error-free false positives, and positive Δ overlap F1, Major recall, or exact top sets, are favorable directions. Interpret every difference as a small-pilot diagnostic, not a general result.

## Structural failures

| Evaluator | Invalid / 37 | Failure reasons |
|---|---|---|
| V0 | 2 | errors[0]_text_not_in_candidate: 2 |
| A | 2 | stage1:errors[0]_text_not_in_candidate: 2 |
| B | 3 | stage1:errors[0]_text_not_in_candidate: 2; stage1:errors[0]_must_be_object: 1 |
| C | 1 | stage1:invalid_json_in_markdown_fence: Expecting ',' delimiter: line 17 column 2 (char 235): 1 |
| E | 14 | stage1_accuracy:errors[0]_text_not_in_candidate: 4; stage1_combined:cross_pass_duplicate: 7; stage1_combined:cross_pass_severity_conflict: 2; stage1_fluency:errors[0]_text_not_in_candidate: 2; stage1_combined:cross_pass_overlap: 1 |
