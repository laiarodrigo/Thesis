# Gemma 31B G0 zero-shot versus G1 eight-shot

Paired exploratory comparison on the frozen 96-judgment DEV subset. The two reserved demonstration documents are excluded from evaluation; the 84 newly selected judgments are reported separately from the 12 previously inspected pilot judgments.

## Frozen design and coverage

- Demonstration documents: `ted2020-2203, ted2020-27951`.
- Evaluation documents: `ted2020-18219, ted2020-2433, ted2020-24360, ted2020-2609, ted2020-1663, ted2020-2860, ted2020-1886, ted2020-946`.
- Evaluation sample: 24 complete sentence groups × 4 candidates = 96 judgments.
- Execution hardware: G0 used g09 physical GPUs 3,7 (NVIDIA RTX 6000 Ada Generation); G1 used g07 physical GPUs 2,3 (NVIDIA RTX A6000). The frozen model and deterministic BF16 settings are identical, but different GPU architectures are a minor numerical reproducibility confound.
- MPP uses the official `span-mt-metrics-eval` implementation with severity weights Minor=0.25 and Major=1.0 when that optional dependency is installed. Otherwise MPP is reported as unavailable without substitution. `[MISSING]` is evaluated separately because it is not a candidate-side non-empty span.
- PDP is the published Pairwise Difference Pearson calculation over both directions of every within-sentence candidate pair.

| Slice | Expected | G0 valid | G1 valid | Common valid |
|---|---|---|---|---|
| All 96 | 96 | 95 | 95 | 95 |
| New 84 | 84 | 83 | 83 | 83 |
| Prior pilot 12 | 12 | 12 | 12 | 12 |

## All 96: common-valid paired results

| Evaluator | N | MAE | Bias | Pearson | Spearman | MPP F1 | Exact F1 | Overlap F1 | Major recall | Error-free FP; MCC | Missing | PDP | Top-set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | 95 | 43.821 | 41.000 | 0.382 | 0.449 | unavailable | 0.229 | 0.365 | 0.169 | 3/24; MCC 0.467 | R 0.000; P NA | 0.504 (23 groups) | 9/23 exact; 21/23 overlap |
| G1 | 95 | 40.474 | 34.789 | 0.352 | 0.346 | unavailable | 0.225 | 0.360 | 0.197 | 3/24; MCC 0.360 | R 0.176; P 1.000 | 0.469 (23 groups) | 4/23 exact; 19/23 overlap |

## New 84: common-valid paired results

| Evaluator | N | MAE | Bias | Pearson | Spearman | MPP F1 | Exact F1 | Overlap F1 | Major recall | Error-free FP; MCC | Missing | PDP | Top-set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | 83 | 45.265 | 43.120 | 0.371 | 0.446 | unavailable | 0.225 | 0.367 | 0.150 | 2/20; MCC 0.458 | R 0.000; P NA | 0.492 (20 groups) | 8/20 exact; 18/20 overlap |
| G1 | 83 | 42.048 | 36.988 | 0.337 | 0.343 | unavailable | 0.228 | 0.380 | 0.217 | 2/20; MCC 0.367 | R 0.143; P 1.000 | 0.442 (20 groups) | 3/20 exact; 17/20 overlap |

## Prior pilot 12: common-valid paired results

| Evaluator | N | MAE | Bias | Pearson | Spearman | MPP F1 | Exact F1 | Overlap F1 | Major recall | Error-free FP; MCC | Missing | PDP | Top-set |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | 12 | 33.833 | 26.333 | 0.587 | 0.625 | unavailable | 0.261 | 0.348 | 0.273 | 1/4; MCC 0.625 | R 0.000; P NA | 0.558 (3 groups) | 1/3 exact; 3/3 overlap |
| G1 | 12 | 29.583 | 19.583 | 0.500 | 0.434 | unavailable | 0.200 | 0.200 | 0.091 | 1/4; MCC 0.354 | R 0.333; P 1.000 | 0.579 (3 groups) | 1/3 exact; 2/3 overlap |

## Structural and stage-specific failures

| Evaluator | Invalid | Stage 1 | Stage 2 | Truncated | Confirmed source-only | Reasons |
|---|---|---|---|---|---|---|
| G0 | 1 | 1 | 0 | 0 | 1 records / 1 spans | stage1:errors[2]_text_not_in_candidate: 1 |
| G1 | 1 | 1 | 0 | 0 | 1 records / 1 spans | stage1:errors[2]_text_not_in_candidate: 1 |

## Per-system paired results on all common-valid judgments

| Candidate system | Evaluator | N | MAE | Bias | Overlap F1 | Major recall |
|---|---|---|---|---|---|---|
| copy | G0 | 24 | 67.292 | 67.292 | 0.421 | 0.171 |
| copy | G1 | 24 | 54.375 | 53.542 | 0.338 | 0.171 |
| B4-D-G-PTBR | G0 | 24 | 34.250 | 32.417 | 0.240 | 0.182 |
| B4-D-G-PTBR | G1 | 24 | 32.708 | 27.292 | 0.231 | 0.182 |
| B4-D-GF-PTBR | G0 | 23 | 46.217 | 38.652 | 0.361 | 0.200 |
| B4-D-GF-PTBR | G1 | 23 | 44.130 | 32.826 | 0.459 | 0.400 |
| gemma-4-E4B-it | G0 | 24 | 27.625 | 25.542 | 0.333 | 0.111 |
| gemma-4-E4B-it | G1 | 24 | 30.833 | 25.417 | 0.308 | 0.111 |

## Interpretation limits

Formal gold contains spans, severity, and score, but no error-category labels. BP-retention, proper-name/meaning-change, and unnecessary-rewrite findings therefore cannot be separated reliably without a later qualitative review; private development comments are not used here. Missing-content behavior and human-error-free acceptance are reported directly. Valid Stage 2 judgments are structurally consistent with validated Stage 1 annotations because the frozen parser rejects a score of 100 when errors exist and rejects a non-100 score when Stage 1 is empty.
