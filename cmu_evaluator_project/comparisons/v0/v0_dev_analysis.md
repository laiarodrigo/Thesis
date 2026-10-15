# Official V0 DEV analysis

Predictions: `cmu_evaluator_project/runtime/evaluator_dev/v0_official_dev_predictions.jsonl`

## Coverage and scope

- Canonical DEV judgments: **388** (97 source sentences × 4 systems).
- Structurally valid predictions used for candidate-level agreement: **365/388**; excluded: **23**.
- Four-candidate ranking coverage: **80/97 sentences** (320 candidates); excluded sentences: **17**.
- Human Pearmut spans were converted from inclusive UTF-16 offsets to exclusive Python code-point offsets with the comparison viewer's verified converter. Gemma spans were validated as exclusive Python offsets.

All agreement metrics below exclude structurally invalid predictions. Denominators are shown explicitly.

## Score agreement

| Slice | Valid / total | MAE | Mean Gemma−human | Pearson | Spearman | Human mean | Gemma mean |
|---|---|---|---|---|---|---|---|
| Overall | 365 / 388 | 49.647 | 45.619 | 0.132 | 0.035 | 45.644 | 91.263 |
| copy | 94 / 97 | 72.021 | 69.681 | 0.037 | 0.036 | 21.862 | 91.543 |
| B4-D-G-PTBR | 89 / 97 | 45.562 | 42.303 | 0.169 | 0.008 | 50.506 | 92.809 |
| B4-D-GF-PTBR | 92 / 97 | 48.326 | 43.870 | 0.197 | 0.250 | 44.293 | 88.163 |
| gemma-4-E4B-it | 90 / 97 | 31.667 | 25.556 | 0.153 | 0.102 | 67.056 | 92.611 |

### Paired score distributions

| Scores | N | Mean | SD | Min | Q1 | Median | Q3 | Max |
|---|---|---|---|---|---|---|---|---|
| Human | 365 | 45.644 | 42.464 | 0.000 | 5.000 | 30.000 | 100.000 | 100.000 |
| Gemma V0 | 365 | 91.263 | 8.649 | 55.000 | 90.000 | 95.000 | 95.000 | 100.000 |

| Scores | 0–20 | 21–40 | 41–60 | 61–80 | 81–100 | Denominator |
|---|---|---|---|---|---|---|
| Human | 170 | 27 | 23 | 24 | 121 | 365 |
| Gemma V0 | 0 | 0 | 2 | 29 | 334 | 365 |

## Error-span agreement

Exact matching requires identical exclusive boundaries. Overlap matching requires at least one shared code point; two `[MISSING]` spans match only at the same anchor. Every result uses maximum-cardinality one-to-one matching, with total IoU as the secondary criterion.

| Matching | Matched | Predicted spans | Human spans | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Exact boundary | 78 | 485 | 523 | 0.161 | 0.149 | 0.155 |
| Exact boundary + severity | 50 | 485 | 523 | 0.103 | 0.096 | 0.099 |
| Any overlap | 167 | 485 | 523 | 0.344 | 0.319 | 0.331 |
| Any overlap + severity | 105 | 485 | 523 | 0.216 | 0.201 | 0.208 |

### Minor/Major agreement

| Severity | Matched overlap | Predicted | Human | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Minor | 89 | 451 | 252 | 0.197 | 0.353 | 0.253 |
| Major | 16 | 34 | 271 | 0.471 | 0.059 | 0.105 |

- Severity agreement among the **167** severity-agnostic overlap matches: **0.629** (105/167).

| Human severity | Predicted Minor | Predicted Major | Matched denominator |
|---|---|---|---|
| Minor | 89 | 8 | 97 |
| Major | 54 | 16 | 70 |

- Missed human Major spans: **255/271**. False-positive predicted Major spans: **18/34**.
- Unmatched predicted spans of either severity: **318/485**. Human-error-free candidates with at least one predicted error: **82/107**.

### Error agreement by candidate system

| System | Valid / total | Exact F1 | Overlap F1 | Minor F1 | Major F1 | Missed Major | False-positive spans |
|---|---|---|---|---|---|---|---|
| copy | 94 / 97 | 0.132 | 0.231 | 0.152 | 0.080 | 130 / 136 | 34 / 62 |
| B4-D-G-PTBR | 89 / 97 | 0.138 | 0.229 | 0.138 | 0.102 | 51 / 54 | 97 / 122 |
| B4-D-GF-PTBR | 92 / 97 | 0.201 | 0.497 | 0.375 | 0.203 | 42 / 48 | 91 / 180 |
| gemma-4-E4B-it | 90 / 97 | 0.116 | 0.263 | 0.196 | 0.054 | 32 / 33 | 96 / 121 |

## Four-candidate rankings and Gemma self-preference

Ranking metrics use only **80/97** sentences with all four predictions valid; **17** sentences are excluded.

| Ranking measure | Result | Denominator |
|---|---|---|
| Exact equality of tied top sets | 22 | 80 |
| Any overlap between tied top sets | 42 | 80 |
| Mean top-set Jaccard | 0.410 | 80 |
| Pairwise order/tie relation agreement | 233 | 480 |
| Mean within-sentence Spearman | -0.065 | 63 defined; 17 undefined |
| Human top-set ties | 44 | 80 |
| Gemma top-set ties | 50 | 80 |

| Gemma-generated candidate | Human ranking | Evaluator ranking | Denominator |
|---|---|---|---|
| Member of top set | 56 | 43 | 80 |
| Sole top candidate | 15 | 4 | 80 |
| Tie-adjusted top credit | 31.583 | 17.250 | 80 |

Across all valid candidates, the Gemma-generated system's mean score residual is **25.556** points (N=90); the pooled non-Gemma residual is **52.185** (N=275), a Gemma-minus-other gap of **-26.630** points.
Within the 80 complete sentences, Gemma's residual minus the mean residual of the other three systems averages **-25.971** points (median -22.500); it is positive in **18/80** sentences.

These are descriptive self-preference diagnostics, not a causal or significance claim.

## Failure analysis

Structurally invalid predictions excluded from agreement metrics: **23/388**.

| Invalid prediction reason | Error occurrences |
|---|---|
| errors[0]_text_not_in_candidate | 15 |
| errors[1]_text_ambiguous_in_candidate | 4 |
| errors[2]_text_ambiguous_in_candidate | 4 |
| errors[0]_text_ambiguous_in_candidate | 3 |
| errors[3]_text_ambiguous_in_candidate | 1 |
| errors[2]_text_not_in_candidate | 1 |

### Largest score disagreements

| Evaluation | Passage / position / system | Human | Gemma | Gemma−human | Candidate |
|---|---|---|---|---|---|
| dev-v0-0381 | ted2020-946_s43-s51 / pos. 8 / copy | 0 | 100 | +100 | Então, enquanto subimos a árvore, algumas das coisas que estamos fazendo para conseguir subir na árvore estão realmente… |
| dev-v0-0357 | ted2020-946_s43-s51 / pos. 2 / copy | 0 | 100 | +100 | Então, estávamos falando sobre as diferenças entre as tradições e o que muitas pessoas entendiam como a tragédia da des… |
| dev-v0-0349 | ted2020-1886_s23-s32 / pos. 10 / copy | 0 | 100 | +100 | Lutamos não apenas por espaço para os ônibus, mas lutamos por espaço para as pessoas, e isto é ainda mais difícil. |
| dev-v0-0330 | ted2020-1886_s23-s32 / pos. 5 / B4-D-G-PTBR | 0 | 100 | +100 | Não é necessário ter doutorado. |
| dev-v0-0329 | ted2020-1886_s23-s32 / pos. 5 / copy | 0 | 100 | +100 | Não é necessário ter doutorado. |
| dev-v0-0309 | ted2020-2860_s124-s133 / pos. 10 / copy | 0 | 100 | +100 | Se algum de vocês ou alguém que vocês conhecem vive com HIV, ou se quiserem ver quais recursos vocês têm em suas comuni… |
| dev-v0-0301 | ted2020-2860_s124-s133 / pos. 8 / copy | 0 | 100 | +100 | Descobri que, independentemente do risco que eu assumi me expondo daquela maneira, valeu a pena qualquer comentário neg… |
| dev-v0-0289 | ted2020-2860_s124-s133 / pos. 5 / copy | 0 | 100 | +100 | Alguns deles eu conhecia pessoalmente, ou eles sabiam sobre mim, mas muitos outros eram anônimos, |
| dev-v0-0285 | ted2020-2860_s124-s133 / pos. 4 / copy | 0 | 100 | +100 | Esses homens eram dos mais variados tipos e muitos deles não tinham a mesma informação que eu, e me procuravam porque t… |
| dev-v0-0262 | ted2020-1663_s7-s16 / pos. 8 / B4-D-G-PTBR | 0 | 100 | +100 | Isso realmente me animou, dia após dia, quando acordo, a tentar e fazer algo interessante com meu dia. |
| dev-v0-0261 | ted2020-1663_s7-s16 / pos. 8 / copy | 0 | 100 | +100 | Isso realmente me animou, dia após dia, quando acordo, a tentar e fazer algo interessante com meu dia. |
| dev-v0-0245 | ted2020-1663_s7-s16 / pos. 4 / copy | 0 | 100 | +100 | E uma coisa que percebi logo nesse projeto é que se eu não estivesse fazendo nada interessante, provavelmente eu esquec… |
| dev-v0-0205 | ted2020-2609_s82-s91 / pos. 4 / copy | 0 | 100 | +100 | E, sem uma parada sequer, nem baldeações demoradas, você está sentado em um ônibus que vai para o seu bairro. |
| dev-v0-0145 | ted2020-27951_s18-s27 / pos. 8 / copy | 0 | 100 | +100 | Eu nunca soube fazer isso porque não posso me colocar dentro da cabeça das pessoas. |
| dev-v0-0108 | ted2020-2433_s21-s30 / pos. 8 / gemma-4-E4B-it | 0 | 100 | +100 | Verificação de calcinhas? |

### Missed Major errors

At least one human Major span lacks an overlapping predicted Major span in **168** valid candidates. The table shows up to 15, ordered by number of missed spans and score overestimation.

| Evaluation | Passage / position / system | Scores H/G | Missed human Major spans |
|---|---|---|---|
| dev-v0-0205 | ted2020-2609_s82-s91 / pos. 4 / copy | 0 / 100 | parada; baldeações; em um; ônibus |
| dev-v0-0149 | ted2020-27951_s18-s27 / pos. 9 / copy | 0 / 95 | toda vez; sempre tento; escrevendo; esperando |
| dev-v0-0265 | ted2020-1663_s7-s16 / pos. 9 / copy | 0 / 95 | com a qual; embaçar; misturar; sabe |
| dev-v0-0007 | ted2020-18219_s30-s37 / pos. 2 / B4-D-GF-PTBR | 0 / 90 | torna um robô mole é; em primeiro lugar; o seu corpo inteiro; os movimentar |
| dev-v0-0189 | ted2020-24360_s25-s33 / pos. 9 / copy | 0 / 65 | Você; estará melhorando; exercitando seu corpo ; [MISSING] |
| dev-v0-0097 | ted2020-2433_s21-s30 / pos. 6 / copy | 0 / 100 | banheiro; legal; [MISSING] |
| dev-v0-0309 | ted2020-2860_s124-s133 / pos. 10 / copy | 0 / 100 | em suas; a vocês; [MISSING] |
| dev-v0-0069 | ted2020-2203_s26-s32 / pos. 9 / copy | 0 / 95 | havia me; trens; caminhões |
| dev-v0-0098 | ted2020-2433_s21-s30 / pos. 6 / B4-D-G-PTBR | 0 / 95 | banheiro; legal; [MISSING] |
| dev-v0-0165 | ted2020-24360_s25-s33 / pos. 3 / copy | 0 / 95 | esteja sempre ganhando; esteja perdendo; [MISSING] |
| dev-v0-0177 | ted2020-24360_s25-s33 / pos. 6 / copy | 0 / 95 | levar tempo; esporte; você |
| dev-v0-0201 | ted2020-2609_s82-s91 / pos. 3 / copy | 0 / 95 | trem; os vagões; ônibus |
| dev-v0-0266 | ted2020-1663_s7-s16 / pos. 9 / B4-D-G-PTBR | 0 / 95 | com a qual; embaçar; sabes |
| dev-v0-0023 | ted2020-18219_s30-s37 / pos. 6 / B4-D-GF-PTBR | 0 / 90 | é possível apenas; ao invés; é apenas uma entrada |
| dev-v0-0063 | ted2020-2203_s26-s32 / pos. 7 / B4-D-GF-PTBR | 0 / 90 | isso; foi destruído; [MISSING] |

### Human-accepted candidates penalized by Gemma

Gemma predicts at least one error for **82/107** valid candidates that have no human error spans. The table shows the ten largest score penalties.

| Evaluation | Passage / position / system | Scores H/G | Predicted error spans |
|---|---|---|---|
| dev-v0-0061 | ted2020-2203_s26-s32 / pos. 7 / copy | 100 / 65 | Major: ontem destruiu-se |
| dev-v0-0141 | ted2020-27951_s18-s27 / pos. 7 / copy | 100 / 65 | Major: eles tentam escrever algo achando |
| dev-v0-0191 | ted2020-24360_s25-s33 / pos. 9 / B4-D-GF-PTBR | 100 / 75 | Minor: Vai fazer parte; Minor: vai melhorar a sua confiança; Minor: exercitar o seu corpo; Minor: alimentar a sua mente |
| dev-v0-0080 | ted2020-2433_s21-s30 / pos. 1 / gemma-4-E4B-it | 100 / 85 | Minor: ter de usar; Minor: casa de banho masculina |
| dev-v0-0115 | ted2020-2433_s21-s30 / pos. 10 / B4-D-GF-PTBR | 100 / 85 | Minor: à porta das casas de banho |
| dev-v0-0332 | ted2020-1886_s23-s32 / pos. 5 / gemma-4-E4B-it | 100 / 85 | Minor: doutoramento |
| dev-v0-0045 | ted2020-2203_s26-s32 / pos. 3 / copy | 100 / 90 | Minor: Paremos |
| dev-v0-0046 | ted2020-2203_s26-s32 / pos. 3 / B4-D-G-PTBR | 100 / 90 | Minor: Paremos |
| dev-v0-0047 | ted2020-2203_s26-s32 / pos. 3 / B4-D-GF-PTBR | 100 / 90 | Minor: Paremos |
| dev-v0-0048 | ted2020-2203_s26-s32 / pos. 3 / gemma-4-E4B-it | 100 / 90 | Minor: Paremos |

### Explicit unnecessary-rewrite notes

Rodrigo's English comments are development notes, not formal gold. Using only comments that explicitly contain `unnecessary`, **18** valid candidates are identifiable as unnecessary-rewrite cases. Gemma emits no errors for **0/18**, has no span overlapping the human annotations for **3/18**, and scores **14/18** at least 30 points above the human score. The ten largest score overestimates are shown.

| Evaluation | Passage / position / system | Scores H/G | Overlapping spans | Development note |
|---|---|---|---|---|
| dev-v0-0069 | ted2020-2203_s26-s32 / pos. 9 / copy | 0 / 95 | 0 | Unnecessary structural and lexical changes. Clear Brazilian marker. |
| dev-v0-0023 | ted2020-18219_s30-s37 / pos. 6 / B4-D-GF-PTBR | 0 / 90 | 2 | Too many unnecessary changes. Good that respected the "rígidas" change. |
| dev-v0-0367 | ted2020-946_s43-s51 / pos. 4 / B4-D-GF-PTBR | 0 / 90 | 4 | It changed too many things unnecessarily. The major error span introduces grammatical inconsistency. |
| dev-v0-0071 | ted2020-2203_s26-s32 / pos. 9 / B4-D-GF-PTBR | 0 / 85 | 3 | Unnecessary structural and lexical changes. |
| dev-v0-0235 | ted2020-1663_s7-s16 / pos. 1 / B4-D-GF-PTBR | 5 / 90 | 2 | Two completely unnecessary changes. "Em essência" actually makes the sentence look even more Brazilian Portuguese. In this case there was n… |
| dev-v0-0255 | ted2020-1663_s7-s16 / pos. 6 / B4-D-GF-PTBR | 10 / 95 | 0 | Unnecessary paraphrasing. Did not respect the conservative nature of the task. |
| dev-v0-0323 | ted2020-1886_s23-s32 / pos. 3 / B4-D-GF-PTBR | 5 / 90 | 2 | Unnecessary paraphrasing. |
| dev-v0-0351 | ted2020-1886_s23-s32 / pos. 10 / B4-D-GF-PTBR | 5 / 85 | 1 | Unnecessary paraphrasing. |
| dev-v0-0024 | ted2020-18219_s30-s37 / pos. 6 / gemma-4-E4B-it | 15 / 90 | 2 | Too many unnecessary changes. |
| dev-v0-0011 | ted2020-18219_s30-s37 / pos. 3 / B4-D-GF-PTBR | 25 / 90 | 2 | Unnecessary changes. |

### Failure counts by system

| System | Invalid | Large score errors ≥30 | Candidates with missed Major | Human-error-free false positives | Unnecessary-note score overestimates ≥30 |
|---|---|---|---|---|---|
| copy | 3 | 77 | 71 | 7 | 1 |
| B4-D-G-PTBR | 8 | 51 | 41 | 21 | 1 |
| B4-D-GF-PTBR | 5 | 58 | 30 | 19 | 10 |
| gemma-4-E4B-it | 7 | 33 | 26 | 35 | 2 |

## Limitations

- The 23 structurally invalid predictions are excluded rather than repaired; this can change slice coverage and removes any sentence containing one of them from four-way ranking metrics.
- Overlap means any shared code point, so it is more permissive than exact-boundary matching. One-to-one matching prevents duplicate credit.
- The evaluator emits no error categories or rationales. Unnecessary-rewrite analysis is therefore limited to explicit English development-note keywords and human-error-free false positives; those comments are not formal gold.
- Self-preference results are descriptive residual and ranking comparisons on DEV, not a causal test.
