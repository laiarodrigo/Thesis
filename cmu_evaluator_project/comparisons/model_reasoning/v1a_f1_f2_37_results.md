# V1-A reasoning and model-capacity comparison

Configurations: frozen V1-A E4B without thinking, F1 E4B with thinking, and F2 31B without thinking. All use the identical 37 inputs and byte-identical V1-A Stage 1/Stage 2 prompts.

This is an exploratory pilot: formal DEV metrics use only 12 judgments from three complete four-candidate sentence groups.

## Configuration and coverage

| Configuration | Model | Revision | Thinking | Dtype | Protocol | Valid |
|---|---|---|---|---|---|---|
| V1-A | google/gemma-4-E4B-it | ee0ef6023621cff504d758262d4e04895a5af4a2 | False | auto | 888db49e814be9dd74b51d9e99289896d2c8d482534ebf2a5a9f0103705d8537 | 35/37 |
| F1 | google/gemma-4-E4B-it | ee0ef6023621cff504d758262d4e04895a5af4a2 | True | auto | 1ea219447286c089423b361fce10c821efaaee95cd9c4dd24931c553d6ea7d71 | 35/37 |
| F2 | google/gemma-4-31B-it | 842da3794eaa0b77d5f08bae87a17459d91ff475 | False | bfloat16 | 06fec449668da4ae5ff2b4964dcd0fee63b002a7f4d296b3de93f71a2aed693e | 37/37 |

## Synthetic diagnostics

| Configuration | Valid | Correct FP | Meaning change | Omission | Unrelated | Source conflict |
|---|---|---|---|---|---|---|
| V1-A | 23/25 | 0/5 | Major 5/5; lower 5/5; mean Δ -45.000 | [MISSING] 0/3; Major 1/3 | Major 3/5; mean Δ -47.000 | lower 5/5; mean Δ -53.000 |
| F1 | 25/25 | 0/5 | Major 5/5; lower 5/5; mean Δ -65.000 | [MISSING] 5/5; Major 5/5 | Major 5/5; mean Δ -90.000 | lower 5/5; mean Δ -68.000 |
| F2 | 25/25 | 0/5 | Major 5/5; lower 5/5; mean Δ -67.000 | [MISSING] 5/5; Major 5/5 | Major 5/5; mean Δ -100.000 | lower 4/5; mean Δ -65.000 |

## Formal DEV agreement

| Configuration | Valid | MAE | Signed Δ | Pearson | Spearman | Exact F1 (M/P/G) | Overlap F1 (M/P/G) | Major recall | Error-free TN | Top-set agreement |
|---|---|---|---|---|---|---|---|---|---|---|
| V1-A | 12/12 | 45.000 | 33.000 | -0.254 | -0.331 | 0.000 (0/13/14) | 0.000 (0/13/14) | 0.000 (0/11) | 0/4 | 0/3 exact; 0/3 overlap |
| F1 | 10/12 | 51.500 | 35.500 | -0.617 | -0.660 | 0.000 (0/4/12) | 0.000 (0/4/12) | 0.000 (0/10) | 1/4 | 0/1 exact; 1/1 overlap |
| F2 | 12/12 | 31.917 | 24.417 | 0.677 | 0.752 | 0.182 (2/8/14) | 0.273 (3/8/14) | 0.273 (3/11) | 3/4 | 1/3 exact; 3/3 overlap |

## Failure attribution and source-side span mistakes

| Configuration | Invalid | Stage 1 failures | Stage 2 failures | Truncated records | Absent candidate spans | Confirmed BP-source copies |
|---|---|---|---|---|---|---|
| V1-A | 2/37 | 2 | 0 | 0 | 2 records / 2 spans | 2 records / 2 spans |
| F1 | 2/37 | 0 | 2 | 0 | 0 records / 0 spans | 0 records / 0 spans |
| F2 | 0/37 | 0 | 0 | 0 | 0 records / 0 spans | 0 records / 0 spans |

## The 12 DEV judgments

| Passage / sentence | ID | System | Human | V1-A | F1 | F2 |
|---|---|---|---|---|---|---|
| ted2020-24360_s25-s33 / 27 | dev-v0-0165 | copy | 0: esteja sempre ganhando [Major], esteja perdendo [Major], [MISSING] [Major] | 100: no errors | 100: no errors | 55: sua equipe [Major] |
| ted2020-24360_s25-s33 / 27 | dev-v0-0166 | B4-D-G-PTBR | 100: no errors | 70: a tua equipa [Minor] | 85: a tua [Minor] | 100: no errors |
| ted2020-24360_s25-s33 / 27 | dev-v0-0167 | B4-D-GF-PTBR | 60: [MISSING] [Minor] | 70: a tua equipa [Minor] | 100: no errors | 55: não esteja sempre a ganhar ou a perder [Major] |
| ted2020-24360_s25-s33 / 27 | dev-v0-0168 | gemma-4-E4B-it | 100: no errors | 85: a sua [Minor] | 100: no errors | 100: no errors |
| ted2020-2609_s82-s91 / 85 | dev-v0-0205 | copy | 0: parada [Major], baldeações [Major], em um [Major], ônibus [Major] | 100: no errors | 100: no errors | 55: você está sentado [Major] |
| ted2020-2609_s82-s91 / 85 | dev-v0-0206 | B4-D-G-PTBR | 15: baldeações [Major] | 70: estás [Minor] | invalid (stage2:score_must_be_100_when_errors_empty) | 60: baldeações [Major] |
| ted2020-2609_s82-s91 / 85 | dev-v0-0207 | B4-D-GF-PTBR | 100: no errors | 85: paragens [Minor], trocas [Minor], estás [Minor], num [Minor], te leva até ao teu bairro [Minor] | 55: ou trocas demoradas [Major] | 60: sem paragens ou trocas [Major] |
| ted2020-2609_s82-s91 / 85 | dev-v0-0208 | gemma-4-E4B-it | 15: baldeações [Major] | 65: está sentado [Major] | 100: no errors | 60: baldeações [Major] |
| ted2020-946_s43-s51 / 45 | dev-v0-0361 | copy | 0: caindo [Major], se desintegrando [Major] | 100: no errors | 100: no errors | 55: caindo e se desintegrando [Major] |
| ted2020-946_s43-s51 / 45 | dev-v0-0362 | B4-D-G-PTBR | 100: no errors | 88: a cair e desintegrar-se [Minor] | 80: a cair e desintegrar-se [Minor] | 100: no errors |
| ted2020-946_s43-s51 / 45 | dev-v0-0363 | B4-D-GF-PTBR | 65: [MISSING] [Minor] | 88: a cair e desintegrar-se [Minor] | 75: uma substância sólida a cair e desintegrar-se. [Minor] | 88: uma substância sólida [Minor] |
| ted2020-946_s43-s51 / 45 | dev-v0-0364 | gemma-4-E4B-it | 40: Bamiyã [Minor] | 70: a cair e desintegrar-se [Minor] | invalid (stage2:score_must_be_100_when_errors_empty) | 100: no errors |

Stage attribution is structural: a Stage 1 failure prevents Stage 2 from running; a Stage 2 failure occurs after validated Stage 1 annotations. Source-copy counts require the invalid reported substring to be absent from the candidate and present verbatim in the current BP source.
