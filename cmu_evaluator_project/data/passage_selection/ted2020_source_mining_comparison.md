# Comparison of TED2020 source-mining passes

> PT TED remains discovery evidence only; none of these contrasts is a conservative rewrite rule or annotator-visible reference.

## 1–3. Overlap, recovered original passages, and additions

The revised union pool has **100** passages: **0** exact original windows, **0** talks represented in the original 50, and **0** passages supported by both miners.
New passages outside original talks: **100**. Previous negative-audit windows recovered exactly: **5** (`ted2020-1626_s39-s48`, `ted2020-1714_s62-s71`, `ted2020-1773_s83-s91`, `ted2020-431_s12-s21`, `ted2020-473_s1-s10`).

## 4–5. Lemma recovery and independently discovered lexical/orthographic signals

Stable BP lemmas retrospectively matching an old detector include: `você`, `equipe`, `econômico`, `gênero`, `fenômeno`, `prêmio`, `ônibus`, `banheiro`, `esporte`, `acadêmico`.
Strong stable BP surface/lemma signals not matched by the old inventory include: `e`, `e`, `esse`, `eu`, `então`, `então`, `eu`, `eles`, `seu`, `isso`, `eles`, `ele`.

## 6–8. POS-trigram structure and cross-view passages

Strong stable BP POS trigrams include: `PRON VERB ADP`, `PRON VERB DET`, `VERB DET NOUN`, `PRON AUX VERB`, `CCONJ PRON VERB`, `PRON PRON VERB`, `PRON ADV VERB`, `PRON VERB VERB`, `AUX VERB ADP`, `VERB ADP PRON`, `ADV PRON VERB`, `NOUN PRON VERB`.
Passages with POS-trigram evidence: **100**; with both lemma and POS-trigram evidence: **100**.
Passages most strengthened by POS evidence: `ted2020-2118_s89.1-s96` (70.4% of family contribution), `ted2020-18219_s30-s37` (67.9% of family contribution), `ted2020-500_s21-s28` (67.7% of family contribution), `ted2020-1577_s24-s33` (67.0% of family contribution), `ted2020-414_s50-s59` (66.9% of family contribution), `ted2020-1503_s33-s42` (66.8% of family contribution), `ted2020-1654_s36-s45` (66.5% of family contribution), `ted2020-176_s1-s9` (66.5% of family contribution), `ted2020-31_s24-s32` (66.3% of family contribution), `ted2020-2246_s80-s88` (66.3% of family contribution).

## 9–10. Bidirectional divergence and complementary information

- surface_form: D_KL(BP||PT)=0.20975051 bits; D_KL(PT||BP)=0.21393427 bits.
- lemma: D_KL(BP||PT)=0.13896955 bits; D_KL(PT||BP)=0.13699023 bits.
- pos_trigram: D_KL(BP||PT)=0.04101765 bits; D_KL(PT||BP)=0.04457364 bits.

Both directions are retained because asymmetry changes the feature contribution and exposes complementary BP- and PT-characteristic evidence.

## 11–12. Recurrent aligned contrast associations

Lexical/orthographic associations retained: **200**. Leading examples: `esse` ↔ `este` (2620 docs), `essa` ↔ `esta` (1460 docs), `essas` ↔ `estas` (1083 docs), `isso` ↔ `isto` (1872 docs), `esse` ↔ `este` (1341 docs), `isso` ↔ `isto` (1769 docs), `esses` ↔ `estes` (957 docs), `algo` ↔ `coisa` (1008 docs).
Structural POS-trigram associations retained: **80**. Leading examples: `CCONJ DET NOUN` ↔ `NOUN CCONJ DET` (2259 docs), `VERB DET NOUN` ↔ `SCONJ VERB DET` (3093 docs), `NOUN CCONJ NOUN` ↔ `DET NOUN CCONJ` (2299 docs), `NOUN VERB ADP` ↔ `DET NOUN VERB` (2747 docs), `CCONJ DET NOUN` ↔ `DET NOUN CCONJ` (2428 docs), `VERB ADP DET` ↔ `SCONJ VERB ADP` (2689 docs), `NOUN ADJ CCONJ` ↔ `DET NOUN ADJ` (2504 docs), `NOUN ADP NOUN` ↔ `ADP NOUN VERB` (2524 docs).

## 13–14. Stability

Stable frequent evidence was retained when it preserved direction in all folds: `de` (197,122), `e` (172,926), `e` (172,879), `VERB DET NOUN` (143,966), `NOUN ADP NOUN` (101,711), `VERB ADP DET` (89,167), `um` (76,094), `em` (63,301), `eu` (61,098), `estar` (51,548), `DET NOUN PRON` (44,916), `se` (44,902).
Statistically significant candidates with fewer than four stable replications are excluded from passage ranking; leading cases are listed below.

### Strong revised candidates

- `ted2020-255_s20-s29` — 106.67; 137 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-1562_s18.1-s25` — 106.34; 123 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-5_s110.2-s119` — 106.18; 127 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-250_s7-s16` — 105.97; 125 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-2246_s80-s88` — 105.93; 131 families; 9/9 evidence-bearing sentences; HIGH.
- `ted2020-432_s79-s87` — 105.54; 130 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-9_s94-s103` — 105.40; 138 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-946_s43-s51` — 105.25; 126 families; 9/9 evidence-bearing sentences; HIGH.
- `ted2020-1626_s39-s48` — 105.20; 142 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-1503_s33-s42` — 105.14; 119 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-431_s12-s21` — 105.11; 129 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-474_s11-s20` — 105.02; 146 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-1479_s90-s97` — 104.96; 122 families; 9/9 evidence-bearing sentences; HIGH.
- `ted2020-1817_s99-s106` — 104.84; 125 families; 10/10 evidence-bearing sentences; HIGH.
- `ted2020-958_s133-s142` — 104.78; 115 families; 10/10 evidence-bearing sentences; HIGH.

### Statistically strong but unstable examples

- None among retained candidates.

## 15. Source-quality intervention

Source-quality statistics: `{"corpus_best_excluded_by_union_ranking": 3834, "corpus_best_materially_quality_downranked": 284, "documents_with_all_windows_hard_rejected": 34, "hard_rejected_windows": 427, "selected_with_quality_flags": 23}`. Severe failures are rejected; moderate flags apply a bounded score penalty and remain reviewable.

## 16. Previous negative-audit recovery

The revised shortlist recovered **5** exact windows from the preceding deterministic negative audit. The new audit remains diagnostic and requires researcher inspection for residual false negatives.

## 17. Researcher-pool assessment

The revision materially broadens the evidence used for human source review: real Bosque lemmas and UPOS trigrams supplement surface evidence, while bidirectional KLD, document-level stability, aligned recurrence, saturation, and quality controls reduce obvious failure modes. It does **not** establish that every signal is an obligatory BP→EP change, and it does not freeze a benchmark.

## Limitations

- PT and BP TED texts are independent translations, not conservative rewrites.
- Pointwise signals may encode translator conventions despite stability controls.
- Aligned associations use only conservative exact-ID pairs and are not replacement rules.
- Dependency and word-order entropy were omitted to prioritize the required feature views.
- Human review remains necessary before source freezing.
