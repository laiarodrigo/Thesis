# Translation Family Struggle Examples: FRMT and Golden Collection

Generated on 2026-05-11 16:21 UTC.

Method: for each translation family from the recovered tables, choose the stronger representative run on that dataset (best Stage B/Stage C row from the table), then use that run's `translation_summary.json` `worst_sentence_bleu_examples` list as the source of the bad examples. Sentence BLEU is copied directly from the summary JSON. Sentence WER is shown as additional context after stripping an optional leading `BR`/`PT` decoder label.

## FRMT

### Plain translation-only, 4B GPT+FRMT lineage

Chosen model: `4B Two-Staged LoRA r24 - FRMT Eval (GPT+FRMT Stage B)`

Chosen because Stage B BLEU 44.83 / WER 0.445788 beat its Stage C counterpart in the recovered table.

Predictions file: `eval_results/encoder_decoder/compare_staged/frmt_eval/gpt_refresh2_frmt/4b_translation_stageB_r24_fullfrmt/20260322_224206_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/frmt_eval/gpt_refresh2_frmt/4b_translation_stageB_r24_fullfrmt/20260322_224206_translation_summary.json`

1. `source_id=4540` `id=4540` `direction=None` `sentence_bleu=0.00` `sentence_wer=1.000`

Input:
```text
<br-pt> Ordenado.
```
Gold:
```text
ordenou
```
Prediction:
```text
Organizado.
```

2. `source_id=4541` `id=4541` `direction=None` `sentence_bleu=0.00` `sentence_wer=1.000`

Input:
```text
<pt-br> ordenou
```
Gold:
```text
Ordenado.
```
Prediction:
```text
ordenou
```

3. `source_id=5059` `id=5059` `direction=None` `sentence_bleu=0.17` `sentence_wer=1.000`

Input:
```text
<pt-br> Refere-se também que dominam os tubarões-das-galápagos (C. galapagensis) e os tubarões-galha-preta (C.
```
Gold:
```text
Eles também são conhecidos por dominar tubarões de Galápagos (C. galapagensis) e tubarões-gralha-preta (C.
```
Prediction:
```text
fimbriatus).
```

4. `source_id=5058` `id=5058` `direction=None` `sentence_bleu=0.74` `sentence_wer=1.000`

Input:
```text
<br-pt> Eles também são conhecidos por dominar tubarões de Galápagos (C. galapagensis) e tubarões-gralha-preta (C.
```
Gold:
```text
Refere-se também que dominam os tubarões-das-galápagos (C. galapagensis) e os tubarões-galha-preta (C.
```
Prediction:
```text
Carcharhinus melanopterus).
```

5. `source_id=5030` `id=5030` `direction=None` `sentence_bleu=2.71` `sentence_wer=2.500`

Input:
```text
<br-pt> O escritor do episódio, David A.
```
Gold:
```text
O escritor deste episódio, David A.
```
Prediction:
```text
O. Serra, afirmou que o episódio foi inspirado no filme "The Thing From Another World".
```

### Plain translation-only, 4B GPT-Wiki r48 lineage

Chosen model: `4B Stage C Dr. GRPO r48 - FRMT Eval (GPT-Wiki Stage B-BLEU+WER Reward)`

Chosen because Stage C BLEU 41.74 / WER 0.473424 slightly beat Stage B in the recovered table.

Predictions file: `eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r48_stageC_drgrpo_frmt_gpt_wiki_legit_bleu_wer_g07_bf16_run2/20260507_224850_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r48_stageC_drgrpo_frmt_gpt_wiki_legit_bleu_wer_g07_bf16_run2/20260507_224850_translation_summary.json`

1. `source_id=4540` `id=4540` `direction=br2pt` `sentence_bleu=0.00` `sentence_wer=1.000`

Input:
```text
<br-pt> Ordenado.
```
Gold:
```text
ordenou
```
Prediction:
```text
Ordenado.
```

2. `source_id=4541` `id=4541` `direction=pt2br` `sentence_bleu=0.00` `sentence_wer=1.000`

Input:
```text
<pt-br> ordenou
```
Gold:
```text
Ordenado.
```
Prediction:
```text
ordenou
```

3. `source_id=3629` `id=3629` `direction=pt2br` `sentence_bleu=2.72` `sentence_wer=1.214`

Input:
```text
<pt-br> Os fatos de negócios tradicionais normalmente têm cores uniformes ou com riscas; o padrão de quadrados também é aceitável.
```
Gold:
```text
Ternos corporativos tradicionais são geralmente em cores sólidas com listras finas ou xadrez windowpane.
```
Prediction:
```text
Os fatos de negócios tradicionais normalmente têm cores uniformes ou com riscas; o padrão de quadrados também é aceitável.
```

4. `source_id=3628` `id=3628` `direction=br2pt` `sentence_bleu=2.80` `sentence_wer=0.842`

Input:
```text
<br-pt> Ternos corporativos tradicionais são geralmente em cores sólidas com listras finas ou xadrez windowpane.
```
Gold:
```text
Os fatos de negócios tradicionais normalmente têm cores uniformes ou com riscas; o padrão de quadrados também é aceitável.
```
Prediction:
```text
Os ternos corporativos tradicionais são geralmente em cores sólidas com listras finas ou xadrez windowpane.
```

5. `source_id=2771` `id=2771` `direction=pt2br` `sentence_bleu=2.82` `sentence_wer=1.400`

Input:
```text
<pt-br> No final da década de 60 e ao longo da década de 70, eram populares os tons de terra, incluindo o "Harvest Gold" (ouro da colheita), o verde abacate e o tom amêndoa.
```
Gold:
```text
No fim dos anos 1960 e durante os anos 1970, cores em tons terrosos ficaram populares, amarelo-queimado, verde-abacate e amêndoa.
```
Prediction:
```text
Na década de 60 e ao longo da década de 70, eram populares os tons de terra, incluindo o "Harvest Gold" (ouro da colheita), o verde abacate e o tom amêndoa.
```

### With cls, 270M GPT-Wiki+FRMT

Chosen model: `270M Two-Staged Full-FT - FRMT Eval (GPT-Wiki+FRMT Stage B with cls)`

Only Stage B row in the recovered table for this family.

Predictions file: `eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_with_cls/20260422_182519_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_with_cls/20260422_182519_translation_summary.json`

1. `source_id=4540` `id=4540` `direction=None` `sentence_bleu=0.00` `sentence_wer=1.000`

Input:
```text
<br-pt> Ordenado.
```
Gold:
```text
ordenou
```
Prediction:
```text
Ordenado.
```

2. `source_id=4541` `id=4541` `direction=None` `sentence_bleu=0.00` `sentence_wer=2.000`

Input:
```text
<pt-br> ordenou
```
Gold:
```text
Ordenado.
```
Prediction:
```text
ele ordenou
```

3. `source_id=3628` `id=3628` `direction=None` `sentence_bleu=2.80` `sentence_wer=0.842`

Input:
```text
<br-pt> Ternos corporativos tradicionais são geralmente em cores sólidas com listras finas ou xadrez windowpane.
```
Gold:
```text
Os fatos de negócios tradicionais normalmente têm cores uniformes ou com riscas; o padrão de quadrados também é aceitável.
```
Prediction:
```text
Os ternos corporativos tradicionais são geralmente em cores sólidas com listras finas ou xadrez windowpane.
```

4. `source_id=3629` `id=3629` `direction=None` `sentence_bleu=2.84` `sentence_wer=1.143`

Input:
```text
<pt-br> Os fatos de negócios tradicionais normalmente têm cores uniformes ou com riscas; o padrão de quadrados também é aceitável.
```
Gold:
```text
Ternos corporativos tradicionais são geralmente em cores sólidas com listras finas ou xadrez windowpane.
```
Prediction:
```text
Os fatos de negócios tradicionais geralmente têm cores uniformes ou com riscas; o padrão de quadrados também é aceitável.
```

5. `source_id=1281` `id=1281` `direction=None` `sentence_bleu=2.86` `sentence_wer=0.857`

Input:
```text
<pt-br> Perdeu dez quilogramas e contraiu uma doença na tiroide.
```
Gold:
```text
Ela estava dez quilos (22 libras) mais magra e havia adquirido problemas de tireoide.
```
Prediction:
```text
Ele perdeu dez quilogramas e contraiu uma doença na tiroide.
```

### Label-first + cls, 270M GPT-Wiki+FRMT

Chosen model: `270M Two-Staged Full-FT - FRMT Eval (GPT-Wiki+FRMT Stage B label-first with cls)`

Only Stage B row in the recovered table for this family.

Predictions file: `eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_label_first_with_cls/20260422_182519_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_label_first_with_cls/20260422_182519_translation_summary.json`

1. `source_id=4540` `id=4540` `direction=None` `sentence_bleu=0.00` `sentence_wer=1.000`

Input:
```text
<br-pt> Ordenado.
```
Gold:
```text
ordenou
```
Prediction:
```text
Ordenado.
```

2. `source_id=4541` `id=4541` `direction=None` `sentence_bleu=0.00` `sentence_wer=2.000`

Input:
```text
<pt-br> ordenou
```
Gold:
```text
Ordenado.
```
Prediction:
```text
<pt-br> ordenou
```

3. `source_id=206` `id=206` `direction=None` `sentence_bleu=1.11` `sentence_wer=0.893`

Input:
```text
<br-pt> Como alguns runs — em particular o "rhum agricole" do Caribe francês — também são feitos com esse processo, a cachaça também é conhecida como rum brasileiro.
```
Gold:
```text
Uma vez que alguns runs, em especial o rum agrícola das Caraíbas Francesas, são também produzidos por este processo, a cachaça é conhecida também como o rum brasileiro.
```
Prediction:
```text
As cachaças são também conhecidas como os rumos brasileiros.
```

4. `source_id=3106` `id=3106` `direction=None` `sentence_bleu=1.63` `sentence_wer=0.889`

Input:
```text
<br-pt> Lapelas no estilo shawl, ou xale, vêm da vestimenta noturna informal da era Vitoriana, e não costumam ser vistas, a não ser em smokings e ternos mais formais.
```
Gold:
```text
As lapelas de xaile são um estilo derivado da roupa de noite informal da época vitoriana e, como tal, não é usual nos casacos dos fatos, exceto no caso dos smokings ou dos fatos de jantar.
```
Prediction:
```text
<br-pt> Lapelas de estilo "shawl" ou "xale" vêm da moda noturna do século 19 e não são vistas frequentemente, pois não são em smokings e ternos mais formais.
```

5. `source_id=3701` `id=3701` `direction=None` `sentence_bleu=2.09` `sentence_wer=1.000`

Input:
```text
<pt-br> Apesar de inicialmente ser uma característica dos fatos de campo, utilizado para guardar o bilhete de comboio, atualmente costuma ver-se nos fatos de cidade.
```
Gold:
```text
Originalmente, ele aparecia apenas em ternos rurais e era usado para se guardar a passagem de trem, mas hoje também aparece em ternos urbanos.
```
Prediction:
```text
Apesar de inicialmente ser uma característica dos fatos de campo, usada para guardar o bilhete de voo, hoje em dia, os fatos de cidade são encontrados.
```

### Label-first + cls, 270M, PtBrVId Stage A

Chosen model: `270M Stage C Dr. GRPO - FRMT Eval (GPT-Wiki+FRMT Stage B label-first with cls PtBrVId Stage A BLEU+WER+first-token Reward, gold candidate)`

Chosen because Stage C BLEU 36.66 / WER 0.535877 beat Stage B in the recovered table.

Predictions file: `eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageC_drgrpo_gpt_wiki_frmt_label_first_with_cls_from_stageA_opensubs_plus_ptbrvarid_bleu_wer_first_token_gold/20260426_123354_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageC_drgrpo_gpt_wiki_frmt_label_first_with_cls_from_stageA_opensubs_plus_ptbrvarid_bleu_wer_first_token_gold/20260426_123354_translation_summary.json`

1. `source_id=4540` `id=4540` `direction=None` `sentence_bleu=0.00` `sentence_wer=2.000`

Input:
```text
<br-pt> Ordenado.
```
Gold:
```text
ordenou
```
Prediction:
```text
<br-pt> Ordenado.
```

2. `source_id=4541` `id=4541` `direction=None` `sentence_bleu=0.00` `sentence_wer=2.000`

Input:
```text
<pt-br> ordenou
```
Gold:
```text
Ordenado.
```
Prediction:
```text
<pt-br> ordenou
```

3. `source_id=3217` `id=3217` `direction=None` `sentence_bleu=2.05` `sentence_wer=0.846`

Input:
```text
<pt-br> A Convenção Complementar de 1956 relativa à Abolição da Escravatura, do Tráfico de Escravos e das Instituições e Práticas Análogas à Escravatura define que as "instituições e práticas semelhantes à escravatura" devem incluir: c) Qualquer instituição ou prática segundo a qual: (i) Uma mulher, sem o direito de recusar, seja prometida ou dada em casamento mediante uma retribuição em dinheiro ou em espécie aos pais da rapariga, tutor, família ou qualquer outra pessoa ou grupo; ou (ii) O marido de uma mulher, ou respetivo clã, tem o direito de transferi-la para outra pessoa pelo valor recebido ou outro; ou (iii) Uma mulher cujo marido tenha falecido pode ser herdada por outra pessoa.
```
Gold:
```text
A Convenção Complementar de 1956 sobre a Abolição da Escravatura, o Tráfico de Escravos e Instituições e Práticas Análogas à Escravidão define "instituições e práticas análogas à escravidão" como incluindo: c) Qualquer instituição ou prática em que: (i) Uma mulher, sem o direito de recusar, é prometida ou doada em casamento como pagamento de uma dívida em dinheiro ou em bens de seus pais, guardião, família ou qualquer outra pessoa ou grupo; ou (ii) O marido de uma mulher, sua família, ou seu clã, tem o direito de transferi-la para outra pessoa pelo valor recebido ou de outra forma; ou (iii) Uma mulher, com a morte de seu marido, está sujeita a ser herdada por outra pessoa.
```
Prediction:
```text
<pt-br> A Convenção Complementar de 1956, sobre a Abolição da Escravatura, do Tráfico de Escravos e das Instituições e Práticas Análogas à Escravaturade definem que “instituições e práticas semelhantes à escrav
```

4. `source_id=3216` `id=3216` `direction=None` `sentence_bleu=2.11` `sentence_wer=0.821`

Input:
```text
<br-pt> A Convenção Complementar de 1956 sobre a Abolição da Escravatura, o Tráfico de Escravos e Instituições e Práticas Análogas à Escravidão define "instituições e práticas análogas à escravidão" como incluindo: c) Qualquer instituição ou prática em que: (i) Uma mulher, sem o direito de recusar, é prometida ou doada em casamento como pagamento de uma dívida em dinheiro ou em bens de seus pais, guardião, família ou qualquer outra pessoa ou grupo; ou (ii) O marido de uma mulher, sua família, ou seu clã, tem o direito de transferi-la para outra pessoa pelo valor recebido ou de outra forma; ou (iii) Uma mulher, com a morte de seu marido, está sujeita a ser herdada por outra pessoa.
```
Gold:
```text
A Convenção Complementar de 1956 relativa à Abolição da Escravatura, do Tráfico de Escravos e das Instituições e Práticas Análogas à Escravatura define que as "instituições e práticas semelhantes à escravatura" devem incluir: c) Qualquer instituição ou prática segundo a qual: (i) Uma mulher, sem o direito de recusar, seja prometida ou dada em casamento mediante uma retribuição em dinheiro ou em espécie aos pais da rapariga, tutor, família ou qualquer outra pessoa ou grupo; ou (ii) O marido de uma mulher, ou respetivo clã, tem o direito de transferi-la para outra pessoa pelo valor recebido ou outro; ou (iii) Uma mulher cujo marido tenha falecido pode ser herdada por outra pessoa.
```
Prediction:
```text
<br-pt> A Convenção Complementar de 1956 sobre a Abolição da Escravatura, o Tráfico de Escravos e Instituições e Práticas Análogas à Escravidão definem "instituições e práticas análogicas à escravidão
```

5. `source_id=3033` `id=3033` `direction=None` `sentence_bleu=2.21` `sentence_wer=0.878`

Input:
```text
<pt-br> Em 2016, as mortes de peões eram de: 2,6 por milhão de habitantes nos Países Baixos; 4,3 na Suécia; 4,5 por milhão de habitantes no País de Gales; 5,3 na Nova Zelândia; 6,0 na Alemanha; 7,1 no Reino Unido; 7,5 na Austrália; 8,4 em França; 8,4 em Espanha; 9,4 em Itália; 11,1 em Israel; 13 no Japão; 13,8 na Grécia; 18,5 nos Estados Unidos da América; 22,9 na Polónia e 36,3 na Roménia
```
Gold:
```text
As fatalidades com pedestres em 2016 foram 2,6 por população de um milhão nos Países Baixos, 4,3 na Suécia, 4,5 por população de um milhão no País de Gales, 5,3 na Nova Zelândia, 6,0 na Alemanha; 7,1 no Reino Unido, 7,5 na Austrália, 8,4 na França, 8,4 na Espanha, 9,4 na Itália, 11,1 em Israel, 13 no Japão, 13,8 na Grécia, 18,5 nos Estados Unidos da América, 22,9 na Polônia e 36,3 na Romênia
```
Prediction:
```text
<pt-br> Em 2016, as mortes de peões eram: 2,6 por milhão de habitantes no Países Baixos; 4,3 na Suécia; 1,45 por milhões de habitantes do País de Gales; 5
```

### Label-first + cls equal, 4B

Chosen model: `4B Stage C Dr. GRPO - FRMT Eval (GPT-Wiki+FRMT Stage B label-first with cls equal BLEU+WER+first-token Reward, gold candidate)`

Chosen because Stage C BLEU 37.61 / WER 0.508754 beat Stage B in the recovered table.

Predictions file: `eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r48_stageC_drgrpo_gpt_wiki_frmt_label_first_with_cls_equal_legit_bleu_wer_first_token_gold_g09/20260507_125027_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r48_stageC_drgrpo_gpt_wiki_frmt_label_first_with_cls_equal_legit_bleu_wer_first_token_gold_g09/20260507_125027_translation_summary.json`

1. `source_id=4540` `id=4540` `direction=br2pt` `sentence_bleu=0.00` `sentence_wer=2.000`

Input:
```text
<br-pt> Ordenado.
```
Gold:
```text
ordenou
```
Prediction:
```text
<br-pt> Ordenado.
```

2. `source_id=4541` `id=4541` `direction=pt2br` `sentence_bleu=0.00` `sentence_wer=1.000`

Input:
```text
<pt-br> ordenou
```
Gold:
```text
Ordenado.
```
Prediction:
```text
ordenou
```

3. `source_id=3629` `id=3629` `direction=pt2br` `sentence_bleu=2.35` `sentence_wer=1.286`

Input:
```text
<pt-br> Os fatos de negócios tradicionais normalmente têm cores uniformes ou com riscas; o padrão de quadrados também é aceitável.
```
Gold:
```text
Ternos corporativos tradicionais são geralmente em cores sólidas com listras finas ou xadrez windowpane.
```
Prediction:
```text
<pt-br> Os fatos de negócios tradicionais normalmente têm cores uniformes ou com riscas; o padrão de quadrados também é aceitável.
```

4. `source_id=1280` `id=1280` `direction=br2pt` `sentence_bleu=2.52` `sentence_wer=1.444`

Input:
```text
<br-pt> Ela estava dez quilos (22 libras) mais magra e havia adquirido problemas de tireoide.
```
Gold:
```text
Perdeu dez quilogramas e contraiu uma doença na tiroide.
```
Prediction:
```text
<br-pt> Ela estava dez quilos (22 libras) mais magra e havia adquirido problemas de tireoide.
```

5. `source_id=3628` `id=3628` `direction=br2pt` `sentence_bleu=2.72` `sentence_wer=0.895`

Input:
```text
<br-pt> Ternos corporativos tradicionais são geralmente em cores sólidas com listras finas ou xadrez windowpane.
```
Gold:
```text
Os fatos de negócios tradicionais normalmente têm cores uniformes ou com riscas; o padrão de quadrados também é aceitável.
```
Prediction:
```text
<br-pt> Ternos corporativos tradicionais são geralmente em cores sólidas com listras finas ou xadrez windowpane.
```

## Golden Collection

### Plain translation-only, 4B GPT+FRMT lineage

Chosen model: `4B Stage C Dr. GRPO r24 - Golden Collection`

Chosen because Stage C BLEU 62.35 / WER 0.277996 beat Stage B in the recovered table.

Predictions file: `eval_results/encoder_decoder/compare_staged/stageC_drgrpo_frmt_gptrefresh2/4b_translation_stageC_r24_golden/20260318_175733_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/stageC_drgrpo_frmt_gptrefresh2/4b_translation_stageC_r24_golden/20260318_175733_translation_summary.json`

1. `source_id=403` `id=403` `direction=None` `sentence_bleu=0.00` `sentence_wer=1.000`

Input:
```text
<pt-br> Com efeito!
```
Gold:
```text
com efeito!
```
Prediction:
```text
De fato.
```

2. `source_id=385` `id=385` `direction=None` `sentence_bleu=4.29` `sentence_wer=0.818`

Input:
```text
<pt-br> E vamos, então, normas de disciplinar a nossa forma de funcionamento.
```
Gold:
```text
E vamos, então, normas de disciplinar a nossa forma de funcionamento.
```
Prediction:
```text
Então, vamos às regras para disciplinar nosso modo de operação.
```

3. `source_id=721` `id=721` `direction=None` `sentence_bleu=5.11` `sentence_wer=0.812`

Input:
```text
<pt-br> - Oww, és adorável. Obrigado!
```
Gold:
```text
- Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Prediction:
```text
- Oww, você é adorável. Obrigado!
```

4. `source_id=665` `id=665` `direction=None` `sentence_bleu=6.27` `sentence_wer=0.857`

Input:
```text
<pt-br> Olhou para o vovô, que se riu.
```
Gold:
```text
Olho veio para vovô que dá risada.
```
Prediction:
```text
Ela olhou para o avô, que sorriu.
```

5. `source_id=799` `id=799` `direction=None` `sentence_bleu=6.27` `sentence_wer=1.000`

Input:
```text
<pt-br> É só para me respeitares, pequena menina.
```
Gold:
```text
E so para me respeita garotinha.
```
Prediction:
```text
Só para você me respeitar, pequena garota.
```

### Plain translation-only, 4B GPT-Wiki r48 lineage

Chosen model: `4B Two-Staged LoRA r48 - Golden Collection (GPT-Wiki Stage B)`

Chosen because Stage B BLEU 88.19 / WER 0.086149 beat the Stage C counterpart in the recovered table.

Predictions file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r48_stageB_gpt_wiki/20260401_101835_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r48_stageB_gpt_wiki/20260401_101835_translation_summary.json`

1. `source_id=721` `id=721` `direction=None` `sentence_bleu=5.11` `sentence_wer=0.812`

Input:
```text
<pt-br> - Oww, és adorável. Obrigado!
```
Gold:
```text
- Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Prediction:
```text
- Oww, você é adorável. Obrigado!
```

2. `source_id=720` `id=720` `direction=None` `sentence_bleu=5.33` `sentence_wer=2.800`

Input:
```text
<br-pt> - Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Gold:
```text
- Oww, és adorável. Obrigado!
```
Prediction:
```text
- Oww tu és um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```

3. `source_id=410` `id=410` `direction=None` `sentence_bleu=6.48` `sentence_wer=0.857`

Input:
```text
<br-pt> disse Raimundo consigo, respirando.
```
Gold:
```text
Disse Raimundo para si mesmo, a respirar.
```
Prediction:
```text
disse Raimundo consigo, respirando.
```

4. `source_id=665` `id=665` `direction=None` `sentence_bleu=6.74` `sentence_wer=0.857`

Input:
```text
<pt-br> Olhou para o vovô, que se riu.
```
Gold:
```text
Olho veio para vovô que dá risada.
```
Prediction:
```text
Olhou para o vovô, que se riu.
```

5. `source_id=664` `id=664` `direction=None` `sentence_bleu=6.89` `sentence_wer=0.857`

Input:
```text
<br-pt> Olho veio para vovô que dá risada.
```
Gold:
```text
Olhou para o vovô, que se riu.
```
Prediction:
```text
Olho veio para vovô que dá risada.
```

### With cls, 270M GPT-Wiki+FRMT

Chosen model: `270M Two-Staged Full-FT - Golden Collection (GPT-Wiki+FRMT Stage B with cls)`

Only Stage B row in the recovered table for this family.

Predictions file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_with_cls/20260422_182519_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_with_cls/20260422_182519_translation_summary.json`

1. `source_id=721` `id=721` `direction=None` `sentence_bleu=2.21` `sentence_wer=0.875`

Input:
```text
<pt-br> - Oww, és adorável. Obrigado!
```
Gold:
```text
- Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Prediction:
```text
- Oww, é adorável.
```

2. `source_id=720` `id=720` `direction=None` `sentence_bleu=3.04` `sentence_wer=3.000`

Input:
```text
<br-pt> - Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Gold:
```text
- Oww, és adorável. Obrigado!
```
Prediction:
```text
-Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```

3. `source_id=798` `id=798` `direction=None` `sentence_bleu=4.92` `sentence_wer=1.000`

Input:
```text
<br-pt> E so para me respeita garotinha.
```
Gold:
```text
É só para me respeitares, pequena menina.
```
Prediction:
```text
E eu sou uma rapariga.
```

4. `source_id=665` `id=665` `direction=None` `sentence_bleu=6.74` `sentence_wer=0.857`

Input:
```text
<pt-br> Olhou para o vovô, que se riu.
```
Gold:
```text
Olho veio para vovô que dá risada.
```
Prediction:
```text
Olhou para o vovô, que se riu.
```

5. `source_id=664` `id=664` `direction=None` `sentence_bleu=6.89` `sentence_wer=0.857`

Input:
```text
<br-pt> Olho veio para vovô que dá risada.
```
Gold:
```text
Olhou para o vovô, que se riu.
```
Prediction:
```text
Olho veio para vovô que dá risada.
```

### Label-first translation-only, 4B GPT-Wiki+FRMT

Chosen model: `4B Two-Staged LoRA r48 - Golden Collection (GPT-Wiki+FRMT Stage B label-first translation-only)`

Only row in the recovered table for this family.

Predictions file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r48_stageB_gpt_wiki_frmt_label_first/20260413_115428_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r48_stageB_gpt_wiki_frmt_label_first/20260413_115428_translation_summary.json`

1. `source_id=720` `id=720` `direction=None` `sentence_bleu=3.42` `sentence_wer=3.250`

Input:
```text
- Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Gold:
```text
Oww, és adorável. Obrigado!
```
Prediction:
```text
Oww tu és um amor, obrigada! - Digo e ele retribui com um abraço.
```

2. `source_id=721` `id=721` `direction=None` `sentence_bleu=3.76` `sentence_wer=0.867`

Input:
```text
- Oww, és adorável. Obrigado!
```
Gold:
```text
Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Prediction:
```text
Oh, você é adorável. Obrigada!
```

3. `source_id=777` `id=777` `direction=None` `sentence_bleu=4.94` `sentence_wer=0.739`

Input:
```text
Ate ele ter a certeza que a estava apta para ser a parceira dele e que não iria sair a correr como uma gata assustada.
```
Gold:
```text
Ate ele ter a certeza que a estava apta para ser sua parceira e que não iria sair correndo feito uma gata assustada.
```
Prediction:
```text
Até que ele tivesse certeza de que ela estava pronta para ser a parceira dele e não fugiria como um gato assustado.
```

4. `source_id=799` `id=799` `direction=None` `sentence_bleu=5.52` `sentence_wer=1.000`

Input:
```text
É só para me respeitares, pequena menina.
```
Gold:
```text
E so para me respeita garotinha.
```
Prediction:
```text
É só para você me respeitar, pequena garota.
```

5. `source_id=665` `id=665` `direction=None` `sentence_bleu=6.27` `sentence_wer=0.857`

Input:
```text
Olhou para o vovô, que se riu.
```
Gold:
```text
Olho veio para vovô que dá risada.
```
Prediction:
```text
Ele olhou para o avô, que riu-se.
```

### Label-first + cls, 270M GPT-Wiki+FRMT

Chosen model: `270M Two-Staged Full-FT - Golden Collection (GPT-Wiki+FRMT Stage B label-first with cls)`

Only Stage B row in the recovered table for this family.

Predictions file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_label_first_with_cls/20260422_182519_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_label_first_with_cls/20260422_182519_translation_summary.json`

1. `source_id=721` `id=721` `direction=None` `sentence_bleu=1.22` `sentence_wer=0.938`

Input:
```text
<pt-br> - Oww, és adorável. Obrigado!
```
Gold:
```text
- Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Prediction:
```text
Oww, ele é adorável.
```

2. `source_id=720` `id=720` `direction=None` `sentence_bleu=2.86` `sentence_wer=3.600`

Input:
```text
<br-pt> - Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Gold:
```text
- Oww, és adorável. Obrigado!
```
Prediction:
```text
Oww és um amor Obrigado viu! Eu falei com o mesmo e o mesmo retribuiu com um abraço.
```

3. `source_id=523` `id=523` `direction=None` `sentence_bleu=3.93` `sentence_wer=0.889`

Input:
```text
<pt-br> Era até estimado, pelo seu génio mulherengo e prestável.
```
Gold:
```text
Era até estimado, pelo seu gênio folgazão e prestativo.
```
Prediction:
```text
<pt-br> Ele era até considerado, por seu génio mulherengo e prestável.
```

4. `source_id=522` `id=522` `direction=None` `sentence_bleu=4.37` `sentence_wer=0.889`

Input:
```text
<br-pt> Era até estimado, pelo seu gênio folgazão e prestativo.
```
Gold:
```text
Era até estimado, pelo seu génio mulherengo e prestável.
```
Prediction:
```text
<br-pt> Ele foi estimado por seu gênio folgazão e prestativo.
```

5. `source_id=665` `id=665` `direction=None` `sentence_bleu=5.52` `sentence_wer=0.857`

Input:
```text
<pt-br> Olhou para o vovô, que se riu.
```
Gold:
```text
Olho veio para vovô que dá risada.
```
Prediction:
```text
Ele olhou para o avô, que se riu.
```

### Label-first + cls, 270M, PtBrVId Stage A

Chosen model: `270M Stage C Dr. GRPO - Golden Collection (GPT-Wiki+FRMT Stage B label-first with cls PtBrVId Stage A BLEU+WER+first-token Reward, gold candidate)`

Chosen because Stage C BLEU 66.29 / WER 0.268006 beat Stage B in the recovered table.

Predictions file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageC_drgrpo_gpt_wiki_frmt_label_first_with_cls_from_stageA_opensubs_plus_ptbrvarid_bleu_wer_first_token_gold/20260426_123354_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageC_drgrpo_gpt_wiki_frmt_label_first_with_cls_from_stageA_opensubs_plus_ptbrvarid_bleu_wer_first_token_gold/20260426_123354_translation_summary.json`

1. `source_id=721` `id=721` `direction=None` `sentence_bleu=2.21` `sentence_wer=0.875`

Input:
```text
<pt-br> - Oww, és adorável. Obrigado!
```
Gold:
```text
- Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Prediction:
```text
Oww, você é adorável.
```

2. `source_id=410` `id=410` `direction=None` `sentence_bleu=4.02` `sentence_wer=1.286`

Input:
```text
<br-pt> disse Raimundo consigo, respirando.
```
Gold:
```text
Disse Raimundo para si mesmo, a respirar.
```
Prediction:
```text
<br-pt> disse Raimundo, com a sua voz, que respirava.
```

3. `source_id=665` `id=665` `direction=None` `sentence_bleu=4.46` `sentence_wer=0.857`

Input:
```text
<pt-br> Olhou para o vovô, que se riu.
```
Gold:
```text
Olho veio para vovô que dá risada.
```
Prediction:
```text
<pt-br> Olhou para o avô, que se riu.
```

4. `source_id=720` `id=720` `direction=None` `sentence_bleu=4.78` `sentence_wer=2.800`

Input:
```text
<br-pt> - Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Gold:
```text
- Oww, és adorável. Obrigado!
```
Prediction:
```text
<br-pt> - Oww és um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```

5. `source_id=664` `id=664` `direction=None` `sentence_bleu=4.93` `sentence_wer=1.000`

Input:
```text
<br-pt> Olho veio para vovô que dá risada.
```
Gold:
```text
Olhou para o vovô, que se riu.
```
Prediction:
```text
<br-pt> Olho veio para avô que dá risada.
```

### Label-first + cls equal, 4B

Chosen model: `4B Stage C Dr. GRPO - Golden Collection (GPT-Wiki+FRMT Stage B label-first with cls equal BLEU+WER+first-token Reward, gold candidate)`

Chosen because Stage C BLEU 79.79 / WER 0.129599 beat Stage B in the recovered table.

Predictions file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r48_stageC_drgrpo_gpt_wiki_frmt_label_first_with_cls_equal_legit_bleu_wer_first_token_gold_g09/20260507_120525_translation_predictions.jsonl`

Summary file: `eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r48_stageC_drgrpo_gpt_wiki_frmt_label_first_with_cls_equal_legit_bleu_wer_first_token_gold_g09/20260507_120525_translation_summary.json`

1. `source_id=361` `id=721` `direction=pt2br` `sentence_bleu=3.68` `sentence_wer=0.875`

Input:
```text
<pt-br> - Oww, és adorável. Obrigado!
```
Gold:
```text
- Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Prediction:
```text
Oww, você é adorável. Obrigado!
```

2. `source_id=361` `id=720` `direction=br2pt` `sentence_bleu=4.55` `sentence_wer=3.000`

Input:
```text
<br-pt> - Oww você é um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```
Gold:
```text
- Oww, és adorável. Obrigado!
```
Prediction:
```text
<br-pt> - Oww tu és um amor Obrigado viu!- Falei e o mesmo retribuiu com um abraço.
```

3. `source_id=333` `id=664` `direction=br2pt` `sentence_bleu=5.30` `sentence_wer=1.000`

Input:
```text
<br-pt> Olho veio para vovô que dá risada.
```
Gold:
```text
Olhou para o vovô, que se riu.
```
Prediction:
```text
<br-pt> Olho veio para vovô que dá risada.
```

4. `source_id=326` `id=650` `direction=br2pt` `sentence_bleu=5.52` `sentence_wer=1.000`

Input:
```text
<br-pt> ー Lembrei e Charlie riu.
```
Gold:
```text
ー Lembrei-me e Charlie riu-se.
```
Prediction:
```text
<br-pt> - Lembrei e o Charlie riu.
```

5. `source_id=305` `id=608` `direction=br2pt` `sentence_bleu=5.67` `sentence_wer=1.000`

Input:
```text
<br-pt> Comente e vote se gostou.
```
Gold:
```text
Comenta e vota que gostaste.
```
Prediction:
```text
<br-pt> Comente e vote se gostou.
```
