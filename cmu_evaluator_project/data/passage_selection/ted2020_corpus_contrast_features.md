# TED2020 corpus-driven BP/PT contrast features

> **Interpretation boundary:** these are source-selection signals. They are not annotation errors, required rewrites, unique EP replacements, candidates, or gold references.

## 1. Corpus basis

- BP: `data/external/TED2020/TED2020/raw/pt_br` — 3,968 documents and 457,273 sentences.
- PT: `data/external/TED2020/TED2020/raw/pt` — 3,246 documents and 367,680 sentences.
- Primary comparison: 3,217 matched TED talks; 751 BP-only and 29 PT-only documents.

## 2. BP/PT overlap and alignment quality

- Exact sentence-ID overlap: 303,025/416,375 (Jaccard 0.728).
- Common documents with segmentation/ID differences: 3,007.
- High-confidence aligned pairs retained: 255,934; rejected/anomalous ID matches: 47,091.
- Shared IDs are used only for recurrent association evidence; no forced alignment is performed.

## 3. NLP pipeline and Stanza model

- Stanza `1.11.1`; Portuguese **Bosque** package.
- Processors: tokenize, mwt, pos, lemma.
- Models: `{"tokenize": "bosque", "mwt": "bosque", "pos": "bosque_nocharlm", "lemma": "bosque_nocharlm"}`.
- Every original TED sentence is passed as one no-split unit; its original ID and exact text remain authoritative.
- Deviation from Alves: Official Stanza Bosque no-charlm POS/lemma models were used for feasible full-corpus CPU analysis; dependency and word-order analyses were not required and were omitted.

## 4. KLD methodology

Methodological basis: Diego Alves (2025), *Information Theory and Linguistic Variation: A Study of Brazilian and European Portuguese*, VarDial 2025 ([ACL Anthology](https://aclanthology.org/2025.vardial-1.2/)). This experiment adapts the analysis for source mining rather than reproducing the paper's FRMT study.

For every feature view, both `D_KL(BP || PT)` and `D_KL(PT || BP)` are computed in bits. Jelinek–Mercer smoothing uses λ=0.05. The background distribution is the pooled matched-talk distribution, which gives non-zero support while retaining corpus composition.

| View | D_KL(BP || PT) | D_KL(PT || BP) | BP events | PT events |
|---|---:|---:|---:|---:|
| surface_form | 0.20975051 | 0.21393427 | 5,167,105 | 5,143,924 |
| lemma | 0.13896955 | 0.13699023 | 5,453,958 | 5,502,068 |
| pos_trigram | 0.04101765 | 0.04457364 | 4,788,507 | 4,819,203 |
| word_bigram | 1.15374651 | 1.12043967 | 4,812,354 | 4,779,645 |
| word_trigram | 2.76173281 | 2.72601718 | 4,465,181 | 4,423,567 |

## 5. Smoothing and significance

Feature-level Pearson χ² tests use `p < 0.001`. Significance is necessary but not sufficient: minimum count, document spread, effect direction, pointwise KLD, weighted log-odds, and stability are also retained.

## 6. Cross-document stability

Matched documents are assigned deterministically to 5 folds by SHA-256 of document ID. Direction is recomputed in five leave-one-fold-out replications; passage-ranking features must preserve direction in at least four.

## 7. BP-characteristic surface forms

### `você`

- BP 29,542 (5717.32/M), 2,558 documents; PT 724 (140.75/M), 298 documents.
- Pointwise KLD: 0.0240636291 bits; weighted log-odds z: 98.637; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0167.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **yes**.
  - `ted2020-1:49` — A cada maré que vem e vai, você encontra mais conchas.
  - `ted2020-1:54` — e então eu vou mostrar novos achados sobre o que você pode fazer.

### `e`

- BP 172,879 (33457.61/M), 3,207 documents; PT 140,934 (27398.15/M), 3,206 documents.
- Pointwise KLD: 0.0091176861 bits; weighted log-odds z: 57.415; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0018.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Eu fui muito aplaudido por esta conferência e quero agradecer a todos pelos muitos comentários delicados sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:4` — E quero dizer sinceramente, porque eu preciso (cara de brincadeira) eu preciso disso (risadas)

### `eu`

- BP 43,162 (8353.23/M), 2,619 documents; PT 26,252 (5103.50/M), 2,595 documents.
- Pointwise KLD: 0.0055749658 bits; weighted log-odds z: 63.370; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0107.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Eu fui muito aplaudido por esta conferência e quero agradecer a todos pelos muitos comentários delicados sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:4` — E quero dizer sinceramente, porque eu preciso (cara de brincadeira) eu preciso disso (risadas)

### `então`

- BP 21,297 (4121.65/M), 2,846 documents; PT 7,919 (1539.49/M), 2,209 documents.
- Pointwise KLD: 0.0054312594 bits; weighted log-odds z: 75.019; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0078.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:33.2` — E então lembrei que poderia haver um punhado de coisas.
  - `ted2020-1:54` — e então eu vou mostrar novos achados sobre o que você pode fazer.

### `eles`

- BP 18,289 (3539.51/M), 2,671 documents; PT 8,085 (1571.76/M), 2,133 documents.
- Pointwise KLD: 0.0038620239 bits; weighted log-odds z: 60.931; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0136.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:28.1` — E eu contei a eles, quase a mesma coisa que eu compartilhei com vocês.
  - `ted2020-1:28.2` — Tipper e eu dirigindo sozinhos, o Shoney, rede de restaurantes familiares barateiros, o que o homem disse - e eles riram.

### `ele`

- BP 18,439 (3568.54/M), 2,581 documents; PT 9,306 (1809.12/M), 2,109 documents.
- Pointwise KLD: 0.0032689515 bits; weighted log-odds z: 53.561; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0122.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:24` — E o homem disse "Ele desceu bastante, não foi?"
  - `ted2020-1:32.1` — Ele agitava um pedaço de papel e gritava, Ligue para Washington!

### `isso`

- BP 30,037 (5813.12/M), 2,959 documents; PT 20,070 (3901.69/M), 2,912 documents.
- Pointwise KLD: 0.0031463769 bits; weighted log-odds z: 43.943; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0040.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:8` — (risadas e aplausos) Eu vou contar uma rápida história para mostrar o que isso significa para mim.
  - `ted2020-1:35.1` — Mas não era isso - minha equipe ficou muito preocupada porque uma das agências nigerianas já havia publicado uma história sobre o meu discurso.

### `em`

- BP 63,301 (12250.77/M), 3,197 documents; PT 52,955 (10294.67/M), 3,188 documents.
- Pointwise KLD: 0.0029085207 bits; weighted log-odds z: 29.876; χ² p: 2e-194.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0014.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:10` — Logo depois que Tipper e eu deixamos (suspiro) a Casa Branca (risadas) estávamos na estrada de nossa casa em Nashville para uma pequena fazenda que temos 80 quilômetros ao leste de Nashville --
  - `ted2020-1:26` — No dia seguinte, continuando a nossa história realmente verdadeira, Eu entrei num G-5 para chegar à África e fazer um discurso na Nigéria, em Lagos, sobre energia.

### `nós`

- BP 17,938 (3471.58/M), 2,481 documents; PT 9,873 (1919.35/M), 2,322 documents.
- Pointwise KLD: 0.0027800972 bits; weighted log-odds z: 47.415; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0323.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:13` — Não havia uma fila de carros atrás de nós.
  - `ted2020-1:39` — (risadas) Nós gostamos de celebrar os nossos sucessos nesta vida.

### `vocês`

- BP 12,204 (2361.86/M), 2,177 documents; PT 5,270 (1024.51/M), 1,658 documents.
- Pointwise KLD: 0.0026499302 bits; weighted log-odds z: 50.752; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0082.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **yes**.
  - `ted2020-1:12` — Eu sei que parece uma pequena coisa para vocês, mas...(risadas) Eu olhei no retrovisor e de repente percebi.
  - `ted2020-1:28.1` — E eu contei a eles, quase a mesma coisa que eu compartilhei com vocês.

### `fato`

- BP 3,388 (655.69/M), 1,539 documents; PT 146 (28.38/M), 76 documents.
- Pointwise KLD: 0.0024700907 bits; weighted log-odds z: 37.177; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0888.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1004:17` — E se olharmos para nossas vidas, de fato fomos nós que pegamos a melhor parte da carne.
  - `ted2020-1004:39` — De fato, esta foi a única guerra na história americana em que o governo negociou a paz concedendo tudo que o inimigo demandou.

### `essa`

- BP 10,109 (1956.41/M), 2,503 documents; PT 4,111 (799.20/M), 1,853 documents.
- Pointwise KLD: 0.0023491670 bits; weighted log-odds z: 48.459; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0187.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-10:8.3` — O que realmente está me preocupando é que a diabetes aumentou em 70 por cento nos últimos 10 anos, e essa pode ser a primeira geração na qual nossas crianças vivem uma expectativa de vida menor do que nós.
  - `ted2020-1000:41` — Então, na virada do último século, comecei a pensar, quão maravilhoso seria se alguém pudesse pegar essa lógica e colocá-la de ponta-cabeça.

### `me`

- BP 12,626 (2443.53/M), 2,179 documents; PT 6,511 (1265.77/M), 1,835 documents.
- Pointwise KLD: 0.0021685406 bits; weighted log-odds z: 43.188; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0083.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:37.2` — "O ex-vice presidente Al Gore anunciou ontem na Nigeria "Eu e minha esposa Tipper abrimos um restaurante familiar barateiro, chamado Shoney, e eles estão tocando o negócio eles mesmos". (risadas) Antes de eu chegar a sol
  - `ted2020-1:83` — Aqueles de vocês que são bons em construir marcas, Eu adoraria ter seu aconselhamento e ajuda em como dizer isso de forma a me conectar com mais pessoas.

### `esse`

- BP 8,944 (1730.95/M), 2,362 documents; PT 3,565 (693.05/M), 1,724 documents.
- Pointwise KLD: 0.0021239024 bits; weighted log-odds z: 46.263; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0233.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1001:13.2` — Eu não sei se é necessário demonstrar isso -- (Risos) -- mas o meu ponto é que esse é o som da auto-destruição.
  - `ted2020-1001:30` — esse é o ponto.

### `pra`

- BP 2,289 (442.99/M), 751 documents; PT 22 (4.28/M), 16 documents.
- Pointwise KLD: 0.0020835587 bits; weighted log-odds z: 21.801; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0602.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-101:10` — (Música) ♫ Atada no vento eu chamava por você ♫ ♫ mas você não ouvia ...♫ ♫ E você é uma planta que precisa solo pobre♫ ♫ e eu o tratei bem demais ♫ ♫ para desistir de flores ... ♫ ♫ Ah, eu fui fértil demais para você ..
  - `ted2020-10173:6` — E estou aqui pra dizer que isso acontece pois o capitalismo, em sua versão moderna, é grandemente mal-compreendido.

### `ela`

- BP 11,723 (2268.78/M), 2,296 documents; PT 5,934 (1153.59/M), 1,704 documents.
- Pointwise KLD: 0.0020694175 bits; weighted log-odds z: 42.522; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0077.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:22` — (risadas) Ela anotou nosso pedido e foi atender um casal no outro reservado. e ela baixou tanto a voz que eu realmente tive que fazer força para escutar o que disse.
  - `ted2020-1:23` — Ela disse "Sim, é o ex-vice presidente Al Gore e sua esposa Tipper".

### `se`

- BP 44,902 (8689.97/M), 3,178 documents; PT 38,102 (7407.19/M), 3,158 documents.
- Pointwise KLD: 0.0018948620 bits; weighted log-odds z: 23.116; χ² p: 1.54e-117.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0050.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:61` — Estes são investimentos que se pagam.
  - `ted2020-1:70` — Captura e sequestro de carbono -- é o que CCS significa -- e deve se transformar numa aplicação matadora que nos permitirá continuar a usar os combustíveis fósseis de uma forma segura.

### `bem`

- BP 11,566 (2238.39/M), 2,557 documents; PT 6,238 (1212.69/M), 2,149 documents.
- Pointwise KLD: 0.0018530208 bits; weighted log-odds z: 39.081; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0152.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:108` — Escolha bem.
  - `ted2020-10:9.4` — Bem, você sabe, a dieta que nós encontramos para reverter doenças cardíacas e o câncer é a dieta da Ásia.

### `elas`

- BP 7,362 (1424.78/M), 2,041 documents; PT 2,913 (566.30/M), 1,357 documents.
- Pointwise KLD: 0.0017618917 bits; weighted log-odds z: 42.189; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0190.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:109` — E eu vou pessoalmente treinar pessoas a fazer esta apresentação, re-colocada com algumas histórias pessoais trocadas por uma abordagem genérica, e -- não são apenas as imagens, é o que elas significam.
  - `ted2020-1:110` — E é como elas se relacionam.

### `um`

- BP 76,094 (14726.62/M), 3,200 documents; PT 69,539 (13518.67/M), 3,198 documents.
- Pointwise KLD: 0.0017237850 bits; weighted log-odds z: 16.547; χ² p: 1.07e-60.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0030.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:16` — Era hora do jantar, e começamos a procurar um lugar para comer
  - `ted2020-1:19` — Assim que entramos na saída e começamos a procurar, encontramos um Shoney.

### `fazendo`

- BP 2,773 (536.66/M), 1,454 documents; PT 310 (60.27/M), 283 documents.
- Pointwise KLD: 0.0015017154 bits; weighted log-odds z: 36.523; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0254.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-10:6` — Nós mostramos alguns meses atrás -- nós publicamos o primeiro estudo mostrando que você pode mesmo parar ou reverter a progressão de câncer de próstata fazendo mudanças na dieta e estilo de vida, e 70 por cento de regres
  - `ted2020-1000:12` — Mesmo que pudéssemos medir o que cada célula está fazendo o tempo todo, ainda teríamos que atribuir sentido aos padrões de atividade registrados, e isto é muito difícil, provavelmente entenderemos estes padrões tão mal q

### `realmente`

- BP 6,224 (1204.54/M), 1,863 documents; PT 2,485 (483.09/M), 1,201 documents.
- Pointwise KLD: 0.0014753467 bits; weighted log-odds z: 38.531; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0292.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de estar neste palco pela segunda vez.
  - `ted2020-1:22` — (risadas) Ela anotou nosso pedido e foi atender um casal no outro reservado. e ela baixou tanto a voz que eu realmente tive que fazer força para escutar o que disse.

### `sobre`

- BP 15,133 (2928.72/M), 2,739 documents; PT 10,448 (2031.13/M), 2,488 documents.
- Pointwise KLD: 0.0014561478 bits; weighted log-odds z: 28.842; χ² p: 1.64e-184.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0054.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Eu fui muito aplaudido por esta conferência e quero agradecer a todos pelos muitos comentários delicados sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:26` — No dia seguinte, continuando a nossa história realmente verdadeira, Eu entrei num G-5 para chegar à África e fazer um discurso na Nigéria, em Lagos, sobre energia.

### `seu`

- BP 10,095 (1953.71/M), 2,604 documents; PT 5,953 (1157.29/M), 2,231 documents.
- Pointwise KLD: 0.0013847375 bits; weighted log-odds z: 32.092; χ² p: 8.29e-231.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0091.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:83` — Aqueles de vocês que são bons em construir marcas, Eu adoraria ter seu aconselhamento e ajuda em como dizer isso de forma a me conectar com mais pessoas.
  - `ted2020-1:93.2` — Você já pensou em neutralizar o seu negócio?

### `essas`

- BP 6,030 (1167.00/M), 2,109 documents; PT 2,587 (502.92/M), 1,419 documents.
- Pointwise KLD: 0.0013193257 bits; weighted log-odds z: 35.841; χ² p: 5.16e-298.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0145.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-10:5.2` — E nos últimos quase 29 anos, fomos capazes de mostrar que pela simples mudança de dieta e estilo de vida, usando essas medidas tecnológicas, caras, avançadas para provar o quão poderosas essas intervenções simples e bási
  - `ted2020-1001:31` — Então, eu venho trabalhando nessa música que menciona essas coisas e que também imagina uma pessoa que foi tão bem sucedida em defender-se do desgosto que eles deixaram de fazer a escritura por si mesmos, se isso for pos

## 8. PT-characteristic surface forms

### `a`

- BP 142,430 (27564.76/M), 3,204 documents; PT 204,554 (39766.14/M), 3,207 documents.
- Pointwise KLD: 0.0197988515 bits; weighted log-odds z: -109.852; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0026.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.

### `os`

- BP 39,034 (7554.33/M), 3,143 documents; PT 55,924 (10871.86/M), 3,182 documents.
- Pointwise KLD: 0.0053774534 bits; weighted log-odds z: -55.706; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0027.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:7` — (Risos) Agora tenho que tirar os sapatos ou as botas para entrar num avião!

### `o`

- BP 120,643 (23348.28/M), 3,205 documents; PT 134,359 (26119.94/M), 3,206 documents.
- Pointwise KLD: 0.0040047189 bits; weighted log-odds z: -28.999; χ² p: 1.45e-180.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0014.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:22.1` — (Risos) Ela anotou o nosso pedido e foi atender um casal numa mesa próxima.

### `isto`

- BP 6,086 (1177.84/M), 1,237 documents; PT 14,914 (2899.34/M), 2,594 documents.
- Pointwise KLD: 0.0035025750 bits; weighted log-odds z: -59.334; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0137.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:51` — Isto é válido só para os EUA.
  - `ted2020-1:84` — Gostaria que os que são bons em construir marcas me dessem o vosso conselho e ajuda para dizer isto de forma a abranger o maior número de pessoas.

### `as`

- BP 35,067 (6786.59/M), 3,129 documents; PT 45,857 (8914.79/M), 3,171 documents.
- Pointwise KLD: 0.0033108052 bits; weighted log-odds z: -38.752; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0047.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:7` — (Risos) Agora tenho que tirar os sapatos ou as botas para entrar num avião!
  - `ted2020-1:57` — A eficiência no uso final da eletricidade e de todas as formas de energia é a mais baixa.

### `que`

- BP 180,901 (35010.13/M), 3,202 documents; PT 191,463 (37221.20/M), 3,204 documents.
- Pointwise KLD: 0.0031194082 bits; weighted log-odds z: -19.377; χ² p: 1.01e-80.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0005.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:7` — (Risos) Agora tenho que tirar os sapatos ou as botas para entrar num avião!

### `facto`

- BP 0 (0.00/M), 0 documents; PT 2,481 (482.32/M), 1,284 documents.
- Pointwise KLD: 0.0024868509 bits; weighted log-odds z: -6.020; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0004.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:80` — Agora podemos, de facto, interferir com a sua psicologia de modo verdadeiramente profundo. como vou mostrar no meu último exemplo, que é direcionado para uma questão familiar.
  - `ted2020-1000:120` — De facto, faz tanto sentido que podemos construir um circuito eletrónico de brinquedos que simule o comportamento da mosca.

### `vos`

- BP 5 (0.97/M), 5 documents; PT 2,299 (446.94/M), 1,238 documents.
- Pointwise KLD: 0.0022535100 bits; weighted log-odds z: -14.150; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0249.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:12.1` — (Risos) Eu sei que vos parece uma coisa sem importância, mas...
  - `ted2020-1:28.1` — Contei-lhes praticamente a mesma história que acabei de vos contar.

### `à`

- BP 9,474 (1833.52/M), 2,660 documents; PT 15,973 (3105.22/M), 3,001 documents.
- Pointwise KLD: 0.0022141976 bits; weighted log-odds z: -40.725; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0062.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:18` — Chegámos à saída 238, Lebanon, Tennessee.
  - `ted2020-1:43.2` — O que podemos fazer em relação à crise climática?

### `esta`

- BP 6,653 (1287.57/M), 1,846 documents; PT 12,340 (2398.95/M), 2,700 documents.
- Pointwise KLD: 0.0020158505 bits; weighted log-odds z: -40.984; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0054.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:56` — Antes de mais, esta era a projeção da contribuição dos EUA para o aquecimento global, com os negócios como sempre.

### `num`

- BP 2,661 (514.99/M), 1,323 documents; PT 7,064 (1373.27/M), 2,495 documents.
- Pointwise KLD: 0.0018025395 bits; weighted log-odds z: -43.157; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0098.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:7` — (Risos) Agora tenho que tirar os sapatos ou as botas para entrar num avião!
  - `ted2020-1:15` — (Risos) Estávamos num Ford Taurus alugado.

### `numa`

- BP 2,403 (465.06/M), 1,293 documents; PT 6,450 (1253.91/M), 2,413 documents.
- Pointwise KLD: 0.0016639200 bits; weighted log-odds z: -41.532; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0176.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:21` — Entrámos, sentámo-nos numa mesa e veio uma empregada, que ficou encantada quando viu Tipper.
  - `ted2020-1:22.1` — (Risos) Ela anotou o nosso pedido e foi atender um casal numa mesa próxima.

### `vossa`

- BP 4 (0.77/M), 4 documents; PT 1,655 (321.74/M), 850 documents.
- Pointwise KLD: 0.0016183522 bits; weighted log-odds z: -12.526; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0316.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:94` — Podem pensar nesta irritante voz interior como uma espécie de equivalente cerebral da Igreja Católica, se forem austríacos como eu, ou do superego, se forem freudianos, ou da vossa mãe, se forem Judeus.
  - `ted2020-1001:26` — Desculpem-me se coloquei esta ideia na vossa cabeça

### `este`

- BP 7,092 (1372.53/M), 1,869 documents; PT 11,771 (2288.33/M), 2,648 documents.
- Pointwise KLD: 0.0015837432 bits; weighted log-odds z: -34.066; χ² p: 1.89e-259.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0148.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:90` — Este "site" tem uma calculadora de carbono.

### `há`

- BP 11,212 (2169.88/M), 2,582 documents; PT 16,112 (3132.24/M), 2,860 documents.
- Pointwise KLD: 0.0015620103 bits; weighted log-odds z: -29.925; χ² p: 1.9e-198.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0097.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:64` — É um alvo fácil e visível de preocupação e é assim que deve ser, mas há mais poluição a causar aquecimento proveniente dos edifícios do que de carros e camiões.
  - `ted2020-1:69` — Com Vinod, John Doerr e outros, muitos de vocês aqui, — há muitas pessoas diretamente envolvidas nisso — esta parcela vai crescer muito mais rapidamente do que as projeções preveem.

### `estas`

- BP 1,815 (351.26/M), 795 documents; PT 5,268 (1024.12/M), 2,044 documents.
- Pointwise KLD: 0.0014630347 bits; weighted log-odds z: -39.338; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0118.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:66` — Vocês conhecem estas células porque são elas que vos deixam frustrados quando tentam esmagar uma mosca.
  - `ted2020-1000:78` — Há hoje centenas de laboratórios a utilizar estas abordagens.

### `vosso`

- BP 2 (0.39/M), 2 documents; PT 1,387 (269.64/M), 751 documents.
- Pointwise KLD: 0.0013696295 bits; weighted log-odds z: -9.990; χ² p: 1.13e-303.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0248.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:84` — Gostaria que os que são bons em construir marcas me dessem o vosso conselho e ajuda para dizer isto de forma a abranger o maior número de pessoas.
  - `ted2020-1:94.2` — Já pensaram em tornar o vosso negócio neutro em carbono?

### `duma`

- BP 17 (3.29/M), 14 documents; PT 1,450 (281.89/M), 827 documents.
- Pointwise KLD: 0.0013049069 bits; weighted log-odds z: -18.389; χ² p: 8.35e-308.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0721.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1002:105` — "Design como meio de ensino" significa que podemos ensinar design nas escolas públicas, em vez duma aprendizagem baseada no design, não como "vamos aprender física construindo um foguetão," mas sim aprender a pensar em d
  - `ted2020-1004:130` — A solução, como lhe chamam, pode ser muito mais difícil para a sociedade dominante do que, por assim dizer, um cheque de 50 dólares, ou uma excursão da igreja para pintar umas casas cobertas de grafitis, ou a oferta duma

### `é`

- BP 82,897 (16043.22/M), 3,159 documents; PT 87,278 (16967.20/M), 3,164 documents.
- Pointwise KLD: 0.0013003552 bits; weighted log-odds z: -11.740; χ² p: 2.46e-31.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0007.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:9` — É uma história verdadeira, cada pedaço desta história.

### `estes`

- BP 1,726 (334.04/M), 795 documents; PT 4,768 (926.92/M), 1,897 documents.
- Pointwise KLD: 0.0012646955 bits; weighted log-odds z: -36.351; χ² p: 5.23e-315.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0181.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:45` — São estes os diapositivos.
  - `ted2020-1:53` — Sei que querem mais más notícias sobre o meio ambiente — estou a brincar — mas estes são os diapositivos de revisão

### `porque`

- BP 12,115 (2344.64/M), 2,510 documents; PT 16,197 (3148.76/M), 2,762 documents.
- Pointwise KLD: 0.0012635490 bits; weighted log-odds z: -24.616; χ² p: 2.13e-134.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0062.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:4` — Digo-o sinceramente, em parte, porque... preciso mesmo disso.
  - `ted2020-1:35.2` — A minha equipa estava muito preocupada porque uma agência nigeriana já tinha publicado a notícia do meu discurso.

### `cancro`

- BP 0 (0.00/M), 0 documents; PT 1,235 (240.09/M), 285 documents.
- Pointwise KLD: 0.0012379125 bits; weighted log-odds z: -5.526; χ² p: 8.65e-272.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0004.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-10:6` — Publicámos há meses o primeiro estudo mostrando que é possível deter ou inverter a progressão do cancro da próstata, alterando a dieta e o estilo de vida e obter uma regressão de 70% no crescimento do tumor, ou a inibiçã
  - `ted2020-10:9.20` — Usamos uma dieta asiática que pode fazer regredir as doenças cardíacas e o cancro.

### `equipa`

- BP 0 (0.00/M), 0 documents; PT 1,205 (234.26/M), 619 documents.
- Pointwise KLD: 0.0012078417 bits; weighted log-odds z: -5.509; χ² p: 3.07e-265.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0004.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:35.2` — A minha equipa estava muito preocupada porque uma agência nigeriana já tinha publicado a notícia do meu discurso.
  - `ted2020-1002:85` — Depois enuncia um problema de matemática — por exemplo, quatro vezes quatro — e um aluno de cada equipa tem de competir para perceber que quatro vezes quatro é 16 e descobrir qual o pneu que tem o número 16, e sentar-se 

### `dos`

- BP 13,783 (2667.45/M), 2,905 documents; PT 17,732 (3447.17/M), 2,999 documents.
- Pointwise KLD: 0.0012040310 bits; weighted log-odds z: -22.651; χ² p: 7.28e-114.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0091.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:56` — Antes de mais, esta era a projeção da contribuição dos EUA para o aquecimento global, com os negócios como sempre.
  - `ted2020-1:64` — É um alvo fácil e visível de preocupação e é assim que deve ser, mas há mais poluição a causar aquecimento proveniente dos edifícios do que de carros e camiões.

### `da`

- BP 38,555 (7461.63/M), 3,168 documents; PT 42,670 (8295.22/M), 3,182 documents.
- Pointwise KLD: 0.0012009298 bits; weighted log-odds z: -15.192; χ² p: 8.94e-52.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0052.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:10.2` — (Risos) ... íamos da nossa casa em Nashville para uma pequena quinta que temos a 80 quilómetros a leste de Nashville
  - `ted2020-1:14` — (Risos) Já ouviram falar da dor do membro amputado?

## 9. BP-characteristic lemmas

### `você`

- BP 41,787 (7661.78/M), 2,869 documents; PT 5,994 (1089.41/M), 1,712 documents.
- Pointwise KLD: 0.0193482142 bits; weighted log-odds z: 141.706; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0151.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **yes**.
  - `ted2020-1:12` — Eu sei que parece uma pequena coisa para vocês, mas...(risadas) Eu olhei no retrovisor e de repente percebi.
  - `ted2020-1:28.1` — E eu contei a eles, quase a mesma coisa que eu compartilhei com vocês.

### `e`

- BP 172,926 (31706.51/M), 3,207 documents; PT 140,945 (25616.73/M), 3,206 documents.
- Pointwise KLD: 0.0092201799 bits; weighted log-odds z: 61.179; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0018.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Eu fui muito aplaudido por esta conferência e quero agradecer a todos pelos muitos comentários delicados sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:4` — E quero dizer sinceramente, porque eu preciso (cara de brincadeira) eu preciso disso (risadas)

### `esse`

- BP 40,243 (7378.68/M), 3,054 documents; PT 17,801 (3235.33/M), 2,852 documents.
- Pointwise KLD: 0.0081744681 bits; weighted log-odds z: 92.055; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0055.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-10:4` — Doenças de coração e vasos sanguíneos ainda matam mais pessoas -- não só nesse país, mas também no mundo -- que todo resto combinado, e ainda assim é completamente prevenível para quase todo mundo.
  - `ted2020-10:5.2` — E nos últimos quase 29 anos, fomos capazes de mostrar que pela simples mudança de dieta e estilo de vida, usando essas medidas tecnológicas, caras, avançadas para provar o quão poderosas essas intervenções simples e bási

### `então`

- BP 21,296 (3904.69/M), 2,846 documents; PT 7,919 (1439.28/M), 2,209 documents.
- Pointwise KLD: 0.0052130072 bits; weighted log-odds z: 76.014; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0079.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:33.2` — E então lembrei que poderia haver um punhado de coisas.
  - `ted2020-1:54` — e então eu vou mostrar novos achados sobre o que você pode fazer.

### `eu`

- BP 61,098 (11202.51/M), 2,699 documents; PT 46,174 (8392.12/M), 2,685 documents.
- Pointwise KLD: 0.0044040457 bits; weighted log-odds z: 47.302; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0086.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Eu fui muito aplaudido por esta conferência e quero agradecer a todos pelos muitos comentários delicados sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:4` — E quero dizer sinceramente, porque eu preciso (cara de brincadeira) eu preciso disso (risadas)

### `seu`

- BP 31,219 (5724.10/M), 3,090 documents; PT 19,805 (3599.56/M), 2,961 documents.
- Pointwise KLD: 0.0035991209 bits; weighted log-odds z: 51.298; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0057.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:23` — Ela disse "Sim, é o ex-vice presidente Al Gore e sua esposa Tipper".
  - `ted2020-1:67` — Outras formas de transporte e sua eficiência são tão importantes quanto os carros e caminhões!

### `isso`

- BP 34,503 (6326.23/M), 3,007 documents; PT 23,321 (4238.59/M), 2,966 documents.
- Pointwise KLD: 0.0034391745 bits; weighted log-odds z: 47.489; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0054.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:4` — E quero dizer sinceramente, porque eu preciso (cara de brincadeira) eu preciso disso (risadas)
  - `ted2020-1:8` — (risadas e aplausos) Eu vou contar uma rápida história para mostrar o que isso significa para mim.

### `eles`

- BP 25,497 (4674.95/M), 2,918 documents; PT 15,044 (2734.24/M), 2,667 documents.
- Pointwise KLD: 0.0033929988 bits; weighted log-odds z: 52.361; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0120.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:28.1` — E eu contei a eles, quase a mesma coisa que eu compartilhei com vocês.
  - `ted2020-1:28.2` — Tipper e eu dirigindo sozinhos, o Shoney, rede de restaurantes familiares barateiros, o que o homem disse - e eles riram.

### `ele`

- BP 32,168 (5898.10/M), 3,040 documents; PT 23,061 (4191.33/M), 2,931 documents.
- Pointwise KLD: 0.0027389162 bits; weighted log-odds z: 39.790; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0060.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:24` — E o homem disse "Ele desceu bastante, não foi?"
  - `ted2020-1:32.1` — Ele agitava um pedaço de papel e gritava, Ligue para Washington!

### `fato`

- BP 3,708 (679.87/M), 1,591 documents; PT 190 (34.53/M), 99 documents.
- Pointwise KLD: 0.0024644631 bits; weighted log-odds z: 40.088; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.1015.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1004:17` — E se olharmos para nossas vidas, de fato fomos nós que pegamos a melhor parte da carne.
  - `ted2020-1004:39` — De fato, esta foi a única guerra na história americana em que o governo negociou a paz concedendo tudo que o inimigo demandou.

### `pra`

- BP 2,286 (419.15/M), 751 documents; PT 22 (4.00/M), 16 documents.
- Pointwise KLD: 0.0019756937 bits; weighted log-odds z: 21.857; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0599.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-101:10` — (Música) ♫ Atada no vento eu chamava por você ♫ ♫ mas você não ouvia ...♫ ♫ E você é uma planta que precisa solo pobre♫ ♫ e eu o tratei bem demais ♫ ♫ para desistir de flores ... ♫ ♫ Ah, eu fui fértil demais para você ..
  - `ted2020-10173:6` — E estou aqui pra dizer que isso acontece pois o capitalismo, em sua versão moderna, é grandemente mal-compreendido.

### `bem`

- BP 11,714 (2147.80/M), 2,567 documents; PT 6,412 (1165.38/M), 2,176 documents.
- Pointwise KLD: 0.0017737345 bits; weighted log-odds z: 39.419; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0141.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:108` — Escolha bem.
  - `ted2020-10:9.4` — Bem, você sabe, a dieta que nós encontramos para reverter doenças cardíacas e o câncer é a dieta da Ásia.

### `ela`

- BP 18,926 (3470.14/M), 2,765 documents; PT 13,183 (2396.01/M), 2,564 documents.
- Pointwise KLD: 0.0017460151 bits; weighted log-odds z: 32.745; χ² p: 3.58e-237.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0085.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:22` — (risadas) Ela anotou nosso pedido e foi atender um casal no outro reservado. e ela baixou tanto a voz que eu realmente tive que fazer força para escutar o que disse.
  - `ted2020-1:23` — Ela disse "Sim, é o ex-vice presidente Al Gore e sua esposa Tipper".

### `bilhão`

- BP 1,718 (315.00/M), 709 documents; PT 12 (2.18/M), 7 documents.
- Pointwise KLD: 0.0015190198 bits; weighted log-odds z: 17.376; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0535.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1015:159` — E logo, nós engolimos aquela coisa aos montes, bilhões de galões.
  - `ted2020-1018:17` — Há uma estimativa, uma estimativa conservadora, há dois anos que a economia americana lucrou 57 bilhões de dólares por ano.

### `sobre`

- BP 15,132 (2774.50/M), 2,739 documents; PT 10,448 (1898.92/M), 2,488 documents.
- Pointwise KLD: 0.0014288450 bits; weighted log-odds z: 29.879; χ² p: 4.75e-198.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0055.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Eu fui muito aplaudido por esta conferência e quero agradecer a todos pelos muitos comentários delicados sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:26` — No dia seguinte, continuando a nossa história realmente verdadeira, Eu entrei num G-5 para chegar à África e fazer um discurso na Nigéria, em Lagos, sobre energia.

### `realmente`

- BP 6,224 (1141.19/M), 1,863 documents; PT 2,486 (451.83/M), 1,201 documents.
- Pointwise KLD: 0.0014170594 bits; weighted log-odds z: 39.078; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0294.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de estar neste palco pela segunda vez.
  - `ted2020-1:22` — (risadas) Ela anotou nosso pedido e foi atender um casal no outro reservado. e ela baixou tanto a voz que eu realmente tive que fazer força para escutar o que disse.

### `compartilhar`

- BP 1,607 (294.65/M), 870 documents; PT 15 (2.73/M), 11 documents.
- Pointwise KLD: 0.0013922571 bits; weighted log-odds z: 18.222; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.1013.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:28.1` — E eu contei a eles, quase a mesma coisa que eu compartilhei com vocês.
  - `ted2020-1002:18` — Talvez a maior coisa com que ele luta, como muitas comunidades semelhantes, é que não há investimento compartilhado e coletivo no futuro das comunidades rurais.

### `algo`

- BP 7,142 (1309.51/M), 2,211 documents; PT 3,600 (654.30/M), 1,369 documents.
- Pointwise KLD: 0.0012248382 bits; weighted log-odds z: 33.976; χ² p: 4.92e-263.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0213.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:91` — Ela tem que decidir entre voltar e continuar no mesmo odor, ou se vai continuar e cruzar a linha média e tentar algo novo.
  - `ted2020-1000:119` — Então, essa anatomia sugere que os corpos pedunculados têm algo a ver com a escolha da ação.

### `câncer`

- BP 1,316 (241.29/M), 292 documents; PT 4 (0.73/M), 3 documents.
- Pointwise KLD: 0.0012070160 bits; weighted log-odds z: 12.045; χ² p: 4.21e-288.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0374.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-10:6` — Nós mostramos alguns meses atrás -- nós publicamos o primeiro estudo mostrando que você pode mesmo parar ou reverter a progressão de câncer de próstata fazendo mudanças na dieta e estilo de vida, e 70 por cento de regres
  - `ted2020-10:9.4` — Bem, você sabe, a dieta que nós encontramos para reverter doenças cardíacas e o câncer é a dieta da Ásia.

### `entender`

- BP 3,805 (697.66/M), 1,613 documents; PT 1,096 (199.20/M), 636 documents.
- Pointwise KLD: 0.0011613383 bits; weighted log-odds z: 36.575; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0182.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:6` — Eu controlo o cérebro com o objetivo de entender como ele funciona.
  - `ted2020-1000:7` — Espere um pouco, você pode dizer, como você pode simplesmente 'ir controlando' o cérebro sem antes, entendê-lo?

### `algum`

- BP 12,868 (2359.39/M), 2,800 documents; PT 9,461 (1719.54/M), 2,592 documents.
- Pointwise KLD: 0.0010152093 bits; weighted log-odds z: 23.406; χ² p: 6.46e-122.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0041.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:44` — Eu quero começar com --- Eu vou mostrar-lhes algumas novas imagens e vou recapitular só quatro ou cinco.
  - `ted2020-1:55` — Mas eu quero trabalhar sobre alguns destes.

### `equipe`

- BP 1,125 (206.27/M), 556 documents; PT 6 (1.09/M), 5 documents.
- Pointwise KLD: 0.0010096681 bits; weighted log-odds z: 13.126; χ² p: 6.3e-245.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0367.
- Retrospective metadata only: old lexical inventory **yes**; any earlier seeded detector **yes**.
  - `ted2020-1:35.1` — Mas não era isso - minha equipe ficou muito preocupada porque uma das agências nigerianas já havia publicado uma história sobre o meu discurso.
  - `ted2020-1002:86` — Assim, o objetivo é ter todos os seus companheiros sentados sobre os pneus e então a sua equipe vence.

### `verdade`

- BP 6,438 (1180.43/M), 2,156 documents; PT 3,590 (652.48/M), 1,612 documents.
- Pointwise KLD: 0.0009456871 bits; weighted log-odds z: 28.486; χ² p: 1.44e-183.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0120.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:9` — É uma história verdadeira - cada molécula é verdade.
  - `ted2020-1:131.1` — É a mais pura verdade.

### `qual`

- BP 6,673 (1223.52/M), 2,280 documents; PT 3,904 (709.55/M), 1,776 documents.
- Pointwise KLD: 0.0009018746 bits; weighted log-odds z: 27.065; χ² p: 3.69e-165.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0090.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-10:1` — Com todas as preocupações legítimas sobre AIDS e gripe aviária -- e ouviremos sobre isso do brilhante Dr. Brilliant hoje mais tarde -- Quero falar sobre a outra pandemia, que é doença cardiovascular, diabetes, hipertensã
  - `ted2020-10:8.3` — O que realmente está me preocupando é que a diabetes aumentou em 70 por cento nos últimos 10 anos, e essa pode ser a primeira geração na qual nossas crianças vivem uma expectativa de vida menor do que nós.

### `achar`

- BP 6,758 (1239.10/M), 1,995 documents; PT 4,014 (729.54/M), 1,517 documents.
- Pointwise KLD: 0.0008883185 bits; weighted log-odds z: 26.607; χ² p: 1.79e-159.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0135.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:134` — Eu não sei quanto a vocês, mas eu acho sensacional ver como noções psicológicas vagas se evaporam e dão lugar a um entendimento físico, mecanicista da mente mesmo que seja a mente de uma mosca.
  - `ted2020-1001:1` — Bem, há muita coisa para ser dita, mas eu acho que vou apenas tocar para começar.

## 10. PT-characteristic lemmas

### `o`

- BP 528,489 (96900.09/M), 3,208 documents; PT 616,276 (112008.07/M), 3,209 documents.
- Pointwise KLD: 0.0221638786 bits; weighted log-odds z: -86.282; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0007.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.

### `a`

- BP 71,739 (13153.57/M), 3,193 documents; PT 121,789 (22135.13/M), 3,204 documents.
- Pointwise KLD: 0.0155947607 bits; weighted log-odds z: -112.530; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0047.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:10.1` — Logo a seguir de a Tipper e eu deixarmos a Casa Branca...

### `este`

- BP 22,900 (4198.79/M), 2,556 documents; PT 46,127 (8383.58/M), 3,150 documents.
- Pointwise KLD: 0.0078149070 bits; weighted log-odds z: -86.060; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0040.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.

### `vós`

- BP 8 (1.47/M), 8 documents; PT 6,088 (1106.49/M), 1,922 documents.
- Pointwise KLD: 0.0056181716 bits; weighted log-odds z: -19.131; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0131.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:8` — (Risos) (Aplausos) Vou contar-vos uma pequena história para ilustrar como tem sido a minha vida.
  - `ted2020-1:12.1` — (Risos) Eu sei que vos parece uma coisa sem importância, mas...

### `vosso`

- BP 6 (1.10/M), 5 documents; PT 4,045 (735.18/M), 1,358 documents.
- Pointwise KLD: 0.0037263562 bits; weighted log-odds z: -16.369; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0207.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:84` — Gostaria que os que são bons em construir marcas me dessem o vosso conselho e ajuda para dizer isto de forma a abranger o maior número de pessoas.
  - `ted2020-1:94.2` — Já pensaram em tornar o vosso negócio neutro em carbono?

### `isto`

- BP 6,500 (1191.80/M), 1,273 documents; PT 16,227 (2949.25/M), 2,627 documents.
- Pointwise KLD: 0.0035830805 bits; weighted log-odds z: -61.846; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0153.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:51` — Isto é válido só para os EUA.
  - `ted2020-1:83.1` — Pensem nisto.

### `facto`

- BP 0 (0.00/M), 0 documents; PT 2,798 (508.54/M), 1,359 documents.
- Pointwise KLD: 0.0026178632 bits; weighted log-odds z: -6.096; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0003.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:80` — Agora podemos, de facto, interferir com a sua psicologia de modo verdadeiramente profundo. como vou mostrar no meu último exemplo, que é direcionado para uma questão familiar.
  - `ted2020-1000:120` — De facto, faz tanto sentido que podemos construir um circuito eletrónico de brinquedos que simule o comportamento da mosca.

### `que`

- BP 180,895 (33167.66/M), 3,202 documents; PT 191,465 (34798.73/M), 3,204 documents.
- Pointwise KLD: 0.0022868528 bits; weighted log-odds z: -15.156; χ² p: 3.41e-50.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0006.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:7` — (Risos) Agora tenho que tirar os sapatos ou as botas para entrar num avião!

### `ter`

- BP 53,058 (9728.35/M), 3,164 documents; PT 61,176 (11118.73/M), 3,179 documents.
- Pointwise KLD: 0.0020290548 bits; weighted log-odds z: -22.755; χ² p: 1.3e-113.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0053.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.

### `um`

- BP 145,655 (26706.29/M), 3,205 documents; PT 152,905 (27790.46/M), 3,204 documents.
- Pointwise KLD: 0.0015141793 bits; weighted log-odds z: -11.173; χ² p: 3.05e-28.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0021.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:7` — (Risos) Agora tenho que tirar os sapatos ou as botas para entrar num avião!

### `de`

- BP 330,449 (60588.84/M), 3,210 documents; PT 339,255 (61659.54/M), 3,211 documents.
- Pointwise KLD: 0.0014797047 bits; weighted log-odds z: -7.634; χ² p: 1.4e-13.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0017.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:4` — Digo-o sinceramente, em parte, porque... preciso mesmo disso.

### `equipa`

- BP 0 (0.00/M), 0 documents; PT 1,439 (261.54/M), 678 documents.
- Pointwise KLD: 0.0013463564 bits; weighted log-odds z: -5.625; χ² p: 3.48e-312.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0003.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:35.2` — A minha equipa estava muito preocupada porque uma agência nigeriana já tinha publicado a notícia do meu discurso.
  - `ted2020-1002:84.1` — Neste jogo, a turma é dividida em duas equipas, cada uma de cada lado do recreio.

### `cancro`

- BP 0 (0.00/M), 0 documents; PT 1,334 (242.45/M), 290 documents.
- Pointwise KLD: 0.0012481163 bits; weighted log-odds z: -5.572; χ² p: 1.46e-289.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0003.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-10:6` — Publicámos há meses o primeiro estudo mostrando que é possível deter ou inverter a progressão do cancro da próstata, alterando a dieta e o estilo de vida e obter uma regressão de 70% no crescimento do tumor, ou a inibiçã
  - `ted2020-10:9.20` — Usamos uma dieta asiática que pode fazer regredir as doenças cardíacas e o cancro.

### `partilhar`

- BP 64 (11.73/M), 43 documents; PT 1,737 (315.70/M), 917 documents.
- Pointwise KLD: 0.0012300321 bits; weighted log-odds z: -25.906; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.1194.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1002:18` — Provavelmente, a maior coisa com que luta, como outras comunidades semelhantes, é que não existe um investimento partilhado e coletivo no futuro das comunidades rurais.
  - `ted2020-1002:89` — Penso que a coisa mais importante no "design para a educação" é partilhar a responsabilidade das soluções com os professores, para que eles tenham o incentive e o desejo de as usar.

### `se`

- BP 50,023 (9171.87/M), 3,190 documents; PT 55,045 (10004.42/M), 3,189 documents.
- Pointwise KLD: 0.0011887952 bits; weighted log-odds z: -14.202; χ² p: 2.22e-45.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0050.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:5` — (Risos) Coloquem-se no meu lugar!
  - `ted2020-1:28.2` — Nós no carro a conduzir, o Shoney's, a rede de restaurantes económicos, o que o homem disse — e eles riram-se.

### `porque`

- BP 12,113 (2220.96/M), 2,510 documents; PT 16,188 (2942.17/M), 2,762 documents.
- Pointwise KLD: 0.0011262702 bits; weighted log-odds z: -23.467; χ² p: 2.88e-122.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0063.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:4` — Digo-o sinceramente, em parte, porque... preciso mesmo disso.
  - `ted2020-1:35.2` — A minha equipa estava muito preocupada porque uma agência nigeriana já tinha publicado a notícia do meu discurso.

### `portanto`

- BP 1,231 (225.71/M), 657 documents; PT 3,732 (678.29/M), 1,404 documents.
- Pointwise KLD: 0.0009954805 bits; weighted log-odds z: -33.489; χ² p: 2e-271.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0306.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:17` — Portanto, estiveram a olhar para cerca de 1% do cérebro de uma barata.
  - `ted2020-1000:25` — Portanto, também nesta situação, para descodificar as informações contidas em padrões como este, não basta observar, temos de rearranjar o padrão.

### `haver`

- BP 16,166 (2964.09/M), 2,815 documents; PT 19,884 (3613.91/M), 2,923 documents.
- Pointwise KLD: 0.0009770862 bits; weighted log-odds z: -18.779; χ² p: 1.11e-78.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0091.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:13` — Não havia um cortejo motorizado atrás de nós.
  - `ted2020-1:64` — É um alvo fácil e visível de preocupação e é assim que deve ser, mas há mais poluição a causar aquecimento proveniente dos edifícios do que de carros e camiões.

### `depois`

- BP 5,415 (992.86/M), 2,131 documents; PT 8,495 (1543.96/M), 2,490 documents.
- Pointwise KLD: 0.0009244814 bits; weighted log-odds z: -25.421; χ² p: 1.08e-144.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0163.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:38` — Três dias depois, recebi uma deliciosa carta de um meu amigo, companheiro e colega Bill Clinton, dizendo: "Parabéns pelo novo restaurante, Al!"
  - `ted2020-1:88` — reduzir a nossas emissões de dióxido de carbono nas várias escolhas que podem fazer, e depois comprar ou adquirir compensações para o restante que não conseguimos reduzir totalmente.

### `bebé`

- BP 3 (0.55/M), 3 documents; PT 1,004 (182.48/M), 342 documents.
- Pointwise KLD: 0.0009110402 bits; weighted log-odds z: -10.553; χ² p: 1.64e-216.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0478.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1008:5` — Quero dizer, pensem só na enorme variedade — a cara de um bebé, a sinfonia "Harold en Italie" de Berlioz, filmes como "O Feiticeiro de Oz", ou as peças de Chekhov, as paisagens da Califórnia central, a vista do monte Fuj
  - `ted2020-1020:94` — Tem cinco semanas, é o meu bebé, e dá muito trabalho.

### `altura`

- BP 453 (83.06/M), 306 documents; PT 2,328 (423.11/M), 1,316 documents.
- Pointwise KLD: 0.0009033184 bits; weighted log-odds z: -31.709; χ² p: 2.16e-273.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0628.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:113` — A certa altura deste processo, vamos trabalhar com Larry Lessig, publicando, com ferramentas e direitos de uso limitados, para que os jovens possam editá-la e trabalhá-la à sua maneira.
  - `ted2020-1000:40` — Por alturas do final do século passado, comecei a pensar: Não seria maravilhoso se pegássemos nesta lógica e a invertêssemos?

### `aspeto`

- BP 2 (0.37/M), 2 documents; PT 963 (175.03/M), 611 documents.
- Pointwise KLD: 0.0008818058 bits; weighted log-odds z: -9.389; χ² p: 2.55e-208.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0298.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:13` — Observem o aspeto que a atividade cerebral pode ter.
  - `ted2020-1004:25` — Se as fronteiras deste tratado tivessem sido assumidas — e há uma base legal para isso — então os EUA teriam este aspeto hoje em dia.

### `económico`

- BP 2 (0.37/M), 2 documents; PT 942 (171.21/M), 445 documents.
- Pointwise KLD: 0.0008621742 bits; weighted log-odds z: -9.354; χ² p: 8.54e-204.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0340.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:20` — — uma rede de restaurantes familiares económicos, para quem não conhece.
  - `ted2020-1:28.2` — Nós no carro a conduzir, o Shoney's, a rede de restaurantes económicos, o que o homem disse — e eles riram-se.

### `demasiado`

- BP 16 (2.93/M), 14 documents; PT 1,052 (191.20/M), 739 documents.
- Pointwise KLD: 0.0008590154 bits; weighted log-odds z: -16.715; χ² p: 1.33e-218.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0400.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1005:138` — Parece assustador demais, demasiado perigoso, para qualquer empresa se comprometer.
  - `ted2020-1008:74` — Aliás, alguns deles são demasiado grandes para serem usados no esquartejamento de animais.

### `pôr`

- BP 517 (94.79/M), 407 documents; PT 2,332 (423.84/M), 1,270 documents.
- Pointwise KLD: 0.0008362661 bits; weighted log-odds z: -30.813; χ² p: 4.69e-250.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0338.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:8` — "Isso não é pôr a carroça à frente dos bois?"
  - `ted2020-1000:75` — (Risos) De qualquer maneira, como já vão ver, a Susana foi capaz de ativar o motor do voo daquilo que é o equivalente à espinal medula das moscas e pôr alguns destes corpos decapitados a levantar voo e voar.

## 11. BP-characteristic POS trigrams

### `PRON VERB ADP`

- BP 43,647 (9114.95/M), 3,184 documents; PT 30,190 (6264.52/M), 3,148 documents.
- Pointwise KLD: 0.0046429508 bits; weighted log-odds z: 50.481; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0045.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:4` — E quero dizer sinceramente, porque eu preciso (cara de brincadeira) eu preciso disso (risadas)
  - `ted2020-1:6` — Eu voei no Força Aérea 2 por oito anos.

### `PRON VERB DET`

- BP 36,593 (7641.84/M), 3,163 documents; PT 25,273 (5244.23/M), 3,145 documents.
- Pointwise KLD: 0.0039079779 bits; weighted log-odds z: 46.329; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0063.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:22` — (risadas) Ela anotou nosso pedido e foi atender um casal no outro reservado. e ela baixou tanto a voz que eu realmente tive que fazer força para escutar o que disse.
  - `ted2020-1:29` — Eu fiz o meu discurso, voltei para o aeroporto para voltar para casa.

### `VERB DET NOUN`

- BP 143,966 (30064.90/M), 3,205 documents; PT 132,650 (27525.30/M), 3,206 documents.
- Pointwise KLD: 0.0036286228 bits; weighted log-odds z: 23.876; χ² p: 1.69e-122.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0027.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de estar neste palco pela segunda vez.
  - `ted2020-1:7` — E hoje tenho que tirar meus sapatos ou botas para entrar num avião!

### `PRON AUX VERB`

- BP 20,070 (4191.29/M), 2,912 documents; PT 10,864 (2254.31/M), 2,688 documents.
- Pointwise KLD: 0.0035101968 bits; weighted log-odds z: 52.229; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0101.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:8` — (risadas e aplausos) Eu vou contar uma rápida história para mostrar o que isso significa para mim.
  - `ted2020-1:37.2` — "O ex-vice presidente Al Gore anunciou ontem na Nigeria "Eu e minha esposa Tipper abrimos um restaurante familiar barateiro, chamado Shoney, e eles estão tocando o negócio eles mesmos". (risadas) Antes de eu chegar a sol

### `CCONJ PRON VERB`

- BP 17,054 (3561.44/M), 2,816 documents; PT 8,583 (1781.00/M), 2,445 documents.
- Pointwise KLD: 0.0033271357 bits; weighted log-odds z: 52.497; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0123.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:10` — Logo depois que Tipper e eu deixamos (suspiro) a Casa Branca (risadas) estávamos na estrada de nossa casa em Nashville para uma pequena fazenda que temos 80 quilômetros ao leste de Nashville --
  - `ted2020-1:22` — (risadas) Ela anotou nosso pedido e foi atender um casal no outro reservado. e ela baixou tanto a voz que eu realmente tive que fazer força para escutar o que disse.

### `PRON PRON VERB`

- BP 43,220 (9025.78/M), 3,155 documents; PT 34,685 (7197.25/M), 3,123 documents.
- Pointwise KLD: 0.0027851374 bits; weighted log-odds z: 31.659; χ² p: 3.8e-219.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0012.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Eu fui muito aplaudido por esta conferência e quero agradecer a todos pelos muitos comentários delicados sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:8` — (risadas e aplausos) Eu vou contar uma rápida história para mostrar o que isso significa para mim.

### `PRON ADV VERB`

- BP 20,787 (4341.02/M), 2,978 documents; PT 13,117 (2721.82/M), 2,772 documents.
- Pointwise KLD: 0.0027465512 bits; weighted log-odds z: 42.008; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0121.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:20` — Uma rede de restaurantes familiares barateiros, para quem não conhece.
  - `ted2020-1:87` — Reduzir suas emissões de carbono em todas as escolhas que você faz, e então comprar ou adquirir o que faltar do que você não conseguiu reduzir completamente.

### `PRON VERB VERB`

- BP 28,071 (5862.16/M), 3,071 documents; PT 20,876 (4331.84/M), 3,027 documents.
- Pointwise KLD: 0.0024129780 bits; weighted log-odds z: 33.270; χ² p: 2.48e-243.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0075.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:33.1` — E pensei comigo mesmo, no meio da noite, no meio do Atlântico o que poderia haver de errado em Washington?
  - `ted2020-1:41` — Mas eu pensei que como eu estou planejando voltar habitualmente ao TED, talvez eu possa falar sobre isso numa outra ocasião.

### `AUX VERB ADP`

- BP 22,824 (4766.41/M), 3,086 documents; PT 16,917 (3510.33/M), 2,994 documents.
- Pointwise KLD: 0.0019835594 bits; weighted log-odds z: 30.274; χ² p: 4.48e-202.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0069.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:40` — E eu iria falar sobre a informação ecológica.
  - `ted2020-1:92` — E quando o filme for lançado em Maio, ela será atualizada para a versão 2.0 e nós teremos a possibilidade de neutralizar através de cliques.

### `VERB ADP PRON`

- BP 18,264 (3814.13/M), 2,878 documents; PT 12,753 (2646.29/M), 2,725 documents.
- Pointwise KLD: 0.0018942524 bits; weighted log-odds z: 31.779; χ² p: 2.19e-223.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0076.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Eu fui muito aplaudido por esta conferência e quero agradecer a todos pelos muitos comentários delicados sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:4` — E quero dizer sinceramente, porque eu preciso (cara de brincadeira) eu preciso disso (risadas)

### `ADV PRON VERB`

- BP 19,372 (4045.52/M), 2,994 documents; PT 14,046 (2914.59/M), 2,886 documents.
- Pointwise KLD: 0.0018036969 bits; weighted log-odds z: 29.688; χ² p: 9.35e-195.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0063.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:41` — Mas eu pensei que como eu estou planejando voltar habitualmente ao TED, talvez eu possa falar sobre isso numa outra ocasião.
  - `ted2020-1:91` — Com ela, você pode calcular precisamente suas emissões de CO2, e então você recebe opções para reduzi-las.

### `NOUN PRON VERB`

- BP 38,031 (7942.14/M), 3,180 documents; PT 32,674 (6779.96/M), 3,158 documents.
- Pointwise KLD: 0.0017155172 bits; weighted log-odds z: 21.129; χ² p: 1.39e-98.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0029.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:4` — E quero dizer sinceramente, porque eu preciso (cara de brincadeira) eu preciso disso (risadas)
  - `ted2020-1:10` — Logo depois que Tipper e eu deixamos (suspiro) a Casa Branca (risadas) estávamos na estrada de nossa casa em Nashville para uma pequena fazenda que temos 80 quilômetros ao leste de Nashville --

### `NOUN CCONJ NOUN`

- BP 19,658 (4105.25/M), 3,048 documents; PT 14,633 (3036.39/M), 2,936 documents.
- Pointwise KLD: 0.0016846308 bits; weighted log-odds z: 27.721; χ² p: 8.14e-170.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0063.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:7` — E hoje tenho que tirar meus sapatos ou botas para entrar num avião!
  - `ted2020-1:8` — (risadas e aplausos) Eu vou contar uma rápida história para mostrar o que isso significa para mim.

### `DET NOUN PRON`

- BP 44,916 (9379.96/M), 3,180 documents; PT 39,675 (8232.69/M), 3,173 documents.
- Pointwise KLD: 0.0016718435 bits; weighted log-odds z: 19.104; χ² p: 9.06e-81.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0049.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:22` — (risadas) Ela anotou nosso pedido e foi atender um casal no outro reservado. e ela baixou tanto a voz que eu realmente tive que fazer força para escutar o que disse.
  - `ted2020-1:31` — Acordei, eles abriram a porta, eu saí para respirar ar fresco, e vi um homem correndo na pista.

### `VERB ADP NOUN`

- BP 30,753 (6422.25/M), 3,156 documents; PT 25,627 (5317.68/M), 3,134 documents.
- Pointwise KLD: 0.0016536057 bits; weighted log-odds z: 22.445; χ² p: 2.95e-111.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0065.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:29` — Eu fiz o meu discurso, voltei para o aeroporto para voltar para casa.
  - `ted2020-1:37.2` — "O ex-vice presidente Al Gore anunciou ontem na Nigeria "Eu e minha esposa Tipper abrimos um restaurante familiar barateiro, chamado Shoney, e eles estão tocando o negócio eles mesmos". (risadas) Antes de eu chegar a sol

### `SCONJ PRON VERB`

- BP 26,682 (5572.09/M), 3,055 documents; PT 21,733 (4509.67/M), 3,003 documents.
- Pointwise KLD: 0.0016072785 bits; weighted log-odds z: 23.269; χ² p: 1.3e-119.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0114.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:4` — E quero dizer sinceramente, porque eu preciso (cara de brincadeira) eu preciso disso (risadas)
  - `ted2020-1:43.1` — Al Gore: Eu quero focar no que muitos de vocês disseram que gostariam que eu elaborasse.

### `NOUN CCONJ PRON`

- BP 9,077 (1895.58/M), 2,546 documents; PT 5,025 (1042.70/M), 2,169 documents.
- Pointwise KLD: 0.0015308896 bits; weighted log-odds z: 34.040; χ² p: 3.21e-261.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0134.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:111` — E eu vou oferecer um curso neste verão para um grupo de pessoas que foram indicadas por diferentes pessoas para vir e aprender, em massa, em comunidades em todo o país, e nós vamos atualizar este slideshow para todos ele
  - `ted2020-10:8.3` — O que realmente está me preocupando é que a diabetes aumentou em 70 por cento nos últimos 10 anos, e essa pode ser a primeira geração na qual nossas crianças vivem uma expectativa de vida menor do que nós.

### `PRON VERB ADV`

- BP 17,127 (3576.69/M), 2,973 documents; PT 12,612 (2617.03/M), 2,846 documents.
- Pointwise KLD: 0.0015199157 bits; weighted log-odds z: 26.706; χ² p: 6.48e-158.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0136.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:22` — (risadas) Ela anotou nosso pedido e foi atender um casal no outro reservado. e ela baixou tanto a voz que eu realmente tive que fazer força para escutar o que disse.
  - `ted2020-1:23` — Ela disse "Sim, é o ex-vice presidente Al Gore e sua esposa Tipper".

### `CCONJ PRON AUX`

- BP 6,563 (1370.57/M), 2,091 documents; PT 3,135 (650.52/M), 1,635 documents.
- Pointwise KLD: 0.0013751252 bits; weighted log-odds z: 34.356; χ² p: 1.53e-270.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0241.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:37.2` — "O ex-vice presidente Al Gore anunciou ontem na Nigeria "Eu e minha esposa Tipper abrimos um restaurante familiar barateiro, chamado Shoney, e eles estão tocando o negócio eles mesmos". (risadas) Antes de eu chegar a sol
  - `ted2020-1:40` — E eu iria falar sobre a informação ecológica.

### `VERB SCONJ PRON`

- BP 14,698 (3069.43/M), 2,812 documents; PT 10,764 (2233.56/M), 2,607 documents.
- Pointwise KLD: 0.0013271362 bits; weighted log-odds z: 25.124; χ² p: 4.28e-140.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0078.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:43.1` — Al Gore: Eu quero focar no que muitos de vocês disseram que gostariam que eu elaborasse.
  - `ted2020-1:53` — Eu sei que vocês querem mais más notícias sobre o meio ambiente - estou brincando - mas estes são os slides de recapitulação

### `NOUN AUX VERB`

- BP 14,278 (2981.72/M), 2,958 documents; PT 10,362 (2150.15/M), 2,825 documents.
- Pointwise KLD: 0.0013257152 bits; weighted log-odds z: 25.400; χ² p: 3.16e-143.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0044.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:25` — (Risadas) Tenho tido uma série de epifanias.
  - `ted2020-1:69` — muitos de vocês aqui -- muitas pessoas diretamente envolvidas nisso -- esta cunha vai crescer muito mais rápido do que as projeções nos mostram.

### `AUX PRON VERB`

- BP 3,931 (820.92/M), 1,846 documents; PT 1,216 (252.32/M), 908 documents.
- Pointwise KLD: 0.0012889409 bits; weighted log-odds z: 35.967; χ² p: 2.43e-317.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0275.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:113.2` — Isso não significa que se você é um Republicano eu estou te convencendo a ser
  - `ted2020-10:8.3` — O que realmente está me preocupando é que a diabetes aumentou em 70 por cento nos últimos 10 anos, e essa pode ser a primeira geração na qual nossas crianças vivem uma expectativa de vida menor do que nós.

### `CCONJ DET NOUN`

- BP 26,333 (5499.21/M), 3,141 documents; PT 22,365 (4640.81/M), 3,129 documents.
- Pointwise KLD: 0.0012738569 bits; weighted log-odds z: 18.759; χ² p: 2.63e-78.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0032.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:21` — Entramos, sentamos num reservado e uma garçonete veio, e ficou muito perturbada ao ver Tipper.
  - `ted2020-1:23` — Ela disse "Sim, é o ex-vice presidente Al Gore e sua esposa Tipper".

### `VERB PRON VERB`

- BP 11,960 (2497.65/M), 2,788 documents; PT 8,429 (1749.04/M), 2,591 documents.
- Pointwise KLD: 0.0012092202 bits; weighted log-odds z: 25.104; χ² p: 2.98e-140.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0087.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:24` — E o homem disse "Ele desceu bastante, não foi?"
  - `ted2020-1:31` — Acordei, eles abriram a porta, eu saí para respirar ar fresco, e vi um homem correndo na pista.

### `PRON VERB NOUN`

- BP 11,884 (2481.78/M), 2,823 documents; PT 8,603 (1785.15/M), 2,653 documents.
- Pointwise KLD: 0.0011118281 bits; weighted log-odds z: 23.324; χ² p: 3.79e-121.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0110.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:4` — E quero dizer sinceramente, porque eu preciso (cara de brincadeira) eu preciso disso (risadas)
  - `ted2020-1:10` — Logo depois que Tipper e eu deixamos (suspiro) a Casa Branca (risadas) estávamos na estrada de nossa casa em Nashville para uma pequena fazenda que temos 80 quilômetros ao leste de Nashville --

## 12. PT-characteristic POS trigrams

### `DET DET NOUN`

- BP 34,347 (7172.80/M), 3,146 documents; PT 78,653 (16320.75/M), 3,200 documents.
- Pointwise KLD: 0.0180300612 bits; weighted log-odds z: -128.549; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0069.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:5` — (Risos) Coloquem-se no meu lugar!
  - `ted2020-1:8` — (Risos) (Aplausos) Vou contar-vos uma pequena história para ilustrar como tem sido a minha vida.

### `VERB SCONJ VERB`

- BP 33,531 (7002.39/M), 3,132 documents; PT 61,251 (12709.78/M), 3,181 documents.
- Pointwise KLD: 0.0102374001 bits; weighted log-odds z: -88.599; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0048.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:16` — (Risos) Eram horas de jantar e começámos a procurar um lugar para comer.

### `ADP DET DET`

- BP 22,068 (4608.53/M), 3,040 documents; PT 40,593 (8423.18/M), 3,169 documents.
- Pointwise KLD: 0.0068626927 bits; weighted log-odds z: -72.566; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0063.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:5` — (Risos) Coloquem-se no meu lugar!

### `VERB DET DET`

- BP 6,738 (1407.12/M), 2,357 documents; PT 19,110 (3965.39/M), 3,043 documents.
- Pointwise KLD: 0.0054896232 bits; weighted log-odds z: -73.305; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0122.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:22.1` — (Risos) Ela anotou o nosso pedido e foi atender um casal numa mesa próxima.
  - `ted2020-1:84` — Gostaria que os que são bons em construir marcas me dessem o vosso conselho e ajuda para dizer isto de forma a abranger o maior número de pessoas.

### `SCONJ VERB DET`

- BP 31,513 (6580.97/M), 3,146 documents; PT 43,144 (8952.52/M), 3,166 documents.
- Pointwise KLD: 0.0037482055 bits; weighted log-odds z: -41.853; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0050.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:7` — (Risos) Agora tenho que tirar os sapatos ou as botas para entrar num avião!

### `SCONJ VERB ADP`

- BP 15,259 (3186.59/M), 2,910 documents; PT 25,564 (5304.61/M), 3,075 documents.
- Pointwise KLD: 0.0036602971 bits; weighted log-odds z: -50.024; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0086.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:3` — Fiquei muito impressionado com esta conferência e quero agradecer a todos os imensos comentários simpáticos sobre o que eu tinha a dizer naquela noite.
  - `ted2020-1:7` — (Risos) Agora tenho que tirar os sapatos ou as botas para entrar num avião!

### `VERB PRON ADP`

- BP 11,812 (2466.74/M), 2,796 documents; PT 21,561 (4473.98/M), 3,078 documents.
- Pointwise KLD: 0.0035992656 bits; weighted log-odds z: -52.186; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0046.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:5` — (Risos) Coloquem-se no meu lugar!
  - `ted2020-1:21` — Entrámos, sentámo-nos numa mesa e veio uma empregada, que ficou encantada quando viu Tipper.

### `DET NOUN VERB`

- BP 71,398 (14910.28/M), 3,199 documents; PT 81,630 (16938.49/M), 3,200 documents.
- Pointwise KLD: 0.0029515459 bits; weighted log-odds z: -25.292; χ² p: 4.14e-139.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0041.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:13` — Não havia um cortejo motorizado atrás de nós.
  - `ted2020-1:14` — (Risos) Já ouviram falar da dor do membro amputado?

### `DET NOUN ADP`

- BP 183,415 (38303.17/M), 3,207 documents; PT 192,842 (40015.33/M), 3,206 documents.
- Pointwise KLD: 0.0023956982 bits; weighted log-odds z: -13.954; χ² p: 1.35e-42.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0015.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:9` — É uma história verdadeira, cada pedaço desta história.

### `SCONJ VERB SCONJ`

- BP 6,521 (1361.80/M), 2,289 documents; PT 12,712 (2637.78/M), 2,744 documents.
- Pointwise KLD: 0.0023526822 bits; weighted log-odds z: -43.487; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0110.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:41` — Mas, como estou a planear fazer das palestras no TED um hábito para a vida, pensei que talvez pudesse falar sobre isso numa outra ocasião.
  - `ted2020-1:43.1` — Al Gore: Gostaria de me concentrar no que muitos de vocês disseram que gostariam que eu explicasse.

### `VERB PRON DET`

- BP 2,179 (455.05/M), 1,379 documents; PT 7,179 (1489.67/M), 2,467 documents.
- Pointwise KLD: 0.0023505609 bits; weighted log-odds z: -48.527; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0124.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:8` — (Risos) (Aplausos) Vou contar-vos uma pequena história para ilustrar como tem sido a minha vida.
  - `ted2020-1:37.4` — Um deles colocou-me um enorme chapéu branco de cozinheiro, enquanto Tipper dizia: "Mais um hambúrguer com batatas fritas!"

### `VERB PRON SCONJ`

- BP 3,312 (691.66/M), 1,759 documents; PT 8,741 (1813.79/M), 2,586 documents.
- Pointwise KLD: 0.0023410735 bits; weighted log-odds z: -47.302; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0147.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:115` — Significa que, se forem Republicanos, eu não vou tentar convencer-vos a tornarem-se Democratas.
  - `ted2020-1:133` — Um cientista disse-me que o teste que agora enfrentamos é se a combinação de um polegar oponível com um neocórtex é uma combinação viável.

### `NOUN VERB PRON`

- BP 6,653 (1389.37/M), 2,508 documents; PT 12,579 (2610.18/M), 2,941 documents.
- Pointwise KLD: 0.0022221014 bits; weighted log-odds z: -41.676; χ² p: 0.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0127.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:24` — E o homem disse: "Ele deu um grande tombo na vida, não foi?"
  - `ted2020-1:100` — Com o decorrer do tempo, as pessoas fazem o que lhes pagam para fazer.

### `NOUN SCONJ VERB`

- BP 23,644 (4937.66/M), 3,101 documents; PT 30,653 (6360.60/M), 3,141 documents.
- Pointwise KLD: 0.0021940657 bits; weighted log-odds z: -29.422; χ² p: 3.21e-190.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0049.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:7` — (Risos) Agora tenho que tirar os sapatos ou as botas para entrar num avião!

### `PRON ADP DET`

- BP 14,864 (3104.10/M), 2,949 documents; PT 21,153 (4389.31/M), 3,074 documents.
- Pointwise KLD: 0.0020668875 bits; weighted log-odds z: -32.490; χ² p: 5.14e-233.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0043.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:5` — (Risos) Coloquem-se no meu lugar!
  - `ted2020-1:21` — Entrámos, sentámo-nos numa mesa e veio uma empregada, que ficou encantada quando viu Tipper.

### `NOUN ADP DET`

- BP 145,223 (30327.41/M), 3,205 documents; PT 153,158 (31780.77/M), 3,204 documents.
- Pointwise KLD: 0.0020365439 bits; weighted log-odds z: -13.190; χ² p: 1.5e-38.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0022.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:9` — É uma história verdadeira, cada pedaço desta história.

### `DET NOUN ADJ`

- BP 88,826 (18549.83/M), 3,196 documents; PT 96,249 (19971.97/M), 3,197 documents.
- Pointwise KLD: 0.0020183196 bits; weighted log-odds z: -16.188; χ² p: 7.23e-58.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0020.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:9` — É uma história verdadeira, cada pedaço desta história.
  - `ted2020-1:22.1` — (Risos) Ela anotou o nosso pedido e foi atender um casal numa mesa próxima.

### `NOUN VERB SCONJ`

- BP 12,061 (2518.74/M), 2,887 documents; PT 17,783 (3690.03/M), 3,029 documents.
- Pointwise KLD: 0.0019136845 bits; weighted log-odds z: -32.472; χ² p: 2.06e-233.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0070.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:31` — Acordei, abriram a porta, saí para apanhar um pouco de ar fresco, e, de repente, vi um homem a correr na pista.
  - `ted2020-1:41` — Mas, como estou a planear fazer das palestras no TED um hábito para a vida, pensei que talvez pudesse falar sobre isso numa outra ocasião.

### `SCONJ VERB VERB`

- BP 7,162 (1495.66/M), 2,402 documents; PT 11,209 (2325.90/M), 2,697 documents.
- Pointwise KLD: 0.0013927352 bits; weighted log-odds z: -29.242; χ² p: 1.13e-190.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0124.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:19` — Saímos da I-40 e começámos a procurar... encontrámos um restaurante Shoney's.
  - `ted2020-1:41` — Mas, como estou a planear fazer das palestras no TED um hábito para a vida, pensei que talvez pudesse falar sobre isso numa outra ocasião.

### `PRON SCONJ VERB`

- BP 3,719 (776.65/M), 1,888 documents; PT 7,285 (1511.66/M), 2,490 documents.
- Pointwise KLD: 0.0013579955 bits; weighted log-odds z: -33.080; χ² p: 1.19e-248.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0176.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:115` — Significa que, se forem Republicanos, eu não vou tentar convencer-vos a tornarem-se Democratas.
  - `ted2020-10:2.2` — Toda a gente está a começar a comer como nós, a viver como nós, e a morrer como nós.

### `CCONJ VERB PRON`

- BP 5,014 (1047.09/M), 2,132 documents; PT 8,717 (1808.81/M), 2,595 documents.
- Pointwise KLD: 0.0013376190 bits; weighted log-odds z: -30.883; χ² p: 2.29e-214.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0096.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:113` — A certa altura deste processo, vamos trabalhar com Larry Lessig, publicando, com ferramentas e direitos de uso limitados, para que os jovens possam editá-la e trabalhá-la à sua maneira.
  - `ted2020-1:121.1` — e negociá-las.

### `DET NOUN SCONJ`

- BP 25,625 (5351.35/M), 3,107 documents; PT 29,878 (6199.78/M), 3,140 documents.
- Pointwise KLD: 0.0012459802 bits; weighted log-odds z: -17.384; χ² p: 1.97e-67.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0042.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:2.1` — É realmente uma grande honra ter a oportunidade de pisar este palco pela segunda vez.
  - `ted2020-1:7` — (Risos) Agora tenho que tirar os sapatos ou as botas para entrar num avião!

### `SCONJ VERB PRON`

- BP 9,087 (1897.67/M), 2,604 documents; PT 12,694 (2634.05/M), 2,776 documents.
- Pointwise KLD: 0.0011744302 bits; weighted log-odds z: -23.915; χ² p: 3.05e-127.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0123.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:22.2` — Baixou tanto a voz que tive de fazer um grande esforço para perceber o que ela disse:
  - `ted2020-1:37.2` — (Risos) " '... chamado Shoney's, e estamos nós próprios a geri-lo' ".

### `NOUN DET DET`

- BP 1,140 (238.07/M), 858 documents; PT 3,625 (752.20/M), 1,878 documents.
- Pointwise KLD: 0.0011525912 bits; weighted log-odds z: -33.892; χ² p: 1.7e-280.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0295.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1000:5` — (Risos) Mas ao contrário do meu gémeo malvado que anseia dominar o mundo, os meus motivos não são sinistros.
  - `ted2020-1000:19` — Algures, num padrão como este, somos nós, as nossas perceções, as nossas emoções, as nossas memórias, os nossos planos para o futuro.

### `SCONJ VERB ADV`

- BP 4,899 (1023.07/M), 2,176 documents; PT 8,149 (1690.94/M), 2,581 documents.
- Pointwise KLD: 0.0011506089 bits; weighted log-odds z: -27.830; χ² p: 8.23e-174.
- Stability: 5/5 leave-one-fold-out replications; effect SD 0.0068.
- Retrospective metadata only: old lexical inventory **no**; any earlier seeded detector **no**.
  - `ted2020-1:8` — (Risos) (Aplausos) Vou contar-vos uma pequena história para ilustrar como tem sido a minha vida.
  - `ted2020-1:53` — Sei que querem mais más notícias sobre o meio ambiente — estou a brincar — mas estes são os diapositivos de revisão

## 13. Supplemental multiword features

Word bigrams and trigrams remain visible, but passage scoring groups overlaps with their strongest lexical family and uses saturating repetition contributions.

### word_bigram

`que você`, `se você`, `em um`, `em uma`, `você pode`, `de fato`, `você não`, `você tem`, `por que`, `quando você`, `e você`, `e então`, `você está`, `na verdade`, `de sua`, `bilhões de`, `e eu`, `isso é`, `ao redor`, `em sua`

### word_trigram

`o que você`, `que está acontecendo`, `o fato de`, `você tem que`, `e se você`, `ao redor do`, `que você pode`, `fato de que`, `você não pode`, `por que não`, `essa é uma`, `redor do mundo`, `você pode ver`, `vocês podem ver`, `esse é o`, `que você está`, `que você não`, `essa é a`, `e de fato`, `e o que`

## 14. Aligned lexical / orthographic associations

> Recurrent aligned contrast associations are source-selection evidence, not replacement rules.

- `esse` ↔ `este` (lemma; general): 14455 pairs / 2620 documents; P(PT|BP)=0.505; lift=3.98; folds=5/5; fold-support CV=0.054.
- `essa` ↔ `esta` (surface_form; orthographic_near_form_candidate): 3063 pairs / 1460 documents; P(PT|BP)=0.410; lift=11.51; folds=5/5; fold-support CV=0.081.
- `essas` ↔ `estas` (surface_form; orthographic_near_form_candidate): 1845 pairs / 1083 documents; P(PT|BP)=0.406; lift=26.13; folds=5/5; fold-support CV=0.098.
- `isso` ↔ `isto` (lemma; general): 5373 pairs / 1872 documents; P(PT|BP)=0.227; lift=5.09; folds=5/5; fold-support CV=0.051.
- `esse` ↔ `este` (surface_form; orthographic_near_form_candidate): 2531 pairs / 1341 documents; P(PT|BP)=0.382; lift=11.45; folds=5/5; fold-support CV=0.072.
- `isso` ↔ `isto` (surface_form; orthographic_near_form_candidate): 4712 pairs / 1769 documents; P(PT|BP)=0.227; lift=5.54; folds=5/5; fold-support CV=0.061.
- `esses` ↔ `estes` (surface_form; orthographic_near_form_candidate): 1608 pairs / 957 documents; P(PT|BP)=0.410; lift=28.85; folds=5/5; fold-support CV=0.061.
- `algo` ↔ `coisa` (surface_form; general): 1659 pairs / 1008 documents; P(PT|BP)=0.332; lift=15.63; folds=5/5; fold-support CV=0.109.
- `compartilhar` ↔ `partilhar` (lemma; general): 882 pairs / 566 documents; P(PT|BP)=0.718; lift=143.05; folds=5/5; fold-support CV=0.113.
- `apenas` ↔ `só` (surface_form; general): 1561 pairs / 1007 documents; P(PT|BP)=0.272; lift=12.54; folds=5/5; fold-support CV=0.099.
- `apenas` ↔ `só` (lemma; general): 1560 pairs / 1007 documents; P(PT|BP)=0.272; lift=12.48; folds=5/5; fold-support CV=0.098.
- `fato` ↔ `facto` (lemma; general): 1035 pairs / 662 documents; P(PT|BP)=0.376; lift=45.56; folds=5/5; fold-support CV=0.066.
- `um` ↔ `num` (surface_form; general): 3094 pairs / 1705 documents; P(PT|BP)=0.064; lift=3.16; folds=5/5; fold-support CV=0.069.
- `algo` ↔ `coisa` (lemma; general): 1822 pairs / 1087 documents; P(PT|BP)=0.365; lift=8.19; folds=5/5; fold-support CV=0.108.
- `seus` ↔ `os` (surface_form; general): 2597 pairs / 1469 documents; P(PT|BP)=0.589; lift=4.13; folds=5/5; fold-support CV=0.086.
- `bilhão` ↔ `milhão` (lemma; general): 1071 pairs / 561 documents; P(PT|BP)=0.857; lift=66.00; folds=5/5; fold-support CV=0.172.
- `fato` ↔ `facto` (surface_form; orthographic_near_form_candidate): 835 pairs / 585 documents; P(PT|BP)=0.331; lift=45.05; folds=5/5; fold-support CV=0.087.
- `seu` ↔ `o` (surface_form; general): 4435 pairs / 1997 documents; P(PT|BP)=0.631; lift=2.02; folds=5/5; fold-support CV=0.070.
- `em` ↔ `num` (surface_form; general): 2681 pairs / 1549 documents; P(PT|BP)=0.064; lift=3.15; folds=5/5; fold-support CV=0.092.
- `achar` ↔ `pensar` (lemma; general): 1571 pairs / 894 documents; P(PT|BP)=0.339; lift=9.47; folds=5/5; fold-support CV=0.112.
- `bilhões` ↔ `milhões` (surface_form; orthographic_near_form_candidate): 875 pairs / 483 documents; P(PT|BP)=0.866; lift=73.27; folds=5/5; fold-support CV=0.133.
- `existir` ↔ `haver` (lemma; general): 1717 pairs / 889 documents; P(PT|BP)=0.504; lift=9.26; folds=5/5; fold-support CV=0.127.
- `em` ↔ `numa` (surface_form; general): 2372 pairs / 1468 documents; P(PT|BP)=0.056; lift=3.06; folds=5/5; fold-support CV=0.071.
- `maneira` ↔ `forma` (lemma; general): 1088 pairs / 702 documents; P(PT|BP)=0.456; lift=16.66; folds=5/5; fold-support CV=0.128.
- `seu` ↔ `vosso` (lemma; general): 1735 pairs / 884 documents; P(PT|BP)=0.085; lift=8.60; folds=5/5; fold-support CV=0.110.

## 15. Aligned structural associations

> Recurrent aligned contrast associations are source-selection evidence, not replacement rules.

- `CCONJ DET NOUN` ↔ `NOUN CCONJ DET` (pos_trigram; general): 5342 pairs / 2259 documents; P(PT|BP)=0.287; lift=7.02; folds=5/5; fold-support CV=0.051.
- `VERB DET NOUN` ↔ `SCONJ VERB DET` (pos_trigram; general): 21779 pairs / 3093 documents; P(PT|BP)=0.250; lift=2.12; folds=5/5; fold-support CV=0.058.
- `NOUN CCONJ NOUN` ↔ `DET NOUN CCONJ` (pos_trigram; general): 5955 pairs / 2299 documents; P(PT|BP)=0.421; lift=3.57; folds=5/5; fold-support CV=0.044.
- `NOUN VERB ADP` ↔ `DET NOUN VERB` (pos_trigram; general): 8829 pairs / 2747 documents; P(PT|BP)=0.531; lift=2.48; folds=5/5; fold-support CV=0.068.
- `CCONJ DET NOUN` ↔ `DET NOUN CCONJ` (pos_trigram; general): 6467 pairs / 2428 documents; P(PT|BP)=0.347; lift=2.95; folds=5/5; fold-support CV=0.056.
- `VERB ADP DET` ↔ `SCONJ VERB ADP` (pos_trigram; general): 9632 pairs / 2689 documents; P(PT|BP)=0.164; lift=2.30; folds=5/5; fold-support CV=0.069.
- `NOUN ADJ CCONJ` ↔ `DET NOUN ADJ` (pos_trigram; general): 7383 pairs / 2504 documents; P(PT|BP)=0.609; lift=2.54; folds=5/5; fold-support CV=0.069.
- `NOUN ADP NOUN` ↔ `ADP NOUN VERB` (pos_trigram; general): 6899 pairs / 2524 documents; P(PT|BP)=0.116; lift=2.36; folds=5/5; fold-support CV=0.058.
- `DET NOUN AUX` ↔ `NOUN AUX DET` (pos_trigram; general): 2944 pairs / 1643 documents; P(PT|BP)=0.120; lift=5.01; folds=5/5; fold-support CV=0.084.
- `PRON VERB ADP` ↔ `VERB PRON ADP` (pos_trigram; general): 4883 pairs / 2174 documents; P(PT|BP)=0.161; lift=2.67; folds=5/5; fold-support CV=0.069.
- `NOUN ADJ PRON` ↔ `DET NOUN ADJ` (pos_trigram; general): 4875 pairs / 2180 documents; P(PT|BP)=0.628; lift=2.62; folds=5/5; fold-support CV=0.062.
- `NOUN CCONJ NOUN` ↔ `NOUN CCONJ DET` (pos_trigram; general): 2715 pairs / 1591 documents; P(PT|BP)=0.192; lift=4.69; folds=5/5; fold-support CV=0.055.
- `NOUN CCONJ ADV` ↔ `DET NOUN CCONJ` (pos_trigram; general): 2874 pairs / 1690 documents; P(PT|BP)=0.487; lift=4.13; folds=5/5; fold-support CV=0.070.
- `NOUN CCONJ PRON` ↔ `DET NOUN CCONJ` (pos_trigram; general): 2966 pairs / 1644 documents; P(PT|BP)=0.502; lift=4.26; folds=5/5; fold-support CV=0.056.
- `NOUN ADJ CCONJ` ↔ `ADJ CCONJ VERB` (pos_trigram; general): 1722 pairs / 1214 documents; P(PT|BP)=0.142; lift=7.74; folds=5/5; fold-support CV=0.048.
- `ADP NOUN CCONJ` ↔ `NOUN CCONJ ADP` (pos_trigram; general): 2069 pairs / 1326 documents; P(PT|BP)=0.163; lift=5.54; folds=5/5; fold-support CV=0.063.
- `ADJ ADP NOUN` ↔ `DET NOUN ADJ` (pos_trigram; general): 5050 pairs / 2167 documents; P(PT|BP)=0.499; lift=2.08; folds=5/5; fold-support CV=0.089.
- `PRON AUX VERB` ↔ `VERB SCONJ VERB` (pos_trigram; general): 5054 pairs / 2013 documents; P(PT|BP)=0.361; lift=2.32; folds=5/5; fold-support CV=0.077.
- `NOUN ADJ AUX` ↔ `DET NOUN ADJ` (pos_trigram; general): 3427 pairs / 1770 documents; P(PT|BP)=0.684; lift=2.85; folds=5/5; fold-support CV=0.084.
- `PRON VERB ADP` ↔ `PRON ADP DET` (pos_trigram; general): 4141 pairs / 1997 documents; P(PT|BP)=0.136; lift=2.29; folds=5/5; fold-support CV=0.059.
- `NOUN ADJ ADV` ↔ `DET NOUN ADJ` (pos_trigram; general): 3473 pairs / 1838 documents; P(PT|BP)=0.630; lift=2.63; folds=5/5; fold-support CV=0.064.
- `NOUN VERB NOUN` ↔ `DET NOUN VERB` (pos_trigram; general): 3318 pairs / 1847 documents; P(PT|BP)=0.554; lift=2.59; folds=5/5; fold-support CV=0.056.
- `ADP NOUN CCONJ` ↔ `DET NOUN CCONJ` (pos_trigram; general): 3624 pairs / 1904 documents; P(PT|BP)=0.285; lift=2.42; folds=5/5; fold-support CV=0.065.
- `NOUN CCONJ NOUN` ↔ `NOUN CCONJ ADP` (pos_trigram; general): 2087 pairs / 1280 documents; P(PT|BP)=0.147; lift=5.01; folds=5/5; fold-support CV=0.043.
- `VERB CCONJ VERB` ↔ `SCONJ VERB CCONJ` (pos_trigram; general): 1133 pairs / 867 documents; P(PT|BP)=0.154; lift=13.09; folds=5/5; fold-support CV=0.076.

## 16. Stability diagnostics

Stable high-frequency evidence is retained rather than penalized for frequency. Features reversing across folds remain in the machine-readable report with lower confidence and are excluded from passage ranking.

## 17. Optional dependency / word-order results

Not computed. The required surface, real-lemma, and real-POS-trigram views were prioritized; no heuristic dependency or POS labels were invented.

## 18. Translation-bias limitations

Matched talks control topic, but TED translations can encode translator conventions, paraphrase, age, transcription, or segmentation effects. Document spread, fold stability, aligned recurrence, multiple views, and researcher review reduce—but cannot eliminate—this risk.

## 19. Source-mining interpretation boundary

Source mining asks whether a passage is likely to contain useful BP/EP contrastive material. Later cESA annotation asks whether a specific model output requires correction. Statistical distinctiveness in the source does not predetermine the annotation judgment.
