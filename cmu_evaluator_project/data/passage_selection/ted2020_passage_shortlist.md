# TED2020 BP passage shortlist

## Purpose

This was the unranked reading shortlist for choosing one coherent Brazilian Portuguese TED2020 passage for the first CMU evaluator pilot. **Candidate 2 (`ted_candidate_02`) is now the final selection.** No thesis or external candidate outputs have been generated, and no PT-PT material was used.

The shortlist contains ten passages from ten different TED documents. Nine passages contain ten consecutive sentences; one contains nine. Candidate numbering is organizational, not a quality ranking.

## Source structure

The BP source is `data/external/TED2020/TED2020/raw/pt_br/`.

- Each XML file is one transcript; the filename identifies the document, for example `ted2020-2387.xml`.
- Sentence order is the order of `<s>` elements in the XML.
- The original `id` attribute on each `<s>` element identifies the sentence. Decimal IDs such as `22.1` and `22.2` preserve subdivisions in the source.
- `<p>` elements group some sentences but do not create separate documents.
- Stage directions such as laughter or applause occur as sentence text in some transcripts.
- This raw BP package does not provide talk title, speaker, or URL metadata, so the document ID and source filename are the stable recovery keys.

Every passage below copies the BP sentence text exactly from the local XML.

## Candidate 1: `ted_candidate_01`

**Document:** `ted2020-2387`  
**Source file:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2387.xml`  
**Sentence range:** `24`–`33`  
**Sentence count:** 10

**Topic:** A Black woman returns to her goal of earning a PhD in astrophysics after seeing representation in physics.

**Passage:**

1. `24` — Depois de me formar, e apesar de saber que queria um PhD em astrofísica, deixei tudo para trás.
2. `25` — Foi um cartaz que salvou meu sonho, além de algumas pessoas e programas realmente incríveis.
3. `26` — A Sociedade Americana de Física tinha um cartaz lindo encorajando os estudantes de cor a se tornarem físicos.
4. `27` — Fiquei em choque, pois mostrava uma menina negra, em torno dos seus 12 anos, olhando absorta para algumas equações de física.
5. `28` — Lembro-me de pensar que estava olhando direto para o passado, para a menina que primeiro ousou sonhar aquele sonho.
6. `29` — Imediatamente escrevi para a Sociedade e pedi uma cópia daquele cartaz, que até hoje está pendurado no meu escritório.
7. `30` — Descrevi para eles por e-mail minha trajetória educacional e meu desejo de tentar o PhD novamente.
8. `31` — Fui direcionada para um programa das Universidades Fisk e Vanderbilt, ele próprio uma interseção do mestrado e do doutorado nas duas instituições.
9. `32` — Depois de dois anos fora da escola, eles me aceitaram no programa, e me vi novamente a caminho do PhD.
10. `33` — Depois de obter meu mestrado na Fisk, fui para Yale completar meu PhD.

**Quality notes:**

- Coherent chronological personal narrative.
- Clear referents and intact sentence boundaries.
- Contains institution names and the abbreviation “PhD”, but no source corruption.

**Concern to notice:** Several university and organization names appear, though they are not needed to understand the narrative.

## Candidate 2: `ted_candidate_02`

**Document:** `ted2020-2292`  
**Source file:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2292.xml`  
**Sentence range:** `3`–`12`  
**Sentence count:** 10

**Topic:** A childhood sailing experience becomes a long-term dream, followed by years of saving money.

**Passage:**

1. `3` — E com quatro anos de idade, eu tive a oportunidade de navegar pela primeira vez.
2. `4` — Nunca vou me esquecer da emoção quando nos aproximávamos da costa.
3. `5` — Nunca vou me esquecer do sentimento de aventura quando eu subi a bordo do barco e vi sua pequena cabine pela primeira vez.
4. `6` — Mas a sensação mais incrível era aquela de liberdade, a sensação que eu tive quando içamos as velas.
5. `7` — Com quatro anos de idade, foi o maior sentimento de liberdade que eu podia imaginar.
6. `8` — Naquele momento eu decidi que algum dia, de alguma maneira, eu iria navegar ao redor do mundo.
7. `9` — Então eu fiz o que pude na minha vida para alcançar aquele sonho.
8. `10` — Com dez anos, eu economizava o dinheiro da cantina.
9. `11` — Todos os dias durante oito anos, eu comia purê de batatas e feijão, que custavam 4p cada, e o molho era gratuito.
10. `12` — Todos os dias eu empilhava os trocados em cima do meu cofrinho, e quando juntava uma libra, eu colocava dentro e riscava um dos 100 quadrados que eu tinha desenhado num papel.

**Quality notes:**

- Coherent first-person narrative with strong tense and referent continuity.
- Ordinary vocabulary and intact segmentation.
- No stage directions or source corruption.

**Concern to notice:** The last two sentences contain British currency notation and several numbers.

## Candidate 3: `ted_candidate_03`

**Document:** `ted2020-4062`  
**Source file:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-4062.xml`  
**Sentence range:** `80`–`89`  
**Sentence count:** 10

**Topic:** A visual artist explains how lettering, color, and form became a personal voice across languages.

**Passage:**

1. `80` — Gestos e tons fazem parte da mensagem.
2. `81` — Dando forma às letras, posso decidir com mais precisão o que quero dizer e como, além do texto literal.
3. `82` — Posso dizer meu palavrão favorito de uma maneira muito enfeitada e ser bem brega quando falo de amor.
4. `83` — Posso falar alto e de forma imponente ou numa voz suave e poética.
5. `84` — E posso falar da diferença entre Buenos Aires e Berlim, duas cidades que conheço muito bem.
6. `85` — Foi justamente em Berlim que meu trabalho ficou mais colorido, mais expressivo, mais preciso ao contar histórias.
7. `86` — Tudo que eu não conseguia dizer naquelas festas, ali parada, segurando minha taça de vinho, explodia em formas e cores no papel.
8. `87` — Sem perceber, essa limitação da língua se tornou um motor que me impulsionou para aperfeiçoar minhas ferramentas de expressão.
9. `88` — Se não conseguia falar, este era meu jeito de dizer e contar coisas ao mundo.
10. `89` — Desde então, minha busca tem sido a de encontrar minha própria voz e contar histórias com os tons e gestos exatos da minha vontade.

**Quality notes:**

- Cohesive first-person reflection.
- Manageable sentence lengths and clear referents.
- No stage directions, malformed quotations, or obvious corruption.

**Concern to notice:** Sentence 3 contains the word “palavrão”, but no profanity itself.

## Candidate 4: `ted_candidate_04`

**Document:** `ted2020-2511`  
**Source file:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2511.xml`  
**Sentence range:** `31`–`40`  
**Sentence count:** 10

**Topic:** After her mother's illness and death, an artist continues their planned Arctic journey and reflects on Greenland's ice.

**Passage:**

1. `31` — Era minha mãe quem ia liderar essa viagem.
2. `32` — Eu e ela estávamos iniciando o planejamento, já que iríamos juntas, quando ela foi vitimada por um tumor cerebral.
3. `33` — O câncer rapidamente tomou conta de seu corpo e sua mente, e ela faleceu seis meses depois.
4. `34` — Durante a doença, no entanto, sua dedicação à expedição nunca esmoreceu, e eu prometi a ela levar adiante sua jornada final.
5. `35` — A paixão da minha mãe pelo Ártico ecoou na minha experiência na Groenlândia, e eu senti o poder e a fragilidade da paisagem.
6. `36` — A grandiosidade dos icebergs nos dá uma lição de humildade.
7. `37` — Os campos de gelo estão vivos com movimento e som de uma forma que nunca esperei.
8. `38` — Expandi o tamanho das minhas obras para nelas imprimir o mesmo sentimento de reverência que experimentei.
9. `39` — Mesmo assim, apesar de ser evidente a magnitude do gelo, também é evidente a sua vulnerabilidade.
10. `40` — Do nosso barco, pude ver o gelo transpirando sob o sol quente temporão.

**Quality notes:**

- Coherent narrative arc from planning and bereavement to travel and artistic response.
- Clear first-person continuity and intact segmentation.
- No stage directions or multimedia-dependent references.

**Concern to notice:** Contains bereavement and cancer; the final sentence uses the less common adjective “temporão”.

## Candidate 5: `ted_candidate_05`

**Document:** `ted2020-21996`  
**Source file:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-21996.xml`  
**Sentence range:** `20`–`29`  
**Sentence count:** 10

**Topic:** Why impostor feelings persist and how open discussion and feedback can help.

**Passage:**

1. `20` — Todos são suscetíveis a um fenômeno conhecido como ignorância pluralista, em que particularmente duvidamos de nós mesmos, mas achamos que só nós pensamos assim, pois ninguém mais expressa suas dúvidas.
2. `21` — Como é difícil saber o quanto nossos colegas se esforçam, suas dificuldades em certas tarefas, ou o quanto eles duvidam de si mesmos, não há maneira fácil de rejeitar sentimentos de que somos menos capazes do que as pessoas ao nosso redor.
3. `22` — Sentimentos intensos de ser um impostor podem evitar que pessoas compartilhem boas ideias ou se candidatem a empregos e programas em que são excelentes.
4. `23` — Pelo menos até agora, a melhor maneira de combater a síndrome do impostor é falar sobre ela.
5. `24` — Muitas pessoas com síndrome do impostor têm medo de perguntar sobre seu desempenho e confirmar o que temem.
6. `25` — E mesmo quando recebem um feedback positivo, geralmente não conseguem aliviar a sensação de fraude.
7. `26` — Mas, por outro lado, ouvir que um orientador ou mentor já se sentiu um impostor pode aliviar esses sentimentos.
8. `27` — O mesmo vale para os colegas.
9. `28` — Até descobrir que há um termo para esses sentimentos pode ser um alívio incrível.
10. `29` — Ao estar ciente do fenômeno, você poderá combater a sua síndrome do impostor reunindo e revisitando feedback positivo.

**Quality notes:**

- Coherent explanatory passage with a clear progression from cause to coping strategy.
- Natural mix of general statements and direct address.
- No obvious source corruption.

**Concern to notice:** Mental-health-adjacent topic; sentence 8 is deliberately short but complete.

## Candidate 6: `ted_candidate_06`

**Document:** `ted2020-24556`  
**Source file:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-24556.xml`  
**Sentence range:** `20`–`27`  
**Sentence count:** 9

**Topic:** Linux task scheduling is used as an analogy for prioritizing a crowded email inbox.

**Passage:**

1. `20` — Os programadores do sistema operacional Linux encontraram um problema similar em 2003.
2. `21` — O Linux colocava cada uma de suas tarefas em ordem de importância, e às vezes gastava mais tempo ordenando as tarefas do que as executando.
3. `22.1` — A solução encontrada pelos programadores foi substituir esse ordenamento por um número limitado de cestos de prioridades.
4. `22.2` — O sistema era menos preciso sobre o que fazer na sequência, mas mais do que compensou por isso gastando tempo fazendo progresso.
5. `23` — Assim, quanto aos e-mails, insistir em sempre fazer o mais importante primeiro pode nos levar a um colapso.
6. `24` — Organizar uma caixa de entrada três vezes mais cheia do que o normal pode levar nove vezes mais tempo para limpá-la.
7. `25` — O melhor seria responder em ordem cronológica ou até mesmo aleatória!
8. `26` — Surpreendentemente, às vezes, deixar de fazer as coisas em uma ordem perfeita pode ser a melhor forma de fazê-las.
9. `27` — Outra ideia oriunda do escalonamento de computadores tem a ver com uma das maiores questões da vida moderna: interrupções.

**Quality notes:**

- Nine-sentence passage beginning at a clean topic boundary.
- The technical example is explained and returns to an everyday email problem.
- Intact segmentation, including the original 22.1 and 22.2 sentence IDs.

**Concern to notice:** Contains light computing terminology and ends by introducing the next topic, interruptions.

## Candidate 7: `ted_candidate_07`

**Document:** `ted2020-46519`  
**Source file:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-46519.xml`  
**Sentence range:** `9`–`18`  
**Sentence count:** 10

**Topic:** Temujin's rise, military organization, religious worldview, and emergence as Genghis Khan.

**Passage:**

1. `9` — Apesar de perder o pai bem cedo e crescer na pobreza, ele chegou rapidamente ao poder formando alianças estratégicas com outros líderes.
2. `10` — Ao contrário desses cãs, Temujin promovia soldados com base no mérito e distribuía igualmente os espólios entre eles.
3. `11` — A jogada mais genial dele foi espalhar os nômades que conquistou entre seus próprios soldados, para que não pudessem se unir contra ele.
4. `12` — Essas inovações o tornaram irrefreável e, em 1206, ele uniu o povo das tendas de feltro e tornou-se Gengis Khan.
5. `13` — Os mongóis eram xamanistas e acreditavam que os espíritos da natureza e seus ancestrais habitavam o mundo ao seu redor.
6. `14` — Acima de tudo, curvavam-se ao deus do céu Tengri.
7. `15` — Gengis Khan acreditava que Tengri queria que ele conquistasse o mundo inteiro em seu nome.
8. `16` — Com os nômades da planície mongol unidos, isso parecia ao seu alcance.
9. `17` — Qualquer um que resistisse aos mongóis resistia à vontade de Tengri e, por essa insubordinação, seria morto.
10. `18` — Sob Gengis Khan, os mongóis dominaram o norte da China e as terras islâmicas orientais.

**Quality notes:**

- Coherent chronological historical explanation.
- Clear pronoun continuity and complete sentences.
- No stage directions or source corruption.

**Concern to notice:** Contains several historical names, one date, and references to conquest and killing.

## Candidate 8: `ted_candidate_08`

**Document:** `ted2020-1418`  
**Source file:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1418.xml`  
**Sentence range:** `8`–`17`  
**Sentence count:** 10

**Topic:** Public acceptance and controversy surrounding contraception and family planning.

**Passage:**

1. `8` — Mais de um bilhão de pessoas usam métodos contraceptivos sem qualquer hesitação.
2. `9` — Elas querem ter o poder de planejar sua própria vida e criar famílias mais saudáveis, mais instruídas e mais prósperas.
3. `10` — No entanto, para uma ideia que é amplamente aceita no âmbito do privado, o controle de natalidade certamente gera muita polêmica em público.
4. `11` — Algumas pessoas pensam, quando falamos sobre contracepção, que é uma porta para o aborto, o que não é.
5. `12` — Algumas pessoas, sejamos sinceros, ficam constrangidas com o assunto, porque é sobre sexo.
6. `13` — Algumas pessoas se preocupam que o real objetivo do planejamento familiar seja o controle populacional.
7. `14` — Todas essas são questões secundárias ligadas à ideia principal de que homens e mulheres devem ser capazes de decidir quando querem ter um filho.
8. `15` — E, como resultado, a contracepção desapareceu quase que completamente da pauta global da saúde.
9. `16` — As vítimas dessa inércia são os povos da África subsariana e do sul da Ásia.
10. `17` — Aqui, na Alemanha, a proporção de pessoas que usam métodos contraceptivos é de aproximadamente 66%.

**Quality notes:**

- Coherent policy-oriented argument.
- Mostly ordinary vocabulary with intact sentence boundaries.
- No stage directions or malformed text.

**Concern to notice:** Sensitive subject matter includes contraception, abortion, and sex; the final sentence contains a percentage.

## Candidate 9: `ted_candidate_09`

**Document:** `ted2020-29998`  
**Source file:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-29998.xml`  
**Sentence range:** `18`–`27`  
**Sentence count:** 10

**Topic:** The effectiveness, treatment schedule, and possible side effects of electroconvulsive therapy.

**Passage:**

1. `18` — Metade ou mais daqueles que passam por tratamento sentem uma melhora em seus sintomas.
2. `19` — A maioria dos pacientes tratados com ECT tem duas ou três sessões por semana durante várias semanas.
3. `20` — Alguns começam a notar uma melhora em seus sintomas após apenas uma sessão, enquanto outros demoram mais para reagir.
4. `21` — Os pacientes geralmente continuam tratamentos menos frequentes por vários meses a um ano, e alguns precisam de sessões esporádicas de manutenção pelo resto da vida.
5. `22` — A ECT moderna é muito mais segura do que costumava ser, mas os pacientes ainda podem sofrer efeitos colaterais.
6. `23` — Podem sentir dores, fadiga ou náuseas logo após o tratamento.
7. `24` — Alguns têm dificuldade em lembrar o que aconteceu antes de uma sessão, como, por exemplo, o que comeram no jantar da noite anterior.
8. `25` — Raramente, podem ter dificuldade em se lembrar de semanas e meses antes.
9. `26` — Para a maioria dos pacientes, essa perda de memória melhora com o tempo.
10. `27` — O fascinante é que, apesar de seu histórico comprovado, ainda não sabemos exatamente por que a ECT funciona.

**Quality notes:**

- Coherent medical explanation with consistent topic and referents.
- Manageable sentence lengths and intact segmentation.
- No obvious source corruption.

**Concern to notice:** Medical and mental-health content; the abbreviation “ECT” is not expanded inside this excerpt.

## Candidate 10: `ted_candidate_10`

**Document:** `ted2020-58601`  
**Source file:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-58601.xml`  
**Sentence range:** `9`–`18`  
**Sentence count:** 10

**Topic:** How penicillin damages bacterial cell walls and how it moved from laboratory curiosity toward larger-scale use.

**Passage:**

1. `9` — O fungo Penicillium produz penicilina constantemente para se defender de ameaças, como colônias bacterianas próximas que podem consumir seus recursos.
2. `10` — A penicilina destrói muitos tipos de bactérias, interrompendo a síntese de suas paredes celulares.
3. `11` — Essas paredes se fortalecem com uma malha espessa e protetora de açúcares e aminoácidos, que são constantemente quebrados e reconstruídos.
4. `12` — A penicilina se liga a um dos compostos que tece essa malha e impede a reconstrução da parede em uma fase crítica.
5. `13` — Enquanto isso, a penicilina estimula a liberação de moléculas altamente reativas que causam danos adicionais.
6. `14` — Por fim, a estrutura da célula se decompõe completamente.
7. `15` — Esse ataque duplo é letal para uma ampla gama de bactérias, seja em placas de Petri, no corpo ou em outros lugares.
8. `16` — No entanto, não é prejudicial para nossas próprias células, porque elas não têm paredes celulares.
9. `17` — Por mais de uma década após a descoberta de Fleming, a penicilina permaneceu uma curiosidade de laboratório.
10. `18` — Mas, durante a Segunda Guerra Mundial, pesquisadores descobriram como isolar o composto ativo e cultivar o fungo em quantidades maiores.

**Quality notes:**

- Coherent scientific explanation with a clear causal sequence.
- Complete sentences and consistent terminology.
- No stage directions, lists, or obvious corruption.

**Concern to notice:** Some biological terminology is specialized, although the mechanism is explained in the passage.

## Selection boundary

These are candidate passages only. The reader should choose the final passage based on content, suitability, and annotation comfort. No candidate in this file is designated as the final or preferred passage.
