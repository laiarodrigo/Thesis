# TED2020 larger BP passage candidate pool

> Candidate shortlist for researcher review. It is **not** a finalized annotation dataset.

Scanned **3,968 documents / 457,273 sentences** and scored **1,277,112** overlapping 8–10-sentence windows. The conservative threshold admitted 4,125; diversity selection retained **50** non-overlapping, one-per-document windows.

The score is an audit aid, not linguistic ground truth. Strong lexical types dominate; repeated hits are capped. `bp_preferred_ep_attested`, orthographic, and structural signals are shown separately.

## 1. ted2020-1092_s12-s20

- **Document:** `ted2020-1092`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1092.xml`
- **Sentence IDs:** 12, 13, 14, 15, 16, 17, 18.1, 18.2, 19, 20
- **Length:** 10 sentences
- **Score / confidence:** 41.5 / **high**
- **Evidence:** 3 strong lexical hits (3 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 6 structural.
- **Why promising:** verified strong types: baba_childcare, cafe_da_manha, equipe_group; supporting types: address_voce, possessive_without_article, progressive_estar_gerund.
- **Concerns:** Contains a long sentence that may increase annotation burden.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 12 | Então, vocês estão vendo o que é certamente a maior coleção de vídeos caseiros já feita. | — | — | estar + gerund; você/vocês (context-dependent) |
| 13 | (Risos) E o que esses dados representam para nossa família, pessoalmente, o impacto tem sido imenso, e ainda estamos aprendendo seu valor. | — | — | estar + gerund |
| 14 | Inúmeros momentos de expressões naturais, não ensaiados, foram captados lá, e estamos aprendendo como descobri-los e encontrá-los. | — | — | estar + gerund |
| 15 | Mas também há uma razão científica para este projeto, que era usar estes dados naturais e longitudinais para entender o processo de como uma criança aprende linguagem - essa criança sendo meu filho. | — | — | — |
| 16 | Então, com várias cláusulas de privacidade para proteger todos que foram gravados, nós disponibilizamos elementos dos dados para a minha confiável equipe de pesquisadores no MIT, para que pudéssemos separar padrões neste conjunto maciço de dados, tentando entender a influência de ambientes sociais na aquisição da linguagem. | equipe (equipe_group) | — | — |
| 17 | Aqui, estamos vendo uma das primeiras coisas que fizemos. | — | — | estar + gerund |
| 18.1 | Eu e minha esposa fazendo o café da manhã na cozinha. | café da manhã (cafe_da_manha) | — | — |
| 18.2 | E à medida em que avançamos no espaço e no tempo, há um padrão de vida bem cotidiano na cozinha. | — | — | — |
| 19 | Para converter este vídeo opaco de 90 mil horas em algo que poderíamos começar a ver, usamos a análise de movimentos para selecionar, à medida em que avançamos no espaço e no tempo, o que chamamos de vermes do espaço-tempo. | — | — | — |
| 20 | E isso se tornou parte das nossas ferramentas para sermos capazes de olhar e ver onde estão as atividades nos dados e, com isso, traçar um padrão de onde meu filho se movia pela casa, para que pudéssemos focar nossos esforços de transcrição em todo o ambiente de fala em volta do meu filho - todas as palavras que ele ouviu de mim, minha esposa, nossa babá e, com o tempo, as palavras que ele começou a produzir. | babá (baba_childcare) | — | bare possessive (weak heuristic) |

## 2. ted2020-1565_s7.2-s14

- **Document:** `ted2020-1565`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1565.xml`
- **Sentence IDs:** 7.2, 7.3, 8, 9, 10.1, 10.2, 11, 12, 13, 14
- **Length:** 10 sentences
- **Score / confidence:** 40.0 / **high**
- **Evidence:** 2 strong lexical hits (2 types) across 2 sentences; 0 weaker lexical hits; 0 orthographic; 12 structural.
- **Why promising:** verified strong types: cafe_da_manha, caminhao_vehicle; supporting types: possessive_without_article, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 7.2 | Meus porcos transformaram aquele resíduo alimentar em uma deliciosa carne de porco. | — | — | bare possessive (weak heuristic) |
| 7.3 | Eu vendi aquela carne para os pais de meus colegas da escola, e eu fiz um bom dinheiro extra para minha mesada. | — | — | bare possessive (weak heuristic); bare possessive (weak heuristic) |
| 8 | Mas eu notei que a maior parte da comida que eu estava dando aos meus porcos era de fato própria para consumo humano, e eu via apenas na ponta do iceberg, e , certamente, na cadeia de abastecimento de alimentos, em supermercados, verdureiros, padarias, em nosso lares, em fábricas e fazendas, nós estávamos jogando alimentos pelo ralo. | — | — | estar + gerund |
| 9 | Supermercados não queriam nem falar comigo sobre quanto alimento eles estavam desperdiçando. | — | — | estar + gerund |
| 10.1 | Eu estive nos fundos. | — | — | — |
| 10.2 | E eu vi lixeiras cheias de comida sendo fechadas e enviadas em caminhões para aterros sanitários, e eu pensei, realmente há algo mais sensato para fazer com alimentos do que desperdiçá-los. | caminhões (caminhao_vehicle) | — | — |
| 11 | Certa manhã, quando eu estava alimentando meus porcos, eu notei em particular uma vistosa fatia de tomate seco que surgia de vez em quando. | — | — | estar + gerund |
| 12 | Eu a peguei, sentei e tomei meu café da manhã com meus porcos. (risos) Esse foi o primeiro ato do que mais tarde eu aprendi a chamar de freeganismo, realmente uma exibição da injustiça do desperdício de alimentos e a provisão da solução para o desperdício de alimentos, é simplesmente sentar e comê-los, em vez de jogá-los fora. | café da manhã (cafe_da_manha) | — | bare possessive (weak heuristic) |
| 13 | Isso tornou-se, por assim dizer, uma forma de enfrentar grandes empresas no negócio do desperdício alimentar, e expor, mais importante, ao público, que quando nós estamos falando sobre alimentos jogados fora, não estamos falando de coisas podres, não estamos falando de coisas que não servem mais. | — | — | estar + gerund; estar + gerund; estar + gerund |
| 14 | Nós estamos falando de comida fresca e boa que está sendo desperdiçada em uma escala colossal. | — | — | estar + gerund; estar + gerund |

## 3. ted2020-1188_s9.2-s18

- **Document:** `ted2020-1188`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1188.xml`
- **Sentence IDs:** 9.2, 10, 11, 12, 13, 14, 15, 16, 17, 18
- **Length:** 10 sentences
- **Score / confidence:** 39.5 / **high**
- **Evidence:** 7 strong lexical hits (2 types) across 5 sentences; 0 weaker lexical hits; 1 orthographic; 3 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: premio_accent, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 9.2 | A Apple lança o iPhone na China e censura aplicativos do Dalai Lama junto a outros tantos aplicativos politicamente sensíveis, a pedido do governo chinês para sua loja app chinesa. | aplicativos (aplicativo_software); aplicativos (aplicativo_software) | — | — |
| 10 | O chargista político americano Mark Fiore também teve seu aplicativo de sátira censurado nos Estados Unidos porque alguns da equipe da Apple estavam preocupados se ele seria ofensivo a certos grupos. | aplicativo (aplicativo_software); equipe (equipe_group) | — | — |
| 11 | Seu aplicativo só foi readmitido quando ele ganhou o Prêmio Pulitzer. | aplicativo (aplicativo_software) | — | prêmio→prémio |
| 12 | A revista alemã Stern, uma revista de notícias, teve seu aplicativo censurado porque as 'babás da Apple' o julgaram picante demais para seus usuários, a despeito dessa revista ser legalmente vendida nas bancas de toda a Alemanha. | aplicativo (aplicativo_software) | — | — |
| 13 | E mais controversialmente, recentemente, a Apple censurou um aplicativo de protesto palestino depois que o governo israelense demonstrou receio de que ele poderia ser usado para organizar ataques violentos. | aplicativo (aplicativo_software) | — | — |
| 14 | Então, é o seguinte, | — | — | — |
| 15 | temos uma situação onde empresas privadas estão utilizando critérios de censura que são quase sempre um tanto arbitrários e geralmente mais restritos que as normas constitucionais de liberdade de expressão que temos nas democracias. | — | — | estar + gerund |
| 16 | Ou estão respondendo a pedidos de censura de regimes autoritários que não refletem consentimento dos governados. | — | — | estar + gerund |
| 17 | Ou respondem a pedidos e interesses dos governos que não têm jurisdição sobre tantos, ou da maioria, dos usuários e espectadores que estão interagindo com o conteúdo em questão. | — | — | estar + gerund |
| 18 | Então, a situação é esta. | — | — | — |

## 4. ted2020-2042_s100-s108

- **Document:** `ted2020-2042`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2042.xml`
- **Sentence IDs:** 100, 101, 102, 103.1, 103.2, 104, 105, 106, 107, 108
- **Length:** 10 sentences
- **Score / confidence:** 35.0 / **high**
- **Evidence:** 4 strong lexical hits (2 types) across 4 sentences; 0 weaker lexical hits; 0 orthographic; 3 structural.
- **Why promising:** verified strong types: esporte_activity, onibus_vehicle; supporting types: progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 100 | Dezenas de crianças Somália-americana saíram para abraçar o esporte apesar do <i>fatwa</i> contra isso. | esporte (esporte_activity) | — | — |
| 101 | Eles jogaram basquete como Burhan Hassan nunca faria novamente. | — | — | — |
| 102 | Por seus esforços, Sr. Bihi foi condenado ao ostracismo pela liderança do Centro Islâmico Abubakar As-Saddique, com que ele costumava ter boas relações. | — | — | — |
| 103.1 | Ele me disse: "Um dia, vimos o <i>imam</i> na TV chamando-nos infiéis e dizendo: "Essas famílias estão tentando destruir a mesquita." | — | — | estar + gerund |
| 103.2 | Isso está em total desacordo com como Abdirizak Bihi entende o que ele está tentando fazer expondo o recrutamento al Shabaab, que é salvar a religião que eu amo de um pequeno número de extremistas. | — | — | estar + gerund |
| 104 | Agora eu quero contar uma última história, a de uma aluna de direito de 22 anos, da Argélia, chamada Amel Zenoune-Zouani que tinha os mesmos sonhos de uma carreira jurídica que eu tinha nos anos 90. | — | — | — |
| 105 | Ela recusou-se a desistir dos estudos, apesar do fato de que os fundamentalistas lutando contra o Estado argelino na época, ameaçou a todos os que continuaram com sua educação. | — | — | — |
| 106 | Em 26 de janeiro de 1997, Amel embarcou no ônibus em Argel, onde ela estava estudando para ir para casa e passar o Ramadan com sua família, e nunca terminaria a faculdade de direito. | ônibus (onibus_vehicle) | — | estar + gerund |
| 107 | Quando o ônibus alcançou os arredores de sua cidade, ele foi parado em um posto ocupado por homens do Grupo Islâmico Armado. | ônibus (onibus_vehicle) | — | — |
| 108 | Levando sua mochila, Amel foi retirada do ônibus e morta na rua. | ônibus (onibus_vehicle) | — | — |

## 5. ted2020-2279_s83-s91

- **Document:** `ted2020-2279`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2279.xml`
- **Sentence IDs:** 83, 84.1, 84.2, 85, 86, 87, 88, 89, 90, 91
- **Length:** 10 sentences
- **Score / confidence:** 33.5 / **high**
- **Evidence:** 3 strong lexical hits (3 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 6 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group, geladeira_appliance; supporting types: address_voce.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 83 | Na educação, imagine se seus aplicativos de ensino perceberem que você está confuso e desacelerarem, ou que você está entediado e acelerarem, assim como faria um bom professor em sala de aula. | aplicativos (aplicativo_software) | — | você/vocês (context-dependent); você/vocês (context-dependent) |
| 84.1 | E se seu relógio de pulso acompanhasse seu humor ou seu carro percebesse que você está cansado, ou talvez sua geladeira saiba que você está estressada, e se tranca sozinha para que você não coma demais. | geladeira (geladeira_appliance) | — | você/vocês (context-dependent); você/vocês (context-dependent); você/vocês (context-dependent) |
| 84.2 | (Risos) Eu gostaria disso, sim. | — | — | — |
| 85 | E se, quando eu estava em Cambridge, eu tivesse acesso ao meu fluxo de sentimentos em tempo real, e pudesse compartilhá-lo com a minha família em casa de maneira natural, assim como eu faria se estivéssemos todos juntos na sala? | — | — | — |
| 86 | Acredito que daqui a cinco anos, todos os nossos dispositivos vão ter um chip de sentimentos, e nem vamos nos lembrar como era quando não adiantava fazer cara feia para o dispositivo e ele diria: "Hm, você não gostou disso, né?" | — | — | você/vocês (context-dependent) |
| 87 | Nosso maior desafio é que há tantas aplicações para essa tecnologia, que minha equipe percebeu que não podemos satisfazê-las sozinhos, então nós disponibilizamos essa tecnologia a outros desenvolvedores para que construam e usem a criatividade. | equipe (equipe_group) | — | — |
| 88 | Nós reconhecemos que há potenciais riscos e potencial para abuso, mas pessoalmente, após muitos anos fazendo isso, acredito que os benefícios à humanidade de ter tecnologia emocionalmente inteligente superam de longe o potencial de abuso. | — | — | — |
| 89 | E eu os convido todos a participarem da conversa. | — | — | — |
| 90 | Quanto mais pessoas conhecerem essa tecnologia, mais peso terá nossa opinião sobre como usá-la. | — | — | — |
| 91 | E à medida que nossas vidas vão se digitalizando, nós lutamos uma batalha perdida tentando frear nosso uso de dispositivos para reivindicar nossos sentimentos. | — | — | — |

## 6. ted2020-10274_s9-s18

- **Document:** `ted2020-10274`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-10274.xml`
- **Sentence IDs:** 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
- **Length:** 10 sentences
- **Score / confidence:** 34.0 / **high**
- **Evidence:** 3 strong lexical hits (2 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 4 structural.
- **Why promising:** verified strong types: caminhao_vehicle, celular_phone; supporting types: progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 9 | Essa pessoa está aguardando nas docas, prestes a embarcar em um navio que irá cruzar o Atlântico ou o Pacífico. | — | — | estar + gerund |
| 10 | Essa pessoa está carregando um caminhão que cruzará o Rio Grande. | caminhão (caminhao_vehicle) | — | estar + gerund |
| 11 | Ou está esperando em uma estação de trem, prestes a embarcar em um trem que irá cruzar rios e montanhas e deixar pra trás o sul do Jim Crow até chegar onde ela espera encontrar liberdade ao norte. | — | — | estar + gerund |
| 12 | E lá, com essa jovem pessoa, prestes a embarcar naquele navio, naquele barco, naquele trem, naquele caminhão, estão as pessoas que a criaram. | caminhão (caminhao_vehicle) | — | — |
| 13 | A sua mãe e seu pai, suas tias, tios, seus avós, quem quer que tenha cuidado dela até aquele ponto. | — | — | — |
| 14 | As pessoas mais velhas não seriam capazes de realizar a travessia com ela. | — | — | — |
| 15 | E quando ela olhou nos olhos das pessoas que os criaram, não havia garantias de que se veriam em vida novamente. | — | — | — |
| 16 | Lembrem, não havia Skype, e-mail, telefones celulares, nem mesmo serviços telefônicos de longa distância confiáveis. | celulares (celular_phone) | — | — |
| 17 | E mesmo que houvesse, muitas das pessoas que estavam deixando pra trás não possuíam telefones. | — | — | estar + gerund |
| 18 | Essa seria uma separação completa de tudo que conheciam e de todas as pessoas que amavam. | — | — | — |

## 7. ted2020-1381_s25-s34

- **Document:** `ted2020-1381`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1381.xml`
- **Sentence IDs:** 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
- **Length:** 10 sentences
- **Score / confidence:** 35.25 / **high**
- **Evidence:** 4 strong lexical hits (2 types) across 4 sentences; 0 weaker lexical hits; 0 orthographic; 4 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: address_voce, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 25 | Tem um cara no departamento de TI da cidade de Honolulu que viu este aplicativo e percebeu que ele poderia usá-lo, não para neve, mas para pedir cidadãos que adotem sirenas de tsunami. | aplicativo (aplicativo_software) | — | — |
| 26 | É muito importante que essas sirenas funcionem, mas tem gente que rouba as baterias delas. | — | — | — |
| 27 | Então ele está pedindo aos cidadãos que as verifiquem. | — | — | estar + gerund |
| 28 | E aí Seattle decidiu usar o aplicativo para organizar cidadãos para limpar ralos entupidos depois das tempestades. | aplicativo (aplicativo_software) | — | — |
| 29 | E Chicago acabou de o implantar e quer que as pessoas se inscrevam para remover neve das calçadas. | — | — | — |
| 30 | Então, estamos sabendo que 9 cidades estão planejando usar este aplicativo. | aplicativo (aplicativo_software) | — | estar + gerund; estar + gerund |
| 31 | E isso se espalhou sem impedimento algum, organicamente, naturalmente. | — | — | — |
| 32 | Se vocês sabem alguma coisa sobre tecnologia governamental, sabem que normalmente não é assim que funciona. | — | — | você/vocês (context-dependent) |
| 33 | Aquisição de software geralmente leva uns dois anos. | — | — | — |
| 34 | Uma equipe de três pessoas que trabalhou num projeto em Boston no ano passado levou cerca de dois meses e meio. | equipe (equipe_group) | — | — |

## 8. ted2020-2568_s127-s135

- **Document:** `ted2020-2568`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2568.xml`
- **Sentence IDs:** 127, 128, 129, 130, 131, 132, 133, 134.1, 134.2, 135
- **Length:** 10 sentences
- **Score / confidence:** 33.0 / **high**
- **Evidence:** 3 strong lexical hits (3 types) across 3 sentences; 0 weaker lexical hits; 1 orthographic; 0 structural.
- **Why promising:** verified strong types: equipe_group, esporte_activity, onibus_vehicle; supporting types: metro_accent.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 127 | E as duas equipes quebraram a mesma regra, justamente a que dizia que eles não podiam jogar futebol. | equipes (equipe_group) | — | — |
| 128 | Quando alguém os vê jogando, não vê deficiência, vê competência. | — | — | — |
| 129 | O problema é quando termina a partida e saem da quadra | — | — | — |
| 130 | para jogar a nossa partida, em uma sociedade que estabelece regras que não os leva em conta, que não cuida deles. | — | — | — |
| 131 | Com o esporte eu aprendi que a deficiência depende muito das regras do jogo. | esporte (esporte_activity) | — | — |
| 132 | Por isso acredito que, se trocarmos algumas regras da nossa partida, podemos tornar a vida um pouco mais fácil para eles. | — | — | — |
| 133 | Todos sabemos que existem pessoas com deficiência, vemos isso diariamente. | — | — | — |
| 134.1 | Mas talvez por não termos contato direto com elas, não sejamos conscientes da problemática que enfrentam no dia a dia. | — | — | — |
| 134.2 | Como é difícil subir em um ônibus, conseguir um trabalho, pegar o metrô, atravessar uma rua. | ônibus (onibus_vehicle) | — | metrô→metro |
| 135 | É certo que há uma maior responsabilidade social em relação à inclusão de pessoas com deficiência. | — | — | — |

## 9. ted2020-24360_s25-s33

- **Document:** `ted2020-24360`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-24360.xml`
- **Sentence IDs:** 25, 26, 27, 28, 29, 30, 31, 32, 33
- **Length:** 9 sentences
- **Score / confidence:** 34.5 / **high**
- **Evidence:** 6 strong lexical hits (2 types) across 6 sentences; 0 weaker lexical hits; 1 orthographic; 4 structural.
- **Why promising:** verified strong types: equipe_group, esporte_activity; supporting types: academico_accent, address_voce.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 25 | E então há o aprendizado através do fracasso, um dos benefícios mais transformadores e duradouros na prática de esportes. | esportes (esporte_activity) | — | — |
| 26 | A experiência da conviver com a derrota pode construir a resistência e a autoconsciência necessárias para lidar com obstáculos acadêmicos, sociais e físicos. | — | — | acadêmic-→académic- |
| 27 | Mesmo que sua equipe não esteja sempre ganhando, ou esteja perdendo, existe um benefício real através da experiência. | equipe (equipe_group) | — | — |
| 28 | No entanto, nem todo mundo vai gostar de todos os esportes. | esportes (esporte_activity) | — | — |
| 29 | Talvez uma equipe seja muito competitiva ou não competitiva o suficiente. | equipe (equipe_group) | — | — |
| 30 | Também pode levar tempo para encontrar um esporte que seja ideal para você. | esporte (esporte_activity) | — | você/vocês (context-dependent) |
| 31 | É perfeitamente normal. | — | — | — |
| 32 | Mas se você passar um tempo procurando, será capaz de encontrar um esporte que atenda suas necessidades individuais, e, se você encontrar, há muitos benefícios. | esporte (esporte_activity) | — | você/vocês (context-dependent); você/vocês (context-dependent) |
| 33 | Você será parte de uma comunidade solidária, estará melhorando sua confiança, exercitando seu corpo e alimentado sua mente, sem mencionar o divertimento. | — | — | você/vocês (context-dependent) |

## 10. ted2020-2562_s51-s60

- **Document:** `ted2020-2562`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2562.xml`
- **Sentence IDs:** 51, 52, 53, 54, 55, 56, 57, 58, 59, 60
- **Length:** 10 sentences
- **Score / confidence:** 33.0 / **medium**
- **Evidence:** 2 strong lexical hits (2 types) across 1 sentences; 1 weaker lexical hits; 0 orthographic; 5 structural.
- **Why promising:** verified strong types: onibus_vehicle, trem_rail; supporting types: rodovia_road, progressive_estar_gerund.
- **Concerns:** Strong evidence is concentrated in one sentence.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 51 | Abolir combustíveis fósseis significa revolucionar nosso sistema de transporte e passar a mover nossos carros, ônibus e trens a eletricidade, em vez de energia suja. | ônibus (onibus_vehicle); trens (trem_rail) | — | — |
| 52 | E o transporte, é preciso que se diga, se tornou uma questão existencial para nós, costa-riquenhos, porque o modelo que temos não está funcionando para nós. | — | — | estar + gerund |
| 53 | Está prejudicando as pessoas, está prejudicando as empresas e está prejudicando nossa saúde. | — | — | estar + gerund; estar + gerund; estar + gerund |
| 54 | Porque, quando as políticas e a infraestrutura falham, isso é o que acontece diariamente. | — | — | — |
| 55 | Duas horas pela manhã, duas horas à noite. | — | — | — |
| 56 | Não entendo por que temos de aceitar isso como o normal. | — | — | — |
| 57 | É ofensivo ter de gastar nosso tempo assim todo santo dia. | — | — | — |
| 58 | E essa rodovia é na verdade muito boa comparada com o que vemos em outros países onde o tráfico está explodindo. | — | rodovia (rodovia_road) | estar + gerund |
| 59 | Sabem, os costa-riquenhos chamam isso de "presa". | — | — | — |
| 60 | "Presa", como em português, significa emprisionada. | — | — | — |

## 11. ted2020-2327_s50-s59

- **Document:** `ted2020-2327`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2327.xml`
- **Sentence IDs:** 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
- **Length:** 10 sentences
- **Score / confidence:** 34.75 / **high**
- **Evidence:** 6 strong lexical hits (2 types) across 5 sentences; 0 weaker lexical hits; 0 orthographic; 4 structural.
- **Why promising:** verified strong types: equipe_group, esporte_activity; supporting types: address_voce, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 50 | E já há gerações de mulheres que também viveram o trabalho em equipe. | equipe (equipe_group) | — | — |
| 51 | Elas passaram a jogar esportes em equipe como nunca antes. | esportes (esporte_activity); equipe (equipe_group) | — | — |
| 52 | Então você tinha um legado já construído em termos de ser atleta, um legado da sua defesa pela igualdade salarial das atletas e da Fundação de Esportes para Mulheres. | Esportes (esporte_activity) | — | você/vocês (context-dependent) |
| 53 | E agora o que você está buscando com a Iniciativa Liderança Billie Jean King? | — | — | estar + gerund; você/vocês (context-dependent) |
| 54 | BJK: Acho que remete a uma epifania que tive aos 12 anos. | — | — | — |
| 55 | Aos 11 anos, eu queria ser a jogadora de tênis número um do mundo, e um amigo me chamou para jogar e eu disse: "O que é isso?" | — | — | — |
| 56 | Tênis não estava na minha família; basquete sim, outros esportes. | esportes (esporte_activity) | — | — |
| 57 | Avançando para os 12 anos, (Risos) eu finalmente comecei a jogar em torneios nos quais você obtém uma classificação no fim do ano. | — | — | você/vocês (context-dependent) |
| 58 | Eu sonhava acordada com o clube de Tênis de Los Angeles, e comecei a pensar no meu esporte e quão minúsculo ele era, mas também que todo mundo que jogava usava sapatos brancos, roupas brancas, jogava com bolas brancas... todo mundo que jogava era branco. | esporte (esporte_activity) | — | — |
| 59 | E eu disse a mim mesma, aos 12 anos: "Onde estão as outras pessoas?" | — | — | — |

## 12. ted2020-58_s16.2-s24

- **Document:** `ted2020-58`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-58.xml`
- **Sentence IDs:** 16.2, 17.1, 17.2, 18, 19, 20, 21, 22, 23, 24
- **Length:** 10 sentences
- **Score / confidence:** 32.5 / **high**
- **Evidence:** 6 strong lexical hits (2 types) across 6 sentences; 0 weaker lexical hits; 0 orthographic; 2 structural.
- **Why promising:** verified strong types: onibus_vehicle, terno_suit; supporting types: address_voce, possessive_without_article.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 16.2 | Minha esposa e eu, de 37 anos, juntos no ônibus. | ônibus (onibus_vehicle) | — | bare possessive (weak heuristic) |
| 17.1 | Nosso ônibus nos levou de São Francisco para Londres. | ônibus (onibus_vehicle) | — | — |
| 17.2 | Nós mudamos de ônibus na grande lagoa. | ônibus (onibus_vehicle) | — | — |
| 18 | Nós então pegamos mais dois ônibus e atravessamos a Turquia e o Irã, Afeganistão, sobre a Passagem de Khyber, no Paquistão, como qualquer outro jovem médico. | ônibus (onibus_vehicle) | — | — |
| 19 | Somos nós na Passagem de Khyber e este é o nosso ônibus. | ônibus (onibus_vehicle) | — | — |
| 20 | Nós tivemos algumas dificuldades para atravessar a Passagem de Khyber, | — | — | — |
| 21 | mas chegamos na Índia. | — | — | — |
| 22 | E então, como todo mundo da nossa geração, fomos morar em um mosteiro no Himalaia. | — | — | — |
| 23 | (Risos) É como se fosse um programa de residência, para aqueles de vocês que estão na faculdade de medicina. | — | — | você/vocês (context-dependent) |
| 24 | E estudamos com um homem sábio, um guru chamado Karoli Baba, que me falou para me livrar do vestido, vestir um terno de três peças, ir para as Nações Unidas como diplomata e trabalhar na Organização Mundial da Saúde. | terno (terno_suit) | — | — |

## 13. ted2020-2203_s26-s32

- **Document:** `ted2020-2203`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2203.xml`
- **Sentence IDs:** 26, 27, 28.1, 28.2, 29.1, 29.2, 29.3, 30, 31, 32
- **Length:** 10 sentences
- **Score / confidence:** 31.0 / **high**
- **Evidence:** 3 strong lexical hits (3 types) across 2 sentences; 0 weaker lexical hits; 0 orthographic; 2 structural.
- **Why promising:** verified strong types: caminhao_vehicle, celular_phone, trem_rail; supporting types: address_voce, possessive_without_article.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 26 | Mas você precisa ter um dispositivo no topo das árvores, certo? | — | — | você/vocês (context-dependent) |
| 27 | Então se pudermos usar um dispositivo que possa ouvir os sons da floresta, conectá-lo à rede de celulares e enviar um alerta às pessoas em terra, talvez pudéssemos ter uma solução para este problema. | celulares (celular_phone) | — | — |
| 28.1 | Paremos por um instante, para falar sobre salvar a floresta. | — | — | — |
| 28.2 | Todos nós já ouvimos falar disto. | — | — | — |
| 29.1 | Minha geração ouviu falar disto desde quando éramos crianças. | — | — | bare possessive (weak heuristic) |
| 29.2 | Parece que a mensagem nunca muda. | — | — | — |
| 29.3 | "Salvemos as florestas tropicais, isto é muito urgente, ontem destruiu-se o equivalente a tantos campos de futebol." | — | — | — |
| 30 | Entretanto, hoje resta a metade das florestas tropicais e há problemas potencialmente brm mais urgentes, como a mudança climática. | — | — | — |
| 31 | Eis um fato pouco conhecido do qual eu não havia me dado conta: o desmatamento é responsável por mais gases de efeito estufa do que todos os aviões, trens, carros, caminhões e navios do mundo juntos. | trens (trem_rail); caminhões (caminhao_vehicle) | — | — |
| 32 | Ele é também a segunda maior causa da mudança climática. | — | — | — |

## 14. ted2020-1272_s10-s18

- **Document:** `ted2020-1272`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1272.xml`
- **Sentence IDs:** 10, 11, 12, 13, 14, 15.1, 15.2, 16, 17, 18
- **Length:** 10 sentences
- **Score / confidence:** 34.0 / **high**
- **Evidence:** 7 strong lexical hits (2 types) across 6 sentences; 0 weaker lexical hits; 0 orthographic; 1 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 10 | Quero dizer, no caso do futebol, poderiam ir para uma equipe de futebol. | equipe (equipe_group) | — | — |
| 11 | no caso do violino, poderiam ter aulas de violino. | — | — | — |
| 12 | Mas e se quiserem fazer um aplicativo? | aplicativo (aplicativo_software) | — | — |
| 13 | E seus pais, os pais das crianças talvez tenham feito algumas destas coisas quando eram jovens, mas nem todos escreveram aplicativos. | aplicativos (aplicativo_software) | — | — |
| 14 | (Risos) Onde devem ir para descobrir como fazer um aplicativo? | aplicativo (aplicativo_software) | — | — |
| 15.1 | Bem, foi assim que cheguei até ele. | — | — | — |
| 15.2 | Foi isso que eu fiz. | — | — | — |
| 16 | Em primeiro lugar, estive programando em diversas linguagens de programação até chegar ao básico como Python, C, Java, etc. | — | — | estar + gerund |
| 17 | E então a Apple lançou o iPhone, e com ele, o kit para desenvolvimento de programas para o iPhone, e o kit para desenvolvimento de programas é um conjunto de ferramentas para a criação e programação de um aplicativo para o iPhone. | aplicativo (aplicativo_software) | — | — |
| 18 | Isto abriu um mundo de possibilidades inteiramente novo para mim, e depois de brincar um pouquinho com o kit para desenvolvimento de programas fiz alguns aplicativos, fiz alguns aplicativos de teste. | aplicativos (aplicativo_software); aplicativos (aplicativo_software) | — | — |

## 15. ted2020-726_s9-s17

- **Document:** `ted2020-726`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-726.xml`
- **Sentence IDs:** 9, 10, 11, 12.1, 12.2, 13, 14, 15, 16, 17
- **Length:** 10 sentences
- **Score / confidence:** 33.25 / **high**
- **Evidence:** 5 strong lexical hits (2 types) across 4 sentences; 0 weaker lexical hits; 0 orthographic; 4 structural.
- **Why promising:** verified strong types: caminhao_vehicle, onibus_vehicle; supporting types: address_voce, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 9 | Foi só um daqueles potes da sorte onde você pega pela primeira vez. | — | — | você/vocês (context-dependent) |
| 10 | Movendo-se para alguma coisa maior, Este é um raio-x de um ônibus. | ônibus (onibus_vehicle) | — | — |
| 11 | E o ônibus está cheio de gente. | ônibus (onibus_vehicle) | — | — |
| 12.1 | É realmente a mesma pessoa. | — | — | — |
| 12.2 | É só um esqueleto. | — | — | — |
| 13 | E de volta aos anos 60 eles costumavam ensinar estudantes de radiografia a fazer raios-x, felizmente não em você ou eu, mas em pessoas mortas. | — | — | você/vocês (context-dependent) |
| 14 | Então, ainda tenho acesso a uma destas pessoas mortas chamada Frieda; ela está fora daqui, sinto muito, porque ela é muito velha e frágil. | — | — | — |
| 15 | Mas qualquer um naquele ônibus é Frieda. | ônibus (onibus_vehicle) | — | — |
| 16 | E o ônibus está sendo fotografado com um scanner de raio-x de caminhão, que é o tipo de máquina que você tem em fronteiras que checa contrabando e drogas e bombas e outras coisas. | ônibus (onibus_vehicle); caminhão (caminhao_vehicle) | — | estar + gerund; você/vocês (context-dependent) |
| 17 | Bastante óbvio o que é isto. | — | — | — |

## 16. ted2020-26922_s70-s79.1

- **Document:** `ted2020-26922`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-26922.xml`
- **Sentence IDs:** 70, 71, 72, 73, 74, 75, 76, 77, 78, 79.1
- **Length:** 10 sentences
- **Score / confidence:** 33.5 / **high**
- **Evidence:** 4 strong lexical hits (2 types) across 4 sentences; 0 weaker lexical hits; 0 orthographic; 4 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: possessive_without_article, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 70 | Alguns dos aplicativos mais populares do mundo são os de encontros, uma indústria que vale atualmente algo em torno de US$ 3 bilhões por ano. | aplicativos (aplicativo_software) | — | — |
| 71 | É claro que não temos problemas em compartilhar detalhes íntimos com nossa cara-metade. | — | — | — |
| 72 | Mas quem mais está espionando, gravando e compartilhando nossas informações enquanto estamos revelando nossa intimidade? | — | — | estar + gerund; estar + gerund |
| 73 | Minha equipe e eu decidimos investigar. | equipe (equipe_group) | — | bare possessive (weak heuristic) |
| 74 | Para entender o problema de todos os ângulos e fazer realmente um trabalho minucioso, percebi que eu mesmo precisava baixar um dos aplicativos de encontros mais populares do mundo. | aplicativos (aplicativo_software) | — | — |
| 75 | Então, fui para casa falar com minha esposa... | — | — | bare possessive (weak heuristic) |
| 76 | (Risos) com quem eu acabara de me casar. | — | — | — |
| 77 | "Tudo bem se eu criar um perfil num aplicativo de encontros muito popular para fins exclusivamente científicos?" | aplicativo (aplicativo_software) | — | — |
| 78 | (Risos) Eis o que encontramos. | — | — | — |
| 79.1 | Escondida atrás do menu principal havia uma opção pré-selecionada que dava à empresa de encontros acesso a todas as minhas fotos pessoais do Facebook. | — | — | — |

## 17. ted2020-2222_s151-s160

- **Document:** `ted2020-2222`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2222.xml`
- **Sentence IDs:** 151, 152, 153, 154, 155, 156, 157, 158, 159, 160
- **Length:** 10 sentences
- **Score / confidence:** 33.0 / **high**
- **Evidence:** 5 strong lexical hits (2 types) across 4 sentences; 0 weaker lexical hits; 0 orthographic; 3 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: address_voce, possessive_without_article, progressive_estar_gerund.
- **Concerns:** Contains a long sentence that may increase annotation burden.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 151 | Durante os últimos meses, a equipe da StoryCorps trabalhou incessantemente para criar um aplicativo que vai tirar a StoryCorps das cabines para que possa ser experimentada por qualquer um em qualquer lugar a qualquer hora. | aplicativo (aplicativo_software); equipe (equipe_group) | — | — |
| 152 | Lembrem-se, a StoryCorps sempre foi duas pessoas e um facilitador ajudando-os a gravar sua conversa, que será preservada para sempre, mas neste exato momento, estamos lançando uma versão beta pública do aplicativo da StoryCorps. | aplicativo (aplicativo_software) | — | estar + gerund |
| 153 | O aplicativo é um facilitador digital que o guia pelo processo de entrevista da StoryCorps, ajuda a escolher as perguntas, e dá todas as dicas necessárias para gravar uma entrevista significativa para a StoryCorps, e com um toque enviá-la para o nosso arquivo na Biblioteca do Congresso. | aplicativo (aplicativo_software) | — | — |
| 154 | Essa é a parte fácil, a tecnologia. | — | — | — |
| 155 | O verdadeiro desafio é com vocês: pegar essa ferramenta e descobrir como podemos usá-la por todo o país e pelo mundo, para que em vez de gravar milhares de entrevistas da StoryCorps por ano, podemos potencialmente gravar dezenas de milhares ou centenas de milhares ou talvez até mais. | — | — | você/vocês (context-dependent) |
| 156 | Imaginem, por exemplo, uma tarefa de casa nacional onde cada aluno de colegial estudando história nos Estados Unidos grava uma entrevista com alguém mais velho no feriado, assim em um único fim de semana uma geração inteira de vidas e experiências americanas é capturada. | — | — | — |
| 157 | (Aplausos) Ou imaginem mães em lados opostos de conflitos em algum lugar do mundo sentando-se não para falar do conflito mas para descobrir quem elas são como pessoas, e fazendo isso, começar a construir laços de confiança; ou que algum dia se torne tradição em todo o mundo que pessoas sejam honradas com uma entrevista da StoryCorps em seu 75º aniversário; ou que pessoas eu sua comunidade vão a asilos, hospitais, abrigos ou até mesmo prisões munidas com esse aplicativo para honrar as pessoas menos ouvidas em nossa sociedade e lhes perguntam quem são, o que aprenderam na vida, e como querem ser lembradas. | aplicativo (aplicativo_software) | — | — |
| 158 | (Aplausos) Dez anos atrás, eu gravei uma entrevista da StoryCorps com meu pai que era psiquiatra, e se tornou um ativista gay bem conhecido. | — | — | bare possessive (weak heuristic) |
| 159 | Esta foto somos nós na entrevista. | — | — | — |
| 160 | Eu nunca pensei nessa gravação até alguns anos atrás, quando meu pai, que parecia estar em ótimas condições e ainda atendia pacientes 40 horas por semana, foi diagnosticado com câncer. | — | — | — |

## 18. ted2020-49131_s64-s73

- **Document:** `ted2020-49131`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-49131.xml`
- **Sentence IDs:** 64, 65, 66, 67, 68, 69, 70, 71, 72, 73
- **Length:** 10 sentences
- **Score / confidence:** 33.0 / **high**
- **Evidence:** 5 strong lexical hits (2 types) across 5 sentences; 1 weaker lexical hits; 0 orthographic; 0 structural.
- **Why promising:** verified strong types: caminhao_vehicle, equipe_group; supporting types: rodovia_road.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 64 | Houve um acidente fatal quando alguém usou a IA do piloto automático de um Tesla mas, em vez de usá-la em uma rodovia, para onde elas foram programadas, utilizaram-na em vias urbanas. | — | rodovia (rodovia_road) | — |
| 65 | Um caminhão entrou na frente do carro, que não conseguiu frear. | caminhão (caminhao_vehicle) | — | — |
| 66 | Vejam, a IA definitivamente foi treinada para reconhecer caminhões em imagens. | caminhões (caminhao_vehicle) | — | — |
| 67 | Mas o que parece ter acontecido é que a IA foi treinada para reconhecê-los em estradas, onde se espera ver a traseira dos caminhões. | caminhões (caminhao_vehicle) | — | — |
| 68 | Caminhões vindos pela lateral não deveriam aparecer em estradas, então, quando a IA o viu, parece que ela avaliou que se tratava de uma placa de trânsito e que, portanto, seria seguro passar sob ela. | Caminhões (caminhao_vehicle) | — | — |
| 69 | Agora vejamos um erro de IA de um campo diferente. | — | — | — |
| 70 | A Amazon teve que desistir de um algoritmo para seleção de currículos no qual trabalhava ao descobrir que ele aprendeu a discriminar mulheres. | — | — | — |
| 71 | Eles o treinaram com base em exemplos de currículos de pessoas que haviam sido contratadas por eles. | — | — | — |
| 72 | E, desses exemplos, a IA aprendeu a evitar os currículos de pessoas que frequentaram faculdades femininas ou que continham a palavra "feminina" em algum lugar, como "equipe feminina de futebol" ou "Sociedade Feminina de Engenharia". | equipe (equipe_group) | — | — |
| 73 | A IA não sabia que não deveria copiar esse padrão específico que havia observado nos humanos. | — | — | — |

## 19. ted2020-4463_s90-s99

- **Document:** `ted2020-4463`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-4463.xml`
- **Sentence IDs:** 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
- **Length:** 10 sentences
- **Score / confidence:** 31.75 / **high**
- **Evidence:** 3 strong lexical hits (3 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 1 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group, onibus_vehicle; supporting types: possessive_without_article.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 90 | Um aplicativo que sabe compartilhar bem isso é o "Ducky". | aplicativo (aplicativo_software) | — | — |
| 91 | A ideia é registrar nossas ações lá, e poder ver depois o desempenho de nossa equipe ou empresa, para obtermos sinais em tempo real. | equipe (equipe_group) | — | — |
| 92 | Finalmente, a identidade. | — | — | — |
| 93 | Podemos mudar a identidade com melhores histórias. | — | — | — |
| 94 | Nosso cérebro adora histórias. | — | — | — |
| 95 | Precisamos de histórias melhores sobre para onde queremos ir, e de mais histórias dos heróis e das heroínas de todos os tipos que fazem a verdadeira mudança acontecer. | — | — | — |
| 96 | Tenho orgulho de minha cidade de Oslo estar embarcando numa jornada arrojada de eletrificação de todos os transportes, sejam carros, motos ou ônibus. | ônibus (onibus_vehicle) | — | bare possessive (weak heuristic) |
| 97 | Uma das líderes dessa campanha é Christina Bu. | — | — | — |
| 98 | Ela está à frente da Electric Vehicle Association há anos e tem lutado todos os dias. | — | — | — |
| 99 | Reino Unido, França, Índia e China também anunciaram planos para pôr um fim às vendas de carros antigos. | — | — | — |

## 20. ted2020-13517_s9-s18

- **Document:** `ted2020-13517`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-13517.xml`
- **Sentence IDs:** 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
- **Length:** 10 sentences
- **Score / confidence:** 32.25 / **high**
- **Evidence:** 7 strong lexical hits (2 types) across 7 sentences; 0 weaker lexical hits; 0 orthographic; 1 structural.
- **Why promising:** verified strong types: equipe_group, esporte_activity; supporting types: address_voce.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 9 | No entanto, dentro de 70 dias, todos os 33 homens serão trazidos à tona vivos. | — | — | — |
| 10 | Esta notável história é um estudo de caso sobre o poder de formação de equipes. | equipes (equipe_group) | — | — |
| 11 | O que é formação de equipes? | equipes (equipe_group) | — | — |
| 12 | É trabalho em equipe. | equipe (equipe_group) | — | — |
| 13 | É trabalhar e colaborar com as pessoas de todas as origens especialidades, distância, fusos horários, para fazer um trabalho. | — | — | — |
| 14 | Pensem em seu time de esporte favorito porque ele é diferente. | esporte (esporte_activity) | — | — |
| 15 | Time de esportes trabalham juntos: mágica, aquelas jogadas que salvam os jogos. | esportes (esporte_activity) | — | — |
| 16 | Equipes esportivas ganham porque elas treinam. | Equipes (equipe_group) | — | — |
| 17 | Mas só é possível treinar se tiverem os mesmos membros ao longo do tempo. | — | — | — |
| 18 | Vocês podem pensar em formação de equipes. | equipes (equipe_group) | — | você/vocês (context-dependent) |

## 21. ted2020-27692_s63-s72

- **Document:** `ted2020-27692`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-27692.xml`
- **Sentence IDs:** 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
- **Length:** 10 sentences
- **Score / confidence:** 32.5 / **high**
- **Evidence:** 5 strong lexical hits (2 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 2 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 63 | Aposentar aplicativos é algo que todas as empresas fazem. | aplicativos (aplicativo_software) | — | — |
| 64 | Mas talvez a mentalidade de que esses aplicativos são colegas de trabalho poderia ajudá-los a decidir quando e como aposentar esses aplicativos de uma maneira menos destrutiva para o resto da equipe. | aplicativos (aplicativo_software); aplicativos (aplicativo_software); equipe (equipe_group) | — | — |
| 65 | Eu fiz esse experimento com 15 profissionais diferentes e cada vez isso provocou uma ideia. | — | — | — |
| 66 | Às vezes, um pouco mais. | — | — | — |
| 67 | Lembram-se daquela funcionária de quem lhes falei, que todo mundo estava ignorando? | — | — | estar + gerund |
| 68 | Essa foi uma história real contada por Christopher, um gerente de recursos humanos dinâmico de uma grande indústria de bens de consumo. | — | — | — |
| 69 | A tecnologia era uma nova plataforma de RH que tinha sido instalada há 14 meses por um valor alto, mas ninguém a estava usando. | — | — | estar + gerund |
| 70 | Então nós estávamos conversando: se ela realmente fosse uma funcionária tão talentosa com credenciais incríveis, faríamos de tudo para conhecê-la, talvez convidá-la para um café, saber da história dela. | — | — | — |
| 71 | Então, no espírito da experimentação, Christopher marcou reuniões de uma hora, com café opcional, apenas para os membros da equipe dele conhecerem o sistema de RH. | equipe (equipe_group) | — | — |
| 72 | Algumas pessoas clicaram em cada item do menu. | — | — | — |

## 22. ted2020-2433_s21-s30

- **Document:** `ted2020-2433`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2433.xml`
- **Sentence IDs:** 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
- **Length:** 10 sentences
- **Score / confidence:** 30.0 / **medium**
- **Evidence:** 4 strong lexical hits (1 types) across 4 sentences; 1 weaker lexical hits; 1 orthographic; 2 structural.
- **Why promising:** verified strong types: banheiro_room; supporting types: calcinha_underwear, genero_accent, progressive_estar_gerund.
- **Concerns:** Repeated instances of one strong lexical type; repetition is capped in scoring.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 21 | E eu não deveria ter que usar o banheiro masculino, porque eu não sou um homem. | banheiro (banheiro_room) | — | — |
| 22 | Sou uma pessoa trans. | — | — | — |
| 23 | E agora alguns políticos que gostam de incitar o medo estão tentando aprovar leis sobre os banheiros. | — | — | estar + gerund |
| 24 | Ouviram falar disso? | — | — | — |
| 25 | Estão tentando criar leis para forçar pessoas como eu a usar o banheiro que eles consideram apropriado de acordo com o gênero que me foi designado quando nasci. | banheiro (banheiro_room) | — | gênero→género; estar + gerund |
| 26 | E se esses políticos conseguirem o que querem no Arizona, na Califórnia ou na Flórida, ou semana passada mesmo, em Houston, Texas, ou em Ottawa, usar o banheiro masculino também não será uma opção legal para mim. | banheiro (banheiro_room) | — | — |
| 27 | E toda vez que algum desses políticos propõe uma lei dessas, eu me pergunto como se aplicaria uma lei dessas, e quem a aplicaria, sabe? | — | — | — |
| 28 | Verificação de calcinhas? | — | calcinhas (calcinha_underwear) | — |
| 29 | Sério. | — | — | — |
| 30 | Inspeções genitais na porta dos banheiros e vestiários das piscinas públicas? | banheiros (banheiro_room) | — | — |

## 23. ted2020-2609_s82-s91

- **Document:** `ted2020-2609`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2609.xml`
- **Sentence IDs:** 82, 83, 84, 85, 86, 87, 88, 89, 90, 91
- **Length:** 10 sentences
- **Score / confidence:** 30.75 / **high**
- **Evidence:** 4 strong lexical hits (2 types) across 4 sentences; 0 weaker lexical hits; 0 orthographic; 2 structural.
- **Why promising:** verified strong types: onibus_vehicle, trem_rail; supporting types: address_voce, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 82 | O problema dos trens urbanos é que, às vezes, temos que parar 5, 10, 15 vezes antes de chegar ao destino. | trens (trem_rail) | — | — |
| 83 | E se, nesse trem, não tivéssemos que parar? | — | — | — |
| 84 | Nesse trem, os vagões conseguem separar-se dinamicamente enquanto se movem e se tornam ônibus expressos e sem motorista que se movem em vias secundárias. | ônibus (onibus_vehicle) | — | — |
| 85 | E, sem uma parada sequer, nem baldeações demoradas, você está sentado em um ônibus que vai para o seu bairro. | ônibus (onibus_vehicle) | — | você/vocês (context-dependent) |
| 86 | E quanto estiver perto, a seção onde está sentado se separa e o conduz diretamente à sua porta. | — | — | — |
| 87 | É coletivo e individual ao mesmo tempo. | — | — | — |
| 88 | Esse pode ser um dos veículos modulares, compartilhados e sem motorista do futuro. | — | — | — |
| 89 | Agora | — | — | — |
| 90 | como se andar pela cidade com o zumbido de drones, táxis aéreos, ônibus modulares e cabines magnéticas suspensas não fosse exótico o bastante, acho que há outra força em ação que deixará o tráfego urbano fascinante. | ônibus (onibus_vehicle) | — | — |
| 91 | Se pararem pra pensar, a atual geração de carros sem motoristas está tentando ganhar seu espaço em meio à rede de tráfego feita por e para humanos. | — | — | estar + gerund |

## 24. ted2020-1711_s93-s100

- **Document:** `ted2020-1711`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1711.xml`
- **Sentence IDs:** 93, 94, 95, 96, 97.1, 97.2, 98.1, 98.2, 99, 100
- **Length:** 10 sentences
- **Score / confidence:** 30.75 / **high**
- **Evidence:** 4 strong lexical hits (2 types) across 4 sentences; 0 weaker lexical hits; 0 orthographic; 2 structural.
- **Why promising:** verified strong types: equipe_group, esporte_activity; supporting types: address_voce, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 93 | Eu quero usar tecnologia a qual estamos trabalhando e fazer acontecer fazer o sistema de saúde um esporte coletivo coordenado. | esporte (esporte_activity) | — | estar + gerund |
| 94 | Isso é o mais assustador pra mim. | — | — | — |
| 95 | De todos os cuidados que eu tive em todos os hospitais e clínicas pelo mundo, a primeira vez que eu tive uma experiencia real com uma equipe médica foi no Legacy Good Sam nesses últimos 6 meses pra eu chegar até aqui. | equipe (equipe_group) | — | — |
| 96 | Essa é uma foto da minha equipe de graduação do Legacy. | equipe (equipe_group) | — | — |
| 97.1 | Tem algumas pessoas aqui. | — | — | — |
| 97.2 | Vocês vão reconhecer o Dr. Batiuk. | — | — | você/vocês (context-dependent) |
| 98.1 | Acabamos de falar com ele. | — | — | — |
| 98.2 | Essa é a Jenny, uma das enfermeiras, Allison, que ajudou a gerenciar a lista de transplante, e uma outra dúzia de pessoas que não estão na foto. um farmacêutico, um psicólogo, um nutricionista, até mesmo uma conselheira financeira, Lisa que me ajudaram a lidar com todas os aborrecimentos com a seguradora. | — | — | — |
| 99 | Eu chorei no dia da graduação. | — | — | — |
| 100 | Eu deveria estar feliz, porque eu estava tão bem que eu poderia voltar pros meus médicos normais, mas eu chorei porque eu estava de fato conectado com essa equipe. | equipe (equipe_group) | — | — |

## 25. ted2020-17023_s63-s72

- **Document:** `ted2020-17023`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-17023.xml`
- **Sentence IDs:** 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
- **Length:** 10 sentences
- **Score / confidence:** 31.0 / **medium**
- **Evidence:** 6 strong lexical hits (1 types) across 5 sentences; 0 weaker lexical hits; 1 orthographic; 4 structural.
- **Why promising:** verified strong types: aplicativo_software; supporting types: genero_accent, address_voce, possessive_without_article, progressive_estar_gerund.
- **Concerns:** Repeated instances of one strong lexical type; repetition is capped in scoring.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 63 | E não são apenas grandes empresas investigando sua vida, os aplicativos gratuitos e pequenas "startups" também. | aplicativos (aplicativo_software) | — | — |
| 64 | Percebi no meu telefone que toda vez que eu registrava informação de exercício no aplicativo Endomondo, ele compartilhava detalhes como minha localização e gênero com seus anunciantes. | aplicativo (aplicativo_software) | — | gênero→género |
| 65 | WebMD, um aplicativo que verifica sintomas, compartilhava informações ainda mais delicadas, incluindo sintomas, procedimentos e remédios vistos pelos usuários no aplicativo, com terceiros. | aplicativo (aplicativo_software); aplicativo (aplicativo_software) | — | — |
| 66 | Fitbit compartilhava dados com o Yahoo. | — | — | — |
| 67 | Um aplicativo de acompanhamento de gravidez estava vendendo informações sobre o ciclo de ovulação e períodos férteis de suas usuárias para pessoas ou anunciantes como InMobi. | aplicativo (aplicativo_software) | — | estar + gerund |
| 68 | Desde que meu celular esteja ligado, minha localização pode ser rastreada, e não apenas pelos aplicativos mais óbvios, como Google Maps, mas toda uma gama de serviços não relacionados desde o Uber até o Twitter, Photos, Snapchat, TripAdvisor e outros. | aplicativos (aplicativo_software) | — | bare possessive (weak heuristic) |
| 69 | Você não está seguro nem em casa. | — | — | você/vocês (context-dependent) |
| 70 | Em 2015, descobriram que a Samsung estava gravando as pessoas nas casas que tinham uma TV vendida por eles, usando reconhecimento de voz. | — | — | estar + gerund |
| 71 | Eles já mudaram isso e só gravam quando o reconhecimento de voz está ativo. | — | — | — |
| 72 | Mas o fator assustador permanece. | — | — | — |

## 26. ted2020-335_s134-s141.2

- **Document:** `ted2020-335`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-335.xml`
- **Sentence IDs:** 134, 135, 136, 137, 138.1, 138.2, 139, 140, 141.1, 141.2
- **Length:** 10 sentences
- **Score / confidence:** 31.5 / **medium**
- **Evidence:** 2 strong lexical hits (1 types) across 2 sentences; 0 weaker lexical hits; 8 orthographic; 7 structural.
- **Why promising:** verified strong types: equipe_group; supporting types: premio_accent, progressive_estar_gerund.
- **Concerns:** Repeated instances of one strong lexical type; repetition is capped in scoring.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 134 | Alguém está oferecendo 10 milhões de dólares para ir e fazer isso. | — | — | estar + gerund |
| 135 | E cada uma dessas áreas foi algo que nós encontramos o Ansari X PRIZE ajudou a criar um curto-circuito para a inovação. | — | — | — |
| 136 | Então, como organização, montamos juntos um processo de descoberta de prêmio de como chegar a prêmios e escrever as regras, e na verdade estamos olhando para a criação de prêmios em um número de diferentes categorias. | — | — | prêmio→prémio; prêmio→prémio; prêmio→prémio; estar + gerund |
| 137 | Estamos olhando para investir em energia, meio ambiente, nanotecnologia - e eu vou falar mais sobre isso num momento. | — | — | estar + gerund |
| 138.1 | E a maneira que nós estamos fazendo é que estamos criando equipes de prêmios dentro da X PRIZE. | equipes (equipe_group) | — | prêmio→prémio; estar + gerund; estar + gerund |
| 138.2 | Temos uma equipe para prêmios relacionados com espaço. | equipe (equipe_group) | — | prêmio→prémio |
| 139 | Estamos indo atrás de um prêmio orbital. | — | — | prêmio→prémio |
| 140 | Nós estamos olhando para uma série de prêmios de energia. | — | — | prêmio→prémio; estar + gerund |
| 141.1 | Craig Venter acaba de entrar para o nosso conselho e estamos fazendo um prêmio de seqüenciamento do genoma rápido com ele. | — | — | prêmio→prémio; estar + gerund |
| 141.2 | Nós estaremos anunciando mais tarde esta queda, sobre - imagine ser capaz de sequenciar qualquer DNA por menos de USD 1.000, revolucionando a medicina. | — | — | — |

## 27. ted2020-1463_s85-s91

- **Document:** `ted2020-1463`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1463.xml`
- **Sentence IDs:** 85, 86, 87.1, 87.2, 88, 89, 90, 91
- **Length:** 8 sentences
- **Score / confidence:** 30.5 / **high**
- **Evidence:** 3 strong lexical hits (2 types) across 2 sentences; 0 weaker lexical hits; 0 orthographic; 5 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: address_voce, progressive_estar_gerund.
- **Concerns:** Contains a long sentence that may increase annotation burden. Contains 1 standalone stage/audience marker(s).

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 85 | ", como coloca MIchel Foucault. | — | — | — |
| 86 | Apenas para dar-lhes um exemplo prático de Buster Benson. | — | — | — |
| 87.1 | Este é Buster instalando um aparelho para flexões no escritório de sua nova empresa Habit Labs, onde eles estão tentando projetar outros aplicativos como 'Health Month' para as pessoas. | aplicativos (aplicativo_software) | — | estar + gerund |
| 87.2 | E por que ele está fazendo uma coisa dessas? | — | — | estar + gerund |
| 88 | Bem, aqui está o conjunto de proposições que a Habit Labs, a empresa de Buster, elaborou para eles mesmos de como queriam trabalhar juntos, como uma equipe, enquanto estão projetando esses aplicativos -- um conjunto de princípios morais que eles mesmos definiram para trabalhar juntos -- e um deles é: "Nós cuidamos de nossa própria saúde e administramos nossos exercícios físicos." | aplicativos (aplicativo_software); equipe (equipe_group) | — | estar + gerund |
| 89 | Porque, em última análise, como podem perguntar a si mesmos e como podem encontrar uma resposta sobre qual visão de vida boa querem transmitir e criar com seus designs sem fazer o questionamento de qual visão de vida boa vocês mesmos querem viver? | — | — | você/vocês (context-dependent) |
| 90 | E com isto, agradeço a vocês. | — | — | você/vocês (context-dependent) |
| 91 | (Aplausos) | — | — | — |

## 28. ted2020-46587_s40-s49

- **Document:** `ted2020-46587`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-46587.xml`
- **Sentence IDs:** 40, 41, 42, 43, 44, 45, 46, 47, 48, 49
- **Length:** 10 sentences
- **Score / confidence:** 30.25 / **high**
- **Evidence:** 5 strong lexical hits (2 types) across 4 sentences; 0 weaker lexical hits; 0 orthographic; 1 structural.
- **Why promising:** verified strong types: equipe_group, esporte_activity; supporting types: possessive_without_article.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 40 | Devo confessar que, a princípio, isso me pareceu tirar toda a diversão do esporte. | esporte (esporte_activity) | — | — |
| 41 | Não pude vibrar após uma grande vitória contra nossos jogadores mais fortes, porque seis minutos depois, com um novo parceiro, eu ficava para trás novamente. | — | — | — |
| 42 | Por outro lado, nunca me senti desconsolado. | — | — | — |
| 43 | Quando deixei o Japão e comecei a jogar partidas simples de novo com meu arquirrival inglês, notei que, após cada derrota, eu ficava muito desolado. | — | — | bare possessive (weak heuristic) |
| 44 | Mas, após cada vitória, eu também não conseguia dormir, porque sabia que só havia um caminho a percorrer, e era para baixo. | — | — | — |
| 45 | Se eu tentasse fazer negócios no Japão, isso levaria a uma frustração sem fim. | — | — | — |
| 46 | No Japão, diferente de outros lugares, se o placar ainda estiver equilibrado após quatro horas, um jogo de beisebol termina empatado e, como a classificação da liga baseia-se no percentual de vitórias, uma equipe com muitos empates pode terminar à frente de uma equipe com mais vitórias. | equipe (equipe_group); equipe (equipe_group) | — | — |
| 47 | Uma das primeiras vezes que levaram um norte-americano ao Japão para liderar uma equipe profissional de beisebol japonês, Bobby Valentine, em 1995, ele assumiu esse elenco muito medíocre, levou-o a um impressionante segundo lugar e foi imediatamente demitido. | equipe (equipe_group) | — | — |
| 48 | Por quê? | — | — | — |
| 49 | "Bem", disse o porta-voz da equipe, "por causa da ênfase dele em vencer". | equipe (equipe_group) | — | — |

## 29. ted2020-48847_s104-s111

- **Document:** `ted2020-48847`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-48847.xml`
- **Sentence IDs:** 104, 105, 106, 107, 108.1, 108.2, 109, 110.1, 110.2, 111
- **Length:** 10 sentences
- **Score / confidence:** 29.0 / **high**
- **Evidence:** 3 strong lexical hits (2 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 2 structural.
- **Why promising:** verified strong types: esporte_activity, terno_suit; supporting types: progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 104 | E como resultado, um jeito de ter uma dica daquilo que o futuro nos reserva é olhar para esta divertida "caixa de areia" interativa de videogame e se perguntar: "O que os jogadores estão fazendo hoje?" | — | — | estar + gerund |
| 105 | E isso pode lhes dar uma dica sobre o que o futuro reserva para todos nós. | — | — | — |
| 106 | Uma das coisas que já estamos vendo no Twitch é entretenimento multijogador chegando aos esportes. | esportes (esporte_activity) | — | estar + gerund |
| 107 | O Twitch e a NFL se uniram para oferecer futebol norte-americano ao vivo mas, em vez de apresentadores da emissora transmitindo o jogo usando terno, nós temos usuários do Twitch transmitindo em seu próprio canal, interagindo com sua comunidade, numa verdadeira experiência multijogador. | terno (terno_suit) | — | — |
| 108.1 | Então, na verdade, acho que se olharmos para o futuro... | — | — | — |
| 108.2 | Hoje apenas centenas de pessoas são locutores esportivos. | — | — | — |
| 109 | É um número minúsculo de pessoas que têm essa oportunidade. | — | — | — |
| 110.1 | Mas os esportes estão prestes a se tornar multijogador. | esportes (esporte_activity) | — | — |
| 110.2 | E isso significa que quem quiser em qualquer lugar do mundo vai ter a oportunidade de se tornar um locutor esportivo, pelo menos tentar. | — | — | — |
| 111 | E acho que isso vai revelar quantidades incríveis de novos talentos pra todos nós. | — | — | — |

## 30. ted2020-26073_s61-s69

- **Document:** `ted2020-26073`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-26073.xml`
- **Sentence IDs:** 61, 62, 63, 64, 65, 66, 67, 68.1, 68.2, 69
- **Length:** 10 sentences
- **Score / confidence:** 30.25 / **high**
- **Evidence:** 5 strong lexical hits (2 types) across 4 sentences; 0 weaker lexical hits; 0 orthographic; 1 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: possessive_without_article.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 61 | E se seu robô sexual fizer compras irresistíveis pelo aplicativo? | aplicativo (aplicativo_software) | — | — |
| 62 | (Risos) Porque robôs mais capitalismo equivale a questões relacionadas à proteção do consumidor e à privacidade. | — | — | — |
| 63 | Mas não é só por isso que nosso comportamento em relação a essas máquinas pode ser importante. | — | — | — |
| 64 | Alguns anos após essa primeira experiência com o dinossauro robô bebê, fiz um workshop com meu amigo Hannes Gassert. | — | — | bare possessive (weak heuristic) |
| 65 | Pegamos cinco desses dinossauros robôs bebês e os demos a cinco equipes de pessoas. | equipes (equipe_group) | — | — |
| 66 | Pedimos que dessem nomes aos robôs, brincassem e interagissem com eles durante cerca de uma hora. | — | — | — |
| 67 | Depois, mostramos um martelo e uma machadinha e dissemos às equipes para torturarem e matarem os robôs. | equipes (equipe_group) | — | — |
| 68.1 | (Risos) A situação ficou um pouco mais dramática do que esperávamos, porque nenhum dos participantes nem mesmo bateria nos dinossauros robôs bebês. | — | — | — |
| 68.2 | Então, tivemos que improvisar um pouco, e, a certa altura, dissemos: "Está bem, podem salvar o robô de sua equipe se destruírem o da outra equipe". | equipe (equipe_group); equipe (equipe_group) | — | — |
| 69 | (Risos) Nem mesmo isso funcionou; eles não conseguiram. | — | — | — |

## 31. ted2020-1434_s77-s86

- **Document:** `ted2020-1434`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1434.xml`
- **Sentence IDs:** 77, 78, 79, 80, 81, 82, 83, 84, 85, 86
- **Length:** 10 sentences
- **Score / confidence:** 28.0 / **high**
- **Evidence:** 4 strong lexical hits (2 types) across 4 sentences; 0 weaker lexical hits; 0 orthographic; 0 structural.
- **Why promising:** verified strong types: caminhao_vehicle, carona_ride; supporting types: none.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 77 | Podemos usar TI inteligentes para melhorar o trânsito e facilitar o carro compartilhado e a carona. | carona (carona_ride) | — | — |
| 78 | Podemos empregar modelos de crescimento inteligentes e lucrativos que ajudem as pessoas a estarem próximas de onde elas querem ir, para que não tenham que ir para outro lugar. | — | — | — |
| 79 | E podemos usar TI inteligente para fazer o tráfego mais fluido. | — | — | — |
| 80 | Juntas, estas coisas podem nos dar o mesmo ou um melhor acesso com 46 a 84% a menos de viagens, economizando outros 0.4 trilhões de dólares. mais 0.3 trilhões de dólares com o uso mais produtivo de caminhões. | caminhões (caminhao_vehicle) | — | — |
| 81 | Assim daqui a 40 anos, tudo somado, podemos ter mais mobilidade nos E.U.A. sem usar petróleo. | — | — | — |
| 82 | Economizando ou substituíndo barris a 25 dólares ao invés de comprá-los por mais de cem, resulta em uma economia líquida de 4 trilhões supondo todos os custos ocultos como zero. | — | — | — |
| 83 | Então ter mobilidade sem petróleo, para eliminar o petróleo, nós podemos nos tornar eficientes e depois trocar de combustíveis. | — | — | — |
| 84 | Estes veículos gastam o equivalente a 50 a 100km/l podem usar uma combinação de mistura de células de hidrogênio como combustível, eletricidade e biocombustíveis avançados. | — | — | — |
| 85 | Os caminhões e aviões poderão usar hidrogênio ou bio-combustíveis avançados. | caminhões (caminhao_vehicle) | — | — |
| 86 | Os caminhões podem ainda usar gás natural. | caminhões (caminhao_vehicle) | — | — |

## 32. ted2020-1744_s51-s60

- **Document:** `ted2020-1744`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1744.xml`
- **Sentence IDs:** 51, 52, 53, 54, 55, 56, 57, 58, 59, 60
- **Length:** 10 sentences
- **Score / confidence:** 28.75 / **high**
- **Evidence:** 2 strong lexical hits (2 types) across 2 sentences; 0 weaker lexical hits; 0 orthographic; 16 structural.
- **Why promising:** verified strong types: equipe_group, geladeira_appliance; supporting types: address_voce, possessive_without_article, progressive_estar_gerund.
- **Concerns:** Contains a long sentence that may increase annotation burden.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 51 | Cole esse conjunto em sua geladeira. | geladeira (geladeira_appliance) | — | — |
| 52 | Quando você aparece no pronto-socorro com esse envelope, sua admissão é simplificada na prática. | — | — | você/vocês (context-dependent) |
| 53 | Você vai precisar de cuidadores. | — | — | você/vocês (context-dependent) |
| 54 | Você precisará avaliar sua personalidade e situação financeira para determinar se uma comunidade de assistência a idosos ou ficar em casa é sua melhor opção. | — | — | você/vocês (context-dependent) |
| 55 | Em ambos os casos, não se acomode. | — | — | — |
| 56 | Passamos por uma série de cuidadores não tão bons antes de encontrarmos a equipe perfeita, comandada por Marsha, que não deixará você ganhar no bingo só porque você está morrendo, mas irá fazer vídeos de seu rancho para você quando você não puder ir lá, e Caitlin, que não deixará você perder os exercícios matinais, mas sabe quando você precisa ouvir que sua esposa está em boas mãos. | equipe (equipe_group) | — | estar + gerund; você/vocês (context-dependent); você/vocês (context-dependent); você/vocês (context-dependent); você/vocês (context-dependent); você/vocês (context-dependent); você/vocês (context-dependent) |
| 57 | Por fim, últimas palavras. | — | — | — |
| 58 | O que você quer ouvir no final, e de quem você quer ouvir isso? | — | — | você/vocês (context-dependent); você/vocês (context-dependent) |
| 59 | Em minha experiência, você vai querer ouvir que, seja qual for a sua preocupação, vai ficar tudo bem. | — | — | você/vocês (context-dependent); bare possessive (weak heuristic) |
| 60 | Quando você acreditar que está bem para ir, você irá. | — | — | você/vocês (context-dependent); você/vocês (context-dependent) |

## 33. ted2020-2326_s116-s125

- **Document:** `ted2020-2326`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2326.xml`
- **Sentence IDs:** 116, 117, 118, 119, 120, 121, 122, 123, 124, 125
- **Length:** 10 sentences
- **Score / confidence:** 29.5 / **medium**
- **Evidence:** 1 strong lexical hits (1 types) across 1 sentences; 1 weaker lexical hits; 0 orthographic; 7 structural.
- **Why promising:** verified strong types: cafe_da_manha; supporting types: rodovia_road, progressive_estar_gerund.
- **Concerns:** Strong evidence is concentrated in one sentence. Contains a long sentence that may increase annotation burden.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 116 | A educação está mudando da mesma maneira. | — | — | estar + gerund |
| 117 | Os cuidados com a saúde estão mudando do mesmo jeito. | — | — | estar + gerund |
| 118 | Como os serviços governamentais são oferecidos está mudando. | — | — | estar + gerund |
| 119 | Mesmo assim, em Washington, estamos discutindo se chamamos o grupo terrorista que tomou a Síria e o Iraque de ISIS, ISIL ou Estado Islâmico. | — | — | estar + gerund |
| 120 | Estamos tentando determinar o quanto queremos ceder em uma negociação com os iranianos, em um acordo nuclear que lida com tecnologias de 50 anos atrás, quando, na verdade, sabemos que eles agora estão engajados numa guerra cibernética contra nós e o estamos ignorando, em parte porque empresas não estão dispostas a falar sobre os ataques que estão sendo lançados sobre elas. | — | — | estar + gerund; estar + gerund; estar + gerund |
| 121 | E isso nos leva a um outro fracasso que é crucial, e outro fracasso que não poderia ser mais importante para um grupo como esse, porque o crescimento da América e uma real segurança nacional americana e todas as coisas que conduziram ao progresso, mesmo durante a Guerra Fria, fori uma parceria público-privada entre a ciência, tecnologia e governo, que começou quando Thomas Jefferson se sentou sozinho em seu laboratório inventando coisas novas. | — | — | — |
| 122 | Mas foram os canais, as rodovias, o telégrafo; foi o radar e a internet. | — | rodovias (rodovia_road) | — |
| 123 | Foi o Tang, a bebida do café da manhã -- provavelmente não o mais importante desses desenvolvimentos. | café da manhã (cafe_da_manha) | — | — |
| 124 | Mas o que se tinha era uma parceria e um diálogo e o diálogo fracassou. | — | — | — |
| 125 | Fracassou porque em Washington menos governo é considerado mais. | — | — | — |

## 34. ted2020-48104_s35.2-s44

- **Document:** `ted2020-48104`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-48104.xml`
- **Sentence IDs:** 35.2, 36, 37, 38, 39, 40, 41, 42, 43, 44
- **Length:** 10 sentences
- **Score / confidence:** 30.0 / **high**
- **Evidence:** 5 strong lexical hits (2 types) across 5 sentences; 0 weaker lexical hits; 0 orthographic; 0 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: none.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 35.2 | Essa é a equipe que criamos para as meninas. | equipe (equipe_group) | — | — |
| 36 | Depois que recrutamos voluntários de nossa comunidade, oferecemos a eles treinamento, orientação e apoio. | — | — | — |
| 37 | É quando começa nosso trabalho. | — | — | — |
| 38 | Primeiro, identificamos cada menina que não está indo para a escola. | — | — | — |
| 39 | Mas fazemos isso de modo diferente e com alta tecnologia, pelo menos a meu ver. | — | — | — |
| 40 | Cada membro de nossa equipe tem um smartphone, | equipe (equipe_group) | — | — |
| 41 | com o aplicativo Educate Girls, | aplicativo (aplicativo_software) | — | — |
| 42 | que tem tudo de que nossa equipe precisa: | equipe (equipe_group) | — | — |
| 43 | mapas digitais dos locais onde farão a pesquisa, o questionário com todas as perguntas, pequenos guias sobre como melhor realizar a pesquisa para que os dados cheguem até nós em tempo real e com boa qualidade. | — | — | — |
| 44 | Equipados com isso, nossas equipes e nossos voluntários vão de porta em porta, até cada família, para encontrar as meninas que nunca se matricularam ou que deixaram de ir à escola. | equipes (equipe_group) | — | — |

## 35. ted2020-1933_s42-s51

- **Document:** `ted2020-1933`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1933.xml`
- **Sentence IDs:** 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
- **Length:** 10 sentences
- **Score / confidence:** 29.25 / **high**
- **Evidence:** 2 strong lexical hits (2 types) across 2 sentences; 0 weaker lexical hits; 1 orthographic; 5 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: fenomeno_accent, address_voce, progressive_estar_gerund.
- **Concerns:** Contains a long sentence that may increase annotation burden.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 42 | no Dia Nacional de <i>Hackeamento</i> Cívico, junho passado em Oakland, onde eu moro, a equipe do Code for American de Oakland pegou o código aberto de base do Honolulu Answers e transformou em Oakland Answers, e novamente organizamos uma "writeathon" em que nós pegamos as perguntas mais frequentes e pedimos aos cidadãos para respondê-las, e eu coloquei a mão na massa. | equipe (equipe_group) | — | — |
| 43 | eu sou autora dessa resposta, e de algumas outras. | — | — | — |
| 44 | E eu estou tentando articular neste dia a noção de fortalecimento e responsabilidade que eu sinto pelo lugar onde vivo baseada simplesmente neste pequeno ato de participação. | — | — | estar + gerund |
| 45 | E ao unir meu pequeno ato aos milhares de pequenos atos de participação de outros que foram possíveis através do <i>hackeamento</i> cívico, achamos que podemos reenergizar a cidadania e restaurar a confiança no governo. | — | — | — |
| 46 | Vocês podem estar imaginando o que governantes municipais pensam disso. | — | — | você/vocês (context-dependent) |
| 47 | Eles na verdade adoraram. | — | — | — |
| 48 | Como muitos de vocês sabem, é pedido às cidades todos os dias para se fazer mais com menos, e elas estão sempre procurando por soluções inovadoras para problemas intrincados. | — | — | você/vocês (context-dependent) |
| 49 | Então quando você dá aos cidadãos uma maneira de participar além de atender as reuniões de município, as cidades podem realmente capturar a capacidade em seus municípios para fazer os negócios do governo. | — | — | você/vocês (context-dependent) |
| 50 | Agora, eu não quero deixar a impressão de que <i>hackeamento</i> cívico é apenas um fenômeno americano. | — | — | fenômeno→fenómeno |
| 51 | Está acontecendo ao redor do globo, e um dos meus exemplos favoritos é da Cidade do México, onde no começo deste ano, a Câmara de Deputados do México firmou contrato com uma firma desenvolvedora de software para criar um aplicativo que possibilitasse aos legisladores rastrear leis. | aplicativo (aplicativo_software) | — | estar + gerund |

## 36. ted2020-502_s32-s41

- **Document:** `ted2020-502`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-502.xml`
- **Sentence IDs:** 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
- **Length:** 10 sentences
- **Score / confidence:** 29.0 / **high**
- **Evidence:** 2 strong lexical hits (2 types) across 2 sentences; 0 weaker lexical hits; 0 orthographic; 10 structural.
- **Why promising:** verified strong types: caminhao_vehicle, esporte_activity; supporting types: address_voce, possessive_without_article, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 32 | Ou neste, onde todos estavam congelando, menos eu, porque na Áustria estava muito frio Onde fizemos essa filmagem. | — | — | estar + gerund |
| 33 | Aqui todos estão sentados em uma cesta e eu estava sobre o balão, pronto para escorregar com minha prancha de surfe no ceú. | — | — | bare possessive (weak heuristic) |
| 34 | Ou nesse pulo a partir de um caminhão se movendo na estrada. | caminhão (caminhao_vehicle) | — | — |
| 35 | (risadas) Esportes radicas de alto nível como este somente são possíveis se você pratica passo a passo, se você pratica muito suas habilidades e seu conhecimento. | Esportes (esporte_activity) | — | você/vocês (context-dependent); você/vocês (context-dependent) |
| 36 | Naturalmente você precisa estar em condições físicas muito boas, e por isso, tenho treinado muito. | — | — | você/vocês (context-dependent) |
| 37 | Você precisa ter o melhor equipamento possível. | — | — | você/vocês (context-dependent) |
| 38 | E, provavelmente o mais importante é que você precisa trabalhar nas suas habilidades mentais, sua preparação mental. | — | — | você/vocês (context-dependent) |
| 39 | E tudo isso para se chegar o mais próximo possível ao sonho humano de ser apto a voar. | — | — | — |
| 40 | Então para 2009 Estou treinando para meus 2 novos projetos. | — | — | estar + gerund; bare possessive (weak heuristic) |
| 41 | O primeiro é que eu quero estabelecer um novo recorde mundial de voar a partir de um penhasco com meu macacão. | — | — | bare possessive (weak heuristic) |

## 37. ted2020-1753_s143-s149

- **Document:** `ted2020-1753`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1753.xml`
- **Sentence IDs:** 143, 144, 145.1, 145.2, 146.1, 146.2, 146.3, 147, 148, 149
- **Length:** 10 sentences
- **Score / confidence:** 29.25 / **medium**
- **Evidence:** 3 strong lexical hits (2 types) across 1 sentences; 0 weaker lexical hits; 1 orthographic; 4 structural.
- **Why promising:** verified strong types: equipe_group, esporte_activity; supporting types: genero_accent, address_voce, progressive_estar_gerund.
- **Concerns:** Strong evidence is concentrated in one sentence.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 143 | Precisamos de treinamento de liderança, porque, por exemplo, quando um treinador profissional ou um gerente de uma equipe de basebol ou de futebol -- e também trabalhei amplamente nessa área -- faz um comentário sexista, faz um afirmação homofóbica, faz um comentário racista, haverá discussões nos blogs de esportes e programas de esportes no rádio. | esportes (esporte_activity); esportes (esporte_activity); equipe (equipe_group) | — | — |
| 144 | Alguns vão dizer: "Bem, ele precisa de um trabalho de sensibilização". | — | — | — |
| 145.1 | E outras vão dizer: "Ah, pare com isso. | — | — | — |
| 145.2 | Sabe, é o politicamente correto que perdeu o controle, | — | — | — |
| 146.1 | e acabou dizendo uma estupidez. | — | — | — |
| 146.2 | Deixa pra lá." | — | — | — |
| 146.3 | Eu digo que ele não precisa de um trabalho de sensibilização. | — | — | — |
| 147 | Ele precisa de treinamento de liderança, | — | — | — |
| 148 | porque está sendo um líder ruim, porque, numa sociedade com diversidade de gênero e diversidade sexual -- (Aplausos) — e diversidade racial e étnica, se você faz esse tipo de comentário, está falhando em sua liderança. | — | — | gênero→género; estar + gerund; estar + gerund; você/vocês (context-dependent) |
| 149 | Se pudermos argumentar da maneira que estou fazendo, com homens e mulheres poderosos em nossa sociedade, em todos os níveis de autoridade institucional e de poder, isso vai mudar, isso vai mudar o paradigma de pensamento das pessoas. | — | — | estar + gerund |

## 38. ted2020-28518_s54-s63

- **Document:** `ted2020-28518`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-28518.xml`
- **Sentence IDs:** 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
- **Length:** 10 sentences
- **Score / confidence:** 29.5 / **high**
- **Evidence:** 4 strong lexical hits (2 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 3 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: address_voce, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 54 | Três mil funcionários foram reorganizados em 350 equipes multidisciplinares. | equipes (equipe_group) | — | — |
| 55 | Em vez de pessoal de produtos sentado só com pessoal de produtos e engenheiros com engenheiros, uma pessoa do produto e um engenheiro eram agora membros da mesma equipe. | equipe (equipe_group) | — | — |
| 56 | Você poderia ser de uma equipe responsável pela abertura de conta ou pelo aplicativo de celular de banco, etc. | aplicativo (aplicativo_software); equipe (equipe_group) | — | você/vocês (context-dependent) |
| 57 | Na data de ativação dessa nova organização, algumas pessoas estavam apertando as mãos pela primeira vez, só para descobrir que se sentavam a dois minutos de distância um do outro, mas enviavam e-mails e relatórios um ao outro nos últimos dez anos. | — | — | estar + gerund |
| 58 | Ouvíamos alguém dizendo: "Ah, então, você é o cara que eu sempre procurava para as respostas". | — | — | você/vocês (context-dependent) |
| 59 | (Risos) Mas agora, eles tomam café juntos todos os dias. | — | — | — |
| 60 | Se o cara do produto tem uma ideia, ele pode apenas levantá-la para obter informações do engenheiro que está sentado ao lado dele. | — | — | — |
| 61 | Eles podem decidir testar com os clientes imediatamente, sem passagens de controle, sem PowerPoint, sem burocracia, apenas fazendo o que tem que ser feito. | — | — | — |
| 62 | Mas chegar lá não é fácil. | — | — | — |
| 63 | Como se vê, "Do que está disposto a desistir?" | — | — | — |

## 39. ted2020-1715_s94-s102

- **Document:** `ted2020-1715`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1715.xml`
- **Sentence IDs:** 94, 95, 96, 97, 98.1, 98.2, 99, 100, 101, 102
- **Length:** 10 sentences
- **Score / confidence:** 28.75 / **high**
- **Evidence:** 4 strong lexical hits (2 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 2 structural.
- **Why promising:** verified strong types: caminhao_vehicle, onibus_vehicle; supporting types: address_voce, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 94 | Então, Abed me mostrou um foto do seu caminhão amassado, e disse que o batida foi culpa do motorista de um ônibus na pista esquerda, que não o deixou ultrapassar. | ônibus (onibus_vehicle); caminhão (caminhao_vehicle) | — | — |
| 95 | Eu não queria retomar a história do acidente com Abed. | — | — | — |
| 96 | Esperava algo mais simples: trocar uma sobremesa turca por duas palavras e seguir meu caminho. | — | — | — |
| 97 | Não falei que em seu próprio depoimento, na manhã após o acidente, Abed nem sequer mencionou esse motorista de ônibus. | ônibus (onibus_vehicle) | — | — |
| 98.1 | Não, fiquei quieto. | — | — | — |
| 98.2 | Fiquei quieto porque não tinha ido atrás da verdade. | — | — | — |
| 99 | Tinha ido atrás de arrependimento. | — | — | — |
| 100 | Então, agora estava procurando por arrependimento e joguei a verdade em baixo do ônibus. | ônibus (onibus_vehicle) | — | estar + gerund |
| 101 | "Entendo" - eu disse - "que o acidente não foi culpa sua, mas você não lamenta o fato de outros terem sofrido?" | — | — | você/vocês (context-dependent) |
| 102 | Abed disse três rápidas palavras. | — | — | — |

## 40. ted2020-23845_s21-s30

- **Document:** `ted2020-23845`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-23845.xml`
- **Sentence IDs:** 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
- **Length:** 10 sentences
- **Score / confidence:** 29.0 / **high**
- **Evidence:** 4 strong lexical hits (2 types) across 3 sentences; 1 weaker lexical hits; 0 orthographic; 0 structural.
- **Why promising:** verified strong types: caminhao_vehicle, equipe_group; supporting types: rodovia_road.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 21 | Em 1965, a viagem trilha abaixo podia ser feita por caminhão. | caminhão (caminhao_vehicle) | — | — |
| 22 | Milhares de caminhões fornecidos pela China e pela Rússia assumiram a tarefa em meio a ferozes bombardeios dos B-52 e os motoristas dos caminhões tornaram-se conhecidos como pilotos do chão. | caminhões (caminhao_vehicle); caminhões (caminhao_vehicle) | — | — |
| 23 | À medida que o tráfego trilha abaixo aumentou, também cresceram os bombardeios dos E.U.A. | — | — | — |
| 24 | Eles dirigiam à noite ou bem de manhãzinha para evitar ataques aéreos, e vigias estavam prontos para avisar os motoristas da artilharia aérea inimiga. | — | — | — |
| 25 | Vilas ao longo da trilha organizaram equipes para garantir o fluxo do tráfego e ajudar os motoristas a consertar os estragos causados por ataques aéreos. | equipes (equipe_group) | — | — |
| 26 | O grito de guerra deles era: "Tudo por nossos irmãos do Sul!" | — | — | — |
| 27 | e "Não nos preocuparemos com nossas casas se os veículos não tiverem passado." | — | — | — |
| 28 | Famílias doaram suas portas e camas de madeira para consertar estradas. | — | — | — |
| 29 | As forças vietnamitas usaram até mesmo o logro para fazer aviões dos E.U.A. bombardear encostas de montanhas a fim de conseguir cascalho para usar na construção e manutenção das rodovias. | — | rodovias (rodovia_road) | — |
| 30 | A penetrante poeira vermelha se infiltrava em todas as brechas e fendas. | — | — | — |

## 41. ted2020-1452_s19-s27

- **Document:** `ted2020-1452`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1452.xml`
- **Sentence IDs:** 19, 20, 21, 22, 23.1, 23.2, 24, 25, 26, 27
- **Length:** 10 sentences
- **Score / confidence:** 27.25 / **high**
- **Evidence:** 2 strong lexical hits (2 types) across 2 sentences; 1 weaker lexical hits; 0 orthographic; 7 structural.
- **Why promising:** verified strong types: banheiro_room, esporte_activity; supporting types: xicara_cup, address_voce.
- **Concerns:** Contains a long sentence that may increase annotation burden.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 19 | E embora seja leve, tão leve que você pode levantar com a mão e mover, tem lugar para quatro pessoas dormirem. | — | — | você/vocês (context-dependent) |
| 20 | E você pode organizar estas coisas para ficarem maiores para acampamentos e até um layout de uma cidade ou você pode, em essência, colocar os vagões em círculos, e formar esses centros circulares, o que resulta numa área comunitária semi-privada para as pessoas sairem, para que não estejam na verdade presas dentro dessas unidades | — | — | você/vocês (context-dependent); você/vocês (context-dependent) |
| 21 | Agora, isso muda fundamentalmente a forma como reagimos aos desastres, porque se foram as condições horríveis dentro de uma arena de esportes ou um ginásio, na quais as pessoas estão empilhadas nestas camas de campanha. | esportes (esporte_activity) | — | — |
| 22 | Agora temos quarteirões instantâneos do lado de fora | — | — | — |
| 23.1 | O Exo é projetado para ser simplesmente, basicamente como uma xícara de café. | — | xícara (xicara_cup) | — |
| 23.2 | Eles podem de fato ser empilhados então temos um transporte e armazenamento extremamente eficientes. | — | — | — |
| 24 | Na verdade, 15 Exos podem caber em uma única caminhonete | — | — | — |
| 25 | Isso significa que o Exo pode ser transportado e montado mais rápido do que qualquer opção de moradia disponível hoje em dia. | — | — | — |
| 26 | Mas sou obsessivo, então não podia simplesmente parar ali, então eu comecei a modificar as beliches, assim você pode retirar as beliches, e colocar mesas e prateleiras assim a mesma unidade pode agora ser usada como escritório ou armazém. | — | — | você/vocês (context-dependent) |
| 27 | As portas podem ser trocadas, e você pode colocar um painel rígido com uma janela nele para controle de temperatura ou um módulo conector que permite que você conecte múltiplas unidades juntas, o que dá a você espaços maiores e separando os espaços em comodos, então agora esse mesmo kit com partes, esta unidade pode servir como sala, quarto ou banheiro, ou escritório, moradia ou armazém seguro. | banheiro (banheiro_room) | — | você/vocês (context-dependent); você/vocês (context-dependent); você/vocês (context-dependent) |

## 42. ted2020-1987_s15-s24

- **Document:** `ted2020-1987`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1987.xml`
- **Sentence IDs:** 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
- **Length:** 10 sentences
- **Score / confidence:** 29.0 / **high**
- **Evidence:** 3 strong lexical hits (2 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 2 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 15 | Aqui temos uma demostração de um bebê de três meses passando por um teste de visão preciso, usando um aplicativo e um rastreador ocular. | aplicativo (aplicativo_software) | — | — |
| 16 | Estamos fazendo vários testes na comunidade e nas escolas, e com as lições que aprendemos no campo, percebemos que é muito importante passar a informação sem termos médicos para que as pessoas entendam o que estamos examinando e o que isso significa. | — | — | estar + gerund; estar + gerund |
| 17 | Aqui, por exemplo, usamos nosso aplicativo Sight-Sim depois que sua visão foi medida, para mostrar aos pais e professores como aquela pessoa enxerga o mundo, para que possam entender como se sentem e ajudá-los. | aplicativo (aplicativo_software) | — | — |
| 18 | Uma vez descoberto que alguém tem baixa visão, o próximo desafio é entender por quê, e para ser capaz de fazer isso, precisamos ter acesso ao interior do olho. | — | — | — |
| 19 | Tradicionalmente, seria necessário um equipamento caro para examinar uma área chamada retina. | — | — | — |
| 20 | A retina é a parte única do olho que tem muita informação sobre o corpo e sua saúde. | — | — | — |
| 21 | Desenvolvemos hardware de baixo custo impresso em 3D que pode ser produzido por menos de cinco dólares, que pode ser acoplado a um smartphone e possibilita obter uma visão da parte preta do olho com uma qualidade altíssima. | — | — | — |
| 22 | E a beleza disso é que qualquer um pode fazer. | — | — | — |
| 23 | Em nossos testes em mais de 2.500 pessoas, o smartphone com a modificação se compara a uma câmera que é imensamente mais cara e imensamente mais difícil de transportar. | — | — | — |
| 24 | Quando nos mudamos para o Quênia, fomos com 150 mil dólares em equipamentos, uma equipe de 15 pessoas, e era disso que precisávamos para oferecer atendimento médico. | equipe (equipe_group) | — | — |

## 43. ted2020-36844_s127-s136

- **Document:** `ted2020-36844`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-36844.xml`
- **Sentence IDs:** 127, 128, 129, 130, 131, 132, 133, 134, 135, 136
- **Length:** 10 sentences
- **Score / confidence:** 28.75 / **high**
- **Evidence:** 3 strong lexical hits (2 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 4 structural.
- **Why promising:** verified strong types: aplicativo_software, equipe_group; supporting types: address_voce, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 127 | Acho que a ideia básica é comercializar algo que você sabe que será bom no futuro, mas ainda é um pouco ruim. | — | — | você/vocês (context-dependent) |
| 128 | No nosso caso, o produto era um aplicativo iOS. | aplicativo (aplicativo_software) | — | — |
| 129 | Criamos um escritório no quintal do Dom e, em 1º de agosto de 2018, lançamos a primeira versão. | — | — | — |
| 130 | Houve mais de 30 mil downloads nas primeiras 24 horas, e tivemos mais de 30 mil comentários perguntando quando a versão Android ficaria pronta. | — | — | — |
| 131 | Apesar de o aplicativo ser bem imperfeito e ter sido criado por um só homem, era disso que precisávamos para as pessoas investirem. | aplicativo (aplicativo_software) | — | — |
| 132 | Aprendemos muito com os usuários zangados e os investidores apavorantes. | — | — | — |
| 133 | Ao longo dos últimos seis meses, aumentamos a equipe para 16 pessoas. | equipe (equipe_group) | — | — |
| 134 | De lá para cá, estamos criando a segunda versão, que lançaremos em abril. | — | — | estar + gerund |
| 135 | Na indústria criativa há uma expressão comum que acho que todos conhecemos bem. | — | — | — |
| 136 | "O importante não é o que você sabe, mas quem você conhece." | — | — | você/vocês (context-dependent); você/vocês (context-dependent) |

## 44. ted2020-1236_s121-s130

- **Document:** `ted2020-1236`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1236.xml`
- **Sentence IDs:** 121, 122, 123, 124, 125, 126, 127, 128, 129, 130
- **Length:** 10 sentences
- **Score / confidence:** 28.25 / **medium**
- **Evidence:** 2 strong lexical hits (1 types) across 2 sentences; 1 weaker lexical hits; 1 orthographic; 5 structural.
- **Why promising:** verified strong types: caminhao_vehicle; supporting types: rodovia_road, fenomeno_accent, address_voce, progressive_estar_gerund.
- **Concerns:** Repeated instances of one strong lexical type; repetition is capped in scoring.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 121 | Eles estão tomando essas bolsas e roupas como um senso de identidade e status social. | — | — | estar + gerund |
| 122 | E esta é uma garota dizendo explicitamente num programa de namoro na TV que ela prefere chorar numa BMW a sorrir numa bicicleta. | — | — | — |
| 123 | Mas, é claro, realmente temos jovens que ainda preferem sorrir, seja numa BMW ou numa bicicleta. | — | — | — |
| 124 | Na foto seguinte, vocês veem um fenômeno muito popular chamado casamento "nu" ou união "pelada". | — | — | fenômeno→fenómeno; você/vocês (context-dependent) |
| 125 | Não quer dizer que eles estarão nus na cerimônia, mas mostra que esses jovens casais estão prontos para casar sem casa, sem carro, sem anel de diamante, sem festa de casamento, para demonstrar seu comprometimento com o amor verdadeiro. | — | — | — |
| 126 | E ainda, as pessoas estão fazendo o bem através da mídia social. | — | — | estar + gerund |
| 127 | E a primeira foto nos mostra que esse caminhão carregando 500 cães, abandonados e sequestrados, para processamento alimentar foi flagrado e parado na rodovia com todo o país observando via microblog. | caminhão (caminhao_vehicle) | rodovia (rodovia_road) | — |
| 128 | As pessoas estavam doando dinheiro, comida para cães e oferecendo trabalho voluntário para parar aquele caminhão. | caminhão (caminhao_vehicle) | — | estar + gerund |
| 129 | E depois de horas de negociação, 500 cães foram resgatados. | — | — | — |
| 130 | E aqui as pessoas também estão ajudando a encontrar crianças desaparecidas. | — | — | estar + gerund |

## 45. ted2020-1886_s23-s32

- **Document:** `ted2020-1886`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-1886.xml`
- **Sentence IDs:** 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
- **Length:** 10 sentences
- **Score / confidence:** 28.5 / **medium**
- **Evidence:** 7 strong lexical hits (1 types) across 6 sentences; 0 weaker lexical hits; 1 orthographic; 1 structural.
- **Why promising:** verified strong types: onibus_vehicle; supporting types: metro_accent, progressive_estar_gerund.
- **Concerns:** Repeated instances of one strong lexical type; repetition is capped in scoring.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 23 | De fato, quando eu me tornei prefeito, aplicando o princípio democrático de que o bem público prevalece sobre o interesse privado, de que um ônibus com 100 pessoas tem direito a 100 vezes mais espaço rodoviário do que um carro, implementamos um sistema de transporte coletivo baseado em ônibus em corredores exclusivos. | ônibus (onibus_vehicle); ônibus (onibus_vehicle) | — | — |
| 24 | Nós o chamamos TransMilênio, a fim de tornar os ônibus mais atraentes. | ônibus (onibus_vehicle) | — | — |
| 25 | É também um símbolo democrático muito bonito, porque, enquanto os ônibus passam, carros caros ficam presos no trânsito, o que é claramente quase uma imagem da democracia funcionando. | ônibus (onibus_vehicle) | — | — |
| 26 | De fato, não é apenas uma questão de equidade. | — | — | — |
| 27 | Não é necessário ter doutorado. | — | — | — |
| 28 | Um grupo de crianças de 12 anos de idade perceberia em 20 minutos que o meio mais eficiente de utilizar o escaço espaço rodoviário é com corredores exclusivos para ônibus. | ônibus (onibus_vehicle) | — | — |
| 29 | De fato, ônibus não são atraentes, mas são os únicos meios possíveis para proporcionar transporte coletivo a todas as áreas das cidades que crescem e se desenvolvem rapidamente. | ônibus (onibus_vehicle) | — | — |
| 30 | Eles também têm grande capacidade. | — | — | — |
| 31 | Por exemplo, esse sistema em Cantão está levando mais passageiros a seus destinos do que todas as linhas de metrô na China, com exceção de uma linha em Pequim, por uma fração do custo. | — | — | metrô→metro; estar + gerund |
| 32 | Lutamos não apenas por espaço para os ônibus, mas lutamos por espaço para as pessoas, e isto é ainda mais difícil. | ônibus (onibus_vehicle) | — | — |

## 46. ted2020-960_s21-s29

- **Document:** `ted2020-960`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-960.xml`
- **Sentence IDs:** 21, 22, 23, 24.1, 24.2, 25, 26, 27, 28, 29
- **Length:** 10 sentences
- **Score / confidence:** 27.0 / **high**
- **Evidence:** 3 strong lexical hits (2 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 4 structural.
- **Why promising:** verified strong types: cafe_da_manha, trem_rail; supporting types: address_voce.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 21 | Se você olhar ao seu redor no banheiro, vai ver vários outros produtos como shampoo, condicionador, creme anti-rugas, loção hidratante, mas também creme dental. | — | — | você/vocês (context-dependent) |
| 22 | Então, antes do café da manhã, você já encontrou o porco inúmeras vezes. | café da manhã (cafe_da_manha) | — | você/vocês (context-dependent) |
| 23 | Então, no café da manhã, o porco que eu acompanhei, os pelos do porco, ou proteínas dos pelos extraídos do porco, foram utilizadas como melhorador da farinha. | café da manhã (cafe_da_manha) | — | — |
| 24.1 | (Risos) Sim, é o que o fabricante diz: é "melhorador." | — | — | — |
| 24.2 | Você entende, claro. | — | — | você/vocês (context-dependent) |
| 25 | Em manteiga e vários outros produtos com baixo teor de gordura, quando você retira a gordura, acaba retirando também sabor e textura. | — | — | você/vocês (context-dependent) |
| 26 | Então, eles adicionam gelatina para recuperar a mesma textura. | — | — | — |
| 27 | Bom, no caminho para o trabalho, no asfalto ou nos edifícios por onde passa, pode muito bem haver concreto celular, o que é um tipo de concreto bastante leve que é feito de proteínas de ossos e é totalmente reutilizável. | — | — | — |
| 28 | Nos freios de trens, ao menos nos alemães, há uma parte do freio feita de cinzas de ossos. | trens (trem_rail) | — | — |
| 29 | E em vários tipos de sobremesas, como mousse de chocolate, tiramissu, pudim de baunilha, todos aqueles produtos refrigerados do supermercado, adicionam gelatina para dar boa aparência. | — | — | — |

## 47. ted2020-31849_s49-s58.1

- **Document:** `ted2020-31849`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-31849.xml`
- **Sentence IDs:** 49, 50, 51, 52, 53, 54, 55, 56, 57, 58.1
- **Length:** 10 sentences
- **Score / confidence:** 28.0 / **high**
- **Evidence:** 3 strong lexical hits (2 types) across 3 sentences; 0 weaker lexical hits; 0 orthographic; 3 structural.
- **Why promising:** verified strong types: equipe_group, onibus_vehicle; supporting types: address_voce, possessive_without_article, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 49 | Eu não sabia o que era ser astronauta: o treinamento, as simulações, todas essas coisas que preparam você para a contagem regressiva: três, dois, um, decolar. | — | — | você/vocês (context-dependent) |
| 50 | Em 2007, eu estava no ônibus espacial Atlantis, partindo do planeta, viajando a 28 mil quilômetros por hora. | ônibus (onibus_vehicle) | — | — |
| 51 | Oito minutos e meio depois, os motores principais pararam de repente, e estávamos flutuando no espaço. | — | — | — |
| 52 | Saio e flutuo até a janela e consigo ver o Caribe. | — | — | — |
| 53 | Preciso de novas definições de azul para descrever as cores que vejo. | — | — | — |
| 54 | Azul, índigo, azul-marinho, azul-marinho médio, turquesa não fazem justiça ao que meus olhos veem. | — | — | — |
| 55 | Meu trabalho nessa missão foi instalar o laboratório de Columbus de US$ 2 bilhões. | — | — | bare possessive (weak heuristic) |
| 56 | Era um laboratório de pesquisa para pesquisa de materiais, em humanos. | — | — | — |
| 57 | Cheguei ao compartimento de carga do ônibus espacial, tirei o grande módulo usei o braço robótico e o anexei à estação espacial. | ônibus (onibus_vehicle) | — | — |
| 58.1 | A equipe europeia estava aguardando há dez anos por essa instalação. | equipe (equipe_group) | — | estar + gerund |

## 48. ted2020-2259_s47-s56

- **Document:** `ted2020-2259`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-2259.xml`
- **Sentence IDs:** 47, 48, 49, 50, 51, 52, 53, 54, 55, 56
- **Length:** 10 sentences
- **Score / confidence:** 28.0 / **high**
- **Evidence:** 2 strong lexical hits (2 types) across 2 sentences; 0 weaker lexical hits; 0 orthographic; 6 structural.
- **Why promising:** verified strong types: caminhao_vehicle, equipe_group; supporting types: address_voce, progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 47 | Então, vou contar a vocês sobre Roswell, Novo México, em 24 de outubro. | — | — | você/vocês (context-dependent) |
| 48 | Tínhamos uma equipe incrível, que acordou no meio da madrugada. | equipe (equipe_group) | — | — |
| 49 | Aí está o traje. | — | — | — |
| 50 | Ele está sendo levado num trator que vocês logo verão, e quero mostrar a vocês um vídeo do lançamento de fato. | — | — | estar + gerund; você/vocês (context-dependent); você/vocês (context-dependent) |
| 51 | Roswell é ótima para lançamento de balões, mas é fantástica para pouso de paraquedas, especialmente quando o pouso é a 113 km do lugar de onde você partiu. | — | — | você/vocês (context-dependent) |
| 52 | Aquele é um caminhão com hélio, no fundo. | caminhão (caminhao_vehicle) | — | — |
| 53 | Está escuro. | — | — | — |
| 54 | Já passei cerca de uma hora e meia treinando a respiração. | — | — | — |
| 55 | Aqui, estou vestindo o traje. | — | — | estar + gerund |
| 56 | Leva cerca de uma hora para vesti-lo. | — | — | — |

## 49. ted2020-9473_s3-s12

- **Document:** `ted2020-9473`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-9473.xml`
- **Sentence IDs:** 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
- **Length:** 10 sentences
- **Score / confidence:** 28.0 / **high**
- **Evidence:** 2 strong lexical hits (2 types) across 2 sentences; 0 weaker lexical hits; 0 orthographic; 3 structural.
- **Why promising:** verified strong types: equipe_group, onibus_vehicle; supporting types: progressive_estar_gerund.
- **Concerns:** No automatic source-quality concern; manual discourse review is still required.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 3 | Estava esperando por um milagre. | — | — | estar + gerund |
| 4 | Naquele verão, tive o privilégio de trabalhar como estagiária no Centro Espacial Kennedy da NASA e o milagre que eu estava esperando era o lançamento do ônibus espacial Columbia que levava o observatório de Raio-X Chandra, um telescópio para os cientistas observarem os limites dos buracos negros. | ônibus (onibus_vehicle) | — | estar + gerund |
| 5 | O céu brilhava com as luzes, | — | — | — |
| 6 | como se fosse dia, no meio da noite. | — | — | — |
| 7 | Logo pudemos sentir o ruído e a vibração dos motores no nosso peito. | — | — | — |
| 8 | Aquilo não era um milagre; era o resultado do esforço de milhares de pessoas que trabalharam juntas para fazer de um sonho uma realidade. | — | — | — |
| 9 | Eu queria fazer parte daquela equipe. | equipe (equipe_group) | — | — |
| 10 | Decidi me matricular numa universidade para estudar engenharia aeroespacial. | — | — | — |
| 11 | No ano seguinte, comecei meu curso de engenharia no MIT e entrei num projeto de construção de robôs espaciais. | — | — | — |
| 12 | Tudo estava correndo como planejara, exceto que houve um problema. | — | — | estar + gerund |

## 50. ted2020-213_s36-s42.2

- **Document:** `ted2020-213`
- **Source:** `data/external/TED2020/TED2020/raw/pt_br/ted2020-213.xml`
- **Sentence IDs:** 36, 37, 38.1, 38.2, 39, 40.1, 40.2, 41, 42.1, 42.2
- **Length:** 10 sentences
- **Score / confidence:** 28.0 / **medium**
- **Evidence:** 4 strong lexical hits (1 types) across 4 sentences; 0 weaker lexical hits; 1 orthographic; 4 structural.
- **Why promising:** verified strong types: onibus_vehicle; supporting types: metro_accent, address_voce, progressive_estar_gerund.
- **Concerns:** Repeated instances of one strong lexical type; repetition is capped in scoring.

| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |
| --- | --- | --- | --- | --- |
| 36 | Vocês podem vir quando estiverem em Curitiba e tomar um café lá. | — | — | você/vocês (context-dependent) |
| 37 | E esta é a evolução do sistema. | — | — | — |
| 38.1 | A parte do design que fez a diferença é o tubo de embarque. | — | — | — |
| 38.2 | O tubo de embarque dá ao ônibus a mesma performance de um metrô. | ônibus (onibus_vehicle) | — | metrô→metro |
| 39 | Eis porque, estou tentando dizer, é como metronizar o ônibus. | ônibus (onibus_vehicle) | — | estar + gerund |
| 40.1 | E este é o design do ônibus. | ônibus (onibus_vehicle) | — | — |
| 40.2 | E você pode pagar antes de entrar no ônibus que vai tomar. | ônibus (onibus_vehicle) | — | você/vocês (context-dependent) |
| 41 | E os deficientes podem usar isto como um sistema normal. | — | — | — |
| 42.1 | Mas o que estou tentando dizer, é a principal contribuição às emissões de carbono vem dos carros. | — | — | estar + gerund |
| 42.2 | Mais de 50 por cento. | — | — | — |

