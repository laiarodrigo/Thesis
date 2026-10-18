# TED2020 BP lexical verification audit

This is the lexical audit used by `rank_ted2020_bp_passages.py`. Classification is **sense-specific**, not string-specific. A Portuguese dictionary entry was not treated as proof of unmarked European-Portuguese use: regional labels, definitions, equivalents, and clearly Portugal-authored usage were inspected. Conversely, a `.pt` hit was rejected when it was a quotation, Brazilian import, bilingual legal text, or product name.

Only `strongly_bp_marked` receives strong ranking weight. `bp_preferred_ep_attested` is retained as weak evidence. The last two classes do not contribute to lexical ranking.

Counts: **17 strongly_bp_marked**, **8 bp_preferred_ep_attested**, **6 shared_not_useful**, and **5 uncertain** (36 candidates investigated).

## Verified inventory

| BP item | TED example / exact sense | Likely EP form | Classification | BR dictionary evidence | PT dictionary evidence | `.br` usage evidence | `.pt` usage evidence | Confidence | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ônibus | passenger road vehicle (“subir em um ônibus”) | autocarro | strongly_bp_marked | [Michaelis: collective passenger vehicle][M-onibus] | [Priberam sense 3: Brasil; PT equivalent *autocarro*][P-onibus] | [Brazilian Ministry: ônibus/trem/barco][BR-transport] | [Portugal fair-play case: *autocarro*][PT-boleia] | high | The obsolete/rail senses of *ónibus* are not the TED sense. |
| trem | railway train (“aviões, trens, carros”) | comboio | strongly_bp_marked | [Michaelis sense 7][M-trem] | [Priberam labels the railway sense Brasil][P-trem]; [*comboio* entry][P-comboio] | [Trensurb: *trem–ônibus*][BR-train] | [gov.pt: Comboios de Portugal][PT-comboio] | high | Detector uses only exact `trem/trens`, not prefixes such as *tremendo*. |
| geladeira | refrigerator appliance (“talvez sua geladeira saiba…”) | frigorífico | strongly_bp_marked | [Michaelis search entry][M-geladeira] | [Priberam: Brasil; PT equivalent *frigorífico*][P-geladeira] | [Brazilian government campaign materials][BR-domestic] | [Azores government equipment inventory][PT-domestic] | high | Appliance sense only. |
| celular | mobile telephone (“telefones celulares”) | telemóvel | strongly_bp_marked | [Michaelis search entry][M-celular] | [Priberam noun sense: Brasil; PT *telemóvel*][P-celular] | [Receita Federal: apps for celular][BR-app] | [gov.pt: aplicação on a telemóvel][PT-app] | high | Biological adjective *celular* is shared and excluded by a local context guard. |
| banheiro | sanitary room (“usar o banheiro”) | casa de banho | strongly_bp_marked | [Michaelis search entry][M-banheiro] | [Priberam separates Brazilian room senses from Portuguese person senses][P-banheiro] | [Brazilian government materials][BR-domestic] | [Azores government: *casa de banho*][PT-domestic] | high | Portuguese *banheiro* can denote a bath attendant/lifeguard; detector requires room context. |
| camiseta | T-shirt/top (“camiseta personalizada”) | t-shirt / camisola | strongly_bp_marked | [Michaelis search entry][M-camiseta] | [Priberam sense 1: Brasil][P-camiseta] | [Rondônia government: *camisetas*][BR-carona] | [Portuguese industrial bulletin: *camisolas*][PT-clothes] | high | Only the T-shirt sense is strong; other historical garment senses are not. |
| pedágio | road toll | portagem | strongly_bp_marked | [Michaelis search entry][M-pedagio] | [Priberam: modern road sense Brasil; PT *portagem*][P-pedagio] | [ANTT natural usage][BR-road] | [gov.pt traffic guide: *portagem*][PT-road] | high | Modern road-toll sense only. |
| acostamento | road shoulder | berma | strongly_bp_marked | [Michaelis search entry][M-acostamento] | [Priberam: Brasil; PT *berma*][P-acostamento] | [ANTT natural road usage][BR-road] | [Priberam *berma*: Portugal][P-berma] | high | Exact road sense. |
| caminhão | goods vehicle (“carregando um caminhão”) | camião | strongly_bp_marked | [Michaelis search entry][M-caminhao] | [Priberam: Brasil; PT *camião*][P-caminhao] | [Brazilian road regulator][BR-road] | [gov.pt vehicle registration: *camião*][PT-camiao] | high | This combines a lexical/national spelling contrast. |
| aplicativo | computer/mobile application | aplicação / app | strongly_bp_marked | [Michaelis software senses][M-aplicativo] | [Priberam computing sense: Brasil; PT *aplicação*][P-aplicativo] | [Receita Federal: aplicativos][BR-app] | [gov.pt: aplicação móvel][PT-app] | high | Adjectival “applicable” uses are excluded by matching the noun form in TED contexts. |
| esporte | sporting activity | desporto | strongly_bp_marked | [Michaelis sporting senses][M-esporte] | [Priberam sporting senses: Brasil][P-esporte] | [Brazilian municipal sports report][BR-sport] | [Portuguese IPDJ: desporto][PT-sport] | high | Sporting sense only. |
| equipe | team/group | equipa | strongly_bp_marked | [Michaelis search entry][M-equipe] | [Priberam gives BR *equipe*, PT *equipa*][P-equipe] | [Brazilian municipal report: equipe][BR-sport] | [Portuguese IPDJ: equipa][PT-sport] | high | A systematic national form difference; repeated hits are capped. |
| terno | matching jacket-and-trousers suit | fato | strongly_bp_marked | [Michaelis search entry][M-terno] | [Priberam clothing sense: Brasil][P-terno] | [Brazilian Casa Militar dress guide][BR-terno] | [Portuguese heritage shop: men’s *fato*][PT-fato] | high | Adjective *terno* “tender” and groups of three are excluded with a clothing-context guard. |
| café da manhã | first meal of the day | pequeno-almoço | strongly_bp_marked | [Michaelis search entry][M-cafe] | [Priberam: Brasil/Cabo Verde; synonym *pequeno-almoço*][P-cafe] | [Brazilian municipal usage][BR-sport] | [Portuguese IPDJ natural usage][PT-sport] | high | Exact multiword expression. |
| babá | child-care worker | ama | strongly_bp_marked | [Michaelis search entry][M-baba] | [Priberam child-care sense: Brasil][P-baba] | [Brazilian labour rule: employee caring for children][BR-baba] | [gov.pt: regulated child-care profession *ama*][PT-ama] | high | Cake and historical address senses are excluded by a child-context guard. |
| grampeador | paper-fastening tool | agrafador | strongly_bp_marked | [Michaelis search entry][M-grampeador] | [Priberam: Brasil; PT *agrafador*][P-grampeador] | [Brazilian public procurement][BR-grampeador] | [Portuguese public procurement][PT-agrafador] | high | Exact office-tool sense. |
| carona | free ride in a vehicle | boleia | strongly_bp_marked | [Michaelis search entry][M-carona] | [Priberam vehicle-ride sense: Brasil; PT *boleia*][P-carona] | [Goiás police: natural *dar carona*][BR-carona2] | [Portuguese judicial text: *dar boleia*][PT-boleia2] | high | Saddle, gatecrasher, and other senses are excluded with vehicle/action context. |
| xícara | cup / cupful | chávena | bp_preferred_ep_attested | [Michaelis search entry][M-xicara] | [Priberam has the same sense without a regional label][P-xicara] | [Brazilian public nutrition recipe][BR-xicara] | [Portuguese cultural heritage usage][PT-chavena] | medium | Familiar contrast, but the evidence does not justify “inappropriate in EP”. Weak only. |
| mamadeira | baby bottle | biberão | bp_preferred_ep_attested | [Michaelis search entry][M-mamadeira] | [Priberam defines the same sense without a Brazil label][P-mamadeira] | [Brazilian Ministry of Health][BR-mamadeira] | [Portuguese DGS infant-feeding guidance][PT-biberao] | medium | Downgraded after attempted falsification. |
| rodovia | road/highway | estrada / autoestrada | bp_preferred_ep_attested | [Michaelis search entry][M-rodovia] | [Priberam definition is unlabelled][P-rodovia] | [ANTT natural usage][BR-road] | [gov.pt uses *autoestrada*][PT-road] | medium | Strong preference, but the Portuguese dictionary does not license “BP-only”. |
| sorvete | ice cream | gelado | bp_preferred_ep_attested | [Michaelis search entry][M-sorvete] | [Priberam same sense, no Brazil label][P-sorvete] | [Brazilian state tax schedule][BR-frozen] | [Portuguese labour bulletin: *gelados*][PT-gelado] | medium | Do not call required solely from the string. |
| garçom | restaurant waiter | empregado de mesa | bp_preferred_ep_attested | [Michaelis search entry][M-garcom] | [Priberam same sense without a regional label][P-garcom] | [Brazilian Senac restaurant vacancy][BR-garcom] | [Portuguese public employment listing][PT-waiter] | medium | Strong frequency difference, but the unlabelled Portuguese dictionary entry prevents a strong label. |
| calcinha | women’s underwear | cuecas | bp_preferred_ep_attested | [Michaelis search entry][M-calcinha] | [Priberam same sense without a regional label][P-calcinha] | [Brazilian retailer natural usage][BR-calcinha] | [Portuguese official CPV: *cuecas de mulher*][PT-cuecas] | medium | Explicit negative result: not promoted to strong. |
| cofrinho | small money box | mealheiro | bp_preferred_ep_attested | [Michaelis search entry][M-cofrinho] | [Priberam sense 1 simply “pequeno cofre”, unlabelled][P-cofrinho] | [Central Bank of Brazil][BR-cofrinho] | [Portuguese school usage: *mealheiro*][PT-mealheiro] | medium | Only the anatomical slang sense is marked Brazil. |
| picolé | ice lolly | gelado de pau | bp_preferred_ep_attested | [Michaelis search entry][M-picole] | [Priberam defines it without a regional label][P-picole] | [Brazilian state tax schedule][BR-frozen] | [Portuguese usage supports *gelado*, but no clean official same-string attestation was found][PT-gelado] | low | Sparse Portugal-context evidence; kept weak and should not drive selection alone. |
| economizar | save money/resources | poupar | shared_not_useful | [Michaelis search entry][M-economizar] | [Priberam unlabelled entry][P-economizar] | Normal Brazilian use | Same verb is standard in Portuguese lexicography | high | Pilot copy received 100; *poupar* is an alternative, not a required substitution. |
| todo mundo | everybody | toda a gente | shared_not_useful | [Michaelis search entry][M-todo] | [Priberam phrase search][P-todo] | Common Brazilian phrase | Portugal-authored occurrences exist; inventory itself called this preference-led | medium-high | Frequency does not make the source erroneous. |
| ao redor | around | em redor / à volta | shared_not_useful | [Michaelis search entry][M-redor] | [Priberam unlabelled entry][P-redor] | Normal Brazilian use | Normal Portuguese dictionary locution | high | No strong variety restriction. |
| em cima | on top / above | em cima | shared_not_useful | [Michaelis search entry][M-cima] | [Priberam unlabelled locution][P-cima] | Normal Brazilian use | Normal EP locution | high | Proposed substitutes are stylistic/contextual. |
| estacionamento | car park / parking | estacionamento | shared_not_useful | [Michaelis search entry][M-estacionamento] | [Priberam unlabelled entry][P-estacionamento] | Normal Brazilian use | Normal Portugal usage | high | Shared exact sense. |
| cantina | cafeteria/canteen | cantina | shared_not_useful | [Michaelis search entry][M-cantina] | [Priberam unlabelled entry][P-cantina] | Normal Brazilian use | Portuguese government documents use *cantina* naturally | high | Shared; institution details vary. |
| prefeito | Brazilian municipal officeholder | presidente da câmara (only in a Portugal institution) | uncertain | [Michaelis search entry][M-prefeito] | [Priberam includes historical/foreign senses][P-prefeito] | Natural Brazil-specific institution | A Portugal-local office is different | high (exclusion) | Cultural facts must not be localized automatically; excluded from ranking. |
| delegacia | Brazilian police station/institution | esquadra (context-dependent) | uncertain | [Michaelis search entry][M-delegacia] | [Priberam entry][P-delegacia] | Natural Brazilian institutional term | Portugal institution is not always referentially substitutable | high (exclusion) | Do not rewrite a Brazilian place into a Portuguese institution. |
| vestibular | Brazilian university entrance examination | concurso/provas de acesso (context-dependent) | uncertain | [Michaelis search entry][M-vestibular] | [Priberam multiple senses][P-vestibular] | Natural Brazilian education context | Referentially Brazil-specific | high (exclusion) | Cultural/institutional reference, not a safe lexical adaptation. |
| legal | slang “cool/nice” | fixe / giro | uncertain | [Michaelis multiple senses][M-legal] | [Priberam multiple senses][P-legal] | Natural BP slang | String overwhelmingly collides with shared legal adjective | low | Automated exact-sense disambiguation is unreliable; excluded. |
| time | sports team | equipa | uncertain | [Michaelis multiple senses][M-time] | [Priberam entry][P-time] | Natural BP sports use | Collides with English titles/names and other senses in TED | low | Not used by scanner; *equipe* supplies a safer signal. |

## Interpretation safeguards

- A strong label means “strong evidence for selection in this exact sense,” not that every occurrence demands one unique replacement.
- All strong detector rules are exact-word or exact-phrase matches. `celular`, `banheiro`, `terno`, `babá`, and `carona` additionally require local sense cues.
- Repetition of a single marker is capped. The shortlist rewards distinct verified types and distribution across sentences.
- Orthographic signals such as `purê→puré`, `metrô→metro`, and `prêmio→prémio` are reported separately; they are not counted as lexical types.

[M-onibus]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/%C3%B4nibus/
[M-trem]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/trem/
[M-aplicativo]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/aplicativo/
[M-esporte]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/esporte/
[M-geladeira]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/geladeira/
[M-celular]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/celular/
[M-banheiro]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/banheiro/
[M-camiseta]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/camiseta/
[M-pedagio]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/ped%C3%A1gio/
[M-acostamento]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/acostamento/
[M-caminhao]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/caminh%C3%A3o/
[M-equipe]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/equipe/
[M-terno]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/terno/
[M-cafe]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/caf%C3%A9-da-manh%C3%A3/
[M-baba]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/bab%C3%A1/
[M-grampeador]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/grampeador/
[M-carona]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/carona/
[M-xicara]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/x%C3%ADcara/
[M-mamadeira]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/mamadeira/
[M-rodovia]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/rodovia/
[M-sorvete]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/sorvete/
[M-garcom]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/gar%C3%A7om/
[M-calcinha]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/calcinha/
[M-cofrinho]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/cofrinho/
[M-picole]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/picol%C3%A9/
[M-economizar]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/economizar/
[M-todo]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/todo/
[M-redor]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/redor/
[M-cima]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/cima/
[M-estacionamento]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/estacionamento/
[M-cantina]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/cantina/
[M-prefeito]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/prefeito/
[M-delegacia]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/delegacia/
[M-vestibular]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/vestibular/
[M-legal]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/legal/
[M-time]: https://michaelis.uol.com.br/moderno-portugues/busca/portugues-brasileiro/time/
[P-onibus]: https://dicionario.priberam.org/%C3%B4nibus
[P-trem]: https://dicionario.priberam.org/trem
[P-comboio]: https://dicionario.priberam.org/comboio
[P-geladeira]: https://dicionario.priberam.org/geladeira
[P-celular]: https://dicionario.priberam.org/celular
[P-banheiro]: https://dicionario.priberam.org/banheiro
[P-camiseta]: https://dicionario.priberam.org/camiseta
[P-pedagio]: https://dicionario.priberam.org/ped%C3%A1gio
[P-acostamento]: https://dicionario.priberam.org/acostamento
[P-berma]: https://dicionario.priberam.org/berma
[P-caminhao]: https://dicionario.priberam.org/caminh%C3%A3o
[P-aplicativo]: https://dicionario.priberam.org/aplicativo
[P-esporte]: https://dicionario.priberam.org/esporte
[P-equipe]: https://dicionario.priberam.org/equipe
[P-terno]: https://dicionario.priberam.org/terno
[P-cafe]: https://dicionario.priberam.org/caf%C3%A9%20da%20manh%C3%A3
[P-baba]: https://dicionario.priberam.org/bab%C3%A1
[P-grampeador]: https://dicionario.priberam.org/grampeador
[P-carona]: https://dicionario.priberam.org/carona
[P-xicara]: https://dicionario.priberam.org/x%C3%ADcara
[P-mamadeira]: https://dicionario.priberam.org/mamadeira
[P-rodovia]: https://dicionario.priberam.org/rodovia
[P-sorvete]: https://dicionario.priberam.org/sorvete
[P-gelado]: https://dicionario.priberam.org/gelado
[P-garcom]: https://dicionario.priberam.org/gar%C3%A7om
[P-calcinha]: https://dicionario.priberam.org/calcinha
[P-cofrinho]: https://dicionario.priberam.org/cofrinho
[P-picole]: https://dicionario.priberam.org/picol%C3%A9
[P-economizar]: https://dicionario.priberam.org/economizar
[P-todo]: https://dicionario.priberam.org/todo%20o%20mundo
[P-redor]: https://dicionario.priberam.org/redor
[P-cima]: https://dicionario.priberam.org/em%20cima
[P-estacionamento]: https://dicionario.priberam.org/estacionamento
[P-cantina]: https://dicionario.priberam.org/cantina
[P-prefeito]: https://dicionario.priberam.org/prefeito
[P-delegacia]: https://dicionario.priberam.org/delegacia
[P-vestibular]: https://dicionario.priberam.org/vestibular
[P-legal]: https://dicionario.priberam.org/legal
[P-time]: https://dicionario.priberam.org/time
[BR-transport]: https://www.gov.br/transportes/pt-br/servicos/servicos/passe-livre/
[BR-train]: https://www.gov.br/trensurb/pt-br/acesso-a-informacao/perguntas-frequentes
[BR-domestic]: https://www.gov.br/mulheres/pt-br/acesso-a-informacao/acoes-e-programas-1/acoes-e-programas-de-gestoes-anteriores/politicas-para-mulheres/arquivo/sobre/a-secretaria/logomarcas/campanha-ligue-180/campanha-ligue-180
[BR-app]: https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/download/app
[BR-road]: https://www.gov.br/antt/pt-br/assuntos/rodovias/cnso-rodovias
[BR-sport]: https://www.igaracudotiete.sp.gov.br/portal/noticias/0/3/1561/7-meia-maratona-rustica-superacao-e-adrenalina-em-igaracu-do-tiete/
[BR-terno]: https://www.casamilitar.pr.gov.br/Pagina/Tipos-de-Trajes
[BR-grampeador]: https://www.coffito.gov.br/nsite/wp-content/uploads/2022/11/COFFITO-EDITAL-PREGAO-RP-MATERIAL-DE-EXPEDIENTE.pdf
[BR-carona]: https://rondonia.ro.gov.br/servidores-de-rondonia-sao-convidados-a-dar-carona-solidaria-no-dia-mundial-sem-carro/
[BR-carona2]: https://goias.gov.br/policiacivil/homem-e-preso-em-planaltina-apos-dar-carona-para-ex-companheira-do-autor-e-ser-alvo-de-tiros-em-razao-de-ciumes/
[BR-baba]: https://www.gov.br/trabalho-e-emprego/pt-br/assuntos/legislacao/colegiados-do-trabalho/copy_of_INSTRUONORMATIVAN2DE8DENOVEMBRODE2021versoretificada.pdf
[BR-xicara]: https://nutricao.sibbr.gov.br/ficha/bin/view/FN/Recipe/233_arroz_de_puta_rica
[BR-mamadeira]: https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/a/aleitamento-materno
[BR-frozen]: https://www.sefaz.pb.gov.br/legislacao/243-portarias/portarias-2011/2710-portaria-n-094-gser-2011
[BR-calcinha]: https://www.magazineluiza.com.br/calcinha/moda-e-acessorios/s/md/ccnh/
[BR-cofrinho]: https://www.bcb.gov.br/cedulasemoedas
[BR-garcom]: https://pb.senac.br/matriculas/matricula/vagas/view.php?id=1029
[PT-comboio]: https://www.gov.pt/servicos/consultar-os-servicos-da-cp
[PT-road]: https://www.gov.pt/guias/regras-de-transito-e-mobilidade-em-portugal
[PT-domestic]: https://www.azores.gov.pt/NR/rdonlyres/BF69C0E9-1051-4268-9167-5868746DD5B0/1112741/SADRelat%C3%B3rioCarateriza%C3%A7%C3%A3oAvalia%C3%A7%C3%A3oSatisfa%C3%A7%C3%A3oUtente.pdf
[PT-app]: https://www.gov.pt/plataformas-digitais/aplicacao-movel-gov-pt
[PT-camiao]: https://www.gov.pt/servicos/registar-veiculo-automovel-servico-automovel-online
[PT-sport]: https://ipdj.gov.pt/documents/20123/117806/03_ESPECIFICO_rugby.pdf/f9264c31-b457-943b-7be0-4137075278bd
[PT-clothes]: https://inpi.justica.gov.pt/LinkClick.aspx?fileticket=QXRmv3pWrSQ%3D&portalid=6
[PT-fato]: https://www.comerciocomhistoria.gov.pt/listings/casa-amorim-3839/
[PT-agrafador]: https://www.base.gov.pt/Base4/en/resultados/?ext=.pdf&id=1014113&type=doc_documentos
[PT-boleia]: https://www.apcvd.gov.pt/wp-content/uploads/2020/07/Apresenta%C3%A7%C3%A3o-APCVD_%C3%89tica_pub.pdf
[PT-boleia2]: https://cej.justica.gov.pt/Portals/30/Ficheiros/ingresso/nas_magistraturas/tribunais_judiciais/32_curso_docs/32_Prova_Penal_Acad_1cham.pdf
[PT-ama]: https://www2.gov.pt/pt-PT/fichas-de-enquadramento/como-iniciar-a-atividade-de-ama
[PT-chavena]: https://www.comerciocomhistoria.gov.pt/listings/cafetaria-sical-5641/
[PT-biberao]: https://www.dgs.pt/em-destaque/dgs-publica-primeiras-orientacoes-para-a-alimentacao-nas-creches-pdf.aspx
[PT-gelado]: https://bte.gep.msess.gov.pt/completos/2006/bte1_2006.pdf
[PT-waiter]: https://iefponline.iefp.pt/IEFP/pesquisas/detalheOfertas2.do?idOferta=589450492
[PT-cuecas]: https://www.sgmf.gov.pt/media/dados/PDF/UMC/UMC_RegulamentoCE.pdf
[PT-mealheiro]: https://aesbvizela.edu.gov.pt/?p=3378
