#!/usr/bin/env python3
"""
Apply manual lexical review decisions and propagate safe variants.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Decision:
    span: str
    status: str
    note: str


def parse_decision_block(raw: str, status: str, note: str) -> list[Decision]:
    decisions: list[Decision] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "=>" not in line:
            continue
        line = re.sub(r"\s*\([^)]*\)\s*$", "", line).rstrip(",").strip()
        if "=>" not in line:
            continue
        lhs, rhs = [part.strip() for part in line.split("=>", 1)]
        if not lhs or not rhs:
            continue
        decisions.append(Decision(f"{lhs} => {rhs}", status, note))
    return decisions


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = REPO_ROOT / "data" / "wikipedia_pt_variant_csv"
VERIFIED_CSV = BASE_DIR / "pt_variant_non_approved_lexical_user_verified_pairs.csv"
REVIEW_CSV = BASE_DIR / "pt_variant_non_approved_lexical_unique_pairs_review.csv"

LEADING_FUNCTION_WORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "um",
    "uma",
    "uns",
    "umas",
}


MANUAL_DECISIONS: list[Decision] = [
    Decision("reduzir => diminuir", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("em torno => ao redor", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("producao => fabricacao", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("composta => formada", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("farmaco => medicamento", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("privadas => particulares", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("destacou => ressaltou", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("escravatura => escravidao", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("aprendizagem automatica => aprendizado de maquina", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("fiscais => tributarios", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("repletas => cheias", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("automovel => carro", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("conferencia => palestra", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("aves => passaros", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("mostraram => demonstraram", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("provar => experimentar", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("em workshops => de oficinas", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("habitantes => moradores", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("so => apenas", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("atuacoes => apresentacoes", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("desempenharam => tiveram", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("criar => formar", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("ramos => galhos", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("creatinofosfato => creatina fosfato", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("global => mundial", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("essenciais => fundamentais", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("inicio => comeco", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("permaneceu => ficou", "verified_no_translate", "approved by user on 2026-03-25"),
    Decision("ao => o", "verified_translate", "approved by user on 2026-03-25"),
    Decision("aceder a => acessar", "verified_translate", "approved by user on 2026-03-25"),
    Decision("desportiva => esportiva", "verified_translate", "approved by user on 2026-03-25"),
    Decision("neutroes => neutrons", "verified_translate", "approved by user on 2026-03-25"),
    Decision("destas => dessas", "verified_translate", "approved by user on 2026-03-25"),
    Decision("estes => esses", "verified_translate", "approved by user on 2026-03-25"),
    Decision("no => em", "verified_translate", "approved by user on 2026-03-25"),
    Decision("descolagem => decolagem", "verified_translate", "approved by user on 2026-03-25"),
    Decision("espetaculos => shows", "verified_translate", "approved by user on 2026-03-25"),
    Decision("mestria => maestria", "verified_translate", "approved by user on 2026-03-25"),
    Decision("patinagem => patinacao", "verified_translate", "approved by user on 2026-03-25"),
    Decision("planear => planejar", "verified_translate", "approved by user on 2026-03-25"),
    Decision("imunitario => imunologico", "verified_translate", "approved by user on 2026-03-25"),
    Decision("lipidos => lipidios", "verified_translate", "approved by user on 2026-03-25"),
    Decision("excecional => excepcional", "verified_translate", "approved by user on 2026-03-25"),
    Decision("viragem => virada", "verified_translate", "approved by user on 2026-03-25"),
    Decision("sedans => sedas", "verified_translate", "approved by user on 2026-03-25"),
    Decision("treinador => tecnico", "verified_translate", "approved by user on 2026-03-25"),
    Decision("parlamento => congresso", "verified_translate", "approved by user on 2026-03-25"),
    Decision("experiencias => experimentos", "verified_translate", "approved by user on 2026-03-25"),
    Decision("num => de um", "verified_translate", "approved by user on 2026-03-25"),
    Decision("fotografias => fotos", "verified_translate", "approved by user on 2026-03-25"),
    Decision("da desflorestacao => do desmatamento", "verified_translate", "approved by user on 2026-03-25"),
    Decision("aplicacoes => aplicativos", "verified_translate", "approved by user on 2026-03-25"),
    Decision("alimentar => de alimentos", "verified_translate", "approved by user on 2026-03-25"),
    Decision("detras => tras", "verified_translate", "approved by user on 2026-03-25"),
    Decision("automovel => automotiva", "verified_translate", "approved by user on 2026-03-25"),
    Decision("a estudar => estudando", "verified_translate", "approved by user on 2026-03-25"),
    Decision("numa => de uma", "verified_translate", "approved by user on 2026-03-25"),
    Decision("rasto => rastro", "verified_translate", "approved by user on 2026-03-25"),
    Decision("liquenes => liquens", "verified_translate", "approved by user on 2026-03-25"),
    Decision("eletroes => eletrons", "verified_translate", "approved by user on 2026-03-25"),
    Decision("fios => fio", "verified_translate", "approved by user on 2026-03-25"),
    Decision("gerir => gerenciar", "verified_translate", "approved by user on 2026-03-25"),
    Decision("desafiantes => desafiadores", "verified_translate", "approved by user on 2026-03-25"),
    Decision("aguardente => cachaca", "verified_translate", "approved by user on 2026-03-25"),
    Decision("glicossomas => glicossomos", "verified_translate", "approved by user on 2026-03-25"),
    Decision("formacao => treinamento", "verified_translate", "approved by user on 2026-03-25"),
    Decision("nos => em", "verified_translate", "approved by user on 2026-03-25"),
    Decision("escuteiros => escoteiros", "verified_translate", "approved by user on 2026-03-25"),
]

MANUAL_DECISIONS.extend(
    parse_decision_block(
        """
        entoacao => entonacao
        equipas => times
        petisco => salgado
        saltar => pular
        seccao => secao
        sobro => sobreiro
        terramotos => terremotos
        travagem => frenagem
        calcite => calcita
        metropolitano => metro
        culturas => lavouras
        empresariais => corporativos
        espinafres => espinafre
        europa => america latina
        europeias => brasileiras
        frescos => afrescos
        junto a => ao redor da
        lide => lead
        neerlandes => holandes
        realizacao => direcao
        rede => malha
        regressar => retornar
        trajes => roupas
        utiliza => usa
        a conferencia => o congresso
        acolhe => recebe
        aguardado => esperado
        album => disco
        armazenagem => armazenamento
        assegurar => garantir
        atravessava => cortava
        atuacao => apresentacao
        azoto => nitrogenio
        ballet => bale
        baniuas => baniwas
        broa => pao
        cantigas => cancoes
        cativar => encantar
        cereais => graos
        conducao => direcao
        conservacao => preservacao
        desenhador => designer
        corporal => do corpo
        detecao => deteccao
        dieta => alimentacao
        discutiu => debateu
        do jogo => da partida
        durante o => no
        eleicao => escolha
        encenador => diretor
        encerra => fecha
        enquanto => como
        entoacao => entonacao
        equipas => times
        espacos => areas
        especiarias => temperos
        europeia => brasileira
        excelente => otima
        exigem => demandam
        gastronomia => culinaria
        exitos => sucessos
        grafico => de imagens
        gruta => caverna
        inaugurou => abriu
        inclui => conta com
        influenciar => impactar
        interior => interno
        junto a => perto da
        ligeiro => pequeno
        ligando => conectando
        melhorar => aprimorar
        montagem => edicao
        opiaceos => opioides
        notavel => marcante
        outrora => antes
        outrora => antigamente
        outrora => que ja foi
        paineis => placas
        passatempos => hobbies
        pela utilizacao => pelo uso
        pelos seus mercados => por suas feiras
        plantacao => plantio
        porto => rio de janeiro
        principiantes => iniciantes
        processadores => editores
        rapidas => expressas
        regio => real
        relevancia => importancia
        regioes => areas
        requerem => exigem
        retirou => tirou
        restauro => restauracao
        portuguesa => brasileira
        sedan => seda
        sitcom => serie
        sope => pe
        solar => do sol
        vasta gama => ampla variedade
        vila => cidadezinha
        tiras => tirinhas
        trajes coloridos => roupas coloridas
        topo => alto
        um mercado => uma feira
        um cabaz => uma cesta
        varios => diversos
        websites => sites
        a sua volta => ao seu redor
        abrangendo => cobrindo
        abordam => tratam de
        acabados de colher => recem colhidos
        acesos => acalorados
        acolher => sediar
        afetou => atingiu
        agricultores => pecuaristas
        aguardadas => esperadas
        aldeias => cidades
        alisar => passar
        alterando => mudando
        alunos => estudantes
        angariar => arrecadar
        """,
        "verified_no_translate",
        "approved by user on 2026-03-25",
    )
)

MANUAL_DECISIONS.extend(
    [
        Decision(
            "conservacao => preservacao",
            "verified_translate",
            "approved by user on 2026-03-25",
        ),
        Decision(
            "detecao => deteccao",
            "verified_translate",
            "approved by user on 2026-03-25",
        ),
        Decision(
            "terramotos => terremotos",
            "verified_translate",
            "approved by user on 2026-03-25",
        ),
        Decision(
            "entoacao => entonacao",
            "verified_translate",
            "approved by user on 2026-03-25",
        ),
        Decision(
            "equipas => times",
            "verified_translate",
            "approved by user on 2026-03-25",
        ),
        Decision(
            "petisco => salgado",
            "verified_translate",
            "approved by user on 2026-03-25",
        ),
        Decision(
            "saltar => pular",
            "verified_translate",
            "approved by user on 2026-03-25",
        ),
        Decision(
            "seccao => secao",
            "verified_translate",
            "approved by user on 2026-03-25",
        ),
        Decision(
            "sobro => sobreiro",
            "verified_translate",
            "approved by user on 2026-03-25",
        ),
        Decision(
            "travagem => frenagem",
            "verified_translate",
            "approved by user on 2026-03-25",
        ),
        Decision(
            "ao => a",
            "verified_translate",
            "assistant-reviewed on 2026-03-25 from current 'ao servico -> a servico' contexts",
        ),
        Decision(
            "aos => os",
            "verified_translate",
            "assistant-reviewed on 2026-03-25 from current 'ate aos -> ate os' contexts",
        ),
        Decision(
            "aos => nos",
            "verified_translate",
            "assistant-reviewed on 2026-03-25 from current 'aos fins de semana -> nos fins de semana' context",
        ),
        Decision(
            "aos => que os",
            "verified_no_translate",
            "assistant-reviewed on 2026-03-25; avoid unnecessary syntactic expansion",
        ),
        Decision(
            "ao => por",
            "verified_no_translate",
            "assistant-reviewed on 2026-03-25; avoid unnecessary paraphrase",
        ),
        Decision(
            "ao longo do => durante o",
            "verified_no_translate",
            "assistant-reviewed on 2026-03-25; both forms are natural so keep literal wording",
        ),
        Decision(
            "ao mercado => a feira",
            "verified_no_translate",
            "assistant-reviewed on 2026-03-25; follows reviewed mercado/feira no-translate pattern",
        ),
    ]
)

MANUAL_DECISIONS.extend(
    parse_decision_block(
        """
        catorze => quatorze
        bilhetes => ingressos
        colagenio => colageno
        compilacao => coletanea
        conceptual => conceitual
        de autor => autorais
        decorreu numa => aconteceu em uma
        desinfecao => desinfeccao
        enchem => lotam
        modelacao => modelagem
        """,
        "verified_translate",
        "approved by user on 2026-03-25",
    )
)

MANUAL_DECISIONS.extend(
    parse_decision_block(
        """
        medio => meia
        concebida => projetada
        """,
        "verified_no_translate",
        "approved by user on 2026-03-25",
    )
)

MANUAL_DECISIONS.extend(
    parse_decision_block(
        """
        a desenvolver => desenvolvendo
        vastidao => imensidao
        autoestrada => rodovia
        avancado => atacante
        a => de
        a explorar => explorando
        antitussicos => antitussigenos
        ao => no
        ao detalhe => aos detalhes
        as => os
        banda => trilha
        biliao => bilhao
        cientistas => pesquisadores
        altifalantes => alto falantes
        cinesseriado => seriado
        coentros => coentro
        colidores => colisionadores
        comboio partiu => trem saiu
        concertinas => sanfonas
        coro => coral
        consciencia => conscientizacao
        da ansa => de alca
        da costa => do litoral
        de => da
        de => do mundo
        de => dos
        deslocacao => deslocamento
        desportistas => esportistas
        embraiagem => embreagem
        empilhadoras => empilhadeiras
        espinal => espinhal
        factos => fatos
        fecundacao => fertilizacao
        ferroviaria => de trem
        fissuras => rachaduras
        gerem => gerenciam
        grelha => grade
        hamsteres => hamsters
        harmonica => gaita
        hidroeletrica => hidreletrica
        intemporais => atemporais
        libertam => liberam
        maitacas => maritacas
        marisco => frutos do mar
        medalheiros => medalhistas
        mediterranicas => mediterraneas
        miudo => menino
        acedemos => acessamos
        alpinistas => montanhistas
        acediam a => acessavam
        alta energia => altas energias
        nefronios => nefrons
        o alojamento => a hospedagem
        na evidencia => em evidencias
        morfogenico => morfogenetico
        o estudo arqueologico => a pesquisa arqueologica
        pardos => marrons
        plasmideos => plasmidios
        polipeptidos => polipeptideos
        por ser => sendo
        reflorestacao => reflorestamento
        reside => esta
        vinhas => vinhedos
        adeptos encheram => torcedores lotaram
        todas as semanas => toda semana
        ainda => tambem
        sinalizados => sinalizadas
        a nova aplicacao => o novo aplicativo
        uma monitorizacao => um monitoramento
        se pelos => por
        ainda => ate
        ananas => abacaxi
        antigenio => antigeno
        ao => com o
        anuncios publicitarios => comerciais
        a d => dona
        a afetar => afetando
        a chegar => chegando
        a criar => criando
        a debater => debatendo
        a desaparecer => desaparecendo
        a inspirar => inspirando
        a investigar => pesquisando
        a liderar => liderando
        """,
        "verified_translate",
        "approved by user on 2026-03-25",
    )
)


INFLECTIONAL_SUFFIXES = tuple(
    sorted(
        [
            "ariamos",
            "eriamos",
            "iriamos",
            "ariamos",
            "eriamos",
            "iriamos",
            "aremos",
            "eremos",
            "iremos",
            "arias",
            "erias",
            "irias",
            "ariam",
            "eriam",
            "iriam",
            "aria",
            "eria",
            "iria",
            "assemos",
            "essemos",
            "issemos",
            "avamos",
            "iamos",
            "eramos",
            "assem",
            "essem",
            "issem",
            "asse",
            "esse",
            "isse",
            "arei",
            "erei",
            "irei",
            "ava",
            "ia",
            "aram",
            "eram",
            "iram",
            "avam",
            "iam",
            "arei",
            "erei",
            "irei",
            "ados",
            "adas",
            "idos",
            "idas",
            "ando",
            "endo",
            "indo",
            "ado",
            "ada",
            "ido",
            "ida",
            "oes",
            "aes",
            "eis",
            "ais",
            "is",
            "es",
            "ns",
            "ei",
            "eu",
            "iu",
            "ou",
            "am",
            "em",
            "os",
            "as",
            "o",
            "a",
        ],
        key=len,
        reverse=True,
    )
)


def normalize(text: str) -> str:
    text = " ".join(text.strip().lower().split())
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def split_span(span: str) -> tuple[str, str]:
    lhs, rhs = span.split("=>", 1)
    return lhs.strip(), rhs.strip()


def phrase_tokens(text: str) -> list[str]:
    return [tok for tok in re.split(r"\s+", normalize(text)) if tok]


def inflectional_token_key(token: str) -> str:
    for suffix in INFLECTIONAL_SUFFIXES:
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def phrase_key(text: str) -> tuple[str, ...]:
    return tuple(inflectional_token_key(tok) for tok in phrase_tokens(text))


def content_phrase_key(text: str) -> tuple[str, ...]:
    tokens = list(phrase_key(text))
    while tokens and tokens[0] in LEADING_FUNCTION_WORDS:
        tokens.pop(0)
    return tuple(tokens)


def contains_subsequence(sequence: tuple[str, ...], subseq: tuple[str, ...]) -> bool:
    if not subseq:
        return False
    if len(subseq) > len(sequence):
        return False
    for idx in range(len(sequence) - len(subseq) + 1):
        if sequence[idx : idx + len(subseq)] == subseq:
            return True
    return False


def meaningful_for_containment(text: str) -> bool:
    tokens = phrase_tokens(text)
    return len(tokens) > 1 or any(len(tok) >= 5 for tok in tokens)


def is_infinitive(token: str) -> bool:
    return token.endswith(("ar", "er", "ir")) and len(token) >= 4


def is_gerund(token: str) -> bool:
    return token.endswith(("ando", "endo", "indo")) and len(token) >= 6


def matches_a_infinitive_to_gerund(lhs_tokens: list[str], rhs_tokens: list[str]) -> bool:
    if len(lhs_tokens) != 2:
        return False
    if lhs_tokens[0] != "a" or not is_infinitive(lhs_tokens[1]):
        return False
    if not rhs_tokens:
        return False
    return is_gerund(rhs_tokens[0])


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))


def provenance_score(note: str) -> int:
    note_n = normalize(note)
    if "approved by user on 2026-03-25" in note_n:
        return 6
    if "approved by user" in note_n:
        return 5
    if "assistant-reviewed" in note_n:
        return 4
    if "auto-derived" in note_n:
        return 3
    if "bulk-approved from first 180" in note_n:
        return 2
    return 1


def main() -> None:
    verified_rows = load_csv_rows(VERIFIED_CSV)
    review_rows = load_csv_rows(REVIEW_CSV)

    verified_by_span: dict[str, dict[str, str]] = {
        str(row["span"]).strip(): {
            "span": str(row["span"]).strip(),
            "status": str(row["status"]).strip(),
            "notes": str(row["notes"]).strip(),
        }
        for row in verified_rows
        if str(row.get("span", "")).strip()
    }

    upserted_exact = 0
    overridden_exact = 0
    for decision in MANUAL_DECISIONS:
        existing = verified_by_span.get(decision.span)
        if existing is None:
            upserted_exact += 1
        elif existing["status"] != decision.status or existing["notes"] != decision.note:
            overridden_exact += 1
        verified_by_span[decision.span] = {
            "span": decision.span,
            "status": decision.status,
            "notes": decision.note,
        }

    seed_rows = list(verified_by_span.values())
    seed_exact: dict[str, list[dict[str, str]]] = {}
    seed_key: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict[str, str]]] = {}
    reverse_seed_exact: dict[str, list[dict[str, str]]] = {}
    containment_seeds: list[dict[str, object]] = []
    for row in seed_rows:
        span = row["span"]
        lhs, rhs = split_span(span)
        seed_exact.setdefault(normalize(span), []).append(row)
        key = (phrase_key(lhs), phrase_key(rhs))
        seed_key.setdefault(key, []).append(row)
        reverse_seed_exact.setdefault(normalize(f"{rhs} => {lhs}"), []).append(row)
        if row["status"] == "verified_translate" or (
            meaningful_for_containment(lhs) and meaningful_for_containment(rhs)
        ):
            containment_seeds.append(
                {
                    "lhs_key": phrase_key(lhs),
                    "rhs_key": phrase_key(rhs),
                    "lhs_content_key": content_phrase_key(lhs),
                    "rhs_content_key": content_phrase_key(rhs),
                    "status": row["status"],
                    "notes": row["notes"],
                    "seed_span": span,
                }
            )

    derived_additions = 0
    conflicting_candidates = 0
    default_no_translate_additions = 0
    for row in review_rows:
        current_status = str(row.get("review_status", "")).strip()
        current_note = str(row.get("notes", "")).strip()
        is_assistant_default = current_note.startswith(
            "assistant default on 2026-03-25"
        )
        if current_status != "unreviewed" and not is_assistant_default:
            continue
        span = str(row.get("span", "")).strip()
        if not span:
            continue
        if span in verified_by_span and not is_assistant_default:
            continue
        lhs, rhs = split_span(span)
        norm_span = normalize(span)
        lhs_tokens = phrase_tokens(lhs)
        rhs_tokens = phrase_tokens(rhs)
        lhs_key = phrase_key(lhs)
        rhs_key = phrase_key(rhs)
        lhs_content_key = content_phrase_key(lhs)
        rhs_content_key = content_phrase_key(rhs)
        matches: list[tuple[str, str, int]] = []

        for seed in seed_exact.get(norm_span, []):
            matches.append(
                (
                    seed["status"],
                    f"exact reviewed span {seed['span']}",
                    provenance_score(seed["notes"]),
                )
            )
        for seed in reverse_seed_exact.get(norm_span, []):
            matches.append(
                (
                    seed["status"],
                    f"reverse of reviewed span {seed['span']}",
                    provenance_score(seed["notes"]) - 1,
                )
            )
        for seed in seed_key.get((lhs_key, rhs_key), []):
            matches.append(
                (
                    seed["status"],
                    f"inflectional variant of {seed['span']}",
                    provenance_score(seed["notes"]),
                )
            )
        for seed in containment_seeds:
            if contains_subsequence(lhs_key, seed["lhs_key"]) and contains_subsequence(
                rhs_key, seed["rhs_key"]
            ):
                matches.append(
                    (
                        str(seed["status"]),
                        f"containing phrase variant of {seed['seed_span']}",
                        provenance_score(str(seed["notes"])),
                    )
                )
            elif seed["lhs_content_key"] and seed["rhs_content_key"] and contains_subsequence(
                lhs_content_key, seed["lhs_content_key"]
            ) and contains_subsequence(rhs_content_key, seed["rhs_content_key"]):
                matches.append(
                    (
                        str(seed["status"]),
                        f"content phrase variant of {seed['seed_span']}",
                        provenance_score(str(seed["notes"])) - 1,
                    )
                )
        if matches_a_infinitive_to_gerund(lhs_tokens, rhs_tokens):
            matches.append(
                (
                    "verified_translate",
                    "generic pt-PT a + infinitive to pt-BR gerund pattern",
                    5,
                )
            )
        if "∅" in span:
            matches.append(
                (
                    "verified_no_translate",
                    "assistant default for null insertion/deletion pair",
                    4,
                )
            )

        if matches:
            matches.sort(key=lambda item: item[2], reverse=True)
            best_score = matches[0][2]
            best = [item for item in matches if item[2] == best_score]
            best_statuses = {status for status, _, _ in best}
            if len(best_statuses) != 1:
                conflicting_candidates += 1
                continue
            status, note_source, _ = best[0]
            verified_by_span[span] = {
                "span": span,
                "status": status,
                "notes": f"auto-derived from {note_source}",
            }
            derived_additions += 1
            continue

        status = "verified_no_translate"
        note_source = (
            "assistant default on 2026-03-25: no reviewed translate pattern matched; "
            "prefer minimal lexical change"
        )
        verified_by_span[span] = {
            "span": span,
            "status": status,
            "notes": note_source,
        }
        default_no_translate_additions += 1

    final_rows = sorted(verified_by_span.values(), key=lambda row: normalize(row["span"]))
    with VERIFIED_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["span", "status", "notes"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(
        f"exact_upserts={upserted_exact} exact_overrides={overridden_exact} "
        f"derived_additions={derived_additions} default_no_translate_additions={default_no_translate_additions} "
        f"conflicts_skipped={conflicting_candidates} "
        f"verified_total={len(final_rows)}"
    )


if __name__ == "__main__":
    main()
