#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.encoder_decoder.single_task_models.t5gemma2_4b.frmt_stageb_filter import (
    diff_metrics,
    normalize_space,
    tokenize,
    is_word,
)
from scripts.merge_and_flag_wikipedia_variant_csv import suspicious_diff


MANUAL_TRANSLATE_SPANS = {
    "estupefacientes => entorpecentes",
    "da => de",
    "policias => policiais",
    "por => em",
    "seleccao => selecao",
    "suspendem => sustam",
    "camara municipal => prefeitura",
    "demolicao => derrubada",
    "agrafos => grampos",
    "de => que",
    "apanhar => pegar",
    "regressasse => retornasse",
    "implementacao => implantacao",
    "ao espectaculo => o show",
    "planeamento => planejamento",
    "prestacoes => parcelas",
    "coletiva => juridica",
    "feita => feito",
    "orcamental => orcamentario",
    "autoridade tributaria => receita",
    "acionista => acionaria",
    "rapazes => garotos",
    "o triturar => tritura lo",
    "a => em",
    "neste => nesse",
    "pelas => por",
    "criadores => criatorios",
    "quinta => fazenda",
    "leilao => remate",
    "a equipa => o time",
    "expropriar => desapropriar",
    "se tornou => virou",
    "a caixa de velocidades => o cambio",
    "de => da",
    "autorizacao => liberacao",
    "dos => de",
    "nos dois jogos => nas duas partidas",
    "o jogo => a partida",
    "um jogo => uma partida",
    "este aspeto => esse aspecto",
    "parlamento => congresso",
    "orcamental => orcamentaria",
    "a plantacao prejudicada => o plantio prejudicado",
    "encontrado => achado",
    "confecao => confeccao",
    "cerco => assedio",
    "para => pela",
    "cenario => quadro",
    "do rendimento => da renda",
    "em => com",
    "utilizador => usuario",
    "treinador => tecnico",
    "guarda redes => goleiro",
    "marca => grife",
    "procurador => promotor",
    "amigavel => amistoso",
    "jugoslavia => iugoslavia",
    "surf => surfe",
    "de => pela",
    "desde o => do",
    "ao => o",
    "uma nova relacao => um novo relacionamento",
    "duplicaram => dobraram",
    "a credito => no crediario",
    "este => esse",
    "continuar a aprofundar => seguir aprofundando",
    "esta => essa",
    "laboral => trabalhista",
    "fiscal => impositiva",
    "ficamos a saber => fica sabendo",
    "dispoe => dispoem",
    "uma constipacao => um resfriado",
    "esta a matar => ta matando",
    "deprimido => depressivo",
    "para => apos",
    "a agarrar => agarra la",
    "desinteressada da => desinteresse pela",
    "emygdio => emigdio",
    "paes => pais",
    "encontravam => achava",
    "imprescindivel => imprecindivel",
    "protocolizado => protocolado",
    "contactos => contatos",
    "justificacao => justificativa",
    "governacao => governanca",
    "desporto => esporte",
    "aspeto => aspecto",
    "incumprimento => descumprimento",
    "adeptos => torcedores",
    "despistou se => saiu da pista",
    "camionetas => caminhoes",
    "traduziram se numa => resultaram em uma",
    "acesso => acesse",
    "doseada => dosada",
    "provocatoria => provocativa",
    "loicas => loucas",
    "rastos => rastros",
    "presidente da camara => prefeito",
    "presidentes de camara => prefeitos",
    "deputados => parlamentares",
    "causa => questao",
    "reforma => aposentadoria",
    "relativamente => em relacao",
    "rendimento => renda",
    "judicial => judiciario",
    "felicitar => parabenizar",
    "diz => fala",
    "toda a gente => todo mundo",
    "laborais => trabalhistas",
    "enquadramento => arcabouco",
    "seguranca => previdencia",
    "concelhos => municipios",
}

MANUAL_NO_TRANSLATE_SPANS = {
    "actividade => atividade",
    "ustamente => justo",
    "director => diretor",
    "interpor => entrar com",
    "evasao => sonegacao",
    "xx => 20",
    "aceites => aceitas",
    "sector => setor",
    "lugares => cadeiras",
    "seccoes => secoes",
    "optimista => otimista",
    "e valida => vale",
    "apesar de => em que pese",
    "cinzentos => cinzas",
    "de 1 => do 1o",
    "das financas => da fazenda",
    "1 => 1o",
    "cmvm => cvm",
    "destes sacos => destas sacas",
    "preparadas => preparada",
    "enganar => engodar",
    "actual => atual",
    "accao indemnizatoria => acao indenizatoria",
    "11 a => 11a",
    "fazerem vista grossa => fazer vistas grossa",
    "assente => calcado",
    "ajustamento => ajuste",
    "obsoleta => defasada",
    "empresas => empresa",
    "baptizou => batizou",
    "pagamento => recebimento",
    "melhor => mais",
    "vias rapida => pistas expressa",
    "reaberta => liberada",
    "clientes => fregueses",
    "ambiente => clima",
    "amesterdao => amsterda",
    "media dimensao => porte medio",
    "envolve desde os funcionarios da limpeza ate os => envolvem de faxineiros a",
    "avancar na => ir em direcao da",
    "pronto pagamento => vista",
    "foi => deu",
    "bem sucedido => certo",
    "inspectora => inspetora",
    "ter => manter",
    "ng => ninguem",
    "como => cm",
    "atentos => de olho",
    "e => se faz",
    "perto => proxima",
    "d aquella => daquela",
    "a procurar => tentando descobrir",
    "de todo => completamente",
    "ansiado => ansioso",
    "a curar => cura la",
    "realisacao d esse => realizacao desse",
    "cotovello => cotovelo",
    "instinctivamente => instintivamente",
    "a sala => ao salao",
    "predio arrendado => imovel alugado",
    "congressistas => congressitas",
    "ovid => covid",
    "objeitvo => objetivo",
    "cambrigde => cambridge",
    "escolas municipais => escola municipal",
    "17horas => 17 horas",
    "agregados => reunidos",
    "majoracao => aumento",
    "13 30h => 13h30min",
    "sito a => situada na",
    "utilizadoem => utilizado em",
    "8 => 8h",
    "caracter => carater",
    "n o => no",
    "efectivamente => efetivamente",
    "indemnizacao => indenizacao",
    "email => e mail",
    "respeita => diz respeito",
    "defice => deficit",
    "sns => sus",
    "ctt => correios",
    "factores => fatores",
    "do homem => humanos",
    "porque => pq",
    "seleccao => selecao",
    "das financas => da fazenda",
    "excepcao => excecao",
    "infra estruturas => infraestruturas",
    "exactamente => exatamente",
    "portugues => brasileiro",
    "estrangeiro => exterior",
    "activa => ativa",
    "espectaculo => espetaculo",
    "funcionarios => servidores",
    "concurso publico => licitacao",
    "presidente => prefeito",
    "adoptada => adotada",
    "deteve => prendeu",
    "visto => assistido",
    "recue perante => hesite em pagar",
    "euro deputados => eurodeputados",
    "vai prestar => dara",
    "emprego => trabalho",
    "entupiu => ficou congestionado",
    "empreitada => obra esta prestes a ser liberado",
    "sacar acrescidas => obter maiores",
    "que correm => atuais",
    "cimeira => cupula",
    "restantes => demais",
    "sida => aids",
    "assembleia da republica => camara dos deputados",
    "predio => imovel",
    "servico nacional => sistema unico",
    "contrafaccao => falsificacao",
    "atenta => considerando",
    "escassos => poucos",
    "ao abrigo do => com base no",
    "aquando => a epoca",
    "salientar => ressaltar",
    "altura => epoca",
    "nato => otan",
    "pois => portanto",
    "alinea => inciso",
    "aumento => reajuste",
    "reputa => considera",
    "caso julgado => coisa julgada",
    "levada a cabo => realizada",
    "lugar => posicao",
    "conversa => conversaco",
    "muito => mui",
    "muitas vezes => muita ve",
    "nhanhan => nhanha",
    "casaco => paleto",
    "moscovo a ver => moscou vendo",
    "alegadas => supostas",
    "reclacionados => relacionado",
    "a segunda volta => o segundo turno",
    "acedeu a => aceitou",
    "obteve => recebeu",
    "actuacao combustiva => atuacao explosiva",
    "caras => rostos",
    "convivio => de confraternizacao",
    "a performance => o desempenho",
    "causa => risco",
    "decorrem => estao em andamento",
    "um milhar de => mil",
    "interpares => inter pares",
    "resolvemos => decidimos",
    "atencao => conta",
    "lavradores => agricultores",
    "elementos => integrantes",
    "autarca => politico",
    "populares => membros do partido",
    "condicoes => qualidades",
    "parecida => apessoada",
    "posta de parte => descartada",
    "nos lugares => nas posicoes",
    "a assistir o => ao ajudar",
    "nos 24o => nas 24a",
    "25o => 25a",
    "lugares => posicoes",
    "assistencia => plateia",
    "disponibilizou => colocou",
    "daquele inspeccao => daquela inspecao",
    "actual adminsitrador => atual administrador",
    "transaccoes efectuadas => transacoes efetuadas",
    "disputar => ser disputada",
    "neo realista => neorrealista",
    "bater => vencer",
    "recebe => atende",
    "perante si propria => consigo mesma",
    "levados a cabo => realizados",
    "enquandramento => enquadramento",
}

DEMONSTRATIVE_TRANSLATE_SPANS = {
    "este => esse",
    "esta => essa",
    "isto => isso",
    "estes => esses",
    "estas => essas",
    "deste => desse",
    "desta => dessa",
    "destes => desses",
    "destas => dessas",
    "neste => nesse",
    "nesta => nessa",
    "nestes => nesses",
    "nestas => nessas",
}

MANUAL_WRONG_SPANS = {
    "acampamentos => acompanhamentos",
}


def parse_args() -> argparse.Namespace:
    default_dir = REPO_ROOT / "data" / "ptbrvarid" / "translated_stageb_pairs_r48_500_t50"
    parser = argparse.ArgumentParser(
        description=(
            "Run GPT-Wiki-style quality control over PtBrVId translated pairs and "
            "flag likely copy / paraphrase / suspicious adaptation candidates."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_dir / "translated_pairs.csv",
        help="Input translated pairs CSV.",
    )
    parser.add_argument(
        "--output-all-csv",
        type=Path,
        default=default_dir / "translated_pairs_qc.csv",
        help="Full QC report CSV.",
    )
    parser.add_argument(
        "--output-flagged-csv",
        type=Path,
        default=default_dir / "translated_pairs_qc_flagged.csv",
        help="Flagged-only QC report CSV.",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=default_dir / "translated_pairs_qc_summary.json",
        help="Summary JSON output.",
    )
    parser.add_argument(
        "--output-pairs-csv",
        type=Path,
        default=default_dir / "translated_pairs_qc_pairs.csv",
        help="GPT-style pairs CSV with decision columns.",
    )
    parser.add_argument(
        "--near-copy-overlap",
        type=float,
        default=0.97,
        help="Structural overlap threshold for near-copy flags.",
    )
    parser.add_argument(
        "--paraphrase-score",
        type=float,
        default=0.18,
        help="Paraphrase score threshold for paraphrase_candidate.",
    )
    parser.add_argument(
        "--max-non-marker-with-markers",
        type=int,
        default=2,
        help="Allowed non-marker changed tokens when marker tokens are present.",
    )
    return parser.parse_args()


def qc_reason(
    source_text: str,
    target_text: str,
    metrics: dict[str, object],
    *,
    near_copy_overlap: float,
    paraphrase_score_threshold: float,
    max_non_marker_with_markers: int,
) -> tuple[bool, str, str]:
    source_text = normalize_space(source_text)
    target_text = normalize_space(target_text)
    if not source_text or not target_text:
        return True, "invalid_empty_text", "empty_source_or_target"

    if source_text == target_text:
        return True, "exact_copy", "possible_no_translation_needed"

    suspicious, suspicious_reason, suspicious_spans = suspicious_diff(source_text, target_text)
    changed_spans_preview = metrics["changed_spans_preview"]
    if suspicious:
        changed_spans_preview = " || ".join(suspicious_spans)

    marker_changed_tokens = int(metrics["marker_changed_tokens"])
    non_marker_changed_tokens = int(metrics["non_marker_changed_tokens"])
    structural_overlap = float(metrics["structural_overlap"])
    paraphrase_score = float(metrics["paraphrase_score"])

    if marker_changed_tokens == 0:
        if structural_overlap >= near_copy_overlap:
            return True, "near_copy_no_marker_change", "possible_no_translation_needed"
        if suspicious_reason == "reordering_artifact":
            return True, suspicious_reason, changed_spans_preview
        return True, "no_variant_signal", changed_spans_preview

    if paraphrase_score > paraphrase_score_threshold:
        return True, "paraphrase_candidate", changed_spans_preview

    if non_marker_changed_tokens - marker_changed_tokens > max_non_marker_with_markers:
        return True, "non_approved_lexical_change", changed_spans_preview

    if suspicious and suspicious_reason in {
        "reordering_artifact",
        "word_order_or_structure_change",
        "non_approved_lexical_change",
    }:
        return True, suspicious_reason, changed_spans_preview

    return False, "", changed_spans_preview


def pair_variant_texts(row: dict[str, str]) -> tuple[str, str]:
    source_label = str(row.get("source_label", ""))
    source_text = str(row.get("source_text", ""))
    translated_text = str(row.get("translated_text", ""))
    if source_label == "pt-BR":
        return translated_text, source_text
    if source_label == "pt-PT":
        return source_text, translated_text
    raise ValueError(f"Unsupported source_label: {source_label!r}")


def word_count(text: str) -> int:
    return sum(1 for tok in tokenize(text) if is_word(tok))


def load_gpt_span_decisions() -> dict[str, str]:
    base_dir = REPO_ROOT / "data" / "wikipedia_pt_variant_csv"
    decisions: dict[str, str] = {}

    review_path = base_dir / "pt_variant_non_approved_lexical_unique_pairs_review.csv"
    if review_path.exists():
        with review_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                span = normalize_space(str(row.get("span", "")))
                status = normalize_space(str(row.get("review_status", "")))
                if span and status:
                    decisions[span] = status

    verified_path = base_dir / "pt_variant_non_approved_lexical_user_verified_pairs.csv"
    if verified_path.exists():
        with verified_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                span = normalize_space(str(row.get("span", "")))
                status = normalize_space(str(row.get("status", "")))
                if span and status:
                    decisions[span] = status

    return decisions


def pair_spans_and_reason(pt_pt: str, pt_br: str) -> tuple[str, list[str]]:
    suspicious, reason, spans = suspicious_diff(pt_pt, pt_br)
    if suspicious:
        return reason, spans
    if normalize_space(pt_pt) == normalize_space(pt_br):
        return "exact_copy", []
    return "", []


def is_progressive_translate_span(span: str) -> bool:
    left, _, right = span.partition("=>")
    left = normalize_space(left)
    right = normalize_space(right)
    if not left.startswith("a "):
        return False
    if not right:
        return False
    gerund_suffixes = ("ando", "endo", "indo")
    return right.endswith(gerund_suffixes)


def manual_span_decision(span: str) -> tuple[str, str]:
    span = normalize_space(span)
    if not span:
        return "", ""
    if span in MANUAL_WRONG_SPANS:
        return "", "matched_manual_wrong_span"
    if span in MANUAL_TRANSLATE_SPANS:
        return "verified_translate", "matched_manual_span"
    if span in MANUAL_NO_TRANSLATE_SPANS:
        return "verified_no_translate", "matched_manual_span"
    if "∅" in span:
        return "verified_translate", "matched_manual_null_rule"
    if is_progressive_translate_span(span):
        return "verified_translate", "matched_manual_progressive_rule"
    if span in DEMONSTRATIVE_TRANSLATE_SPANS:
        return "verified_translate", "matched_manual_demonstrative_rule"
    return "", ""


def span_token_count(text: str) -> int:
    return sum(1 for tok in tokenize(text) if is_word(tok))


def normalized_tokens(span: str) -> tuple[str, str]:
    left, _, right = span.partition("=>")
    return normalize_space(left), normalize_space(right)


def exception_span_reason(span: str) -> str:
    left, right = normalized_tokens(span)
    if ("∅" in left or "∅" in right) and max(span_token_count(left), span_token_count(right)) >= 8:
        return "exception_long_null_span"
    return ""


def is_suffix_only_change(left: str, right: str) -> bool:
    changed = [op for op in SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not changed:
        return False
    return all(i2 == len(left) and j2 == len(right) for _, _, i2, _, j2 in changed)


def fallback_span_decision(span: str) -> tuple[str, str]:
    left, right = normalized_tokens(span)
    if not left or not right:
        return "", ""
    if "∅" in left or "∅" in right:
        return "verified_translate", "default_translate_fallback"

    left_compact = left.replace(" ", "")
    right_compact = right.replace(" ", "")
    if (
        left_compact == right_compact
        and any(ch.isdigit() for ch in left_compact)
        and left != right
    ):
        return "verified_no_translate", "fallback_spacing_normalization"

    left_wc = span_token_count(left)
    right_wc = span_token_count(right)
    if left_wc == 1 and right_wc == 1:
        ratio = SequenceMatcher(a=left, b=right, autojunk=False).ratio()
        if ratio >= 0.72 and not is_suffix_only_change(left, right):
            return "verified_no_translate", "fallback_spelling_variant"

    return "verified_translate", "default_translate_fallback"


def pair_decision_from_gpt(spans: list[str], gpt_span_decisions: dict[str, str]) -> tuple[str, str]:
    if not spans:
        return "", "no_span_match"

    matched: list[str] = []
    reasons: list[str] = []
    for span in spans:
        exception_reason = exception_span_reason(span)
        if exception_reason:
            return "", exception_reason
        manual_status, manual_reason = manual_span_decision(span)
        if manual_reason == "matched_manual_wrong_span":
            return "", manual_reason
        if manual_status:
            matched.append(manual_status)
            reasons.append(manual_reason)
            continue
        gpt_status = gpt_span_decisions.get(span, "")
        if gpt_status:
            matched.append(gpt_status)
            reasons.append("matched_gpt_span")
            continue
        fallback_status, fallback_reason = fallback_span_decision(span)
        if fallback_status:
            matched.append(fallback_status)
            reasons.append(fallback_reason)
            continue
        return "", "unmatched_gpt_span"

    distinct = sorted(set(matched))
    if len(distinct) != 1:
        return "", "mixed_span_decisions"

    reason = reasons[0] if len(set(reasons)) == 1 else "matched_multiple_span_rules"
    return distinct[0], reason


def main() -> None:
    args = parse_args()
    args.output_all_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_flagged_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    gpt_span_decisions = load_gpt_span_decisions()

    base_fieldnames = [
        "sample_id",
        "dataset",
        "split",
        "domain",
        "source_label",
        "target_variant",
        "direction",
        "source_text",
        "translated_text",
        "batch_name",
        "attempts",
    ]
    metric_fieldnames = [
        "src_words",
        "tgt_words",
        "changed_spans",
        "changed_word_tokens",
        "marker_changed_tokens",
        "non_marker_changed_tokens",
        "edit_ratio",
        "structural_overlap",
        "paraphrase_score",
        "changed_spans_preview",
        "marker_preview",
    ]
    qc_fieldnames = [
        "flagged",
        "reason",
        "reason_detail",
    ]
    output_fieldnames = base_fieldnames + metric_fieldnames + qc_fieldnames
    pair_fieldnames = [
        "span",
        "count",
        "decision",
        "reason",
    ]

    total_rows = 0
    flagged_rows = 0
    reason_counts: Counter[str] = Counter()
    domain_reason_counts: Counter[tuple[str, str]] = Counter()
    decision_counts: Counter[str] = Counter()
    unique_span_counts: Counter[str] = Counter()

    with (
        args.input_csv.open("r", encoding="utf-8", newline="") as input_fh,
        args.output_all_csv.open("w", encoding="utf-8", newline="") as all_fh,
        args.output_flagged_csv.open("w", encoding="utf-8", newline="") as flagged_fh,
    ):
        reader = csv.DictReader(input_fh)
        all_writer = csv.DictWriter(all_fh, fieldnames=output_fieldnames)
        flagged_writer = csv.DictWriter(flagged_fh, fieldnames=output_fieldnames)
        all_writer.writeheader()
        flagged_writer.writeheader()

        for row in reader:
            total_rows += 1
            source_text = str(row.get("source_text", ""))
            target_text = str(row.get("translated_text", ""))
            metrics = diff_metrics(source_text, target_text)
            flagged, reason, reason_detail = qc_reason(
                source_text,
                target_text,
                metrics,
                near_copy_overlap=args.near_copy_overlap,
                paraphrase_score_threshold=args.paraphrase_score,
                max_non_marker_with_markers=args.max_non_marker_with_markers,
            )
            output_row = {
                "sample_id": str(row.get("sample_id", "")),
                "dataset": str(row.get("dataset", "")),
                "split": str(row.get("split", "")),
                "domain": str(row.get("domain", "")),
                "source_label": str(row.get("source_label", "")),
                "target_variant": str(row.get("target_variant", "")),
                "direction": str(row.get("direction", "")),
                "source_text": source_text,
                "translated_text": target_text,
                "batch_name": str(row.get("batch_name", "")),
                "attempts": str(row.get("attempts", "")),
                "src_words": metrics["src_words"],
                "tgt_words": metrics["tgt_words"],
                "changed_spans": metrics["changed_spans"],
                "changed_word_tokens": metrics["changed_word_tokens"],
                "marker_changed_tokens": metrics["marker_changed_tokens"],
                "non_marker_changed_tokens": metrics["non_marker_changed_tokens"],
                "edit_ratio": f"{float(metrics['edit_ratio']):.4f}",
                "structural_overlap": f"{float(metrics['structural_overlap']):.4f}",
                "paraphrase_score": f"{float(metrics['paraphrase_score']):.4f}",
                "changed_spans_preview": str(metrics["changed_spans_preview"]),
                "marker_preview": str(metrics["marker_preview"]),
                "flagged": "1" if flagged else "0",
                "reason": reason,
                "reason_detail": reason_detail,
            }
            all_writer.writerow(output_row)
            pt_pt, pt_br = pair_variant_texts(row)
            _, pair_spans = pair_spans_and_reason(pt_pt, pt_br)
            for span in pair_spans:
                unique_span_counts[span] += 1
            if flagged:
                flagged_rows += 1
                reason_counts[reason] += 1
                domain_reason_counts[(output_row["domain"], reason)] += 1
                flagged_writer.writerow(output_row)

    with args.output_pairs_csv.open("w", encoding="utf-8", newline="") as pairs_fh:
        pairs_writer = csv.DictWriter(pairs_fh, fieldnames=pair_fieldnames)
        pairs_writer.writeheader()
        for span, count in unique_span_counts.most_common():
            decision, pair_reason = pair_decision_from_gpt([span], gpt_span_decisions)
            if decision:
                decision_counts[decision] += 1
            pairs_writer.writerow(
                {
                    "span": span,
                    "count": str(count),
                    "decision": decision,
                    "reason": pair_reason,
                }
            )

    summary = {
        "input_csv": str(args.input_csv.resolve()),
        "output_all_csv": str(args.output_all_csv.resolve()),
        "output_flagged_csv": str(args.output_flagged_csv.resolve()),
        "output_pairs_csv": str(args.output_pairs_csv.resolve()),
        "total_rows": total_rows,
        "flagged_rows": flagged_rows,
        "flagged_share": round(flagged_rows / total_rows, 4) if total_rows else 0.0,
        "reason_counts": dict(sorted(reason_counts.items())),
        "decision_counts": dict(sorted((k, v) for k, v in decision_counts.items() if k)),
        "domain_reason_counts": {
            f"{domain}:{reason}": count
            for (domain, reason), count in sorted(domain_reason_counts.items())
        },
        "thresholds": {
            "near_copy_overlap": args.near_copy_overlap,
            "paraphrase_score": args.paraphrase_score,
            "max_non_marker_with_markers": args.max_non_marker_with_markers,
        },
    }
    args.output_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "input_csv": str(args.input_csv),
                "total_rows": total_rows,
                "flagged_rows": flagged_rows,
                "output_all_csv": str(args.output_all_csv),
                "output_flagged_csv": str(args.output_flagged_csv),
                "output_pairs_csv": str(args.output_pairs_csv),
                "output_summary_json": str(args.output_summary_json),
                "reason_counts": dict(sorted(reason_counts.items())),
                "decision_counts": dict(sorted((k, v) for k, v in decision_counts.items() if k)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
