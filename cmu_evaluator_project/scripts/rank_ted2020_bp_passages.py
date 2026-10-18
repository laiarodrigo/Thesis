#!/usr/bin/env python3
"""Rank TED2020 pt-BR passage windows for conservative BP->EP evaluation.

The detector is deliberately rule based.  Only externally verified, sense-
appropriate lexical items contribute substantial weight.  Ambiguous strings
use small context guards; weaker structural and orthographic observations are
reported separately.  Source strings are copied directly from the XML.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


Classification = str


@dataclass(frozen=True)
class Signal:
    signal_id: str
    label: str
    classification: Classification
    ep_form: str
    pattern: re.Pattern[str]
    sense: str
    guard: Callable[[str], bool] | None = None


def rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


def contains_any(*terms: str) -> Callable[[str], bool]:
    context = rx(r"\b(?:" + "|".join(terms) + r")\b")
    return lambda text: bool(context.search(text))


# These classifications and senses are documented, with web evidence, in
# ted2020_bp_lexical_verification.md.  Do not promote a new item to STRONG
# without updating that audit first.
LEXICAL_SIGNALS: tuple[Signal, ...] = (
    Signal("onibus_vehicle", "ônibus", "strongly_bp_marked", "autocarro", rx(r"\bônibus\b"), "road passenger vehicle"),
    Signal("trem_rail", "trem/trens", "strongly_bp_marked", "comboio(s)", rx(r"\btrens?\b"), "railway train"),
    Signal("geladeira_appliance", "geladeira", "strongly_bp_marked", "frigorífico", rx(r"\bgeladeiras?\b"), "refrigerating appliance"),
    Signal("celular_phone", "celular", "strongly_bp_marked", "telemóvel", rx(r"\bcelulares?\b"), "mobile telephone", contains_any("telefone", "telefones", "ligar", "ligações", "mensagem", "mensagens", "texto", "textos", "app", "aplicativo", "aplicativos", "smartphone", "dispositivo", "dispositivos", "tela", "bateria", "wi-fi", "wifi")),
    Signal("banheiro_room", "banheiro", "strongly_bp_marked", "casa de banho", rx(r"\bbanheiros?\b"), "sanitary room", contains_any("casa", "quarto", "porta", "hotel", "público", "públicos", "usar", "usando", "ir", "fui", "entro", "entrei", "pia", "vaso", "chuveiro", "limpar")),
    Signal("camiseta_tshirt", "camiseta", "strongly_bp_marked", "t-shirt/camisola", rx(r"\bcamisetas?\b"), "T-shirt or short-sleeved top"),
    Signal("pedagio_road", "pedágio", "strongly_bp_marked", "portagem", rx(r"\bpedágios?\b"), "road toll"),
    Signal("acostamento_road", "acostamento", "strongly_bp_marked", "berma", rx(r"\bacostamentos?\b"), "road shoulder"),
    Signal("caminhao_vehicle", "caminhão", "strongly_bp_marked", "camião", rx(r"\bcaminh(?:ão|ões)\b"), "goods vehicle"),
    Signal("aplicativo_software", "aplicativo", "strongly_bp_marked", "aplicação/app", rx(r"\baplicativos?\b"), "computer/mobile application"),
    Signal("esporte_activity", "esporte", "strongly_bp_marked", "desporto", rx(r"\besportes?\b"), "sporting activity"),
    Signal("equipe_group", "equipe", "strongly_bp_marked", "equipa", rx(r"\bequipes?\b"), "team/group; national spelling distinction"),
    Signal("terno_suit", "terno", "strongly_bp_marked", "fato", rx(r"\bternos?\b"), "matching jacket-and-trousers suit", contains_any("vestir", "vestia", "vestindo", "usava", "usando", "gravata", "casaco", "calça", "calças", "roupa", "traje")),
    Signal("cafe_da_manha", "café da manhã", "strongly_bp_marked", "pequeno-almoço", rx(r"\bcafé(?:s)? da manhã\b"), "first meal of the day"),
    Signal("baba_childcare", "babá", "strongly_bp_marked", "ama", rx(r"\bbabás?\b"), "person caring for children", contains_any("criança", "crianças", "bebê", "bebês", "filho", "filha", "filhos", "filhas", "cuidar", "cuidava", "cuidando")),
    Signal("grampeador_tool", "grampeador", "strongly_bp_marked", "agrafador", rx(r"\bgrampeador(?:a|as|es)?\b"), "paper-fastening tool"),
    Signal("carona_ride", "carona", "strongly_bp_marked", "boleia", rx(r"\bcaronas?\b"), "free ride in a vehicle", contains_any("dar", "dei", "dava", "pegar", "peguei", "pego", "oferecer", "ofereceu", "carro", "veículo", "dirigir", "viagem", "casa")),
    # Useful but intentionally lower-weight: Portuguese dictionaries attest the
    # same sense without a Brazil-only label or the evidence is preference-led.
    Signal("xicara_cup", "xícara", "bp_preferred_ep_attested", "chávena", rx(r"\bxícaras?\b"), "drinking cup"),
    Signal("mamadeira_bottle", "mamadeira", "bp_preferred_ep_attested", "biberão", rx(r"\bmamadeiras?\b"), "baby bottle"),
    Signal("rodovia_road", "rodovia", "bp_preferred_ep_attested", "estrada/autoestrada", rx(r"\brodovias?\b"), "road/highway"),
    Signal("sorvete_icecream", "sorvete", "bp_preferred_ep_attested", "gelado", rx(r"\bsorvetes?\b"), "ice cream"),
    Signal("garcom_waiter", "garçom", "bp_preferred_ep_attested", "empregado de mesa", rx(r"\bgarçom|garçons\b"), "restaurant waiter"),
    Signal("calcinha_underwear", "calcinha", "bp_preferred_ep_attested", "cuecas", rx(r"\bcalcinhas?\b"), "women's underwear"),
    Signal("cofrinho_moneybox", "cofrinho", "bp_preferred_ep_attested", "mealheiro", rx(r"\bcofrinhos?\b"), "small money box"),
    Signal("picolé_ice_lolly", "picolé", "bp_preferred_ep_attested", "gelado de pau", rx(r"\bpicolés?\b"), "ice lolly"),
)


ORTHOGRAPHIC_SIGNALS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("pure_accent", "purê→puré", rx(r"\bpurês?\b")),
    ("metro_accent", "metrô→metro", rx(r"\bmetrô\b")),
    ("economico_accent", "econômic-→económic-", rx(r"\beconômic\w*\b")),
    ("academico_accent", "acadêmic-→académic-", rx(r"\bacadêmic\w*\b")),
    ("premio_accent", "prêmio→prémio", rx(r"\bprêmios?\b")),
    ("fenomeno_accent", "fenômeno→fenómeno", rx(r"\bfenômenos?\b")),
    ("genero_accent", "gênero→género", rx(r"\bgêneros?\b")),
)


STRUCTURAL_SIGNALS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("progressive_estar_gerund", "estar + gerund", rx(r"\b(?:estou|está|estamos|estão|estava|estavam|estive|esteve|estiveram|estaria|estariam)\s+(?:\w+mente\s+)?\w+(?:ando|endo|indo)\b")),
    ("address_voce", "você/vocês (context-dependent)", rx(r"\bvocês?\b")),
    ("possessive_without_article", "bare possessive (weak heuristic)", rx(r"(?:^|[,:;.!?]\s+|\b(?:de|em|com|para|por|sem)\s+)m(?:eu|inha|eus|inhas)\s+\w+")),
)


def sentence_hits(text: str) -> dict[str, list[dict[str, str]]]:
    lexical: list[dict[str, str]] = []
    for signal in LEXICAL_SIGNALS:
        matches = list(signal.pattern.finditer(text))
        if matches and (signal.guard is None or signal.guard(text)):
            lexical.extend(
                {
                    "signal_id": signal.signal_id,
                    "label": signal.label,
                    "classification": signal.classification,
                    "matched_text": match.group(0),
                    "sense": signal.sense,
                    "likely_ep_form": signal.ep_form,
                }
                for match in matches
            )
    orthographic = [
        {"signal_id": sid, "label": label, "matched_text": match.group(0)}
        for sid, label, pattern in ORTHOGRAPHIC_SIGNALS
        for match in pattern.finditer(text)
    ]
    structural = [
        {"signal_id": sid, "label": label, "matched_text": match.group(0)}
        for sid, label, pattern in STRUCTURAL_SIGNALS
        for match in pattern.finditer(text)
    ]
    return {"lexical": lexical, "orthographic": orthographic, "structural": structural}


def read_document(path: Path) -> list[dict[str, object]]:
    root = ET.parse(path).getroot()
    rows: list[dict[str, object]] = []
    for position, node in enumerate(root.findall(".//s"), start=1):
        # itertext keeps text represented by XML character/entity nodes without
        # stripping, normalising, correcting, or repunctuating it.
        text = "".join(node.itertext())
        rows.append(
            {
                "sentence_id": str(node.attrib["id"]),
                "position": position,
                "source_bp": text,
                "evidence": sentence_hits(text),
            }
        )
    return rows


def flatten(rows: Iterable[dict[str, object]], kind: str) -> list[dict[str, str]]:
    return [hit for row in rows for hit in row["evidence"][kind]]  # type: ignore[index]


def build_window(document_id: str, source_path: str, rows: list[dict[str, object]]) -> dict[str, object]:
    lexical = flatten(rows, "lexical")
    strong = [h for h in lexical if h["classification"] == "strongly_bp_marked"]
    preferred = [h for h in lexical if h["classification"] == "bp_preferred_ep_attested"]
    orthographic = flatten(rows, "orthographic")
    structural = flatten(rows, "structural")
    strong_types = sorted({h["signal_id"] for h in strong})
    preferred_types = sorted({h["signal_id"] for h in preferred})
    orthographic_types = sorted({h["signal_id"] for h in orthographic})
    structural_types = sorted({h["signal_id"] for h in structural})
    strong_sentence_count = sum(
        any(h["classification"] == "strongly_bp_marked" for h in row["evidence"]["lexical"])  # type: ignore[index]
        for row in rows
    )
    useful_sentence_count = sum(
        bool(row["evidence"]["lexical"] or row["evidence"]["orthographic"] or row["evidence"]["structural"])  # type: ignore[index]
        for row in rows
    )
    distribution_bonus = min(strong_sentence_count, 4) * 1.5 + min(useful_sentence_count, 6) * 0.5
    score_components = {
        "distinct_strong_lexical": 7.0 * len(strong_types),
        "strong_lexical_hits_capped": 1.5 * min(len(strong), 6),
        "distinct_preferred_lexical": 2.0 * len(preferred_types),
        "preferred_lexical_hits_capped": 0.5 * min(len(preferred), 4),
        "distinct_orthographic": 1.5 * len(orthographic_types),
        "progressive": 2.0 * sum(h["signal_id"] == "progressive_estar_gerund" for h in structural),
        "weak_structural": 0.25 * sum(h["signal_id"] != "progressive_estar_gerund" for h in structural),
        "distribution": distribution_bonus,
    }
    raw_score = round(sum(score_components.values()), 3)
    concerns: list[str] = []
    if strong_sentence_count < 2:
        concerns.append("Strong evidence is concentrated in one sentence.")
    if len(strong_types) == 1 and len(strong) > 1:
        concerns.append("Repeated instances of one strong lexical type; repetition is capped in scoring.")
    if any(len(str(row["source_bp"])) > 350 for row in rows):
        concerns.append("Contains a long sentence that may increase annotation burden.")
    stage_markers = sum(bool(re.fullmatch(r"\s*\([^)]{1,35}\)\s*", str(row["source_bp"]))) for row in rows)
    if stage_markers:
        concerns.append(f"Contains {stage_markers} standalone stage/audience marker(s).")
    return {
        "document_id": document_id,
        "source_xml": source_path,
        "window_id": f"{document_id}_s{rows[0]['sentence_id']}-s{rows[-1]['sentence_id']}",
        "sentence_ids": [row["sentence_id"] for row in rows],
        "sentence_count": len(rows),
        "sentences": rows,
        "evidence_counts": {
            "strong_lexical_hits": len(strong),
            "distinct_strong_lexical_types": len(strong_types),
            "preferred_ep_attested_lexical_hits": len(preferred),
            "distinct_preferred_ep_attested_types": len(preferred_types),
            "orthographic_hits": len(orthographic),
            "structural_hits": len(structural),
            "sentences_with_strong_signal": strong_sentence_count,
            "sentences_with_any_reported_signal": useful_sentence_count,
        },
        "distinct_signal_ids": {
            "strong_lexical": strong_types,
            "preferred_ep_attested": preferred_types,
            "orthographic": orthographic_types,
            "structural": structural_types,
        },
        "score_components": score_components,
        "raw_score": raw_score,
        "concerns": concerns,
    }


def qualifies(window: dict[str, object]) -> bool:
    c = window["evidence_counts"]
    strong_types = c["distinct_strong_lexical_types"]  # type: ignore[index]
    strong_sentences = c["sentences_with_strong_signal"]  # type: ignore[index]
    support = (
        c["distinct_preferred_ep_attested_types"]  # type: ignore[index]
        + c["orthographic_hits"]  # type: ignore[index]
        + sum(s == "progressive_estar_gerund" for s in window["distinct_signal_ids"]["structural"])  # type: ignore[index]
    )
    return (strong_types >= 2 and strong_sentences >= 2) or (strong_types >= 1 and strong_sentences >= 1 and support >= 2)


def diversity_select(candidates: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    # One window per document prevents overlapping near-duplicates.  Greedy
    # novelty favours new strong types while preserving the transparent score.
    remaining = candidates[:]
    selected: list[dict[str, object]] = []
    used_docs: set[str] = set()
    seen_strong: Counter[str] = Counter()
    seen_support: Counter[str] = Counter()
    while remaining and len(selected) < limit:
        best_i = -1
        best_key: tuple[float, float, str] | None = None
        for i, window in enumerate(remaining):
            if window["document_id"] in used_docs:
                continue
            ids = window["distinct_signal_ids"]
            new_strong = sum(1 / math.sqrt(1 + seen_strong[x]) for x in ids["strong_lexical"])  # type: ignore[index]
            new_support = sum(
                1 / math.sqrt(1 + seen_support[x])
                for group in ("preferred_ep_attested", "orthographic", "structural")
                for x in ids[group]  # type: ignore[index]
            )
            utility = float(window["raw_score"]) + 2.5 * new_strong + 0.5 * new_support
            key = (utility, float(window["raw_score"]), str(window["window_id"]))
            if best_key is None or key > best_key:
                best_i, best_key = i, key
        if best_i < 0:
            break
        chosen = remaining.pop(best_i)
        chosen["diversity_utility_at_selection"] = round(best_key[0], 3)  # type: ignore[index]
        selected.append(chosen)
        used_docs.add(str(chosen["document_id"]))
        ids = chosen["distinct_signal_ids"]
        seen_strong.update(ids["strong_lexical"])  # type: ignore[index]
        for group in ("preferred_ep_attested", "orthographic", "structural"):
            seen_support.update(ids[group])  # type: ignore[index]
    for rank, window in enumerate(selected, start=1):
        window["rank"] = rank
        strong_types = window["evidence_counts"]["distinct_strong_lexical_types"]  # type: ignore[index]
        strong_sentences = window["evidence_counts"]["sentences_with_strong_signal"]  # type: ignore[index]
        window["confidence"] = "high" if strong_types >= 2 and strong_sentences >= 2 else "medium"
    return selected


def markdown(data: dict[str, object]) -> str:
    meta = data["scan_summary"]
    lines = [
        "# TED2020 larger BP passage candidate pool",
        "",
        "> Candidate shortlist for researcher review. It is **not** a finalized annotation dataset.",
        "",
        f"Scanned **{meta['documents_scanned']:,} documents / {meta['sentences_scanned']:,} sentences** and scored **{meta['windows_scored']:,}** overlapping 8–10-sentence windows. "
        f"The conservative threshold admitted {meta['windows_above_threshold']:,}; diversity selection retained **{meta['windows_shortlisted']}** non-overlapping, one-per-document windows.",
        "",
        "The score is an audit aid, not linguistic ground truth. Strong lexical types dominate; repeated hits are capped. `bp_preferred_ep_attested`, orthographic, and structural signals are shown separately.",
        "",
    ]
    for window in data["passages"]:  # type: ignore[assignment]
        counts = window["evidence_counts"]
        lines.extend(
            [
                f"## {window['rank']}. {window['window_id']}",
                "",
                f"- **Document:** `{window['document_id']}`",
                f"- **Source:** `{window['source_xml']}`",
                f"- **Sentence IDs:** {', '.join(window['sentence_ids'])}",
                f"- **Length:** {window['sentence_count']} sentences",
                f"- **Score / confidence:** {window['raw_score']} / **{window['confidence']}**",
                f"- **Evidence:** {counts['strong_lexical_hits']} strong lexical hits ({counts['distinct_strong_lexical_types']} types) across {counts['sentences_with_strong_signal']} sentences; "
                f"{counts['preferred_ep_attested_lexical_hits']} weaker lexical hits; {counts['orthographic_hits']} orthographic; {counts['structural_hits']} structural.",
                f"- **Why promising:** verified strong types: {', '.join(window['distinct_signal_ids']['strong_lexical']) or 'none'}; supporting types: "
                f"{', '.join(window['distinct_signal_ids']['preferred_ep_attested'] + window['distinct_signal_ids']['orthographic'] + window['distinct_signal_ids']['structural']) or 'none'}.",
                f"- **Concerns:** {' '.join(window['concerns']) if window['concerns'] else 'No automatic source-quality concern; manual discourse review is still required.'}",
                "",
                "| ID | Exact BP source | Strong lexical evidence | Weaker lexical evidence | Other evidence |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in window["sentences"]:
            lex = row["evidence"]["lexical"]
            strong = [f"{h['matched_text']} ({h['signal_id']})" for h in lex if h["classification"] == "strongly_bp_marked"]
            weak = [f"{h['matched_text']} ({h['signal_id']})" for h in lex if h["classification"] != "strongly_bp_marked"]
            other = [h["label"] for h in row["evidence"]["orthographic"] + row["evidence"]["structural"]]
            text = str(row["source_bp"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['sentence_id']} | {text} | {'; '.join(strong) or '—'} | {'; '.join(weak) or '—'} | {'; '.join(other) or '—'} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/external/TED2020/TED2020/raw/pt_br"))
    parser.add_argument("--output-dir", type=Path, default=Path("cmu_evaluator_project/data/passage_selection"))
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    admitted: list[dict[str, object]] = []
    windows_scored = 0
    documents = 0
    sentences = 0
    for path in sorted(args.corpus.glob("*.xml")):
        rows = read_document(path)
        documents += 1
        sentences += len(rows)
        document_id = path.stem
        source_path = path.as_posix()
        for length in (8, 9, 10):
            for start in range(0, len(rows) - length + 1):
                windows_scored += 1
                window = build_window(document_id, source_path, rows[start : start + length])
                # Keeping only threshold-passing windows makes the exhaustive
                # scan memory-bounded; every possible window is still scored.
                if qualifies(window):
                    admitted.append(window)

    admitted.sort(key=lambda w: (-float(w["raw_score"]), str(w["window_id"])))
    for raw_rank, window in enumerate(admitted, start=1):
        window["raw_rank"] = raw_rank
    selected = diversity_select(admitted, args.limit)
    payload: dict[str, object] = {
        "status": "candidate_pool_for_review_not_final_dataset",
        "source_language": "pt-BR",
        "target_language": "pt-PT",
        "pt_pt_ted_reference_used": False,
        "window_lengths_scored": [8, 9, 10],
        "scan_summary": {
            "documents_scanned": documents,
            "sentences_scanned": sentences,
            "windows_scored": windows_scored,
            "windows_above_threshold": len(admitted),
            "windows_shortlisted": len(selected),
        },
        "scoring_formula": {
            "distinct_strong_lexical": "7.0 each",
            "strong_lexical_hits": "1.5 each, capped at 6",
            "distinct_bp_preferred_ep_attested": "2.0 each",
            "preferred_hits": "0.5 each, capped at 4",
            "distinct_orthographic": "1.5 each",
            "progressive_estar_gerund": "2.0 each",
            "other_weak_structural": "0.25 each",
            "distribution": "1.5 per strong-bearing sentence (cap 4) + 0.5 per any-signal sentence (cap 6)",
        },
        "passages": selected,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ted2020_larger_pool_shortlist.json"
    md_path = args.output_dir / "ted2020_larger_pool_shortlist.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["scan_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
