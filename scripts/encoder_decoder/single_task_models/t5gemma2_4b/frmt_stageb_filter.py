#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)

# Keep Stage B aligned with the conservative Stage C row filter.
MARKER_VARIANTS = {
    "autocarro",
    "ônibus",
    "comboio",
    "trem",
    "telemóvel",
    "celular",
    "sumo",
    "suco",
    "bolachas",
    "biscoitos",
    "pequeno",
    "almoço",
    "manhã",
    "fixe",
    "legal",
    "miúdo",
    "miúda",
    "miúdos",
    "miúdas",
    "garoto",
    "garota",
    "garotos",
    "garotas",
    "rapaz",
    "rapazes",
    "moço",
    "moços",
    "rapariga",
    "raparigas",
    "moça",
    "moças",
    "fato",
    "terno",
    "carrinha",
    "caminhonete",
    "frigorífico",
    "geladeira",
    "gelado",
    "sorvete",
    "chávena",
    "xícara",
    "bicha",
    "fila",
    "propina",
    "mensalidade",
    "lembrar",
    "recordar",
    "ecrã",
    "tela",
    "ficheiro",
    "arquivo",
    "ordenador",
    "esferográfica",
    "caneta",
    "sítio",
    "site",
    "sítios",
    "sites",
    "tu",
    "você",
    "vocês",
    "consigo",
    "connosco",
    "conosco",
    "contigo",
}


@dataclass(frozen=True)
class FrmtFilterConfig:
    max_changed_spans: int = 4
    max_edit_ratio: float = 0.30
    min_structural_overlap: float = 0.72
    max_paraphrase_score: float = 0.18
    max_non_marker_changed: int = 6
    max_non_marker_over_marker_gap: int = 2


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def extract_task_prefix(text: str) -> str | None:
    match = TASK_PREFIX_RE.match(text or "")
    if not match:
        return None
    return match.group(1).strip().lower()


def strip_task_prefix(text: str) -> str:
    return normalize_space(TASK_PREFIX_RE.sub("", text or ""))


def infer_direction(row: dict[str, Any]) -> str:
    for key in ("task", "direction"):
        value = normalize_space(str(row.get(key) or "")).casefold()
        if value in {"translate_br2pt", "br2pt"}:
            return "translate_br2pt"
        if value in {"translate_pt2br", "pt2br"}:
            return "translate_pt2br"
    prefix = extract_task_prefix(str(row.get("input_text") or ""))
    if prefix == "br-pt":
        return "translate_br2pt"
    if prefix == "pt-br":
        return "translate_pt2br"
    return "translation"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_space(text))


def is_word(token: str) -> bool:
    return bool(token) and any(ch.isalnum() for ch in token)


def normalize_token(token: str) -> str:
    return token.casefold()


def is_marker_token(token: str) -> bool:
    return normalize_token(token) in MARKER_VARIANTS


def contains_marker_tokens(text: str) -> bool:
    return any(is_marker_token(tok) for tok in tokenize(text) if is_word(tok))


def sequence_overlap(src_tokens: list[str], tgt_tokens: list[str]) -> float:
    if not src_tokens and not tgt_tokens:
        return 1.0
    ratio = SequenceMatcher(
        a=[normalize_token(t) for t in src_tokens],
        b=[normalize_token(t) for t in tgt_tokens],
        autojunk=False,
    ).ratio()
    return float(ratio)


def diff_metrics(source_text: str, target_text: str) -> dict[str, Any]:
    src_tokens = tokenize(source_text)
    tgt_tokens = tokenize(target_text)
    matcher = SequenceMatcher(
        a=[normalize_token(t) for t in src_tokens],
        b=[normalize_token(t) for t in tgt_tokens],
        autojunk=False,
    )

    changed_spans = 0
    changed_word_tokens = 0
    marker_changed_tokens = 0
    non_marker_changed_tokens = 0
    changed_previews: list[str] = []
    marker_hits: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_spans += 1
        src_chunk = src_tokens[i1:i2]
        tgt_chunk = tgt_tokens[j1:j2]
        changed_previews.append(
            f"{' '.join(src_chunk[:8]) or '<empty>'} => {' '.join(tgt_chunk[:8]) or '<empty>'}"
        )

        for tok in [*src_chunk, *tgt_chunk]:
            if not is_word(tok):
                continue
            changed_word_tokens += 1
            if is_marker_token(tok):
                marker_changed_tokens += 1
                marker_hits.append(tok)
            else:
                non_marker_changed_tokens += 1

    src_words = sum(1 for tok in src_tokens if is_word(tok))
    tgt_words = sum(1 for tok in tgt_tokens if is_word(tok))
    denom = max(src_words, tgt_words, 1)

    return {
        "src_words": src_words,
        "tgt_words": tgt_words,
        "changed_spans": changed_spans,
        "changed_word_tokens": changed_word_tokens,
        "marker_changed_tokens": marker_changed_tokens,
        "non_marker_changed_tokens": non_marker_changed_tokens,
        "edit_ratio": changed_word_tokens / denom,
        "structural_overlap": sequence_overlap(src_tokens, tgt_tokens),
        "paraphrase_score": non_marker_changed_tokens / denom,
        "changed_spans_preview": " | ".join(changed_previews[:4]),
        "marker_preview": ", ".join(sorted(dict.fromkeys(marker_hits))[:12]),
    }


def evaluate_frmt_translation_row(
    row: dict[str, Any],
    *,
    source_path: str,
    line_no: int,
    config: FrmtFilterConfig,
) -> dict[str, Any]:
    raw_input_text = str(row.get("input_text") or "")
    source_text = strip_task_prefix(raw_input_text)
    target_text = normalize_space(str(row.get("target_text") or ""))
    metrics = diff_metrics(source_text, target_text)
    bucket = str(row.get("bucket") or "n/a")
    direction = infer_direction(row)
    record_id = str(row.get("id") or f"{Path(source_path).name}:{line_no}")

    passes_changed_spans = metrics["changed_spans"] <= config.max_changed_spans
    passes_edit_ratio = metrics["edit_ratio"] <= config.max_edit_ratio
    passes_structural_overlap = metrics["structural_overlap"] >= config.min_structural_overlap
    passes_paraphrase_score = metrics["paraphrase_score"] <= config.max_paraphrase_score
    passes_non_marker_changed = (
        metrics["non_marker_changed_tokens"] <= config.max_non_marker_changed
    )
    non_marker_over_marker_gap = (
        metrics["non_marker_changed_tokens"] - metrics["marker_changed_tokens"]
    )
    passes_non_marker_over_marker_gap = (
        metrics["marker_changed_tokens"] == 0
        or non_marker_over_marker_gap <= config.max_non_marker_over_marker_gap
    )

    if not source_text or not target_text:
        decision = "drop"
        decision_reason = "invalid_empty_text"
        keep_reason = ""
    elif source_text == target_text:
        if contains_marker_tokens(source_text):
            decision = "drop"
            decision_reason = "equal_with_marker_token"
            keep_reason = ""
        else:
            decision = "keep"
            decision_reason = "equal_marker_safe"
            keep_reason = "frmt_equal_marker_safe"
    elif (
        passes_changed_spans
        and passes_edit_ratio
        and passes_structural_overlap
        and passes_paraphrase_score
        and passes_non_marker_changed
        and passes_non_marker_over_marker_gap
    ):
        decision = "keep"
        decision_reason = "passes_core_filter"
        keep_reason = "frmt_filtered"
    else:
        decision = "drop"
        decision_reason = "fails_core_filter"
        keep_reason = ""

    return {
        "record_id": record_id,
        "source_path": source_path,
        "line_no": line_no,
        "bucket": bucket,
        "direction": direction,
        "decision": decision,
        "decision_reason": decision_reason,
        "keep_reason": keep_reason,
        "passes_changed_spans": passes_changed_spans,
        "passes_edit_ratio": passes_edit_ratio,
        "passes_structural_overlap": passes_structural_overlap,
        "passes_paraphrase_score": passes_paraphrase_score,
        "passes_non_marker_changed": passes_non_marker_changed,
        "passes_non_marker_over_marker_gap": passes_non_marker_over_marker_gap,
        "non_marker_over_marker_gap": non_marker_over_marker_gap,
        "source_text": source_text,
        "target_text": target_text,
        **metrics,
    }
