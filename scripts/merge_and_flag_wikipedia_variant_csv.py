#!/usr/bin/env python3
"""
Merge Wikipedia pt-variant CSV batches and flag likely unnecessary cross-variant changes.

This script does not modify the original batch CSVs. It writes:
1) a merged CSV with source metadata
2) a CSV report of suspicious rows
3) a JSON summary with counts and common suspicious replacements
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
ALNUM_RE = re.compile(r"\w", flags=re.UNICODE)

# Narrow allowlist: only very strong, expected pt-PT/pt-BR variant pairs.
APPROVED_PHRASE_PAIRS = {
    frozenset({"autocarro", "onibus"}),
    frozenset({"autocarros", "onibus"}),
    frozenset({"comboio", "trem"}),
    frozenset({"comboios", "trens"}),
    frozenset({"telemovel", "celular"}),
    frozenset({"telemoveis", "celulares"}),
    frozenset({"altifalante", "altofalante"}),
    frozenset({"altifalantes", "altofalantes"}),
    frozenset({"controlo", "controle"}),
    frozenset({"controlos", "controles"}),
    frozenset({"iman", "ima"}),
    frozenset({"imanes", "imas"}),
    frozenset({"infecao", "infeccao"}),
    frozenset({"infecoes", "infeccoes"}),
    frozenset({"monitorizacao", "monitoramento"}),
    frozenset({"monitorizacoes", "monitoramentos"}),
}

STOPWORDS = {
    "a", "o", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "e", "ou", "que",
    "se", "ao", "aos", "à", "às", "como", "mais", "menos", "muito", "muitos", "muitas",
    "muita", "sua", "seu", "suas", "seus", "sendo", "ainda", "já", "também", "sobre",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_dir = repo_root / "data" / "wikipedia_pt_variant_csv"
    parser = argparse.ArgumentParser(
        description=(
            "Merge pt_variant_prompts_wikipedia_batch_*.csv files and flag likely "
            "unnecessary lexical/syntactic changes between pt_PT and pt_BR."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_dir,
        help="Directory containing pt_variant_prompts_wikipedia_batch_*.csv files.",
    )
    parser.add_argument(
        "--input-glob",
        default="pt_variant_prompts_wikipedia_batch_*.csv",
        help="Glob used inside --input-dir.",
    )
    parser.add_argument(
        "--output-merged-csv",
        type=Path,
        default=default_dir / "pt_variant_prompts_wikipedia_merged.csv",
        help="Path for the merged CSV.",
    )
    parser.add_argument(
        "--output-flagged-csv",
        type=Path,
        default=default_dir / "pt_variant_prompts_wikipedia_merged_flagged_unnecessary_changes.csv",
        help="Path for the suspicious-row report CSV.",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=default_dir / "pt_variant_prompts_wikipedia_merged_flagged_unnecessary_changes_summary.json",
        help="Path for the JSON summary.",
    )
    return parser.parse_args()


def fold(text: str) -> str:
    raw = unicodedata.normalize("NFKD", (text or "").lower())
    out = "".join(ch for ch in raw if not unicodedata.combining(ch))
    return out.replace("-", "").replace("‑", "")


def is_word(token: str) -> bool:
    return bool(ALNUM_RE.search(token or ""))


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def canonical_phrase(tokens: Iterable[str]) -> str:
    return " ".join(fold(tok) for tok in tokens if is_word(tok)).strip()


def content_token_multiset(tokens: Iterable[str]) -> Counter[str]:
    out: Counter[str] = Counter()
    for tok in tokens:
        if not is_word(tok):
            continue
        key = fold(tok)
        if key and key not in STOPWORDS:
            out[key] += 1
    return out


def extract_batch_number(path: Path) -> int:
    match = re.search(r"_batch_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def suspicious_diff(pt_pt: str, pt_br: str) -> tuple[bool, str, list[str]]:
    pt_tokens = tokenize(pt_pt)
    br_tokens = tokenize(pt_br)
    matcher = difflib.SequenceMatcher(
        a=[fold(tok) for tok in pt_tokens],
        b=[fold(tok) for tok in br_tokens],
        autojunk=False,
    )

    suspicious_spans: list[str] = []
    structured_spans: list[tuple[str, str, list[str], list[str]]] = []
    only_order_like = True

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left_tokens = [tok for tok in pt_tokens[i1:i2] if is_word(tok)]
        right_tokens = [tok for tok in br_tokens[j1:j2] if is_word(tok)]
        left_phrase = canonical_phrase(left_tokens)
        right_phrase = canonical_phrase(right_tokens)

        if left_phrase == right_phrase:
            continue
        if left_phrase and right_phrase and frozenset({left_phrase, right_phrase}) in APPROVED_PHRASE_PAIRS:
            continue

        left_counter = content_token_multiset(left_tokens)
        right_counter = content_token_multiset(right_tokens)
        if left_counter != right_counter:
            only_order_like = False

        suspicious_spans.append(f"{left_phrase or '∅'} => {right_phrase or '∅'}")
        structured_spans.append((left_phrase, right_phrase, left_tokens, right_tokens))

    if not suspicious_spans:
        return False, "", []

    overall_counter_equal = content_token_multiset(pt_tokens) == content_token_multiset(br_tokens)
    inserted_counter: Counter[str] = Counter()
    deleted_counter: Counter[str] = Counter()
    only_insert_delete = True
    for left_phrase, right_phrase, left_tokens, right_tokens in structured_spans:
        if left_phrase and right_phrase:
            only_insert_delete = False
            continue
        if left_phrase:
            deleted_counter.update(content_token_multiset(left_tokens))
        if right_phrase:
            inserted_counter.update(content_token_multiset(right_tokens))

    if (
        overall_counter_equal
        and only_insert_delete
        and (inserted_counter or deleted_counter)
        and inserted_counter == deleted_counter
    ):
        reason = "reordering_artifact"
    else:
        reason = (
            "word_order_or_structure_change"
            if only_order_like
            else "non_approved_lexical_change"
        )
    return True, reason, suspicious_spans


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    input_paths = sorted(args.input_dir.glob(args.input_glob), key=extract_batch_number)
    if not input_paths:
        raise SystemExit(f"No input CSVs matched {args.input_glob!r} under {args.input_dir}")

    args.output_merged_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_flagged_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    merged_fieldnames = [
        "merged_id",
        "source_batch",
        "source_file",
        "source_row_id",
        "pt_PT",
        "pt_BR",
        "pt_PT_words",
        "pt_BR_words",
    ]
    flagged_fieldnames = merged_fieldnames + [
        "reason",
        "suspicious_span_count",
        "suspicious_spans",
    ]

    total_rows = 0
    flagged_rows = 0
    reason_counts: Counter[str] = Counter()
    span_counts: Counter[str] = Counter()

    with (
        args.output_merged_csv.open("w", encoding="utf-8", newline="") as merged_fh,
        args.output_flagged_csv.open("w", encoding="utf-8", newline="") as flagged_fh,
    ):
        merged_writer = csv.DictWriter(merged_fh, fieldnames=merged_fieldnames)
        flagged_writer = csv.DictWriter(flagged_fh, fieldnames=flagged_fieldnames)
        merged_writer.writeheader()
        flagged_writer.writeheader()

        merged_id = 1
        for path in input_paths:
            batch = extract_batch_number(path)
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    merged_row = {
                        "merged_id": str(merged_id),
                        "source_batch": str(batch),
                        "source_file": path.name,
                        "source_row_id": str(row.get("id", "")).strip(),
                        "pt_PT": str(row.get("pt_PT", "")).strip(),
                        "pt_BR": str(row.get("pt_BR", "")).strip(),
                        "pt_PT_words": str(row.get("pt_PT_words", "")).strip(),
                        "pt_BR_words": str(row.get("pt_BR_words", "")).strip(),
                    }
                    merged_writer.writerow(merged_row)
                    total_rows += 1

                    suspicious, reason, spans = suspicious_diff(
                        merged_row["pt_PT"],
                        merged_row["pt_BR"],
                    )
                    if suspicious:
                        flagged_rows += 1
                        reason_counts[reason] += 1
                        for span in spans:
                            span_counts[span] += 1
                        flagged_writer.writerow(
                            {
                                **merged_row,
                                "reason": reason,
                                "suspicious_span_count": str(len(spans)),
                                "suspicious_spans": " || ".join(spans),
                            }
                        )

                    merged_id += 1

    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "input_file_count": len(input_paths),
        "merged_csv": str(args.output_merged_csv.resolve()),
        "flagged_csv": str(args.output_flagged_csv.resolve()),
        "total_rows": total_rows,
        "flagged_rows": flagged_rows,
        "flagged_share": round(flagged_rows / total_rows, 4) if total_rows else 0.0,
        "reason_counts": dict(sorted(reason_counts.items())),
        "top_suspicious_spans": [
            {"span": span, "count": count}
            for span, count in span_counts.most_common(200)
        ],
    }
    args.output_summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Merged {len(input_paths)} CSV files into {args.output_merged_csv} "
        f"(rows={total_rows})."
    )
    print(
        f"Flagged {flagged_rows} suspicious rows into {args.output_flagged_csv} "
        f"and wrote summary to {args.output_summary_json}."
    )


if __name__ == "__main__":
    main()
