#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.merge_and_flag_wikipedia_variant_csv import (
    APPROVED_PHRASE_PAIRS,
    canonical_phrase,
    fold,
    is_word,
    tokenize,
)
from scripts.ptbrvarid.qc_translated_pairs import (
    exception_span_reason,
    fallback_span_decision,
    load_gpt_span_decisions,
    manual_span_decision,
)

DEFAULT_DIR = REPO_ROOT / "data" / "ptbrvarid" / "translated_stageb_pairs_r48_500_t50"

NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", "%", ")", "]", "}", "»"}
NO_SPACE_AFTER = {"(", "[", "{", "«"}
APOSTROPHES = {"'", "’"}
QUOTE_TOKENS = {'"', "“", "”"}

# Spelling overrides where "correct form" is not the pt-PT side.
SPELLING_REPLACEMENT_OVERRIDES = {
    "dois => dous": "dois",
    "gallego => galego": "galego",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply reviewed no-translate/spelling decisions from translated_pairs_qc_pairs.csv "
            "to translated_pairs.csv and rewrite the row-level pt_PT/pt_BR texts."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_DIR / "translated_pairs.csv",
        help="Row-level translated pairs CSV to update in place.",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=DEFAULT_DIR / "translated_pairs_qc_pairs.csv",
        help="Retained for compatibility; decisions are resolved from qc_translated_pairs.py.",
    )
    parser.add_argument(
        "--backup-csv",
        type=Path,
        default=DEFAULT_DIR / "translated_pairs.pre_qc_apply_backup.csv",
        help="Backup path written once before updating the main CSV.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=DEFAULT_DIR / "translated_pairs_qc_applied_changes.csv",
        help="Audit report of rows changed by the apply step.",
    )
    return parser.parse_args()


def pair_variant_texts(row: dict[str, str]) -> tuple[str, str]:
    source_label = str(row.get("source_label", ""))
    source_text = str(row.get("source_text", ""))
    translated_text = str(row.get("translated_text", ""))
    if source_label == "pt-BR":
        return translated_text, source_text
    if source_label == "pt-PT":
        return source_text, translated_text
    raise ValueError(f"Unsupported source_label: {source_label!r}")


def detokenize(tokens: list[str]) -> str:
    out = ""
    prev = ""
    quote_open = False
    prev_was_open_quote = False
    for tok in tokens:
        if tok in QUOTE_TOKENS:
            if not out:
                out = tok
                quote_open = True
                prev_was_open_quote = True
            elif quote_open:
                out += tok
                quote_open = False
                prev_was_open_quote = False
            else:
                if prev not in NO_SPACE_AFTER and prev not in QUOTE_TOKENS and prev not in APOSTROPHES:
                    out += " "
                out += tok
                quote_open = True
                prev_was_open_quote = True
            prev = tok
            continue

        if not out:
            out = tok
        elif (
            tok in NO_SPACE_BEFORE
            or prev in NO_SPACE_AFTER
            or prev_was_open_quote
            or tok in APOSTROPHES
            or prev in APOSTROPHES
        ):
            out += tok
        else:
            out += " " + tok
        prev = tok
        prev_was_open_quote = False
    return out.strip()


def canonical_word_tokens(tokens: list[str]) -> list[str]:
    return [tok for tok in tokens if is_word(tok)]


def span_for_slices(left_tokens: list[str], right_tokens: list[str]) -> str:
    left_phrase = canonical_phrase(canonical_word_tokens(left_tokens))
    right_phrase = canonical_phrase(canonical_word_tokens(right_tokens))
    return f"{left_phrase or '∅'} => {right_phrase or '∅'}"


def is_suspicious_span(left_tokens: list[str], right_tokens: list[str]) -> bool:
    left_words = canonical_word_tokens(left_tokens)
    right_words = canonical_word_tokens(right_tokens)
    left_phrase = canonical_phrase(left_words)
    right_phrase = canonical_phrase(right_words)
    if left_phrase == right_phrase:
        return False
    if left_phrase and right_phrase and frozenset({left_phrase, right_phrase}) in APPROVED_PHRASE_PAIRS:
        return False
    return True


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def common_suffix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(reversed(a), reversed(b)):
        if ca != cb:
            break
        n += 1
    return n


def archaic_penalty(word: str) -> int:
    w = fold(word)
    penalty = 0
    for pat in ("ph", "th", "rh", "yh", "ll", "nn", "ff", "mm", "pp", "tt"):
        penalty += 4 * w.count(pat)
    for pat in ("y", "k", "w"):
        penalty += 2 * w.count(pat)
    if "ct" in w and "fact" not in w:
        penalty += 2 * w.count("ct")
    if "pt" in w and "opt" not in w:
        penalty += 1 * w.count("pt")
    return penalty


def spelling_score(text: str) -> tuple[int, int, int]:
    compact = fold(text)
    vowel_count = sum(ch in "aeiou" for ch in compact)
    # Lower is better.
    return (
        archaic_penalty(text),
        -vowel_count,
        -len(compact),
    )


def looks_like_spelling_case(span: str, reason: str) -> bool:
    if reason in {"fallback_spelling_variant", "fallback_spacing_normalization"}:
        return True
    left, _, right = span.partition("=>")
    left = left.strip()
    right = right.strip()
    if not left or not right or "∅" in left or "∅" in right:
        return False
    left_words = left.split()
    right_words = right.split()
    if len(left_words) == len(right_words) == 1:
        a = fold(left_words[0])
        b = fold(right_words[0])
        if a == b:
            return True
        ratio = difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()
        if ratio >= 0.72 and common_prefix_len(a, b) >= 2:
            return True
    if left.replace(" ", "") == right.replace(" ", ""):
        return True
    return False


def choose_spelling_replacement(span: str) -> str:
    if span in SPELLING_REPLACEMENT_OVERRIDES:
        return SPELLING_REPLACEMENT_OVERRIDES[span]
    left, _, right = span.partition("=>")
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    if left.replace(" ", "") == right.replace(" ", ""):
        return left if len(left) >= len(right) else right
    left_score = spelling_score(left)
    right_score = spelling_score(right)
    return left if left_score <= right_score else right


def replacement_tokens_for_no_translate(
    span: str,
    reason: str,
    left_tokens: list[str],
    right_tokens: list[str],
) -> list[str]:
    if looks_like_spelling_case(span, reason):
        replacement = choose_spelling_replacement(span)
        replacement_tokens = tokenize(replacement)
        if replacement_tokens:
            return replacement_tokens
    return left_tokens


def resolve_span_decision(span: str, gpt_span_decisions: dict[str, str]) -> tuple[str, str]:
    exception = exception_span_reason(span)
    if exception:
        return "", exception
    manual_decision, manual_reason = manual_span_decision(span)
    if manual_reason:
        return manual_decision, manual_reason
    gpt_decision = gpt_span_decisions.get(span, "")
    if gpt_decision:
        return gpt_decision, "matched_gpt_span"
    return fallback_span_decision(span)


def rewrite_pair_texts(
    pt_pt: str,
    pt_br: str,
    gpt_span_decisions: dict[str, str],
) -> tuple[str, str, list[tuple[str, str, str]]]:
    pt_tokens = tokenize(pt_pt)
    br_tokens = tokenize(pt_br)
    matcher = difflib.SequenceMatcher(
        a=[fold(tok) for tok in pt_tokens],
        b=[fold(tok) for tok in br_tokens],
        autojunk=False,
    )

    out_pt: list[str] = []
    out_br: list[str] = []
    applied: list[tuple[str, str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left_slice = pt_tokens[i1:i2]
        right_slice = br_tokens[j1:j2]
        if tag == "equal":
            out_pt.extend(left_slice)
            out_br.extend(right_slice)
            continue

        if not is_suspicious_span(left_slice, right_slice):
            out_pt.extend(left_slice)
            out_br.extend(right_slice)
            continue

        span = span_for_slices(left_slice, right_slice)
        decision, reason = resolve_span_decision(span, gpt_span_decisions)
        if decision == "verified_no_translate":
            repl_tokens = replacement_tokens_for_no_translate(span, reason, left_slice, right_slice)
            out_pt.extend(repl_tokens)
            out_br.extend(repl_tokens)
            applied.append((span, decision, reason))
            continue

        out_pt.extend(left_slice)
        out_br.extend(right_slice)

    return detokenize(out_pt), detokenize(out_br), applied


def row_from_variant_texts(row: dict[str, str], pt_pt: str, pt_br: str) -> dict[str, str]:
    updated = dict(row)
    updated["pt_PT"] = pt_pt
    updated["pt_BR"] = pt_br
    if str(row.get("source_label", "")) == "pt-BR":
        updated["source_text"] = pt_br
        updated["translated_text"] = pt_pt
    else:
        updated["source_text"] = pt_pt
        updated["translated_text"] = pt_br
    return updated


def main() -> None:
    args = parse_args()
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)

    gpt_span_decisions = load_gpt_span_decisions()

    with args.input_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys()) if rows else []

    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")

    if not args.backup_csv.exists():
        args.backup_csv.write_text(args.input_csv.read_text(encoding="utf-8"), encoding="utf-8")

    updated_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, str]] = []
    changed_rows = 0
    applied_counter: Counter[str] = Counter()

    for row in rows:
        pt_pt, pt_br = pair_variant_texts(row)
        new_pt_pt, new_pt_br, applied = rewrite_pair_texts(pt_pt, pt_br, gpt_span_decisions)
        if applied:
            changed_rows += 1
            for span, _, reason in applied:
                applied_counter[reason] += 1
            report_rows.append(
                {
                    "sample_id": str(row.get("sample_id", "")),
                    "domain": str(row.get("domain", "")),
                    "source_label": str(row.get("source_label", "")),
                    "direction": str(row.get("direction", "")),
                    "applied_spans": " || ".join(span for span, _, _ in applied),
                    "applied_reasons": " || ".join(reason for _, _, reason in applied),
                    "old_pt_PT": pt_pt,
                    "new_pt_PT": new_pt_pt,
                    "old_pt_BR": pt_br,
                    "new_pt_BR": new_pt_br,
                }
            )
        updated_rows.append(row_from_variant_texts(row, new_pt_pt, new_pt_br))

    with args.input_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    report_fieldnames = [
        "sample_id",
        "domain",
        "source_label",
        "direction",
        "applied_spans",
        "applied_reasons",
        "old_pt_PT",
        "new_pt_PT",
        "old_pt_BR",
        "new_pt_BR",
    ]
    with args.report_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=report_fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(
        f"Applied reviewed no-translate changes to {changed_rows} rows. "
        f"Top reasons: {dict(applied_counter.most_common(10))}"
    )
    print(f"Backup: {args.backup_csv}")
    print(f"Report: {args.report_csv}")


if __name__ == "__main__":
    main()
