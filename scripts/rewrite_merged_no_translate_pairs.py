#!/usr/bin/env python3
"""
Safely rewrite the merged Wikipedia variant CSV from reviewed lexical decisions.

Rules:
- only touch rows flagged as non_approved_lexical_change
- ignore rows flagged as word_order_or_structure_change/reordering_artifact
- apply only exact suspicious spans present on that row
- skip any span containing `∅`
- skip function-word/structure pairs such as `a => de`, `em => de`
- for verified_no_translate spans, equalize the pair on that row
- for verified_translate spans, enforce lhs on pt_PT and rhs on pt_BR
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import unicodedata
from pathlib import Path

TRUSTED_NO_TRANSLATE_NOTE_PREFIXES = (
    "approved by user",
    "assistant-reviewed",
    "bulk-approved",
)
AUTO_DERIVED_NOTE_PREFIX = "auto-derived from "
AUTO_DERIVED_SOURCE_MARKERS = (
    "exact reviewed span ",
    "reverse of reviewed span ",
    "inflectional variant of ",
    "containing phrase variant of ",
    "content phrase variant of ",
)
PREFERRED_CANONICAL_TOKEN_DIRECTIONS = {
    ("culturas", "lavouras"): "lhs",
    ("sobro", "sobreiro"): "rhs",
}
PREFERRED_CANONICAL_PHRASE_DIRECTIONS = {
    ("cinema em casa", "home theater"): "rhs",
    ("nivel", "fase"): "lhs",
}
PREFERRED_SURFACE_REPLACEMENTS = {
    ("entoacao", "entonacao"): ("entoação", "entonação"),
    ("seccao", "secao"): ("secção", "seção"),
}
FUNCTION_WORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "num",
    "numa",
    "numas",
    "nuns",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "por",
    "se",
    "sem",
    "sob",
    "sobre",
    "um",
    "uma",
    "umas",
    "uns",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "data" / "wikipedia_pt_variant_csv"
    parser = argparse.ArgumentParser(
        description="Normalize reviewed no-translate lexical spans in the merged CSV."
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=base_dir / "pt_variant_prompts_wikipedia_merged.csv",
    )
    parser.add_argument(
        "--flagged-csv",
        type=Path,
        default=base_dir / "pt_variant_prompts_wikipedia_merged_flagged_unnecessary_changes.csv",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=base_dir / "pt_variant_non_approved_lexical_unique_pairs_review.csv",
    )
    parser.add_argument(
        "--backup-csv",
        type=Path,
        default=base_dir / "pt_variant_prompts_wikipedia_merged.pre_no_translate_cleanup.bak.csv",
    )
    return parser.parse_args()


def normalize_charwise(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    mapping: list[int] = []
    for idx, char in enumerate(text):
        lowered = char.lower()
        normalized = "".join(
            c
            for c in unicodedata.normalize("NFD", lowered)
            if unicodedata.category(c) != "Mn"
        )
        if not normalized:
            continue
        for out_char in normalized:
            chars.append(out_char)
            mapping.append(idx)
    return "".join(chars), mapping


def normalize_text(text: str) -> str:
    return normalize_charwise(text)[0]


def is_sentence_initial(text: str, start: int) -> bool:
    idx = start - 1
    while idx >= 0 and text[idx].isspace():
        idx -= 1
    if idx < 0:
        return True
    return text[idx] in ".!?;:\n\"'([{"


def match_case(replacement: str, source_surface: str) -> str:
    if not source_surface:
        return replacement
    if source_surface.isupper():
        return replacement.upper()
    if source_surface[0].isupper() and source_surface[1:] == source_surface[1:].lower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def find_span_surface(
    text: str,
    target: str,
    *,
    whole_phrase: bool = False,
    skip_mid_sentence_titlecase: bool = False,
) -> tuple[int, int, str] | None:
    normalized_text, mapping = normalize_charwise(text)
    normalized_target = normalize_text(target)
    if not normalized_target:
        return None
    if whole_phrase:
        pattern = re.compile(
            rf"(?<![0-9a-z_]){re.escape(normalized_target)}(?![0-9a-z_])"
        )
        for match in pattern.finditer(normalized_text):
            start = mapping[match.start()]
            end = mapping[match.end() - 1] + 1
            surface = text[start:end]
            if (
                skip_mid_sentence_titlecase
                and surface[:1].isupper()
                and not is_sentence_initial(text, start)
            ):
                continue
            return start, end, surface
        return None
    match_index = normalized_text.find(normalized_target)
    if match_index < 0:
        return None
    start = mapping[match_index]
    end = mapping[match_index + len(normalized_target) - 1] + 1
    return start, end, text[start:end]


def replace_first_normalized(
    text: str,
    target: str,
    replacement: str,
    *,
    whole_phrase: bool = False,
    skip_mid_sentence_titlecase: bool = False,
) -> tuple[str, bool]:
    match = find_span_surface(
        text,
        target,
        whole_phrase=whole_phrase,
        skip_mid_sentence_titlecase=skip_mid_sentence_titlecase,
    )
    if match is None:
        return text, False
    start, end, surface = match
    replacement_surface = match_case(replacement, surface)
    return text[:start] + replacement_surface + text[end:], True


def replace_all_directional(
    text: str,
    wrong_side_target: str,
    canonical_replacement: str,
    *,
    skip_mid_sentence_titlecase: bool = True,
) -> tuple[str, int]:
    replacements = 0
    current = text
    while True:
        updated, did_replace = replace_first_normalized(
            current,
            wrong_side_target,
            canonical_replacement,
            whole_phrase=True,
            skip_mid_sentence_titlecase=skip_mid_sentence_titlecase,
        )
        if not did_replace:
            break
        current = updated
        replacements += 1
    return current, replacements


def preferred_surface_pair(lhs: str, rhs: str) -> tuple[str, str]:
    key = (normalize_text(lhs), normalize_text(rhs))
    return PREFERRED_SURFACE_REPLACEMENTS.get(key, (lhs, rhs))


def split_span(span: str) -> tuple[str, str]:
    lhs, rhs = span.split("=>", 1)
    return lhs.strip(), rhs.strip()


def count_words(text: str) -> int:
    return len(text.split())


def phrase_tokens(text: str) -> list[str]:
    return [tok for tok in re.split(r"\s+", normalize_text(text)) if tok]


def content_tokens(text: str) -> list[str]:
    return [tok for tok in phrase_tokens(text) if tok not in FUNCTION_WORDS]


def is_safe_review_pair(lhs: str, rhs: str) -> bool:
    if not lhs or not rhs or lhs == "∅" or rhs == "∅":
        return False
    lhs_content = content_tokens(lhs)
    rhs_content = content_tokens(rhs)
    if not lhs_content or not rhs_content:
        return False
    return True


def extract_auto_derived_source_span(notes: str) -> str | None:
    normalized_notes = notes.strip().lower()
    if not normalized_notes.startswith(AUTO_DERIVED_NOTE_PREFIX):
        return None
    original_notes = notes.strip()
    for marker in AUTO_DERIVED_SOURCE_MARKERS:
        idx = normalized_notes.find(marker)
        if idx >= 0:
            source = original_notes[idx + len(marker) :].strip()
            return source or None
    return None


def build_trusted_no_translate_map(
    review_meta_by_span: dict[str, dict[str, str]]
) -> dict[str, bool]:
    trusted_cache: dict[str, bool] = {}

    def is_trusted(span: str, stack: tuple[str, ...] = ()) -> bool:
        if span in trusted_cache:
            return trusted_cache[span]
        meta = review_meta_by_span.get(span)
        if not meta or meta["status"] != "verified_no_translate":
            trusted_cache[span] = False
            return False

        notes = meta["notes"].strip()
        normalized_notes = notes.lower()
        if normalized_notes.startswith(TRUSTED_NO_TRANSLATE_NOTE_PREFIXES):
            trusted_cache[span] = True
            return True

        if normalized_notes.startswith("auto-derived"):
            trusted_cache[span] = True
            return True

        source_span = extract_auto_derived_source_span(notes)
        if not source_span or source_span in stack:
            trusted_cache[span] = False
            return False

        trusted = is_trusted(source_span, stack + (span,))
        trusted_cache[span] = trusted
        return trusted

    for span in review_meta_by_span:
        is_trusted(span)
    return trusted_cache


def has_unsafe_auto_derived_source(
    span: str,
    review_meta_by_span: dict[str, dict[str, str]],
    stack: tuple[str, ...] = (),
) -> bool:
    meta = review_meta_by_span.get(span)
    if not meta:
        return False
    source_span = extract_auto_derived_source_span(meta["notes"])
    if not source_span or source_span in stack:
        return False
    if "=>" not in source_span:
        return False
    lhs, rhs = split_span(source_span)
    if not is_safe_review_pair(lhs, rhs):
        return True
    return has_unsafe_auto_derived_source(
        source_span,
        review_meta_by_span,
        stack + (span,),
    )


def preferred_canonical_side(lhs: str, rhs: str) -> str | None:
    lhs_n = normalize_text(lhs)
    rhs_n = normalize_text(rhs)
    exact = PREFERRED_CANONICAL_PHRASE_DIRECTIONS.get((lhs_n, rhs_n))
    if exact is not None:
        return exact
    for (lhs_token, rhs_token), side in PREFERRED_CANONICAL_TOKEN_DIRECTIONS.items():
        if lhs_token in lhs_n and rhs_token in rhs_n:
            return side
    return None


def build_safe_span_infos(
    review_meta_by_span: dict[str, dict[str, str]]
) -> tuple[dict[str, dict[str, str | int]], dict[str, dict[str, str | int]]]:
    safe_no_translate: dict[str, dict[str, str | int]] = {}
    safe_translate: dict[str, dict[str, str | int]] = {}
    trusted_no_translate_by_span = build_trusted_no_translate_map(review_meta_by_span)

    for span, meta in review_meta_by_span.items():
        if "∅" in span:
            continue
        lhs, rhs = split_span(span)
        if not is_safe_review_pair(lhs, rhs):
            continue
        if has_unsafe_auto_derived_source(span, review_meta_by_span):
            continue

        info: dict[str, str | int] = {
            "span": span,
            "lhs": lhs,
            "rhs": rhs,
            "sort_len": max(len(lhs), len(rhs)),
        }

        if meta["status"] == "verified_translate":
            safe_translate[span] = info
        elif meta["status"] == "verified_no_translate" and trusted_no_translate_by_span.get(span):
            safe_no_translate[span] = info

    return safe_no_translate, safe_translate


def main() -> None:
    args = parse_args()
    for path in (args.merged_csv, args.flagged_csv, args.review_csv):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    review_meta_by_span: dict[str, dict[str, str]] = {}
    with args.review_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            span = str(row.get("span", "")).strip()
            status = str(row.get("review_status", "")).strip()
            notes = str(row.get("notes", "")).strip()
            if span:
                review_meta_by_span[span] = {"status": status, "notes": notes}

    safe_no_translate_spans, safe_translate_spans = build_safe_span_infos(review_meta_by_span)

    flagged_by_id: dict[str, dict[str, str]] = {}
    with args.flagged_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            merged_id = str(row.get("merged_id", "")).strip()
            if merged_id:
                flagged_by_id[merged_id] = row

    if not args.backup_csv.exists():
        shutil.copy2(args.merged_csv, args.backup_csv)

    with args.merged_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys()) if rows else []

    changed_rows = 0
    changed_pt = 0
    changed_br = 0
    changed_pairs = 0
    skipped_rows_structure = 0
    skipped_null_pairs = 0
    skipped_unsafe_pairs = 0

    for row in rows:
        merged_id = str(row.get("merged_id", "")).strip()
        flagged = flagged_by_id.get(merged_id)
        if not flagged:
            continue
        if str(flagged.get("reason", "")).strip() in {"word_order_or_structure_change", "reordering_artifact"}:
            skipped_rows_structure += 1
            continue
        if str(flagged.get("reason", "")).strip() != "non_approved_lexical_change":
            continue

        suspicious_spans = [
            s.strip()
            for s in str(flagged.get("suspicious_spans", "")).split(" || ")
            if s.strip()
        ]
        if not suspicious_spans:
            continue

        original_pt = str(row["pt_PT"])
        original_br = str(row["pt_BR"])
        current_pt = original_pt
        current_br = original_br

        row_span_infos: list[dict[str, str | int]] = []
        seen_row_spans: set[str] = set()
        for suspicious_span in suspicious_spans:
            if "∅" in suspicious_span:
                skipped_null_pairs += 1
                continue
            info = safe_no_translate_spans.get(suspicious_span) or safe_translate_spans.get(suspicious_span)
            if info is None:
                skipped_unsafe_pairs += 1
                continue
            if suspicious_span not in seen_row_spans:
                row_span_infos.append(info)
                seen_row_spans.add(suspicious_span)

        row_span_infos.sort(
            key=lambda item: (int(item["sort_len"]), len(str(item["span"]))),
            reverse=True,
        )

        for idx, info in enumerate(row_span_infos):
            span = str(info["span"])
            lhs = str(info["lhs"])
            rhs = str(info["rhs"])
            status = review_meta_by_span[span]["status"]

            lhs_pref, rhs_pref = preferred_surface_pair(lhs, rhs)
            source_pt_match = find_span_surface(
                original_pt,
                lhs_pref,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            ) or find_span_surface(
                original_pt,
                lhs,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            )
            source_br_match = find_span_surface(
                original_br,
                rhs_pref,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            ) or find_span_surface(
                original_br,
                rhs,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            )
            lhs_surface = source_pt_match[2] if source_pt_match is not None else lhs_pref
            rhs_surface = source_br_match[2] if source_br_match is not None else rhs_pref

            if status == "verified_no_translate":
                preferred_side = preferred_canonical_side(lhs, rhs)
                if preferred_side == "lhs":
                    use_pt_as_canonical = True
                elif preferred_side == "rhs":
                    use_pt_as_canonical = False
                else:
                    use_pt_as_canonical = ((int(merged_id) + idx) % 2 == 0)
                if use_pt_as_canonical:
                    updated_br, replace_count = replace_all_directional(current_br, rhs, lhs_surface)
                    if replace_count:
                        current_br = updated_br
                        changed_br += replace_count
                        changed_pairs += replace_count
                else:
                    updated_pt, replace_count = replace_all_directional(current_pt, lhs, rhs_surface)
                    if replace_count:
                        current_pt = updated_pt
                        changed_pt += replace_count
                        changed_pairs += replace_count
            elif status == "verified_translate":
                updated_pt, pt_count = replace_all_directional(current_pt, rhs, lhs_surface)
                updated_br, br_count = replace_all_directional(current_br, lhs, rhs_surface)
                if pt_count:
                    current_pt = updated_pt
                    changed_pt += pt_count
                    changed_pairs += pt_count
                if br_count:
                    current_br = updated_br
                    changed_br += br_count
                    changed_pairs += br_count

        if current_pt != row["pt_PT"] or current_br != row["pt_BR"]:
            row["pt_PT"] = current_pt
            row["pt_BR"] = current_br
            row["pt_PT_words"] = str(count_words(current_pt))
            row["pt_BR_words"] = str(count_words(current_br))
            changed_rows += 1

    with args.merged_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Rewrote merged CSV: rows_changed={changed_rows} "
        f"pt_replacements={changed_pt} br_replacements={changed_br} "
        f"pair_replacements={changed_pairs} skipped_structure_rows={skipped_rows_structure} "
        f"skipped_null_pairs={skipped_null_pairs} skipped_unsafe_pairs={skipped_unsafe_pairs} "
        f"backup={args.backup_csv}"
    )


if __name__ == "__main__":
    main()
