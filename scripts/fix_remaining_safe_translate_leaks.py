#!/usr/bin/env python3
"""
Fix remaining wrong-side occurrences for safe verified_translate pairs.

- skips `∅` pairs
- skips function-word / structure-only pairs
- preserves proper names via mid-sentence titlecase protection
- only rewrites rows where the wrong-side phrase is actually detected
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.rewrite_merged_no_translate_pairs as rw


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
    base_dir = Path(__file__).resolve().parents[1] / "data" / "wikipedia_pt_variant_csv"
    parser = argparse.ArgumentParser(
        description="Fix remaining wrong-side safe lexical translate pairs on the merged CSV."
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=base_dir / "pt_variant_prompts_wikipedia_merged.csv",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=base_dir / "pt_variant_non_approved_lexical_unique_pairs_review.csv",
    )
    return parser.parse_args()


def phrase_tokens(text: str) -> list[str]:
    return [tok for tok in re.split(r"\s+", rw.normalize_text(text)) if tok]


def content_tokens(text: str) -> list[str]:
    return [tok for tok in phrase_tokens(text) if tok not in FUNCTION_WORDS]


def is_safe_pair(lhs: str, rhs: str) -> bool:
    if lhs == "∅" or rhs == "∅":
        return False
    return bool(content_tokens(lhs)) and bool(content_tokens(rhs))


def compile_phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![0-9A-Za-zÀ-ÿ]){re.escape(phrase.lower())}(?![0-9A-Za-zÀ-ÿ])"
    )


def find_canonical_surface(rows: list[dict[str, str]], column: str, target: str) -> str:
    for row in rows:
        match = rw.find_span_surface(
            str(row[column]),
            target,
            whole_phrase=True,
            skip_mid_sentence_titlecase=True,
        )
        if match is not None:
            return match[2]
    return target


def has_unsafe_auto_derived_source(
    span: str,
    review_meta_by_span: dict[str, dict[str, str]],
    stack: tuple[str, ...] = (),
) -> bool:
    meta = review_meta_by_span.get(span)
    if not meta:
        return False
    source_span = rw.extract_auto_derived_source_span(str(meta.get("notes", "")))
    if not source_span or source_span in stack or "=>" not in source_span:
        return False
    lhs, rhs = rw.split_span(source_span)
    if not is_safe_pair(lhs, rhs):
        return True
    return has_unsafe_auto_derived_source(
        source_span,
        review_meta_by_span,
        stack + (span,),
    )


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.merged_csv.open("r", encoding="utf-8", newline="")))
    fieldnames = list(rows[0].keys()) if rows else []

    review_meta_by_span: dict[str, dict[str, str]] = {}
    with args.review_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            span = str(row.get("span", "")).strip()
            if span:
                review_meta_by_span[span] = {
                    "status": str(row.get("review_status", "")).strip(),
                    "notes": str(row.get("notes", "")).strip(),
                }

    pair_infos: list[dict[str, object]] = []
    for span, meta in review_meta_by_span.items():
        if str(meta.get("status", "")).strip() != "verified_translate":
            continue
        if " => " not in span:
            continue
        raw_lhs, raw_rhs = [part.strip() for part in span.split(" => ", 1)]
        if not is_safe_pair(raw_lhs, raw_rhs):
            continue
        if has_unsafe_auto_derived_source(span, review_meta_by_span):
            continue
        lhs_pref, rhs_pref = rw.preferred_surface_pair(raw_lhs, raw_rhs)
        pair_infos.append(
            {
                "raw_lhs": raw_lhs,
                "raw_rhs": raw_rhs,
                "lhs": find_canonical_surface(rows, "pt_PT", lhs_pref),
                "rhs": find_canonical_surface(rows, "pt_BR", rhs_pref),
                "lhs_pat": compile_phrase_pattern(rw.normalize_text(raw_lhs)),
                "rhs_pat": compile_phrase_pattern(rw.normalize_text(raw_rhs)),
                "lhs_norm": rw.normalize_text(raw_lhs),
                "rhs_norm": rw.normalize_text(raw_rhs),
            }
        )

    normalized_rows: list[tuple[str, str, str]] = []
    for row in rows:
        normalized_rows.append(
            (
                str(row["merged_id"]),
                rw.normalize_text(str(row["pt_PT"])),
                rw.normalize_text(str(row["pt_BR"])),
            )
        )

    row_pairs: dict[str, list[dict[str, object]]] = {}
    for info in pair_infos:
        rhs_lower = str(info["rhs_norm"])
        lhs_lower = str(info["lhs_norm"])
        rhs_pat = info["rhs_pat"]
        lhs_pat = info["lhs_pat"]
        for merged_id, pt_norm, br_norm in normalized_rows:
            if rhs_lower in pt_norm and rhs_pat.search(pt_norm):
                row_pairs.setdefault(merged_id, []).append(info)
                continue
            if lhs_lower in br_norm and lhs_pat.search(br_norm):
                row_pairs.setdefault(merged_id, []).append(info)

    rows_changed = 0
    pt_replacements = 0
    br_replacements = 0
    rows_by_id = {str(row["merged_id"]): row for row in rows}

    for merged_id, infos in row_pairs.items():
        row = rows_by_id[merged_id]
        current_pt = str(row["pt_PT"])
        current_br = str(row["pt_BR"])
        row_changed = False

        seen_spans: set[tuple[str, str]] = set()
        deduped_infos: list[dict[str, object]] = []
        for info in infos:
            key = (str(info["raw_lhs"]), str(info["raw_rhs"]))
            if key not in seen_spans:
                deduped_infos.append(info)
                seen_spans.add(key)

        for info in deduped_infos:
            pt_count = 0
            br_count = 0
            current_pt, pt_count = rw.replace_all_directional(
                current_pt,
                str(info["raw_rhs"]),
                str(info["lhs"]),
            )
            current_br, br_count = rw.replace_all_directional(
                current_br,
                str(info["raw_lhs"]),
                str(info["rhs"]),
            )
            if pt_count:
                pt_replacements += pt_count
                row_changed = True
            if br_count:
                br_replacements += br_count
                row_changed = True

        if row_changed:
            row["pt_PT"] = current_pt
            row["pt_BR"] = current_br
            row["pt_PT_words"] = str(rw.count_words(current_pt))
            row["pt_BR_words"] = str(rw.count_words(current_br))
            rows_changed += 1

    with args.merged_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Fixed remaining safe translate leaks: rows_changed={rows_changed} "
        f"pt_replacements={pt_replacements} br_replacements={br_replacements}"
    )


if __name__ == "__main__":
    main()
