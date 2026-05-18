#!/usr/bin/env python3
"""
Enforce verified translate pairs directionally on the merged Wikipedia variant CSV.

- lhs of `lhs => rhs` must stay on pt_PT
- rhs of `lhs => rhs` must stay on pt_BR
- uses whole-phrase matching with case preservation
- skips mid-sentence titlecase matches to avoid rewriting proper names like
  "Los Angeles Times"
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
    base_dir = REPO_ROOT / "data" / "wikipedia_pt_variant_csv"
    parser = argparse.ArgumentParser(
        description="Enforce verified translate pairs directionally on the merged CSV."
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


def is_safe_lexical_translate_pair(lhs: str, rhs: str) -> bool:
    if lhs == "∅" or rhs == "∅":
        return False
    lhs_content = content_tokens(lhs)
    rhs_content = content_tokens(rhs)
    if not lhs_content or not rhs_content:
        return False
    return True


def find_canonical_surface(
    rows: list[dict[str, str]],
    column: str,
    target: str,
) -> str:
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


def main() -> None:
    args = parse_args()
    if not args.merged_csv.exists():
        raise SystemExit(f"Missing merged CSV: {args.merged_csv}")
    if not args.review_csv.exists():
        raise SystemExit(f"Missing review CSV: {args.review_csv}")

    review_rows = list(csv.DictReader(args.review_csv.open("r", encoding="utf-8", newline="")))
    rows = list(csv.DictReader(args.merged_csv.open("r", encoding="utf-8", newline="")))
    fieldnames = list(rows[0].keys()) if rows else []

    translate_pairs = []
    for row in review_rows:
        if str(row.get("review_status", "")).strip() != "verified_translate":
            continue
        span = str(row.get("span", "")).strip()
        if not span or "∅" in span:
            continue
        raw_lhs, raw_rhs = rw.split_span(span)
        if not is_safe_lexical_translate_pair(raw_lhs, raw_rhs):
            continue
        lhs_pref, rhs_pref = rw.preferred_surface_pair(raw_lhs, raw_rhs)
        lhs = find_canonical_surface(rows, "pt_PT", lhs_pref)
        rhs = find_canonical_surface(rows, "pt_BR", rhs_pref)
        translate_pairs.append((raw_lhs, raw_rhs, lhs, rhs, span))

    rows_changed = 0
    pt_replacements = 0
    br_replacements = 0
    for row in rows:
        current_pt = str(row["pt_PT"])
        current_br = str(row["pt_BR"])
        current_pt_lower = current_pt.lower()
        current_br_lower = current_br.lower()
        row_changed = False
        for raw_lhs, raw_rhs, lhs, rhs, _ in translate_pairs:
            pt_count = 0
            br_count = 0
            updated_pt = current_pt
            updated_br = current_br
            if raw_rhs.lower() in current_pt_lower:
                updated_pt, pt_count = rw.replace_all_directional(current_pt, raw_rhs, lhs)
            if raw_lhs.lower() in current_br_lower:
                updated_br, br_count = rw.replace_all_directional(current_br, raw_lhs, rhs)
            if pt_count:
                current_pt = updated_pt
                current_pt_lower = current_pt.lower()
                pt_replacements += pt_count
                row_changed = True
            if br_count:
                current_br = updated_br
                current_br_lower = current_br.lower()
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
        f"Enforced translate direction on merged CSV: rows_changed={rows_changed} "
        f"pt_replacements={pt_replacements} br_replacements={br_replacements}"
    )


if __name__ == "__main__":
    main()
