#!/usr/bin/env python3
"""
Verify remaining mismatches for safe auto-derived reviewed pairs on the merged CSV.
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

import scripts.fix_remaining_safe_auto_derived_pairs as auto_fix
import scripts.rewrite_merged_no_translate_pairs as rw


def parse_args() -> argparse.Namespace:
    base_dir = REPO_ROOT / "data" / "wikipedia_pt_variant_csv"
    parser = argparse.ArgumentParser(
        description="Verify remaining safe auto-derived review-pair mismatches on the merged CSV."
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


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.merged_csv.open("r", encoding="utf-8", newline="")))
    review_meta_by_span = auto_fix.load_review_meta(args.review_csv)

    function_words = set(rw.FUNCTION_WORDS)

    def normalize_phrase(text: str) -> str:
        return rw.normalize_text(text)

    def is_safe_pair(lhs: str, rhs: str) -> bool:
        if lhs == "∅" or rhs == "∅":
            return False
        lhs_tokens = [tok for tok in re.split(r"\s+", normalize_phrase(lhs)) if tok and tok not in function_words]
        rhs_tokens = [tok for tok in re.split(r"\s+", normalize_phrase(rhs)) if tok and tok not in function_words]
        return bool(lhs_tokens) and bool(rhs_tokens)

    def has_unsafe_source(span: str, stack: tuple[str, ...] = ()) -> bool:
        meta = review_meta_by_span.get(span)
        if not meta:
            return False
        source = rw.extract_auto_derived_source_span(str(meta.get("notes", "")))
        if not source or source in stack or "=>" not in source:
            return False
        lhs, rhs = rw.split_span(source)
        if not is_safe_pair(lhs, rhs):
            return True
        return has_unsafe_source(source, stack + (span,))

    no_translate_infos: list[dict[str, object]] = []
    translate_infos: list[dict[str, object]] = []
    for span, meta in review_meta_by_span.items():
        notes = str(meta.get("notes", "")).lower()
        if "auto-derived" not in notes or "=>" not in span or "∅" in span:
            continue
        lhs, rhs = rw.split_span(span)
        if not is_safe_pair(lhs, rhs) or has_unsafe_source(span):
            continue
        info = {
            "span": span,
            "lhs_norm": normalize_phrase(lhs),
            "rhs_norm": normalize_phrase(rhs),
        }
        if str(meta.get("status", "")) == "verified_no_translate":
            no_translate_infos.append(info)
        elif str(meta.get("status", "")) == "verified_translate":
            translate_infos.append(info)

    remaining_no_translate: list[tuple[str, str]] = []
    remaining_translate: list[tuple[str, str]] = []
    ignored_rows = 0

    for row in rows:
        pt = str(row["pt_PT"])
        br = str(row["pt_BR"])
        if auto_fix.row_has_contextual_exception(pt, br):
            ignored_rows += 1
            continue
        pt_norm = normalize_phrase(pt)
        br_norm = normalize_phrase(br)

        for info in no_translate_infos:
            pt_has_lhs = str(info["lhs_norm"]) in pt_norm
            pt_has_rhs = str(info["rhs_norm"]) in pt_norm
            br_has_lhs = str(info["lhs_norm"]) in br_norm
            br_has_rhs = str(info["rhs_norm"]) in br_norm
            if (pt_has_lhs and br_has_rhs) or (pt_has_rhs and br_has_lhs):
                remaining_no_translate.append((str(row["merged_id"]), info["span"]))

        for info in translate_infos:
            pt_wrong = str(info["rhs_norm"]) in pt_norm
            br_wrong = str(info["lhs_norm"]) in br_norm
            if pt_wrong or br_wrong:
                remaining_translate.append((str(row["merged_id"]), info["span"]))

    print(
        "Remaining safe auto-derived mismatches: "
        f"no_translate={len(remaining_no_translate)} "
        f"translate={len(remaining_translate)} "
        f"context_rows_ignored={ignored_rows}"
    )
    for label, items in (
        ("NO_TRANSLATE", remaining_no_translate[:20]),
        ("TRANSLATE", remaining_translate[:20]),
    ):
        print(label)
        for merged_id, span in items:
            print(f"  {merged_id}: {span}")


if __name__ == "__main__":
    main()
