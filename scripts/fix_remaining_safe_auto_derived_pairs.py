#!/usr/bin/env python3
"""
Apply safe auto-derived reviewed pairs across the merged Wikipedia variant CSV.

This is a second-pass cleanup for the merged CSV only.

- operates on `auto-derived` review entries
- skips `∅` pairs
- skips pairs whose source chain includes unsafe function-word-only spans
- skips known contextual exceptions where a literal rewrite would be wrong
- for verified_translate pairs, enforces lhs on pt_PT and rhs on pt_BR
- for verified_no_translate pairs, equalizes both sides while preserving
  preferred canonical directions for explicit exceptions
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


CONTEXTUAL_EXCEPTION_SUBSTRINGS = (
    "los angeles times",
    "trilha dos tupiniquins",
    "sociedade internacional para a consciencia de krishna",
    "camara dos deputados",
    "mar mediterraneo",
    "mediterraneo",
    "confederacao africana de voleibol",
)


def parse_args() -> argparse.Namespace:
    base_dir = REPO_ROOT / "data" / "wikipedia_pt_variant_csv"
    parser = argparse.ArgumentParser(
        description="Apply safe auto-derived reviewed pairs across the merged CSV."
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


def row_has_contextual_exception(pt_text: str, br_text: str) -> bool:
    combined = rw.normalize_text(pt_text) + " || " + rw.normalize_text(br_text)
    return any(marker in combined for marker in CONTEXTUAL_EXCEPTION_SUBSTRINGS)


def load_review_meta(review_csv: Path) -> dict[str, dict[str, str]]:
    review_meta: dict[str, dict[str, str]] = {}
    with review_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            span = str(row.get("span", "")).strip()
            if not span:
                continue
            review_meta[span] = {
                "status": str(row.get("review_status", "")).strip(),
                "notes": str(row.get("notes", "")).strip(),
            }
    return review_meta


def build_auto_pair_infos(
    rows: list[dict[str, str]],
    review_meta_by_span: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    safe_no_translate, safe_translate = rw.build_safe_span_infos(review_meta_by_span)
    no_translate_infos: list[dict[str, str]] = []
    translate_infos: list[dict[str, str]] = []

    def canonical_surface(column: str, preferred: str, fallback: str) -> str:
        for row in rows:
            match = rw.find_span_surface(
                str(row[column]),
                preferred,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            )
            if match is not None:
                return match[2]
        return preferred or fallback

    for pool, output, status in (
        (safe_no_translate, no_translate_infos, "verified_no_translate"),
        (safe_translate, translate_infos, "verified_translate"),
    ):
        for span, info in pool.items():
            meta = review_meta_by_span.get(span, {})
            notes = str(meta.get("notes", "")).strip().lower()
            if "auto-derived" not in notes:
                continue
            lhs = str(info["lhs"])
            rhs = str(info["rhs"])
            lhs_pref, rhs_pref = rw.preferred_surface_pair(lhs, rhs)
            output.append(
                {
                    "span": span,
                    "status": status,
                    "lhs": lhs,
                    "rhs": rhs,
                    "lhs_surface": canonical_surface("pt_PT", lhs_pref, lhs),
                    "rhs_surface": canonical_surface("pt_BR", rhs_pref, rhs),
                }
            )

    no_translate_infos.sort(key=lambda item: max(len(item["lhs"]), len(item["rhs"])), reverse=True)
    translate_infos.sort(key=lambda item: max(len(item["lhs"]), len(item["rhs"])), reverse=True)
    return no_translate_infos, translate_infos


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.merged_csv.open("r", encoding="utf-8", newline="")))
    fieldnames = list(rows[0].keys()) if rows else []
    review_meta_by_span = load_review_meta(args.review_csv)

    no_translate_infos, translate_infos = build_auto_pair_infos(rows, review_meta_by_span)

    rows_changed = 0
    pt_replacements = 0
    br_replacements = 0

    for row in rows:
        current_pt = str(row["pt_PT"])
        current_br = str(row["pt_BR"])
        original_pt = current_pt
        original_br = current_br

        if not row_has_contextual_exception(current_pt, current_br):
            for info in no_translate_infos:
                lhs = info["lhs"]
                rhs = info["rhs"]
                pt_has_lhs = rw.find_span_surface(
                    current_pt, lhs, whole_phrase=True, skip_mid_sentence_titlecase=True
                )
                pt_has_rhs = rw.find_span_surface(
                    current_pt, rhs, whole_phrase=True, skip_mid_sentence_titlecase=True
                )
                br_has_lhs = rw.find_span_surface(
                    current_br, lhs, whole_phrase=True, skip_mid_sentence_titlecase=True
                )
                br_has_rhs = rw.find_span_surface(
                    current_br, rhs, whole_phrase=True, skip_mid_sentence_titlecase=True
                )

                mismatch = (pt_has_lhs and br_has_rhs) or (pt_has_rhs and br_has_lhs)
                if not mismatch:
                    continue

                preferred_side = rw.preferred_canonical_side(lhs, rhs)
                if preferred_side == "rhs":
                    if pt_has_lhs:
                        current_pt, count = rw.replace_all_directional(
                            current_pt, lhs, str(info["rhs_surface"])
                        )
                        pt_replacements += count
                    if br_has_lhs:
                        current_br, count = rw.replace_all_directional(
                            current_br, lhs, str(info["rhs_surface"])
                        )
                        br_replacements += count
                else:
                    if pt_has_rhs:
                        current_pt, count = rw.replace_all_directional(
                            current_pt, rhs, str(info["lhs_surface"])
                        )
                        pt_replacements += count
                    if br_has_rhs:
                        current_br, count = rw.replace_all_directional(
                            current_br, rhs, str(info["lhs_surface"])
                        )
                        br_replacements += count

            for info in translate_infos:
                lhs = info["lhs"]
                rhs = info["rhs"]
                current_pt, pt_count = rw.replace_all_directional(
                    current_pt, rhs, str(info["lhs_surface"])
                )
                current_br, br_count = rw.replace_all_directional(
                    current_br, lhs, str(info["rhs_surface"])
                )
                pt_replacements += pt_count
                br_replacements += br_count

        if current_pt != original_pt or current_br != original_br:
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
        "Applied safe auto-derived review pairs: "
        f"rows_changed={rows_changed} "
        f"pt_replacements={pt_replacements} "
        f"br_replacements={br_replacements}"
    )


if __name__ == "__main__":
    main()
