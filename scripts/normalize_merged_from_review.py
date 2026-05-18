#!/usr/bin/env python3
"""
Rebuild and normalize the merged Wikipedia variant CSV line by line.

This script:
- expects the merged CSV and flagged CSV to have been freshly rebuilt from batch CSVs
- updates only the merged CSV
- applies reviewed safe lexical pairs row by row, anchored to each row's suspicious spans
- skips `∅`, structure-only/function-word-only pairs, and known contextual exceptions
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

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
        description="Normalize the merged CSV line by line from the reviewed pair table."
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
    return parser.parse_args()


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


def load_flagged_spans(flagged_csv: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with flagged_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("reason", "")).strip() != "non_approved_lexical_change":
                continue
            merged_id = str(row.get("merged_id", "")).strip()
            spans = [
                span.strip()
                for span in str(row.get("suspicious_spans", "")).split(" || ")
                if span.strip()
            ]
            if merged_id and spans:
                out[merged_id] = spans
    return out


def row_has_contextual_exception(pt_text: str, br_text: str) -> bool:
    combined = rw.normalize_text(pt_text) + " || " + rw.normalize_text(br_text)
    return any(marker in combined for marker in CONTEXTUAL_EXCEPTION_SUBSTRINGS)


def phrase_contains(container: str, candidate: str) -> bool:
    return rw.find_span_surface(
        container,
        candidate,
        whole_phrase=True,
        skip_mid_sentence_titlecase=False,
    ) is not None


def build_candidate_infos(
    suspicious_spans: list[str],
    pair_catalog: dict[str, dict[str, str | int]],
    pair_index: dict[str, set[str]],
) -> list[dict[str, str | int]]:
    candidates: list[dict[str, str | int]] = []
    seen: set[str] = set()

    for suspicious in suspicious_spans:
        exact = pair_catalog.get(suspicious)
        if exact is not None and suspicious not in seen:
            candidates.append(exact)
            seen.add(suspicious)

        if "=>" not in suspicious:
            continue
        left_phrase, right_phrase = rw.split_span(suspicious)
        if not left_phrase or not right_phrase or left_phrase == "∅" or right_phrase == "∅":
            continue

        token_candidates: set[str] = set()
        for token in rw.content_tokens(left_phrase) + rw.content_tokens(right_phrase):
            token_candidates.update(pair_index.get(token, set()))

        for span in token_candidates:
            if span in seen:
                continue
            info = pair_catalog[span]
            lhs = str(info["lhs"])
            rhs = str(info["rhs"])
            if (
                phrase_contains(left_phrase, lhs)
                and phrase_contains(right_phrase, rhs)
            ) or (
                phrase_contains(left_phrase, rhs)
                and phrase_contains(right_phrase, lhs)
            ):
                candidates.append(info)
                seen.add(span)

    candidates.sort(
        key=lambda item: (int(item["sort_len"]), len(str(item["span"]))),
        reverse=True,
    )
    return candidates


def apply_verified_no_translate(
    pt_text: str,
    br_text: str,
    lhs: str,
    rhs: str,
    lhs_surface: str,
    rhs_surface: str,
) -> tuple[str, str, int, int]:
    pt_has_lhs = rw.find_span_surface(pt_text, lhs, whole_phrase=True, skip_mid_sentence_titlecase=False)
    pt_has_rhs = rw.find_span_surface(pt_text, rhs, whole_phrase=True, skip_mid_sentence_titlecase=False)
    br_has_lhs = rw.find_span_surface(br_text, lhs, whole_phrase=True, skip_mid_sentence_titlecase=False)
    br_has_rhs = rw.find_span_surface(br_text, rhs, whole_phrase=True, skip_mid_sentence_titlecase=False)

    preferred_side = rw.preferred_canonical_side(lhs, rhs)
    if preferred_side == "lhs":
        canonical = "lhs"
    elif preferred_side == "rhs":
        canonical = "rhs"
    elif pt_has_lhs and br_has_rhs:
        canonical = "lhs"
    elif pt_has_rhs and br_has_lhs:
        canonical = "rhs"
    elif pt_has_lhs or br_has_lhs:
        canonical = "lhs"
    elif pt_has_rhs or br_has_rhs:
        canonical = "rhs"
    else:
        canonical = "lhs"

    pt_replacements = 0
    br_replacements = 0
    current_pt = pt_text
    current_br = br_text

    if canonical == "lhs":
        current_pt, pt_replacements = rw.replace_all_directional(
            current_pt,
            rhs,
            lhs_surface,
            skip_mid_sentence_titlecase=False,
        )
        current_br, br_replacements = rw.replace_all_directional(
            current_br,
            rhs,
            lhs_surface,
            skip_mid_sentence_titlecase=False,
        )
    else:
        current_pt, pt_replacements = rw.replace_all_directional(
            current_pt,
            lhs,
            rhs_surface,
            skip_mid_sentence_titlecase=False,
        )
        current_br, br_replacements = rw.replace_all_directional(
            current_br,
            lhs,
            rhs_surface,
            skip_mid_sentence_titlecase=False,
        )

    return current_pt, current_br, pt_replacements, br_replacements


def apply_verified_translate(
    pt_text: str,
    br_text: str,
    lhs: str,
    rhs: str,
    lhs_surface: str,
    rhs_surface: str,
) -> tuple[str, str, int, int]:
    current_pt, pt_replacements = rw.replace_all_directional(
        pt_text,
        rhs,
        lhs_surface,
        skip_mid_sentence_titlecase=False,
    )
    current_br, br_replacements = rw.replace_all_directional(
        br_text,
        lhs,
        rhs_surface,
        skip_mid_sentence_titlecase=False,
    )
    return current_pt, current_br, pt_replacements, br_replacements


def main() -> None:
    args = parse_args()
    review_meta = load_review_meta(args.review_csv)
    safe_no_translate, safe_translate = rw.build_safe_span_infos(review_meta)
    pair_catalog = {**safe_no_translate, **safe_translate}
    pair_index: dict[str, set[str]] = {}
    for span, info in pair_catalog.items():
        lhs = str(info["lhs"])
        rhs = str(info["rhs"])
        tokens = set(rw.content_tokens(lhs) + rw.content_tokens(rhs))
        for token in tokens:
            pair_index.setdefault(token, set()).add(span)
    flagged_spans_by_id = load_flagged_spans(args.flagged_csv)

    with args.merged_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys()) if rows else []

    rows_changed = 0
    pt_replacements = 0
    br_replacements = 0

    for row in rows:
        merged_id = str(row["merged_id"])
        suspicious_spans = flagged_spans_by_id.get(merged_id)
        if not suspicious_spans:
            continue

        original_pt = str(row["pt_PT"])
        original_br = str(row["pt_BR"])
        if row_has_contextual_exception(original_pt, original_br):
            continue

        current_pt = original_pt
        current_br = original_br
        candidate_infos = build_candidate_infos(
            suspicious_spans,
            pair_catalog,
            pair_index,
        )

        for info in candidate_infos:
            span = str(info["span"])
            lhs = str(info["lhs"])
            rhs = str(info["rhs"])
            status = review_meta[span]["status"]

            lhs_pref, rhs_pref = rw.preferred_surface_pair(lhs, rhs)
            source_pt_match = rw.find_span_surface(
                current_pt,
                lhs_pref,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            ) or rw.find_span_surface(
                current_pt,
                lhs_pref,
                whole_phrase=True,
                skip_mid_sentence_titlecase=False,
            ) or rw.find_span_surface(
                original_pt,
                lhs_pref,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            ) or rw.find_span_surface(
                original_pt,
                lhs_pref,
                whole_phrase=True,
                skip_mid_sentence_titlecase=False,
            ) or rw.find_span_surface(
                current_pt,
                lhs,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            ) or rw.find_span_surface(
                current_pt,
                lhs,
                whole_phrase=True,
                skip_mid_sentence_titlecase=False,
            )
            source_br_match = rw.find_span_surface(
                current_br,
                rhs_pref,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            ) or rw.find_span_surface(
                current_br,
                rhs_pref,
                whole_phrase=True,
                skip_mid_sentence_titlecase=False,
            ) or rw.find_span_surface(
                original_br,
                rhs_pref,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            ) or rw.find_span_surface(
                original_br,
                rhs_pref,
                whole_phrase=True,
                skip_mid_sentence_titlecase=False,
            ) or rw.find_span_surface(
                current_br,
                rhs,
                whole_phrase=True,
                skip_mid_sentence_titlecase=True,
            ) or rw.find_span_surface(
                current_br,
                rhs,
                whole_phrase=True,
                skip_mid_sentence_titlecase=False,
            )

            lhs_surface = source_pt_match[2] if source_pt_match is not None else lhs_pref
            rhs_surface = source_br_match[2] if source_br_match is not None else rhs_pref

            if status == "verified_no_translate":
                current_pt, current_br, pt_count, br_count = apply_verified_no_translate(
                    current_pt,
                    current_br,
                    lhs,
                    rhs,
                    lhs_surface,
                    rhs_surface,
                )
            elif status == "verified_translate":
                current_pt, current_br, pt_count, br_count = apply_verified_translate(
                    current_pt,
                    current_br,
                    lhs,
                    rhs,
                    lhs_surface,
                    rhs_surface,
                )
            else:
                continue

            pt_replacements += pt_count
            br_replacements += br_count

        if current_pt != original_pt or current_br != original_br:
            row["pt_PT"] = current_pt
            row["pt_BR"] = current_br
            row["pt_PT_words"] = str(rw.count_words(current_pt))
            row["pt_BR_words"] = str(rw.count_words(current_br))
            rows_changed += 1

    preferred_canonical_infos: list[dict[str, str]] = []
    for span, info in safe_no_translate.items():
        lhs = str(info["lhs"])
        rhs = str(info["rhs"])
        preferred_side = rw.preferred_canonical_side(lhs, rhs)
        if preferred_side is None:
            continue
        lhs_pref, rhs_pref = rw.preferred_surface_pair(lhs, rhs)
        preferred_canonical_infos.append(
            {
                "lhs": lhs,
                "rhs": rhs,
                "preferred_side": preferred_side,
                "lhs_surface": lhs_pref,
                "rhs_surface": rhs_pref,
            }
        )

    for row in rows:
        current_pt = str(row["pt_PT"])
        current_br = str(row["pt_BR"])
        if row_has_contextual_exception(current_pt, current_br):
            continue
        original_pt = current_pt
        original_br = current_br

        for info in preferred_canonical_infos:
            lhs = info["lhs"]
            rhs = info["rhs"]
            if info["preferred_side"] == "lhs":
                current_pt, pt_count = rw.replace_all_directional(current_pt, rhs, str(info["lhs_surface"]))
                current_br, br_count = rw.replace_all_directional(current_br, rhs, str(info["lhs_surface"]))
            else:
                current_pt, pt_count = rw.replace_all_directional(current_pt, lhs, str(info["rhs_surface"]))
                current_br, br_count = rw.replace_all_directional(current_br, lhs, str(info["rhs_surface"]))
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
        "Normalized merged CSV line by line: "
        f"rows_changed={rows_changed} "
        f"pt_replacements={pt_replacements} "
        f"br_replacements={br_replacements}"
    )


if __name__ == "__main__":
    main()
