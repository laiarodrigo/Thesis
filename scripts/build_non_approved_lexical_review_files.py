#!/usr/bin/env python3
"""
Annotate unique non_approved_lexical_change spans with user review status.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "data" / "wikipedia_pt_variant_csv"
    parser = argparse.ArgumentParser(
        description=(
            "Build annotated review files for non_approved_lexical_change spans."
        )
    )
    parser.add_argument(
        "--unique-pairs-csv",
        type=Path,
        default=base_dir / "pt_variant_non_approved_lexical_unique_pairs.csv",
    )
    parser.add_argument(
        "--verified-pairs-csv",
        type=Path,
        default=base_dir / "pt_variant_non_approved_lexical_user_verified_pairs.csv",
    )
    parser.add_argument(
        "--output-review-csv",
        type=Path,
        default=base_dir / "pt_variant_non_approved_lexical_unique_pairs_review.csv",
    )
    parser.add_argument(
        "--output-unmatched-verified-csv",
        type=Path,
        default=base_dir / "pt_variant_non_approved_lexical_user_verified_pairs_not_currently_flagged.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.unique_pairs_csv.exists():
        raise SystemExit(f"Missing unique pairs CSV: {args.unique_pairs_csv}")
    if not args.verified_pairs_csv.exists():
        raise SystemExit(f"Missing verified pairs CSV: {args.verified_pairs_csv}")

    verified: dict[str, dict[str, str]] = {}
    with args.verified_pairs_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            span = str(row.get("span", "")).strip()
            if not span:
                continue
            verified[span] = {
                "status": str(row.get("status", "")).strip() or "verified_translate",
                "notes": str(row.get("notes", "")).strip(),
            }

    seen_verified: set[str] = set()
    review_rows: list[dict[str, str]] = []
    with args.unique_pairs_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            span = str(row.get("span", "")).strip()
            count = str(row.get("count", "")).strip()
            status = "unreviewed"
            notes = ""
            if span in verified:
                status = verified[span]["status"]
                notes = verified[span]["notes"]
                seen_verified.add(span)
            out_row = {
                "count": count,
                "span": span,
                "review_status": status,
                "notes": notes,
            }
            review_rows.append(out_row)

    unmatched_verified_rows = []
    for span, meta in verified.items():
        if span in seen_verified:
            continue
        unmatched_verified_rows.append(
            {
                "span": span,
                "status": meta["status"],
                "notes": meta["notes"],
                "currently_flagged": "no",
            }
        )

    args.output_review_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_review_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["count", "span", "review_status", "notes"],
        )
        writer.writeheader()
        writer.writerows(review_rows)

    with args.output_unmatched_verified_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["span", "status", "notes", "currently_flagged"],
        )
        writer.writeheader()
        writer.writerows(unmatched_verified_rows)

    print(
        f"Wrote review CSV to {args.output_review_csv} "
        f"(total_unique_rows={len(review_rows)}, "
        f"unreviewed_rows={sum(1 for row in review_rows if row['review_status'] == 'unreviewed')})."
    )
    print(
        f"Wrote unmatched verified CSV to {args.output_unmatched_verified_csv} "
        f"(rows={len(unmatched_verified_rows)})."
    )


if __name__ == "__main__":
    main()
