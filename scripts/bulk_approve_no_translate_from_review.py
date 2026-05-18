#!/usr/bin/env python3
"""
Promote top-ranked unreviewed lexical-change spans to verified_no_translate.

The intended workflow is:
1. Use the current review CSV.
2. Treat unreviewed rows in the first N review entries as user-approved
   "do not translate" cases.
3. Also approve conservative inflectional variants of those spans elsewhere
   in the review CSV.
4. Persist the approvals back into the verified pairs CSV.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    base_dir = repo_root / "data" / "wikipedia_pt_variant_csv"
    parser = argparse.ArgumentParser(
        description=(
            "Bulk-approve first-N unreviewed review spans as verified_no_translate "
            "and propagate conservative inflectional variants."
        )
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=base_dir / "pt_variant_non_approved_lexical_unique_pairs_review.csv",
    )
    parser.add_argument(
        "--verified-pairs-csv",
        type=Path,
        default=base_dir / "pt_variant_non_approved_lexical_user_verified_pairs.csv",
    )
    parser.add_argument(
        "--top-review-rows",
        type=int,
        default=179,
        help=(
            "Number of review data rows to scan from the top. "
            "179 corresponds to file lines 2-180."
        ),
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    text = " ".join(text.strip().lower().split())
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


INFLECTIONAL_SUFFIXES = tuple(
    sorted(
        [
            "ávamos",
            "íamos",
            "áramos",
            "éramos",
            "íramos",
            "aremos",
            "eremos",
            "iremos",
            "ariam",
            "eriam",
            "iriam",
            "aram",
            "eram",
            "iram",
            "avam",
            "iam",
            "ados",
            "adas",
            "idos",
            "idas",
            "ando",
            "endo",
            "indo",
            "ado",
            "ada",
            "ido",
            "ida",
            "ões",
            "ães",
            "ais",
            "eis",
            "óis",
            "is",
            "es",
            "ns",
            "ou",
            "am",
            "em",
            "os",
            "as",
            "o",
            "a",
        ],
        key=len,
        reverse=True,
    )
)


def inflectional_token_key(token: str) -> str:
    token = normalize(token)
    for suffix in INFLECTIONAL_SUFFIXES:
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def phrase_key(text: str) -> tuple[str, ...]:
    tokens = [tok for tok in re.split(r"\s+", normalize(text)) if tok]
    return tuple(inflectional_token_key(tok) for tok in tokens)


def split_span(span: str) -> tuple[str, str]:
    lhs, rhs = span.split("=>", 1)
    return lhs.strip(), rhs.strip()


def load_verified(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    by_span: dict[str, dict[str, str]] = {}
    for row in rows:
        span = str(row.get("span", "")).strip()
        if not span:
            continue
        by_span[span] = {
            "status": str(row.get("status", "")).strip(),
            "notes": str(row.get("notes", "")).strip(),
        }
    return rows, by_span


def main() -> None:
    args = parse_args()
    if not args.review_csv.exists():
        raise SystemExit(f"Missing review CSV: {args.review_csv}")
    if not args.verified_pairs_csv.exists():
        raise SystemExit(f"Missing verified pairs CSV: {args.verified_pairs_csv}")

    review_rows = list(csv.DictReader(args.review_csv.open("r", encoding="utf-8", newline="")))
    verified_rows, verified_by_span = load_verified(args.verified_pairs_csv)

    seed_spans: list[str] = []
    seed_pairs: list[tuple[str, str]] = []
    for row in review_rows[: args.top_review_rows]:
        if str(row.get("review_status", "")).strip() != "unreviewed":
            continue
        span = str(row.get("span", "")).strip()
        if not span:
            continue
        seed_spans.append(span)
        seed_pairs.append(split_span(span))

    seed_norm_spans = {normalize(span) for span in seed_spans}
    seed_keys = {(phrase_key(lhs), phrase_key(rhs)) for lhs, rhs in seed_pairs}

    exact_matches: list[str] = []
    derived_matches: list[str] = []
    for row in review_rows:
        if str(row.get("review_status", "")).strip() != "unreviewed":
            continue
        span = str(row.get("span", "")).strip()
        if not span:
            continue
        if normalize(span) in seed_norm_spans:
            exact_matches.append(span)
            continue
        lhs, rhs = split_span(span)
        if (phrase_key(lhs), phrase_key(rhs)) in seed_keys:
            derived_matches.append(span)

    additions: list[dict[str, str]] = []
    for span in exact_matches:
        if span in verified_by_span:
            continue
        additions.append(
            {
                "span": span,
                "status": "verified_no_translate",
                "notes": "bulk-approved from first 180 unreviewed review lines",
            }
        )
        verified_by_span[span] = additions[-1]

    for span in derived_matches:
        if span in verified_by_span:
            continue
        additions.append(
            {
                "span": span,
                "status": "verified_no_translate",
                "notes": "auto-derived inflectional variant of top-180 no-translate span",
            }
        )
        verified_by_span[span] = additions[-1]

    if additions:
        verified_rows.extend(additions)
        with args.verified_pairs_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["span", "status", "notes"])
            writer.writeheader()
            writer.writerows(verified_rows)

    print(
        f"Seeds={len(seed_spans)} exact_matches={len(exact_matches)} "
        f"derived_matches={len(derived_matches)} additions_written={len(additions)}"
    )


if __name__ == "__main__":
    main()
