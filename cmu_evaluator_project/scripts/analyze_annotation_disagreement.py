#!/usr/bin/env python3
"""Produce descriptive agreement/disagreement summaries for normalized data."""

from __future__ import annotations

import argparse
import itertools
import statistics
from pathlib import Path
from typing import Any

from campaign_common import CampaignError, atomic_write_json, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _span_set(spans: list[dict[str, Any]]) -> set[int]:
    result: set[int] = set()
    for span in spans:
        start, end = span.get("start_i"), span.get("end_i")
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            raise CampaignError(f"invalid span boundary: {span}")
        result.update(range(start, end + 1))
    return result


def analyze(payload: dict[str, Any], score_threshold: float) -> dict[str, Any]:
    if score_threshold < 0:
        raise CampaignError("score threshold cannot be negative")
    records = payload.get("records")
    if not isinstance(records, list):
        raise CampaignError("normalized input records must be a list")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("candidate_id"), str):
            raise CampaignError(f"record {index} has no candidate_id")
        grouped.setdefault(record["candidate_id"], []).append(record)

    rows: list[dict[str, Any]] = []
    for stable_id, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        scores = [record.get("score") for record in group]
        numeric_scores = [float(score) for score in scores if isinstance(score, (int, float))]
        has_error = [bool(record.get("error_spans", [])) for record in group]
        has_major = [record.get("severity") == "major" for record in group]
        severities = {record.get("severity") for record in group if record.get("severity")}
        span_pairs = []
        for left, right in itertools.combinations(group, 2):
            left_spans = left.get("error_spans", [])
            right_spans = right.get("error_spans", [])
            if not isinstance(left_spans, list) or not isinstance(right_spans, list):
                raise CampaignError(f"{stable_id}: error_spans must be lists")
            left_set, right_set = _span_set(left_spans), _span_set(right_spans)
            union = left_set | right_set
            span_pairs.append(
                {
                    "annotators": [left.get("annotator_id"), right.get("annotator_id")],
                    "any_overlap": bool(left_set & right_set),
                    "exact_boundaries": left_spans == right_spans,
                    "character_jaccard": len(left_set & right_set) / len(union) if union else 1.0,
                }
            )
        score_range = max(numeric_scores) - min(numeric_scores) if len(numeric_scores) >= 2 else None
        comparable = [
            {
                "score": record.get("score"),
                "error_spans": record.get("error_spans", []),
                "comment": record.get("comment"),
            }
            for record in group
        ]
        rows.append(
            {
                "candidate_id": stable_id,
                "system_id": group[0].get("system_id"),
                "annotation_count": len(group),
                "annotator_ids": [record.get("annotator_id") for record in group],
                "scores": scores,
                "score_mean_descriptive": statistics.mean(numeric_scores) if numeric_scores else None,
                "score_median_descriptive": statistics.median(numeric_scores) if numeric_scores else None,
                "score_range": score_range,
                "flags": {
                    "no_error_vs_error": len(set(has_error)) > 1,
                    "major_vs_no_major": len(set(has_major)) > 1,
                    "minor_vs_major": "minor" in severities and "major" in severities,
                    "score_difference_above_threshold": (
                        score_range is not None and score_range > score_threshold
                    ),
                    "identical_judgments": all(value == comparable[0] for value in comparable[1:]),
                    "span_boundary_difference": any(
                        not pair["exact_boundaries"] for pair in span_pairs
                    ),
                },
                "span_pair_comparisons": span_pairs,
            }
        )
    return {
        "schema_version": "1.0",
        "analysis_type": "descriptive_only",
        "score_difference_threshold": score_threshold,
        "candidate_count_total": len(grouped),
        "multiply_annotated_candidate_count": len(rows),
        "candidates": rows,
    }


def main() -> None:
    args = parse_args()
    try:
        result = analyze(read_json(args.input), args.score_threshold)
        atomic_write_json(args.output, result, args.overwrite)
        print(
            f"Analyzed {result['multiply_annotated_candidate_count']} multiply annotated "
            "candidate(s); no consensus or adjudication was applied"
        )
    except CampaignError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
