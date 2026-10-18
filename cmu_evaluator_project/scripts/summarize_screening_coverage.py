#!/usr/bin/env python3
"""Report accepted TEST coverage without modifying raw annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_annotation_campaigns import _accepted_coverage
from campaign_common import CampaignError, read_json, require_nonempty_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    return parser.parse_args()


def summarize(manifest: dict) -> dict:
    assignments = manifest.get("assignments")
    annotators = manifest.get("annotators")
    if not isinstance(assignments, list) or not isinstance(annotators, list):
        raise CampaignError("manifest must contain assignments and annotators lists")
    statuses: dict[str, str] = {}
    for row in annotators:
        if not isinstance(row, dict):
            raise CampaignError("annotators entries must be objects")
        user_id = require_nonempty_string(row.get("user_id"), "annotators[].user_id")
        status = require_nonempty_string(row.get("screening_status"), "screening_status")
        if status not in {"pending", "accepted", "rejected"}:
            raise CampaignError(f"invalid screening status for {user_id}: {status}")
        if user_id in statuses:
            raise CampaignError(f"duplicate annotator user_id: {user_id}")
        statuses[user_id] = status
    coverage = _accepted_coverage(assignments, statuses)
    return {
        "schema_version": "1.0",
        "pool_id": manifest.get("pool_id"),
        "required_accepted_annotators_per_test_passage": 2,
        "coverage": coverage,
        "complete_passage_count": sum(
            row["replacement_assignments_needed"] == 0 for row in coverage
        ),
        "replacement_assignment_count": sum(
            row["replacement_assignments_needed"] or 0 for row in coverage
        ),
        "pending_screening_passage_count": sum(
            not row["screening_complete"] for row in coverage
        ),
    }


def main() -> None:
    args = parse_args()
    try:
        print(json.dumps(summarize(read_json(args.assignment_manifest)), indent=2))
    except CampaignError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
