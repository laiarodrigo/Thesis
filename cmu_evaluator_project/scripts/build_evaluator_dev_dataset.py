#!/usr/bin/env python3
"""Join normalized DEV judgments to candidates and build blind V0 inputs."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from campaign_common import (
    CampaignError,
    atomic_write_json,
    atomic_write_text,
    canonical_candidate_index,
    flatten_records,
    read_json,
)


EXPECTED_PASSAGES = 10
EXPECTED_SOURCE_SENTENCES = 97
EXPECTED_SYSTEMS = 4
EXPECTED_CANDIDATES = 388


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--normalized-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v0-input", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _selection_key(record: dict[str, Any]) -> tuple[float, int, str]:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise CampaignError(f"{record.get('candidate_id')}: missing annotation provenance")
    timing = record.get("timing")
    end = timing.get("submission_end") if isinstance(timing, dict) else None
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        end = float("-inf")
    line = provenance.get("raw_line_number")
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        raise CampaignError(f"{record.get('candidate_id')}: invalid raw line number")
    submission_id = provenance.get("submission_id")
    if not isinstance(submission_id, str) or not submission_id:
        raise CampaignError(f"{record.get('candidate_id')}: invalid submission ID")
    return float(end), line, submission_id


def _validate_gold(record: dict[str, Any], candidate_text: str) -> dict[str, Any]:
    score = record.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        raise CampaignError(f"{record.get('candidate_id')}: score must be an integer in [0, 100]")
    raw_spans = record.get("error_spans")
    if not isinstance(raw_spans, list):
        raise CampaignError(f"{record.get('candidate_id')}: error_spans must be a list")
    spans: list[dict[str, Any]] = []
    for index, span in enumerate(raw_spans):
        if not isinstance(span, dict):
            raise CampaignError(f"{record.get('candidate_id')}: span {index} is not an object")
        start, end = span.get("start_i"), span.get("end_i")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start <= end <= len(candidate_text)
        ):
            raise CampaignError(f"{record.get('candidate_id')}: invalid span offsets {start!r}:{end!r}")
        raw_severity = span.get("severity")
        if raw_severity not in {"minor", "major"}:
            raise CampaignError(f"{record.get('candidate_id')}: invalid severity {raw_severity!r}")
        missing = start == end
        spans.append(
            {
                "start": start,
                "end": end,
                "text": "[MISSING]" if missing else candidate_text[start:end],
                "severity": raw_severity.title(),
                "missing": missing,
                "pearmut_category": span.get("category"),
            }
        )
    return {
        "error_spans": spans,
        "score": score,
        "error_free": not spans,
        "has_major_error": any(span["severity"] == "Major" for span in spans),
    }


def _passage_contexts(candidate_dataset: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flatten_records(candidate_dataset):
        passage_id = row.get("passage_id")
        position = row.get("position")
        source = row.get("source_bp")
        if not isinstance(passage_id, str) or not passage_id:
            raise CampaignError("candidate row has no passage_id")
        if not isinstance(position, int) or isinstance(position, bool):
            raise CampaignError(f"{passage_id}: invalid position")
        if not isinstance(source, str) or not source:
            raise CampaignError(f"{passage_id}: missing source_bp")
        contexts[passage_id].append(
            {"sentence_id": str(row.get("sentence_id")), "position": position, "source_bp": source}
        )
    for passage_id, rows in contexts.items():
        rows.sort(key=lambda row: row["position"])
        positions = [row["position"] for row in rows]
        if positions != list(range(1, len(rows) + 1)):
            raise CampaignError(f"{passage_id}: non-contiguous passage positions {positions}")
    return dict(contexts)


def build(candidate_dataset: dict[str, Any], normalized: dict[str, Any]) -> tuple[dict[str, Any], str]:
    candidates_by_id, _ = canonical_candidate_index(candidate_dataset)
    systems = candidate_dataset.get("systems")
    if not isinstance(systems, list):
        raise CampaignError("candidate dataset systems must be a list")
    systems_by_id = {row.get("system_id"): row for row in systems if isinstance(row, dict)}
    source_rows = flatten_records(candidate_dataset)
    source_by_key = {
        (str(row["document_id"]), str(row["passage_id"]), str(row["sentence_id"])): row
        for row in source_rows
    }
    contexts = _passage_contexts(candidate_dataset)

    normalized_records = normalized.get("records")
    if not isinstance(normalized_records, list):
        raise CampaignError("normalized annotations have no records list")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in normalized_records:
        if not isinstance(record, dict):
            raise CampaignError("normalized annotation record is not an object")
        candidate_id = record.get("candidate_id")
        if candidate_id not in candidates_by_id:
            raise CampaignError(f"normalized annotation has unknown candidate_id {candidate_id!r}")
        grouped[candidate_id].append(record)
    if set(grouped) != set(candidates_by_id):
        missing = sorted(set(candidates_by_id) - set(grouped))
        unexpected = sorted(set(grouped) - set(candidates_by_id))
        raise CampaignError(f"annotation/candidate mismatch; missing={missing}, unexpected={unexpected}")

    ordered_candidate_ids: list[str] = []
    for row in source_rows:
        declared = row.get("candidate_ids")
        if not isinstance(declared, dict):
            raise CampaignError(f"{row.get('item_key')}: missing candidate_ids")
        for system in systems:
            ordered_candidate_ids.append(declared[system["system_id"]])

    records: list[dict[str, Any]] = []
    v0_rows: list[dict[str, Any]] = []
    repeat_candidate_count = 0
    superseded_judgment_count = 0
    for ordinal, candidate_id in enumerate(ordered_candidate_ids, 1):
        candidate = candidates_by_id[candidate_id]
        judgments = sorted(grouped[candidate_id], key=_selection_key)
        selected = judgments[-1]
        if len(judgments) > 1:
            repeat_candidate_count += 1
            superseded_judgment_count += len(judgments) - 1
        if len({row.get("annotator_id") for row in judgments}) != 1:
            raise CampaignError(f"{candidate_id}: canonical DEV expects one annotator")
        key = (candidate["document_id"], candidate["passage_id"], candidate["sentence_id"])
        source_row = source_by_key.get(key)
        if source_row is None:
            raise CampaignError(f"{candidate_id}: source row not found")
        candidate_metadata = source_row.get("candidate_provenance", {}).get(candidate["system_id"])
        if not isinstance(candidate_metadata, dict):
            raise CampaignError(f"{candidate_id}: candidate provenance not found")
        system_metadata = systems_by_id.get(candidate["system_id"])
        if not isinstance(system_metadata, dict):
            raise CampaignError(f"{candidate_id}: system metadata not found")
        provenance = selected["provenance"]
        selected_submission = provenance["submission_id"]
        superseded = [row["provenance"]["submission_id"] for row in judgments[:-1]]
        context = copy.deepcopy(contexts[candidate["passage_id"]])
        evaluation_id = f"dev-v0-{ordinal:04d}"
        gold = _validate_gold(selected, candidate["candidate_text"])
        canonical = {
            "evaluation_id": evaluation_id,
            "candidate_id": candidate_id,
            "document_id": candidate["document_id"],
            "passage_id": candidate["passage_id"],
            "sentence_id": candidate["sentence_id"],
            "position": candidate["position"],
            "passage_context": context,
            "current_source_bp": candidate["source_bp"],
            "candidate": {
                "text": candidate["candidate_text"],
                "system_id": candidate["system_id"],
                "system_metadata": copy.deepcopy(system_metadata),
                "candidate_metadata": copy.deepcopy(candidate_metadata),
            },
            "human_gold": gold,
            "development_notes": {
                "english_pearmut_comment": selected.get("comment"),
                "formal_gold": False,
                "allowed_uses": [
                    "failure_analysis",
                    "prompt_design",
                    "retrieval",
                    "example_selection",
                ],
                "prohibited_for_current_item_input": True,
            },
            "annotation_provenance": {
                "annotator_id": selected.get("annotator_id"),
                "assignment_id": selected.get("assignment_id"),
                "campaign_id": selected.get("campaign_id"),
                "selected_submission_id": selected_submission,
                "candidate_submission_count": len(judgments),
                "selection_policy": "latest_submission_end_then_raw_line_then_submission_id",
                "superseded_submission_ids": superseded,
                "selected_record": copy.deepcopy(selected),
            },
        }
        records.append(canonical)
        v0_rows.append(
            {
                "evaluation_id": evaluation_id,
                "document_id": candidate["document_id"],
                "passage_id": candidate["passage_id"],
                "sentence_id": candidate["sentence_id"],
                "position": candidate["position"],
                "passage_context": context,
                "current_source_bp": candidate["source_bp"],
                "candidate_text": candidate["candidate_text"],
            }
        )

    passage_count = len(contexts)
    source_count = len(source_rows)
    system_count = len(systems)
    if (passage_count, source_count, system_count, len(records)) != (
        EXPECTED_PASSAGES,
        EXPECTED_SOURCE_SENTENCES,
        EXPECTED_SYSTEMS,
        EXPECTED_CANDIDATES,
    ):
        raise CampaignError(
            "unexpected DEV counts: "
            f"passages={passage_count}, source_sentences={source_count}, "
            f"systems={system_count}, candidates={len(records)}"
        )
    if len({row["evaluation_id"] for row in records}) != len(records):
        raise CampaignError("duplicate evaluation_id")
    if len({row["candidate_id"] for row in records}) != len(records):
        raise CampaignError("duplicate canonical candidate_id")

    comments = sum(
        isinstance(row["development_notes"]["english_pearmut_comment"], str)
        and bool(row["development_notes"]["english_pearmut_comment"].strip())
        for row in records
    )
    payload = {
        "schema_version": "1.0",
        "dataset_id": "ted2020_dev_01_evaluator",
        "split": "DEV",
        "record_unit": "one candidate judgment after final-state repeat resolution",
        "formal_gold_fields": ["human_gold.error_spans", "human_gold.score"],
        "comment_policy": {
            "formal_gold": False,
            "preserved_for_development": True,
            "excluded_from_current_item_evaluator_input": True,
        },
        "repeat_resolution": {
            "policy": "latest_submission_end_then_raw_line_then_submission_id",
            "all_submissions_remain_in_normalized_annotations": True,
            "repeat_candidate_count": repeat_candidate_count,
            "superseded_judgment_count": superseded_judgment_count,
        },
        "counts": {
            "passages": passage_count,
            "source_sentences": source_count,
            "systems": system_count,
            "candidate_judgments": len(records),
            "preserved_nonempty_english_comments": comments,
            "by_system": dict(Counter(row["candidate"]["system_id"] for row in records)),
        },
        "records": records,
    }
    v0_text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in v0_rows
    )
    return payload, v0_text


def main() -> None:
    args = parse_args()
    try:
        payload, v0_text = build(
            read_json(args.candidate_dataset), read_json(args.normalized_annotations)
        )
        atomic_write_json(args.output, payload, args.overwrite)
        atomic_write_text(args.v0_input, v0_text, args.overwrite)
        print(
            f"Built {args.output}: {payload['counts']['passages']} passages, "
            f"{payload['counts']['source_sentences']} source sentences, "
            f"{payload['counts']['candidate_judgments']} candidate judgments"
        )
        print(f"Built blind V0 input {args.v0_input}: {len(payload['records'])} rows")
    except CampaignError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
