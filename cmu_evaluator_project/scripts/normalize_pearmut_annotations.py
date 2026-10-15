#!/usr/bin/env python3
"""Normalize Pearmut JSONL exports without collapsing individual judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from campaign_common import (
    CampaignError,
    atomic_write_json,
    candidate_id,
    canonical_candidate_index,
    item_key,
    read_json,
    require_nonempty_string,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--assignment-manifest", type=Path, required=True)
    parser.add_argument(
        "--campaign-id",
        action="append",
        default=[],
        help="normalize only this campaign from a multi-campaign manifest; may be repeated",
    )
    parser.add_argument(
        "--raw-export",
        action="append",
        default=[],
        metavar="ASSIGNMENT_ID=PATH",
        help="override/add a raw export for an assignment; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeat-policy",
        choices=("error", "preserve"),
        default="error",
        help="explicit policy for non-identical repeated annotator/candidate submissions",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mapping_by_label(mapping: dict[str, Any], assignment: dict[str, Any]) -> dict[str, str]:
    for field in ("campaign_id", "assignment_id", "annotator_id"):
        expected = assignment.get(field)
        observed = mapping.get(field)
        if observed is not None and observed != expected:
            raise CampaignError(
                f"mapping {field} {observed!r} does not match assignment {expected!r}"
            )
    entries = mapping.get("mappings")
    if not isinstance(entries, list) or not entries:
        raise CampaignError("private mapping has no mappings")
    result: dict[str, str] = {}
    real_seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CampaignError(f"mapping entry {index} must be an object")
        label = require_nonempty_string(entry.get("anonymous_system_id"), "anonymous_system_id")
        system = require_nonempty_string(entry.get("real_system_id"), "real_system_id")
        if label in result or system in real_seen:
            raise CampaignError("private mapping is not one-to-one")
        result[label] = system
        real_seen.add(system)
    return result


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any], str]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CampaignError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise CampaignError(f"expected JSON object at {path}:{line_number}")
                yield line_number, value, line.rstrip("\n")
    except OSError as exc:
        raise CampaignError(f"could not read raw Pearmut export {path}: {exc}") from exc


def _submission_id(campaign_id: str, row: dict[str, Any]) -> str:
    actions = row.get("actions", [])
    if not isinstance(actions, list):
        raise CampaignError("Pearmut actions must be a list")
    relevant_times = [
        action.get("time")
        for action in actions
        if isinstance(action, dict) and action.get("action") in {"load", "submit"}
    ]
    items = row.get("item")
    item_ids = [item.get("item_id") for item in items] if isinstance(items, list) else []
    identity = {
        "campaign_id": campaign_id,
        "pearmut_user_id": row.get("user_id"),
        "item_i": row.get("item_i"),
        "load_submit_times": relevant_times,
        "item_ids": item_ids,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _timing(actions: list[dict[str, Any]], segment_index: int, label: str) -> dict[str, Any]:
    load_times = [a.get("time") for a in actions if a.get("action") == "load" and isinstance(a.get("time"), (int, float))]
    submit_times = [a.get("time") for a in actions if a.get("action") == "submit" and isinstance(a.get("time"), (int, float))]
    candidate_actions = [
        a
        for a in actions
        if a.get("index") == segment_index and a.get("model") == label
    ]
    start = load_times[0] if load_times else None
    end = submit_times[-1] if submit_times else None
    return {
        "submission_start": start,
        "submission_end": end,
        "submission_elapsed_seconds": end - start if start is not None and end is not None else None,
        "candidate_action_count": len(candidate_actions),
        "candidate_first_action": candidate_actions[0].get("time") if candidate_actions else None,
        "candidate_last_action": candidate_actions[-1].get("time") if candidate_actions else None,
    }


def normalize(
    candidate_dataset: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    overrides: dict[str, list[Path]],
    repeat_policy: str,
    campaign_ids: set[str] | None = None,
) -> dict[str, Any]:
    candidates_by_id, items_by_key = canonical_candidate_index(candidate_dataset)
    dataset_systems = {record["system_id"] for record in candidates_by_id.values()}
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise CampaignError("assignment manifest has no assignments")
    records: list[dict[str, Any]] = []
    submission_ids: set[str] = set()
    judgment_keys: dict[tuple[str, str], list[str]] = {}
    status_by_user = {
        row["user_id"]: row.get("screening_status", "pending")
        for row in manifest.get("annotators", [])
        if isinstance(row, dict) and isinstance(row.get("user_id"), str)
    }
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    exports: dict[tuple[str, Path], set[str]] = {}
    assignment_export_keys: set[tuple[str, str, Path]] = set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise CampaignError("each manifest assignment must be an object")
        assignment_id = require_nonempty_string(assignment.get("assignment_id"), "assignment_id")
        annotator_id = require_nonempty_string(assignment.get("annotator_id"), "annotator_id")
        campaign_id = require_nonempty_string(assignment.get("campaign_id"), "campaign_id")
        if campaign_ids and campaign_id not in campaign_ids:
            continue
        declared_user_id = assignment.get("user_id")
        user_id = require_nonempty_string(
            declared_user_id if declared_user_id is not None else annotator_id,
            f"{assignment_id}.user_id",
        )
        if declared_user_id is not None and user_id != annotator_id:
            raise CampaignError(f"{assignment_id}: user_id and annotator_id must match")
        context_user_key = user_id if declared_user_id is not None else "*"
        identity = (campaign_id, context_user_key)
        if identity in contexts:
            raise CampaignError(f"duplicate campaign/user assignment: {campaign_id}/{user_id}")
        mapping_path = Path(require_nonempty_string(assignment.get("mapping_path"), "mapping_path"))
        if not mapping_path.is_absolute():
            mapping_path = manifest_path.parent / mapping_path
        label_to_system = _mapping_by_label(read_json(mapping_path), assignment)
        if set(label_to_system.values()) != dataset_systems:
            raise CampaignError(f"{assignment_id}: mapping systems do not match candidate dataset")
        expected_passages = assignment.get("passage_ids")
        if expected_passages is None and manifest.get("schema_version") != "2.0":
            expected_passages = sorted(
                {record["passage_id"] for record in candidates_by_id.values()}
            )
        if not isinstance(expected_passages, list) or not expected_passages or not all(
            isinstance(value, str) and value for value in expected_passages
        ):
            raise CampaignError(f"{assignment_id}.passage_ids must be a non-empty string list")
        contexts[identity] = {
            "assignment": assignment,
            "assignment_id": assignment_id,
            "annotator_id": annotator_id,
            "user_id": user_id,
            "declared_user_id": declared_user_id,
            "campaign_id": campaign_id,
            "label_to_system": label_to_system,
            "expected_passages": set(expected_passages),
            "screening_status": status_by_user.get(user_id),
            "eligible_for_gold_override": assignment.get("eligible_for_gold"),
        }
        if context_override := assignment.get("eligible_for_gold"):
            if not isinstance(context_override, bool):
                raise CampaignError(
                    f"{assignment_id}.eligible_for_gold must be a boolean when present"
                )
        elif "eligible_for_gold" in assignment and not isinstance(
            assignment["eligible_for_gold"], bool
        ):
            raise CampaignError(
                f"{assignment_id}.eligible_for_gold must be a boolean when present"
            )
        configured_exports = assignment.get("raw_exports", [])
        if not isinstance(configured_exports, list):
            raise CampaignError(f"{assignment_id}.raw_exports must be a list")
        export_paths = overrides.get(assignment_id)
        if export_paths is None:
            export_paths = [Path(value) for value in configured_exports]
        for raw_path in export_paths:
            if not raw_path.is_absolute():
                raw_path = manifest_path.parent / raw_path
            resolved_raw_path = raw_path.resolve()
            export_key = (campaign_id, context_user_key, resolved_raw_path)
            if export_key in assignment_export_keys:
                raise CampaignError(
                    f"duplicate raw submission source for {campaign_id}/{user_id}: {resolved_raw_path}"
                )
            assignment_export_keys.add(export_key)
            exports.setdefault((campaign_id, resolved_raw_path), set()).add(context_user_key)

    if not contexts:
        raise CampaignError("no assignments matched the selected campaign IDs")
    if not exports:
        raise CampaignError("no raw Pearmut exports were configured")
    for (campaign_id, raw_path), allowed_users in exports.items():
        for line_number, row, raw_line in _jsonl(raw_path):
            row_campaign = row.get("campaign_id")
            if row_campaign is not None and row_campaign != campaign_id:
                raise CampaignError(
                    f"{raw_path}:{line_number}: campaign_id {row_campaign!r} does not match {campaign_id!r}"
                )
            user_id = require_nonempty_string(
                row.get("user_id"), f"{raw_path}:{line_number}.user_id"
            )
            context = contexts.get((campaign_id, user_id)) or contexts.get((campaign_id, "*"))
            if context is None:
                raise CampaignError(
                    f"{raw_path}:{line_number}: unexpected user_id {user_id!r} for {campaign_id}"
                )
            if user_id not in allowed_users and "*" not in allowed_users:
                # The shared file may have been configured on only a subset of
                # assignments. Refuse to infer permission for another user.
                raise CampaignError(
                    f"{raw_path}:{line_number}: export is not configured for user_id {user_id!r}"
                )
            assignment = context["assignment"]
            assignment_id = context["assignment_id"]
            annotator_id = context["annotator_id"]
            label_to_system = context["label_to_system"]
            submission_id = _submission_id(campaign_id, row)
            if submission_id in submission_ids:
                raise CampaignError(
                    f"duplicate raw submission detected: {submission_id} at {raw_path}:{line_number}"
                )
            submission_ids.add(submission_id)
            annotations = row.get("annotation")
            items = row.get("item")
            actions = row.get("actions", [])
            if not isinstance(annotations, list) or not isinstance(items, list) or len(annotations) != len(items):
                raise CampaignError(f"{raw_path}:{line_number}: item/annotation alignment failure")
            if not isinstance(actions, list) or not all(isinstance(action, dict) for action in actions):
                raise CampaignError(f"{raw_path}:{line_number}: actions must be objects")
            raw_sha = hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
            for segment_index, (segment, annotation) in enumerate(zip(items, annotations)):
                if not annotation:
                    continue
                if not isinstance(segment, dict) or not isinstance(annotation, dict):
                    raise CampaignError(f"{raw_path}:{line_number}: invalid segment annotation")
                raw_item_key = require_nonempty_string(segment.get("item_id"), "item.item_id")
                try:
                    document_id, passage_id, sentence_id = json.loads(raw_item_key)
                except (ValueError, TypeError) as exc:
                    raise CampaignError(f"unrecognized judged item_id: {raw_item_key}") from exc
                key = item_key(str(document_id), str(passage_id), str(sentence_id))
                canonical_item = items_by_key.get(key)
                if canonical_item is None:
                    raise CampaignError(f"unknown candidate item: {key}")
                if str(passage_id) not in context["expected_passages"]:
                    raise CampaignError(
                        f"{assignment_id}: unexpected passage_id {passage_id!r}"
                    )
                targets = segment.get("tgt")
                if not isinstance(targets, dict):
                    raise CampaignError(f"{key}: Pearmut tgt must be an object")
                for label, judgment in annotation.items():
                    system_id = label_to_system.get(label)
                    if system_id is None:
                        raise CampaignError(
                            f"{assignment_id}: missing anonymization mapping for {label!r}"
                        )
                    if not isinstance(judgment, dict):
                        raise CampaignError(f"{key}/{label}: judgment must be an object")
                    stable_id = candidate_id(str(passage_id), str(sentence_id), system_id)
                    candidate = candidates_by_id.get(stable_id)
                    if candidate is None:
                        raise CampaignError(f"unknown candidate/system: {stable_id}")
                    if targets.get(label) != candidate["candidate_text"]:
                        raise CampaignError(f"candidate text mismatch for {stable_id}")
                    spans = judgment.get("error_spans", [])
                    if not isinstance(spans, list) or not all(isinstance(span, dict) for span in spans):
                        raise CampaignError(f"{stable_id}: error_spans must be a list of objects")
                    severities = [span.get("severity") for span in spans]
                    invalid = [value for value in severities if value not in {"minor", "major"}]
                    if invalid:
                        raise CampaignError(f"{stable_id}: invalid severity values {invalid}")
                    severity = "major" if "major" in severities else "minor" if "minor" in severities else None
                    duplicate_key = (annotator_id, stable_id)
                    previous = judgment_keys.setdefault(duplicate_key, [])
                    if previous and repeat_policy == "error":
                        raise CampaignError(
                            f"repeated judgment for {annotator_id}/{stable_id}; "
                            "rerun with --repeat-policy preserve only after explicit review"
                        )
                    previous.append(submission_id)
                    status = context["screening_status"]
                    eligibility_override = context["eligible_for_gold_override"]
                    records.append(
                        {
                            "pool_id": assignment.get("pool_id", manifest.get("pool_id")),
                            "campaign_id": campaign_id,
                            "user_id": user_id,
                            "annotator_id": annotator_id,
                            "assignment_id": assignment_id,
                            "screening_status": status,
                            "eligible_for_gold": (
                                eligibility_override
                                if eligibility_override is not None
                                else status == "accepted"
                                if status is not None
                                else None
                            ),
                            "document_id": str(document_id),
                            "passage_id": str(passage_id),
                            "sentence_id": str(sentence_id),
                            "candidate_id": stable_id,
                            "anonymous_system_label": label,
                            "system_id": system_id,
                            "score": judgment.get("score"),
                            "error_spans": spans,
                            "severity": severity,
                            "comment": judgment.get("textfield"),
                            "timing": _timing(actions, segment_index, label),
                            "provenance": {
                                "raw_export": str(raw_path),
                                "raw_line_number": line_number,
                                "raw_line_sha256": raw_sha,
                                "submission_id": submission_id,
                                "pearmut_user_id": user_id,
                                "pearmut_item_i": row.get("item_i"),
                                "presentation_label": label,
                                "document_comment": row.get("comment"),
                                "actions": actions,
                                "repeat_resolution_required": len(previous) > 1,
                            },
                        }
                    )
    repeated_keys = {key for key, submissions in judgment_keys.items() if len(submissions) > 1}
    for record in records:
        if (record["annotator_id"], record["candidate_id"]) in repeated_keys:
            record["provenance"]["repeat_resolution_required"] = True
    return {
        "schema_version": "1.0",
        "record_unit": "candidate_id × annotator_id × submission",
        "individual_judgments_preserved": True,
        "repeat_policy": repeat_policy,
        "pool_id": manifest.get("pool_id"),
        "raw_export_count": len(exports),
        "submission_count": len(submission_ids),
        "record_count": len(records),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    try:
        overrides: dict[str, list[Path]] = {}
        for value in args.raw_export:
            if "=" not in value:
                raise CampaignError("--raw-export must be ASSIGNMENT_ID=PATH")
            assignment_id, raw_path = value.split("=", 1)
            overrides.setdefault(
                require_nonempty_string(assignment_id, "raw export assignment_id"), []
            ).append(Path(require_nonempty_string(raw_path, "raw export path")))
        payload = normalize(
            read_json(args.candidate_dataset),
            read_json(args.assignment_manifest),
            args.assignment_manifest.resolve(),
            overrides,
            args.repeat_policy,
            set(args.campaign_id) or None,
        )
        atomic_write_json(args.output, payload, args.overwrite)
        print(
            f"Normalized {payload['record_count']} individual candidate judgments "
            f"from {payload['submission_count']} submission(s)"
        )
    except CampaignError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
