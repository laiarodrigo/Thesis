#!/usr/bin/env python3
"""Generate assignment manifests, private mappings, and Pearmut campaigns."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any

from assignment_allocation import allocate_two_annotators
from build_pearmut_campaign import build_campaign_payload
from campaign_common import (
    CampaignError,
    atomic_write_json,
    canonical_candidate_index,
    load_config,
    read_json,
    require_nonempty_string,
    resolve_path,
    spreadsheet_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--campaign-id",
        action="append",
        default=[],
        help="build only this campaign from a collection config; may be repeated",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_mapping(
    real_system_ids: list[str],
    campaign_id: str,
    assignment_id: str,
    annotator_id: str,
    seed: int,
    prefix: str,
) -> dict[str, Any]:
    shuffled = list(real_system_ids)
    random.Random(seed).shuffle(shuffled)
    return {
        "schema_version": "2.0",
        "researcher_only": True,
        "campaign_id": campaign_id,
        "assignment_id": assignment_id,
        "annotator_id": annotator_id,
        "seed": seed,
        "anonymous_label_prefix": prefix,
        "system_count": len(shuffled),
        "mappings": [
            {
                "real_system_id": system_id,
                "anonymous_system_id": f"{prefix}-{spreadsheet_label(index)}",
            }
            for index, system_id in enumerate(shuffled)
        ],
    }


def _available_passages(dataset: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    documents = dataset.get("documents")
    if not isinstance(documents, list):
        raise CampaignError("candidate dataset documents must be a list")
    for document in documents:
        document_id = require_nonempty_string(document.get("document_id"), "document_id")
        passages = document.get("passages")
        if not isinstance(passages, list):
            raise CampaignError(f"{document_id}.passages must be a list")
        for passage in passages:
            passage_id = require_nonempty_string(passage.get("passage_id"), "passage_id")
            if passage_id in result:
                raise CampaignError(f"passage_id is not globally unique: {passage_id}")
            result[passage_id] = document_id
    return result


def _assignment_seed(seed: int, assignment_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{assignment_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _configured_assignments(
    config: dict[str, Any], path_root: Path, strategy: dict[str, Any]
) -> list[dict[str, Any]]:
    assignments = config.get("assignments")
    if isinstance(assignments, list) and assignments:
        return assignments
    if strategy.get("allocation") != "balanced_exact_coverage":
        raise CampaignError(
            "config.assignments must be populated, or assignment_strategy.allocation "
            "must be balanced_exact_coverage"
        )
    coverage = strategy.get("required_annotators_per_passage")
    if coverage != 2:
        raise CampaignError(
            "balanced_exact_coverage currently requires exactly two annotators per passage"
        )
    if strategy.get("link_granularity") != "one_passage_per_link":
        raise CampaignError(
            "automatic two-annotator allocation requires one_passage_per_link"
        )
    passages = config.get("selected_passage_ids")
    annotators = config.get("annotators")
    seed = strategy.get("seed")
    if not isinstance(passages, list) or not isinstance(annotators, list):
        raise CampaignError("selected_passage_ids and annotators must be lists")
    allocation = allocate_two_annotators(passages, annotators, seed)
    paths = config.get("paths", {})
    campaign_dir = resolve_path(
        path_root, paths.get("campaign_output_dir"), "paths.campaign_output_dir"
    )
    mapping_dir = resolve_path(
        path_root, paths.get("private_mapping_dir"), "paths.private_mapping_dir"
    )
    project_id = require_nonempty_string(
        config.get("campaign_project_id"), "campaign_project_id"
    )
    generated: list[dict[str, Any]] = []
    for passage_index, row in enumerate(allocation, 1):
        for slot, annotator_id in enumerate(row["annotator_ids"], 1):
            assignment_id = f"{project_id}__p{passage_index:03d}__a{slot}"
            generated.append(
                {
                    "annotator_id": annotator_id,
                    "assignment_id": assignment_id,
                    "campaign_id": assignment_id,
                    "passage_ids": [row["passage_id"]],
                    "anonymization_seed": _assignment_seed(seed, assignment_id),
                    "campaign_output": str(campaign_dir / f"{assignment_id}.json"),
                    "mapping_output": str(mapping_dir / f"{assignment_id}.json"),
                    "raw_exports": [],
                }
            )
    return generated


def _build_legacy_outputs(config_path: Path) -> tuple[list[tuple[Path, dict[str, Any]]], Path, dict[str, Any]]:
    config, path_root = load_config(config_path)
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise CampaignError("config.paths must be an object")
    dataset_path = resolve_path(path_root, paths.get("candidate_dataset"), "paths.candidate_dataset")
    instructions_path = resolve_path(path_root, paths.get("pearmut_instructions"), "paths.pearmut_instructions")
    manifest_path = resolve_path(path_root, paths.get("assignment_manifest"), "paths.assignment_manifest")
    dataset = read_json(dataset_path)
    canonical_candidate_index(dataset)
    available_passages = _available_passages(dataset)
    try:
        instructions = instructions_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CampaignError(f"could not read instructions {instructions_path}: {exc}") from exc
    require_nonempty_string(instructions, "Pearmut instructions")

    systems = dataset.get("systems")
    real_system_ids = [
        require_nonempty_string(system.get("system_id"), "dataset system_id")
        for system in systems
    ]
    configured_systems = config.get("candidate_systems")
    if configured_systems is not None and configured_systems != real_system_ids:
        raise CampaignError("candidate_systems must exactly match canonical dataset order")

    settings = config.get("anonymization", {})
    if not isinstance(settings, dict):
        raise CampaignError("config.anonymization must be an object")
    prefix = require_nonempty_string(settings.get("label_prefix", "System"), "anonymization.label_prefix")
    pearmut = config.get("pearmut")
    if not isinstance(pearmut, dict):
        raise CampaignError("config.pearmut must be an object")
    strategy = config.get("assignment_strategy")
    if not isinstance(strategy, dict):
        raise CampaignError("config.assignment_strategy must be an object")
    granularity = strategy.get("link_granularity")
    allowed = {"one_passage_per_link", "multiple_passages_per_link"}
    if granularity not in allowed:
        raise CampaignError(f"assignment_strategy.link_granularity must be one of {sorted(allowed)}")

    assignments = _configured_assignments(config, path_root, strategy)
    outputs: list[tuple[Path, dict[str, Any]]] = []
    manifest_rows: list[dict[str, Any]] = []
    assignment_ids: set[str] = set()
    campaign_ids: set[str] = set()
    campaign_paths: set[Path] = set()
    mapping_paths: set[Path] = set()
    allocations: dict[str, list[str]] = {}
    total_judgments = 0
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            raise CampaignError(f"assignments[{index}] must be an object")
        annotator_id = require_nonempty_string(assignment.get("annotator_id"), f"assignments[{index}].annotator_id")
        assignment_id = require_nonempty_string(assignment.get("assignment_id"), f"assignments[{index}].assignment_id")
        campaign_id = require_nonempty_string(assignment.get("campaign_id"), f"assignments[{index}].campaign_id")
        if assignment_id in assignment_ids:
            raise CampaignError(f"duplicate assignment_id: {assignment_id}")
        if campaign_id in campaign_ids:
            raise CampaignError(f"duplicate campaign_id: {campaign_id}")
        assignment_ids.add(assignment_id)
        campaign_ids.add(campaign_id)
        passage_ids = assignment.get("passage_ids")
        if not isinstance(passage_ids, list) or not passage_ids or not all(
            isinstance(value, str) and value for value in passage_ids
        ):
            raise CampaignError(f"assignments[{index}].passage_ids must be a non-empty string list")
        if len(passage_ids) != len(set(passage_ids)):
            raise CampaignError(f"{assignment_id}: passage_ids contain duplicates")
        unknown = set(passage_ids) - set(available_passages)
        if unknown:
            raise CampaignError(f"{assignment_id}: unknown passage_ids {sorted(unknown)}")
        if granularity == "one_passage_per_link" and len(passage_ids) != 1:
            raise CampaignError(f"{assignment_id}: one_passage_per_link requires exactly one passage")
        seed = assignment.get("anonymization_seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise CampaignError(f"{assignment_id}.anonymization_seed must be an integer")
        campaign_output = resolve_path(path_root, assignment.get("campaign_output"), f"{assignment_id}.campaign_output")
        mapping_output = resolve_path(path_root, assignment.get("mapping_output"), f"{assignment_id}.mapping_output")
        if campaign_output in campaign_paths or mapping_output in mapping_paths:
            raise CampaignError(f"{assignment_id}: output path is reused")
        campaign_paths.add(campaign_output)
        mapping_paths.add(mapping_output)

        mapping = make_mapping(
            real_system_ids, campaign_id, assignment_id, annotator_id, seed, prefix
        )
        campaign = build_campaign_payload(
            dataset, mapping, campaign_id, pearmut, instructions, passage_ids
        )
        campaign["metadata"].update(
            {"assignment_id": assignment_id, "annotator_id": annotator_id}
        )
        outputs.extend([(mapping_output, mapping), (campaign_output, campaign)])
        allocations[assignment_id] = list(passage_ids)
        total_judgments += campaign["metadata"]["candidate_judgment_count"]
        raw_exports = assignment.get("raw_exports", [])
        if not isinstance(raw_exports, list) or not all(
            isinstance(value, str) and value for value in raw_exports
        ):
            raise CampaignError(f"{assignment_id}.raw_exports must be a list of paths")
        manifest_rows.append(
            {
                "annotator_id": annotator_id,
                "assignment_id": assignment_id,
                "campaign_id": campaign_id,
                "passage_ids": list(passage_ids),
                "anonymization_seed": seed,
                "mapping_path": str(mapping_output),
                "campaign_output": str(campaign_output),
                "expected_candidate_judgments": campaign["metadata"]["candidate_judgment_count"],
                "raw_exports": list(raw_exports),
            }
        )

    expected = config.get("expected_counts", {})
    if not isinstance(expected, dict):
        raise CampaignError("expected_counts must be an object")
    checks = {
        "passages": len({passage for values in allocations.values() for passage in values}),
        "assignments": len(manifest_rows),
        "candidate_judgments": total_judgments,
        "individual_annotations": total_judgments,
    }
    for key, observed in checks.items():
        configured = expected.get(key)
        if configured is not None and configured != observed:
            raise CampaignError(f"expected_counts.{key}={configured}, observed {observed}")

    manifest = {
        "schema_version": "1.0",
        "campaign_project_id": require_nonempty_string(
            config.get("campaign_project_id"), "campaign_project_id"
        ),
        "candidate_dataset": str(dataset_path),
        "link_granularity": granularity,
        "system_count": len(real_system_ids),
        "passage_count": checks["passages"],
        "assignment_count": len(manifest_rows),
        "expected_candidate_judgments": total_judgments,
        "expected_individual_annotations": total_judgments,
        "passage_allocations": allocations,
        "assignments": manifest_rows,
    }
    return outputs, manifest_path, manifest


def _collection_assignments(
    config: dict[str, Any],
    group: dict[str, Any],
    path_root: Path,
) -> list[dict[str, Any]]:
    """Return exactly one ordered task assignment for every shared-campaign user."""
    campaign_id = require_nonempty_string(group.get("campaign_id"), "campaigns[].campaign_id")
    project_id = require_nonempty_string(
        group.get("campaign_project_id", config.get("campaign_project_id")),
        f"{campaign_id}.campaign_project_id",
    )
    users = group.get("users")
    if not isinstance(users, list) or not users or not all(
        isinstance(user, str) and user for user in users
    ):
        raise CampaignError(f"{campaign_id}.users must be a non-empty string list")
    if len(users) != len(set(users)):
        raise CampaignError(f"{campaign_id}.users contains duplicate user_id values")
    strategy = group.get("assignment_strategy", {})
    if not isinstance(strategy, dict):
        raise CampaignError(f"{campaign_id}.assignment_strategy must be an object")
    seed = strategy.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CampaignError(f"{campaign_id}.assignment_strategy.seed must be an integer")
    assignments = group.get("assignments")
    if not isinstance(assignments, list):
        raise CampaignError(f"{campaign_id}.assignments must be a list")

    if not assignments:
        if strategy.get("allocation") != "balanced_exact_coverage":
            raise CampaignError(f"{campaign_id} requires assignments or balanced_exact_coverage")
        if strategy.get("required_annotators_per_passage") != 2:
            raise CampaignError(f"{campaign_id} balanced allocation requires two annotators per passage")
        passage_ids = group.get("selected_passage_ids")
        if not isinstance(passage_ids, list):
            raise CampaignError(f"{campaign_id}.selected_passage_ids must be a list")
        allocation = allocate_two_annotators(passage_ids, users, seed)
        passage_by_user: dict[str, str] = {}
        for allocation_row in allocation:
            for user_id in allocation_row["annotator_ids"]:
                if user_id in passage_by_user:
                    raise CampaignError(
                        f"{campaign_id}: {user_id} received more than one one-passage task"
                    )
                passage_by_user[user_id] = str(allocation_row["passage_id"])
        if set(passage_by_user) != set(users):
            raise CampaignError(f"{campaign_id}: balanced allocation did not assign every user")
        assignments = [
            {
                "user_id": user_id,
                "annotator_id": user_id,
                "assignment_id": f"{project_id}__{user_id}",
                "passage_ids": [passage_by_user[user_id]],
            }
            for user_id in users
        ]

    mapping_dir = resolve_path(
        path_root,
        config.get("paths", {}).get("private_mapping_dir"),
        "paths.private_mapping_dir",
    )
    by_user: dict[str, dict[str, Any]] = {}
    assignment_ids: set[str] = set()
    for index, raw in enumerate(assignments):
        if not isinstance(raw, dict):
            raise CampaignError(f"{campaign_id}.assignments[{index}] must be an object")
        user_id = require_nonempty_string(
            raw.get("user_id", raw.get("annotator_id")),
            f"{campaign_id}.assignments[{index}].user_id",
        )
        annotator_id = require_nonempty_string(
            raw.get("annotator_id", user_id),
            f"{campaign_id}.assignments[{index}].annotator_id",
        )
        if annotator_id != user_id:
            raise CampaignError(f"{campaign_id}: annotator_id must equal stable Pearmut user_id")
        if user_id not in users:
            raise CampaignError(f"{campaign_id}: undeclared assignment user_id {user_id}")
        if user_id in by_user:
            raise CampaignError(f"{campaign_id}: duplicate task for user_id {user_id}")
        assignment_id = require_nonempty_string(
            raw.get("assignment_id"), f"{campaign_id}.assignments[{index}].assignment_id"
        )
        if assignment_id in assignment_ids:
            raise CampaignError(f"duplicate assignment_id: {assignment_id}")
        assignment_ids.add(assignment_id)
        passage_ids = raw.get("passage_ids")
        if not isinstance(passage_ids, list) or not passage_ids or not all(
            isinstance(value, str) and value for value in passage_ids
        ):
            raise CampaignError(f"{assignment_id}.passage_ids must be a non-empty string list")
        if len(passage_ids) != len(set(passage_ids)):
            raise CampaignError(f"{assignment_id}.passage_ids contains duplicates")
        if strategy.get("link_granularity") == "one_passage_per_link" and len(passage_ids) != 1:
            raise CampaignError(f"{assignment_id}: one_passage_per_link requires one passage")
        assignment_seed = raw.get("anonymization_seed", _assignment_seed(seed, assignment_id))
        if not isinstance(assignment_seed, int) or isinstance(assignment_seed, bool):
            raise CampaignError(f"{assignment_id}.anonymization_seed must be an integer")
        mapping_output = raw.get("mapping_output", str(mapping_dir / f"{assignment_id}.json"))
        raw_exports = raw.get("raw_exports", group.get("raw_exports", []))
        if not isinstance(raw_exports, list) or not all(
            isinstance(value, str) and value for value in raw_exports
        ):
            raise CampaignError(f"{assignment_id}.raw_exports must be a path list")
        by_user[user_id] = dict(raw) | {
            "campaign_id": campaign_id,
            "user_id": user_id,
            "annotator_id": user_id,
            "assignment_id": assignment_id,
            "passage_ids": list(passage_ids),
            "anonymization_seed": assignment_seed,
            "mapping_output": mapping_output,
            "raw_exports": list(raw_exports),
        }
    if set(by_user) != set(users):
        raise CampaignError(f"{campaign_id}: exactly one task is required for every user")
    return [by_user[user_id] for user_id in users]


def _accepted_coverage(
    assignments: list[dict[str, Any]], statuses: dict[str, str]
) -> list[dict[str, Any]]:
    accepted: dict[str, set[str]] = {}
    provisional: dict[str, set[str]] = {}
    passages = {
        passage_id
        for assignment in assignments
        if assignment.get("collection_role") == "test_provisional"
        for passage_id in assignment["passage_ids"]
    }
    for assignment in assignments:
        if assignment.get("collection_role") != "test_provisional":
            continue
        for passage_id in assignment["passage_ids"]:
            provisional.setdefault(passage_id, set()).add(assignment["user_id"])
        if statuses.get(assignment["user_id"]) != "accepted":
            continue
        for passage_id in assignment["passage_ids"]:
            accepted.setdefault(passage_id, set()).add(assignment["user_id"])
    result = []
    for passage_id in sorted(passages):
        accepted_users = accepted.get(passage_id, set())
        pending_users = sorted(
            user_id
            for user_id in provisional.get(passage_id, set())
            if statuses.get(user_id) == "pending"
        )
        replacements = None if pending_users else max(0, 2 - len(accepted_users))
        result.append(
            {
                "passage_id": passage_id,
                "accepted_user_ids": sorted(accepted_users),
                "accepted_annotation_count": len(accepted_users),
                "pending_user_ids": pending_users,
                "screening_complete": not pending_users,
                "required_accepted_annotation_count": 2,
                "replacement_assignments_needed": replacements,
                "coverage_status": (
                    "pending_screening"
                    if pending_users
                    else "complete"
                    if replacements == 0
                    else "needs_replacement"
                ),
            }
        )
    return result


def _build_collection_outputs(
    config_path: Path, selected_campaign_ids: set[str] | None
) -> tuple[list[tuple[Path, dict[str, Any]]], Path, dict[str, Any]]:
    config, path_root = load_config(config_path)
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise CampaignError("config.paths must be an object")
    instructions_path = resolve_path(
        path_root, paths.get("pearmut_instructions"), "paths.pearmut_instructions"
    )
    manifest_path = resolve_path(
        path_root, paths.get("assignment_manifest"), "paths.assignment_manifest"
    )
    try:
        instructions = instructions_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CampaignError(f"could not read instructions {instructions_path}: {exc}") from exc
    require_nonempty_string(instructions, "Pearmut instructions")
    pearmut = config.get("pearmut")
    if not isinstance(pearmut, dict) or pearmut.get("assignment", "task-based") != "task-based":
        raise CampaignError("collection campaigns require task-based Pearmut assignment")
    settings = config.get("anonymization", {})
    if not isinstance(settings, dict):
        raise CampaignError("config.anonymization must be an object")
    prefix = require_nonempty_string(
        settings.get("label_prefix", "System"), "anonymization.label_prefix"
    )
    pool_id = require_nonempty_string(config.get("pool_id"), "pool_id")
    groups = config.get("campaigns")
    if not isinstance(groups, list) or not groups or not all(
        isinstance(group, dict) for group in groups
    ):
        raise CampaignError("collection config campaigns must be a non-empty object list")
    configured_ids = [
        require_nonempty_string(group.get("campaign_id"), "campaigns[].campaign_id")
        for group in groups
    ]
    if len(configured_ids) != len(set(configured_ids)):
        raise CampaignError("campaign IDs must be unique")
    if selected_campaign_ids:
        unknown = selected_campaign_ids - set(configured_ids)
        if unknown:
            raise CampaignError(f"unknown selected campaign IDs: {sorted(unknown)}")

    annotators = config.get("annotators")
    if not isinstance(annotators, list) or not annotators:
        raise CampaignError("config.annotators must be a non-empty list")
    statuses: dict[str, str] = {}
    annotator_rows: list[dict[str, str]] = []
    for row in annotators:
        if not isinstance(row, dict):
            raise CampaignError("annotators entries must be objects")
        user_id = require_nonempty_string(row.get("user_id"), "annotators[].user_id")
        status = require_nonempty_string(row.get("screening_status", "pending"), "screening_status")
        if status not in {"pending", "accepted", "rejected"}:
            raise CampaignError(f"invalid screening_status for {user_id}: {status}")
        if user_id in statuses:
            raise CampaignError(f"duplicate annotator user_id: {user_id}")
        statuses[user_id] = status
        annotator_rows.append({"user_id": user_id, "screening_status": status})

    outputs: list[tuple[Path, dict[str, Any]]] = []
    manifest_rows: list[dict[str, Any]] = []
    campaign_rows: list[dict[str, Any]] = []
    output_paths: set[Path] = set()
    mapping_paths: set[Path] = set()
    assignment_ids: set[str] = set()
    for group in groups:
        campaign_id = require_nonempty_string(group.get("campaign_id"), "campaigns[].campaign_id")
        if selected_campaign_ids and campaign_id not in selected_campaign_ids:
            role = require_nonempty_string(group.get("role"), f"{campaign_id}.role")
            real_system_ids = group.get("candidate_systems")
            if not isinstance(real_system_ids, list) or not real_system_ids or not all(
                isinstance(value, str) and value for value in real_system_ids
            ):
                raise CampaignError(
                    f"{campaign_id}.candidate_systems is required when planning an unbuilt campaign"
                )
            assignments = _collection_assignments(config, group, path_root)
            campaign_output = resolve_path(
                path_root, group.get("campaign_output"), f"{campaign_id}.campaign_output"
            )
            dataset_path = resolve_path(
                path_root, group.get("candidate_dataset"), f"{campaign_id}.candidate_dataset"
            )
            expected_per_task = group.get("expected_candidate_judgments_per_task")
            if not isinstance(expected_per_task, int) or expected_per_task <= 0:
                raise CampaignError(
                    f"{campaign_id}.expected_candidate_judgments_per_task must be positive"
                )
            for assignment in assignments:
                assignment_id = assignment["assignment_id"]
                if assignment_id in assignment_ids:
                    raise CampaignError(f"duplicate assignment_id: {assignment_id}")
                assignment_ids.add(assignment_id)
                mapping_output = resolve_path(
                    path_root, assignment["mapping_output"], f"{assignment_id}.mapping_output"
                )
                if mapping_output in mapping_paths:
                    raise CampaignError(f"mapping output path reused: {mapping_output}")
                mapping_paths.add(mapping_output)
                outputs.append(
                    (
                        mapping_output,
                        make_mapping(
                            real_system_ids,
                            campaign_id,
                            assignment_id,
                            assignment["user_id"],
                            assignment["anonymization_seed"],
                            prefix,
                        ),
                    )
                )
                manifest_rows.append(
                    {
                        "pool_id": pool_id,
                        "collection_role": role,
                        "build_status": "pending_candidates",
                        "campaign_id": campaign_id,
                        "user_id": assignment["user_id"],
                        "annotator_id": assignment["user_id"],
                        "assignment_id": assignment_id,
                        "passage_ids": list(assignment["passage_ids"]),
                        "anonymization_seed": assignment["anonymization_seed"],
                        "candidate_dataset": str(dataset_path),
                        "mapping_path": str(mapping_output),
                        "campaign_output": str(campaign_output),
                        "expected_candidate_judgments": expected_per_task,
                        "raw_exports": list(assignment["raw_exports"]),
                    }
                )
            campaign_rows.append(
                {
                    "campaign_id": campaign_id,
                    "role": role,
                    "build_status": "pending_candidates",
                    "candidate_dataset": str(dataset_path),
                    "campaign_output": str(campaign_output),
                    "users": list(group["users"]),
                    "task_count": len(group["users"]),
                }
            )
            continue
        role = require_nonempty_string(group.get("role"), f"{campaign_id}.role")
        if role not in {
            "pilot_screening",
            "test_provisional",
            "external_web_smoke_test",
        }:
            raise CampaignError(f"{campaign_id}.role is unsupported: {role}")
        campaign_policy: dict[str, Any] = {}
        if role == "external_web_smoke_test":
            expected_policy = {
                "purpose": "external_web_smoke_test",
                "eligible_for_gold": False,
                "disposable": True,
            }
            for key, expected_value in expected_policy.items():
                if group.get(key) != expected_value:
                    raise CampaignError(
                        f"{campaign_id}.{key} must be {expected_value!r} for a smoke campaign"
                    )
            campaign_policy = expected_policy
        dataset_path = resolve_path(
            path_root, group.get("candidate_dataset"), f"{campaign_id}.candidate_dataset"
        )
        dataset = read_json(dataset_path)
        canonical_candidate_index(dataset)
        available_passages = _available_passages(dataset)
        systems = dataset.get("systems")
        if not isinstance(systems, list) or not systems:
            raise CampaignError(f"{campaign_id}: candidate dataset has no systems")
        real_system_ids = [
            require_nonempty_string(system.get("system_id"), "dataset system_id")
            for system in systems
        ]
        assignments = _collection_assignments(config, group, path_root)
        users = list(group["users"])
        if any(user_id not in statuses for user_id in users):
            raise CampaignError(f"{campaign_id}: campaign user absent from private registry")
        campaign_output = resolve_path(
            path_root, group.get("campaign_output"), f"{campaign_id}.campaign_output"
        )
        if campaign_output in output_paths:
            raise CampaignError(f"campaign output path reused: {campaign_output}")
        output_paths.add(campaign_output)
        tasks_by_user: dict[str, list[Any]] = {}
        campaign_base: dict[str, Any] | None = None
        assigned_judgments = 0
        for assignment in assignments:
            assignment_id = assignment["assignment_id"]
            if assignment_id in assignment_ids:
                raise CampaignError(f"duplicate assignment_id: {assignment_id}")
            assignment_ids.add(assignment_id)
            unknown = set(assignment["passage_ids"]) - set(available_passages)
            if unknown:
                raise CampaignError(f"{assignment_id}: unknown passage_ids {sorted(unknown)}")
            mapping_output = resolve_path(
                path_root, assignment["mapping_output"], f"{assignment_id}.mapping_output"
            )
            if mapping_output in mapping_paths:
                raise CampaignError(f"mapping output path reused: {mapping_output}")
            mapping_paths.add(mapping_output)
            mapping = make_mapping(
                real_system_ids,
                campaign_id,
                assignment_id,
                assignment["user_id"],
                assignment["anonymization_seed"],
                prefix,
            )
            task_campaign = build_campaign_payload(
                dataset, mapping, campaign_id, pearmut, instructions, assignment["passage_ids"]
            )
            if campaign_base is None:
                campaign_base = task_campaign
            tasks_by_user[assignment["user_id"]] = task_campaign["data"][0]
            assigned_judgments += task_campaign["metadata"]["candidate_judgment_count"]
            outputs.append((mapping_output, mapping))
            manifest_rows.append(
                {
                    "pool_id": pool_id,
                    "collection_role": role,
                    "build_status": "built",
                    "campaign_id": campaign_id,
                    "user_id": assignment["user_id"],
                    "annotator_id": assignment["user_id"],
                    "assignment_id": assignment_id,
                    "passage_ids": list(assignment["passage_ids"]),
                    "anonymization_seed": assignment["anonymization_seed"],
                    "candidate_dataset": str(dataset_path),
                    "mapping_path": str(mapping_output),
                    "campaign_output": str(campaign_output),
                    "expected_candidate_judgments": task_campaign["metadata"]["candidate_judgment_count"],
                    "raw_exports": list(assignment["raw_exports"]),
                }
                | campaign_policy
            )
        if campaign_base is None or list(tasks_by_user) != users:
            raise CampaignError(f"{campaign_id}: task order does not match users order")
        campaign_base["info"]["users"] = users
        campaign_base["data"] = [tasks_by_user[user_id] for user_id in users]
        campaign_base["metadata"] = {
            "pool_id": pool_id,
            "collection_role": role,
            "task_count": len(users),
            "user_count": len(users),
            "passage_ids": sorted(
                {passage_id for assignment in assignments for passage_id in assignment["passage_ids"]}
            ),
            "canonical_source_item_count": dataset.get("source_item_count"),
            "canonical_candidate_judgment_count": dataset.get("candidate_judgment_count"),
            "assigned_candidate_judgment_count": assigned_judgments,
        } | campaign_policy
        if len(campaign_base["data"]) != len(campaign_base["info"]["users"]):
            raise CampaignError(f"{campaign_id}: len(tasks) must equal len(info.users)")
        outputs.append((campaign_output, campaign_base))
        campaign_rows.append(
            {
                "campaign_id": campaign_id,
                "role": role,
                "build_status": "built",
                "candidate_dataset": str(dataset_path),
                "campaign_output": str(campaign_output),
                "users": users,
                "task_count": len(users),
            }
            | campaign_policy
        )

    manifest = {
        "schema_version": "2.0",
        "researcher_only": True,
        "pool_id": pool_id,
        "screening_status_values": ["pending", "accepted", "rejected"],
        "raw_annotations_immutable": True,
        "gold_eligibility_rule": config.get(
            "gold_eligibility_rule",
            "TEST records are eligible only when screening_status is accepted",
        ),
        "required_accepted_annotators_per_test_passage": config.get(
            "required_accepted_annotators_per_test_passage", 2
        ),
        "annotators": annotator_rows,
        "campaigns": campaign_rows,
        "assignments": manifest_rows,
        "accepted_coverage": _accepted_coverage(manifest_rows, statuses),
    }
    return outputs, manifest_path, manifest


def build_outputs(
    config_path: Path, selected_campaign_ids: set[str] | None = None
) -> tuple[list[tuple[Path, dict[str, Any]]], Path, dict[str, Any]]:
    config, _ = load_config(config_path)
    if "campaigns" in config:
        return _build_collection_outputs(config_path, selected_campaign_ids)
    if selected_campaign_ids:
        raise CampaignError("--campaign-id requires a collection config")
    return _build_legacy_outputs(config_path)


def main() -> None:
    args = parse_args()
    try:
        outputs, manifest_path, manifest = build_outputs(
            args.config, set(args.campaign_id) or None
        )
        all_paths = [path for path, _ in outputs] + [manifest_path]
        if not args.overwrite:
            existing = [str(path) for path in all_paths if path.exists()]
            if existing:
                raise CampaignError(f"refusing to overwrite existing outputs: {existing}")
        for path, payload in outputs:
            atomic_write_json(path, payload, args.overwrite)
        atomic_write_json(manifest_path, manifest, args.overwrite)
        if manifest.get("schema_version") == "2.0":
            built_count = sum(
                row.get("build_status") == "built" for row in manifest["campaigns"]
            )
            print(
                f"Planned {len(manifest['assignments'])} assignment(s) in "
                f"{len(manifest['campaigns'])} shared campaign(s); built {built_count} campaign(s)"
            )
        else:
            print(
                f"Built {manifest['assignment_count']} assignment campaign(s), "
                f"{manifest['expected_candidate_judgments']} assigned candidate judgments"
            )
        print(f"Assignment manifest: {manifest_path}")
    except CampaignError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
