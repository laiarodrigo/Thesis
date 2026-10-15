#!/usr/bin/env python3
"""Shared, model-independent utilities for CMU annotation campaigns."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


class CampaignError(ValueError):
    """Raised when campaign data fails a reproducibility or alignment check."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"Expected a JSON object in {path}")
    return value


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.resolve()
    config = read_json(config_path)
    configured_root = config.get("path_root", ".")
    if not isinstance(configured_root, str):
        raise CampaignError("config.path_root must be a string")
    return config, (config_path.parent / configured_root).resolve()


def resolve_path(path_root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CampaignError(f"{field} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else path_root / path


def atomic_write_json(path: Path, payload: Any, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise CampaignError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, content: str, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise CampaignError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignError(f"{field} must be a non-empty string")
    return value


def item_key(document_id: str, passage_id: str, sentence_id: str) -> str:
    """Return an unambiguous stable ID without relying on row position."""
    return json.dumps([str(document_id), str(passage_id), str(sentence_id)], ensure_ascii=False, separators=(",", ":"))


def candidate_id(passage_id: str, sentence_id: str, system_id: str) -> str:
    """Return the stable, presentation-label-independent candidate identity."""
    parts = [
        require_nonempty_string(str(passage_id), "passage_id"),
        require_nonempty_string(str(sentence_id), "sentence_id"),
        require_nonempty_string(str(system_id), "system_id"),
    ]
    return "::".join(quote(part, safe="-._~") for part in parts)


def canonical_candidate_index(
    dataset: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Index candidates and source items while validating permanent IDs.

    Older frozen datasets without an explicit ``candidate_ids`` mapping remain
    readable: the deterministic ID is derived and validated in memory.
    """
    systems = dataset.get("systems")
    if not isinstance(systems, list) or not systems:
        raise CampaignError("candidate dataset has no systems")
    system_ids = [
        require_nonempty_string(system.get("system_id"), f"systems[{index}].system_id")
        for index, system in enumerate(systems)
        if isinstance(system, dict)
    ]
    if len(system_ids) != len(systems):
        raise CampaignError("each candidate dataset system must be an object")
    if len(system_ids) != len(set(system_ids)):
        raise CampaignError("candidate dataset contains duplicate system IDs")

    candidates_by_id: dict[str, dict[str, Any]] = {}
    items_by_key: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(flatten_records(dataset)):
        document_id = require_nonempty_string(row.get("document_id"), f"item {row_index}.document_id")
        passage_id = require_nonempty_string(row.get("passage_id"), f"item {row_index}.passage_id")
        sentence_id = require_nonempty_string(str(row.get("sentence_id", "")), f"item {row_index}.sentence_id")
        key = item_key(document_id, passage_id, sentence_id)
        if key in items_by_key:
            raise CampaignError(f"duplicate candidate item: {key}")
        candidates = row.get("candidates")
        if not isinstance(candidates, dict) or set(candidates) != set(system_ids):
            raise CampaignError(f"{key}: candidates do not match dataset systems")
        declared_ids = row.get("candidate_ids")
        if declared_ids is not None and (
            not isinstance(declared_ids, dict) or set(declared_ids) != set(system_ids)
        ):
            raise CampaignError(f"{key}: candidate_ids do not match dataset systems")
        item_record = {
            "item_key": key,
            "document_id": document_id,
            "passage_id": passage_id,
            "sentence_id": sentence_id,
            "position": row.get("position"),
            "source_bp": row.get("source_bp"),
        }
        items_by_key[key] = item_record
        for system_id in system_ids:
            stable_id = candidate_id(passage_id, sentence_id, system_id)
            if declared_ids is not None and declared_ids.get(system_id) != stable_id:
                raise CampaignError(
                    f"{key}/{system_id}: declared candidate_id is not deterministic"
                )
            if stable_id in candidates_by_id:
                previous = candidates_by_id[stable_id]
                raise CampaignError(
                    "candidate_id collision: "
                    f"{stable_id} represents both {previous['item_key']} and {key}"
                )
            candidates_by_id[stable_id] = {
                **item_record,
                "candidate_id": stable_id,
                "system_id": system_id,
                "candidate_text": require_nonempty_string(
                    candidates.get(system_id), f"{key}.candidates[{system_id}]"
                ),
            }
    if not candidates_by_id:
        raise CampaignError("candidate dataset contains no candidates")
    return candidates_by_id, items_by_key


def _passage_records(container: dict[str, Any], document_id: Any, inherited_passage_id: Any = None) -> Iterable[dict[str, Any]]:
    passages = container.get("passages")
    if passages is not None:
        if not isinstance(passages, list):
            raise CampaignError("passages must be a list")
        for passage in passages:
            if not isinstance(passage, dict):
                raise CampaignError("each passage must be an object")
            yield from _passage_records(passage, document_id, passage.get("passage_id", inherited_passage_id))
        return
    rows = container.get("sentences", container.get("items"))
    if not isinstance(rows, list):
        raise CampaignError("each source document/passage must contain sentences or items")
    passage_id = container.get("passage_id", inherited_passage_id)
    for row in rows:
        if not isinstance(row, dict):
            raise CampaignError("each source sentence must be an object")
        yield {**row, "document_id": row.get("document_id", document_id), "passage_id": row.get("passage_id", passage_id)}


def flatten_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten supported single- or multi-document JSON without losing IDs."""
    documents = payload.get("documents")
    if documents is None:
        return list(_passage_records(payload, payload.get("document_id"), payload.get("passage_id")))
    if not isinstance(documents, list):
        raise CampaignError("documents must be a list")
    flattened: list[dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, dict):
            raise CampaignError("each document must be an object")
        flattened.extend(_passage_records(document, document.get("document_id"), document.get("passage_id")))
    return flattened


def canonicalize_source(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = flatten_records(payload)
    if not rows:
        raise CampaignError("canonical source contains no sentence items")
    canonical: list[dict[str, Any]] = []
    seen: set[str] = set()
    order_seen: set[tuple[str, str, int]] = set()
    for index, row in enumerate(rows):
        document_id = require_nonempty_string(row.get("document_id"), f"source row {index}.document_id")
        passage_id = require_nonempty_string(row.get("passage_id"), f"source row {index}.passage_id")
        sentence_id = require_nonempty_string(str(row.get("sentence_id", "")), f"source row {index}.sentence_id")
        source = require_nonempty_string(row.get("source_bp", row.get("source")), f"source row {index}.source_bp")
        position = row.get("position")
        if not isinstance(position, int) or isinstance(position, bool) or position < 1:
            raise CampaignError(f"source row {index}.position must be a positive integer")
        key = item_key(document_id, passage_id, sentence_id)
        if key in seen:
            raise CampaignError(f"duplicate source ID: {document_id}/{passage_id}/{sentence_id}")
        position_key = (document_id, passage_id, position)
        if position_key in order_seen:
            raise CampaignError(f"duplicate source position {position} in {document_id}/{passage_id}")
        seen.add(key)
        order_seen.add(position_key)
        canonical.append({"item_key": key, "document_id": document_id, "passage_id": passage_id, "sentence_id": sentence_id, "position": position, "source_bp": source})
    return canonical


def group_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group ordered items as documents/passages while preserving first-seen order."""
    documents: list[dict[str, Any]] = []
    document_lookup: dict[str, dict[str, Any]] = {}
    passage_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        document_id, passage_id = item["document_id"], item["passage_id"]
        if document_id not in document_lookup:
            document = {"document_id": document_id, "passages": []}
            document_lookup[document_id] = document
            documents.append(document)
        passage_key = (document_id, passage_id)
        if passage_key not in passage_lookup:
            passage = {"passage_id": passage_id, "items": []}
            passage_lookup[passage_key] = passage
            document_lookup[document_id]["passages"].append(passage)
        passage_lookup[passage_key]["items"].append(item)
    for document in documents:
        for passage in document["passages"]:
            positions = [item["position"] for item in passage["items"]]
            if positions != sorted(positions):
                raise CampaignError(f"source order is not ascending in {document['document_id']}/{passage['passage_id']}: {positions}")
    return documents


def index_records(payload: dict[str, Any], name: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, row in enumerate(flatten_records(payload)):
        document_id = require_nonempty_string(row.get("document_id", payload.get("document_id")), f"{name} row {index}.document_id")
        passage_id = require_nonempty_string(row.get("passage_id", payload.get("passage_id")), f"{name} row {index}.passage_id")
        sentence_id = require_nonempty_string(str(row.get("sentence_id", "")), f"{name} row {index}.sentence_id")
        key = item_key(document_id, passage_id, sentence_id)
        if key in indexed:
            raise CampaignError(f"{name}: duplicate ID {document_id}/{passage_id}/{sentence_id}")
        indexed[key] = row
        order.append(key)
    return indexed, order


def spreadsheet_label(index: int) -> str:
    """Return A..Z, AA..AZ, etc., for a zero-based index."""
    if index < 0:
        raise CampaignError("anonymous label index cannot be negative")
    value, label = index + 1, ""
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label
