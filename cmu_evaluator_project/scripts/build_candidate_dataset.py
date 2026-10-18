#!/usr/bin/env python3
"""Build a canonical, model-independent candidate dataset from configuration."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from campaign_common import (
    CampaignError,
    atomic_write_json,
    atomic_write_text,
    candidate_id,
    canonicalize_source,
    group_items,
    index_records,
    load_config,
    read_json,
    require_nonempty_string,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace generated outputs")
    return parser.parse_args()


def system_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    systems = config.get("systems")
    if not isinstance(systems, list) or not systems:
        raise CampaignError("config.systems must be a non-empty list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, system in enumerate(systems):
        if not isinstance(system, dict):
            raise CampaignError(f"config.systems[{index}] must be an object")
        system_id = require_nonempty_string(system.get("system_id"), f"config.systems[{index}].system_id")
        if system_id in seen:
            raise CampaignError(f"duplicate configured system_id: {system_id}")
        seen.add(system_id)
        kind = system.get("kind", "output_file")
        if kind not in {"copy", "output_file"}:
            raise CampaignError(f"{system_id}: unsupported system kind {kind!r}")
        result.append({**system, "system_id": system_id, "kind": kind})
    return result


def technical_flags(row: dict[str, Any], candidate: str) -> list[str]:
    supplied = row.get("technical_flags", [])
    if supplied is None:
        supplied = []
    if not isinstance(supplied, list) or not all(isinstance(flag, str) for flag in supplied):
        raise CampaignError("technical_flags must be a list of strings")
    flags = list(supplied)
    if "\ufffd" in candidate or any(0xD800 <= ord(char) <= 0xDFFF for char in candidate):
        flags.append("malformed_encoding")
    if row.get("control_label_matches_expected") is False:
        flags.append("unexpected_decoder_control_label")
    lowered = candidate.lstrip().lower()
    wrappers = ("rewritten sentence:", "european portuguese:", "português europeu:", "versão:", "here is", "here's")
    if lowered.startswith(wrappers):
        flags.append("possible_wrapper_or_explanation")
    return list(dict.fromkeys(flags))


def output_metadata(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    metadata = {key: copy.deepcopy(value) for key, value in payload.items() if key not in {"sentences", "items", "documents"}}
    metadata["source_output_file"] = str(source_path)
    return metadata


def render_review(dataset: dict[str, Any], display_names: dict[str, str]) -> str:
    lines = [
        f"# {dataset['campaign_id']} candidate review",
        "",
        "Researcher inspection only. No linguistic scores or annotations are included.",
        "Only mechanically detected technical flags are shown.",
        "",
        f"- Source items: {dataset['source_item_count']}",
        f"- Systems: {dataset['system_count']}",
        f"- Candidate judgments: {dataset['candidate_judgment_count']}",
        f"- Technical flags: {len(dataset['technical_flags'])}",
        "",
    ]
    for document in dataset["documents"]:
        lines.extend([f"## Document `{document['document_id']}`", ""])
        for passage in document["passages"]:
            lines.extend([f"### Passage `{passage['passage_id']}`", ""])
            for item in passage["items"]:
                lines.extend([
                    f"#### Sentence `{item['sentence_id']}` (position {item['position']})",
                    "",
                    f"**Source:** {item['source_bp']}",
                    "",
                ])
                for system_id, candidate in item["candidates"].items():
                    flags = item["candidate_provenance"][system_id]["technical_flags"]
                    lines.extend([
                        f"##### {display_names[system_id]}",
                        "",
                        candidate,
                        "",
                        "**Technical flags:** " + (", ".join(flags) if flags else "none"),
                        "",
                    ])
    return "\n".join(lines).rstrip() + "\n"


def apply_selection(rows: list[dict[str, Any]], selection: Any) -> list[dict[str, Any]]:
    """Apply config-only document/passage selection using stable IDs."""
    if selection is None:
        return rows
    if not isinstance(selection, list) or not selection:
        raise CampaignError("config.selection must be a non-empty list when provided")
    requested: set[tuple[str, str]] = set()
    for index, entry in enumerate(selection):
        if not isinstance(entry, dict):
            raise CampaignError(f"selection[{index}] must be an object")
        document_id = require_nonempty_string(entry.get("document_id"), f"selection[{index}].document_id")
        passage_ids = entry.get("passage_ids")
        if not isinstance(passage_ids, list) or not passage_ids:
            raise CampaignError(f"selection[{index}].passage_ids must be a non-empty list")
        for passage_id in passage_ids:
            requested.add((document_id, require_nonempty_string(passage_id, f"selection[{index}].passage_ids")))
    available = {(row["document_id"], row["passage_id"]) for row in rows}
    missing = requested - available
    if missing:
        raise CampaignError(f"configured document/passage selections not found: {sorted(missing)}")
    selected = [row for row in rows if (row["document_id"], row["passage_id"]) in requested]
    if not selected:
        raise CampaignError("document/passage selection produced no source items")
    return selected


def main() -> None:
    args = parse_args()
    try:
        config, path_root = load_config(args.config)
        campaign_id = require_nonempty_string(config.get("campaign_id"), "campaign_id")
        paths = config.get("paths")
        if not isinstance(paths, dict):
            raise CampaignError("config.paths must be an object")
        source_path = resolve_path(path_root, paths.get("source"), "paths.source")
        dataset_path = resolve_path(path_root, paths.get("candidate_dataset"), "paths.candidate_dataset")
        review_path = resolve_path(path_root, paths.get("candidate_review"), "paths.candidate_review")

        source_payload = read_json(source_path)
        source_rows = apply_selection(canonicalize_source(source_payload), config.get("selection"))
        expected_keys = [row["item_key"] for row in source_rows]
        expected_key_set = set(expected_keys)
        specs = system_specs(config)
        copy_count = sum(spec["kind"] == "copy" for spec in specs)
        include_copy = config.get("include_copy_baseline", copy_count > 0)
        if not isinstance(include_copy, bool):
            raise CampaignError("include_copy_baseline must be true or false")
        if include_copy != (copy_count == 1):
            raise CampaignError(
                "include_copy_baseline must agree with exactly one configured copy system"
            )
        display_names = {
            spec["system_id"]: require_nonempty_string(spec.get("display_name", spec["system_id"]), f"{spec['system_id']}.display_name")
            for spec in specs
        }

        output_indexes: dict[str, dict[str, dict[str, Any]]] = {}
        provenance: dict[str, dict[str, Any]] = {}
        for spec in specs:
            system_id = spec["system_id"]
            if spec["kind"] == "copy":
                provenance[system_id] = {"method": "identity", "generation_performed": False}
                continue
            output_path = resolve_path(path_root, spec.get("output_path"), f"systems[{system_id}].output_path")
            payload = read_json(output_path)
            expected_payload_id = spec.get("expected_output_system_id", system_id)
            if payload.get("system_id") != expected_payload_id:
                raise CampaignError(f"{system_id}: output system_id {payload.get('system_id')!r} does not equal {expected_payload_id!r}")
            indexed, observed_order = index_records(payload, system_id)
            observed_keys = set(indexed)
            missing, unexpected = expected_key_set - observed_keys, observed_keys - expected_key_set
            if missing or unexpected:
                raise CampaignError(f"{system_id}: ID alignment failure; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
            if observed_order != expected_keys:
                raise CampaignError(f"{system_id}: item order differs from canonical source")
            output_indexes[system_id] = indexed
            provenance[system_id] = output_metadata(payload, output_path)

        completed: list[dict[str, Any]] = []
        all_flags: list[dict[str, str]] = []
        seen_candidate_ids: set[str] = set()
        for source_row in source_rows:
            item = dict(source_row)
            candidates: dict[str, str] = {}
            candidate_provenance: dict[str, dict[str, Any]] = {}
            for spec in specs:
                system_id = spec["system_id"]
                if spec["kind"] == "copy":
                    candidate, row, raw, flags = source_row["source_bp"], {}, source_row["source_bp"], []
                else:
                    row = output_indexes[system_id][source_row["item_key"]]
                    for field in ("document_id", "passage_id", "sentence_id", "position"):
                        if str(row.get(field, source_row[field])) != str(source_row[field]):
                            raise CampaignError(f"{system_id}: {field} mismatch for {source_row['item_key']}")
                    if row.get("source_bp", row.get("source")) != source_row["source_bp"]:
                        raise CampaignError(f"{system_id}: source text mismatch for {source_row['item_key']}")
                    candidate_field = spec.get("candidate_field", "cleaned_candidate")
                    candidate = require_nonempty_string(row.get(candidate_field), f"{system_id} {source_row['item_key']}.{candidate_field}")
                    raw_field = spec.get("raw_output_field", "raw_model_output")
                    raw = row.get(raw_field)
                    if not isinstance(raw, str):
                        raise CampaignError(f"{system_id}: missing string {raw_field} for {source_row['item_key']}")
                    flags = technical_flags(row, candidate)
                candidates[system_id] = candidate
                excluded = {
                    "document_id", "passage_id", "sentence_id", "position", "source_bp", "source",
                    spec.get("candidate_field", "cleaned_candidate"), spec.get("raw_output_field", "raw_model_output"), "technical_flags",
                }
                row_metadata = {key: copy.deepcopy(value) for key, value in row.items() if key not in excluded}
                candidate_provenance[system_id] = {"raw_model_output": raw, "technical_flags": flags, "generation_metadata": row_metadata}
                all_flags.extend({"item_key": source_row["item_key"], "system_id": system_id, "flag": flag} for flag in flags)
            item["candidates"] = candidates
            item["candidate_ids"] = {
                system_id: candidate_id(
                    source_row["passage_id"], source_row["sentence_id"], system_id
                )
                for system_id in candidates
            }
            for stable_id in item["candidate_ids"].values():
                if stable_id in seen_candidate_ids:
                    raise CampaignError(f"candidate_id collision: {stable_id}")
                seen_candidate_ids.add(stable_id)
            item["candidate_provenance"] = candidate_provenance
            completed.append(item)

        validation = config.get("validation", {})
        if not isinstance(validation, dict):
            raise CampaignError("config.validation must be an object")
        expected_items = validation.get("expected_source_sentence_count")
        expected_systems = validation.get("expected_system_count")
        if expected_items is not None and len(completed) != expected_items:
            raise CampaignError(f"expected {expected_items} source items, observed {len(completed)}")
        if expected_systems is not None and len(specs) != expected_systems:
            raise CampaignError(f"expected {expected_systems} systems, observed {len(specs)}")

        judgment_count = len(completed) * len(specs)
        dataset = {
            "schema_version": "2.1",
            "campaign_id": campaign_id,
            "source_language": source_payload.get("source_language"),
            "target_language": source_payload.get("target_language"),
            "source_item_count": len(completed),
            "system_count": len(specs),
            "candidate_judgment_count": judgment_count,
            "systems": [{"system_id": spec["system_id"], "display_name": display_names[spec["system_id"]], "kind": spec["kind"]} for spec in specs],
            "system_provenance": provenance,
            "technical_flags": all_flags,
            "documents": group_items(completed),
        }
        atomic_write_json(dataset_path, dataset, args.overwrite)
        atomic_write_text(review_path, render_review(dataset, display_names), args.overwrite)
        print(f"Built {dataset_path}: {len(completed)} source items x {len(specs)} systems = {judgment_count} candidates")
        print(f"Built {review_path}")
    except CampaignError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
