#!/usr/bin/env python3
"""Export a canonical candidate dataset to a validated Pearmut cESA campaign."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any

from campaign_common import (
    CampaignError,
    atomic_write_json,
    load_config,
    read_json,
    require_nonempty_string,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing campaign export")
    return parser.parse_args()


def _inline_markdown(text: str) -> str:
    """Render the small, trusted inline subset used by campaign instructions."""
    rendered = html.escape(text, quote=False)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    return rendered


def markdown_instructions_to_html(markdown: str) -> str:
    """Render headings, paragraphs, lists, bold, code, and hard line breaks.

    Pearmut inserts ``info.instructions`` with ``.html()`` and does not parse
    Markdown itself. Keeping this renderer local makes campaign generation
    deterministic and avoids adding a runtime dependency to the reusable
    pipeline.
    """
    output: list[str] = []
    paragraph: list[tuple[str, bool]] = []
    list_kind: str | None = None
    list_items: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        pieces: list[str] = []
        for index, (line, hard_break) in enumerate(paragraph):
            pieces.append(_inline_markdown(line))
            if index < len(paragraph) - 1:
                pieces.append("<br>" if hard_break else " ")
        output.append(f"<p>{''.join(pieces)}</p>")
        paragraph.clear()

    def flush_list() -> None:
        nonlocal list_kind
        if list_kind is None:
            return
        output.append(f"<{list_kind}>")
        output.extend(f"<li>{_inline_markdown(item)}</li>" for item in list_items)
        output.append(f"</{list_kind}>")
        list_items.clear()
        list_kind = None

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line:
            flush_paragraph()
            flush_list()
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        ordered = re.match(r"^\d+\.\s+(.+)$", line)
        unordered = re.match(r"^-\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            level = min(len(heading.group(1)) + 1, 4)
            output.append(f"<h{level}>{_inline_markdown(heading.group(2))}</h{level}>")
        elif ordered or unordered:
            flush_paragraph()
            new_kind = "ol" if ordered else "ul"
            if list_kind is not None and list_kind != new_kind:
                flush_list()
            list_kind = new_kind
            list_items.append((ordered or unordered).group(1))
        elif list_kind is not None:
            list_items[-1] = f"{list_items[-1]} {line.strip()}"
        else:
            paragraph.append((line, raw_line.endswith("  ")))

    flush_paragraph()
    flush_list()
    return "\n".join(output) + "\n"


def mapping_dict(mapping: dict[str, Any], campaign_id: str, real_ids: list[str]) -> dict[str, str]:
    if mapping.get("campaign_id") != campaign_id:
        raise CampaignError("private mapping campaign_id does not match configuration")
    entries = mapping.get("mappings")
    if not isinstance(entries, list):
        raise CampaignError("private mapping mappings must be a list")
    result: dict[str, str] = {}
    anonymous_seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CampaignError(f"private mapping entry {index} must be an object")
        real = require_nonempty_string(entry.get("real_system_id"), f"mapping[{index}].real_system_id")
        anonymous = require_nonempty_string(entry.get("anonymous_system_id"), f"mapping[{index}].anonymous_system_id")
        if real in result or anonymous in anonymous_seen:
            raise CampaignError("private mapping is not one-to-one")
        result[real] = anonymous
        anonymous_seen.add(anonymous)
    if set(result) != set(real_ids):
        raise CampaignError(f"mapping systems do not match dataset: mapping={sorted(result)}, dataset={sorted(real_ids)}")
    if set(result) & anonymous_seen:
        raise CampaignError("anonymous IDs collide with real system IDs")
    return result


PASSAGE_LAYOUT = "passage"
RESUMABLE_LAYOUT = "one_sentence_per_page_with_full_source_context"


def validate_campaign(
    campaign: dict[str, Any],
    protocol: str,
    anonymous_ids: set[str],
    expected_documents: int,
    expected_segments: int,
    expected_judged_items: int,
    expected_judgments: int,
    page_layout: str,
) -> None:
    info = campaign.get("info")
    if not isinstance(info, dict) or info.get("protocol") != protocol:
        raise CampaignError("Pearmut campaign does not use the configured protocol")
    if info.get("show_model_names") is not False:
        raise CampaignError("Pearmut campaign must hide model names")
    if info.get("shuffle") is not False:
        raise CampaignError("Pearmut campaign must disable Pearmut's unseeded shuffle")
    data = campaign.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], list):
        raise CampaignError("expected one task containing one or more Pearmut documents")
    documents = data[0]
    if len(documents) != expected_documents:
        raise CampaignError(f"expected {expected_documents} Pearmut documents, observed {len(documents)}")
    observed_segments = 0
    observed_judged_items = 0
    observed_judgments = 0
    seen_item_ids: set[str] = set()
    for document in documents:
        if not isinstance(document, list) or not document:
            raise CampaignError("each Pearmut document must be a non-empty list")
        judged_items_in_document = 0
        for segment in document:
            observed_segments += 1
            if not isinstance(segment, dict):
                raise CampaignError("each Pearmut segment must be an object")
            item_id = require_nonempty_string(segment.get("item_id"), "Pearmut item_id")
            if item_id in seen_item_ids:
                raise CampaignError(f"duplicate Pearmut item_id: {item_id}")
            seen_item_ids.add(item_id)
            require_nonempty_string(segment.get("src"), f"{item_id}.src")
            if "ref" in segment:
                raise CampaignError(f"{item_id}: reference translations are forbidden")
            targets = segment.get("tgt")
            if not isinstance(targets, dict) or (targets and set(targets) != anonymous_ids):
                raise CampaignError(
                    f"{item_id}: targets must be empty context or contain every anonymous system"
                )
            if not targets:
                if segment.get("context_only") is not True:
                    raise CampaignError(f"{item_id}: empty targets must be marked context_only")
                continue
            if segment.get("context_only") is True:
                raise CampaignError(f"{item_id}: judged items cannot be marked context_only")
            judged_items_in_document += 1
            observed_judged_items += 1
            observed_judgments += len(targets)
            for anonymous_id, candidate in targets.items():
                require_nonempty_string(candidate, f"{item_id}.tgt[{anonymous_id}]")
        if page_layout == RESUMABLE_LAYOUT and judged_items_in_document != 1:
            raise CampaignError(
                "each resumable Pearmut document must contain exactly one judged source item"
            )
        if page_layout == PASSAGE_LAYOUT and judged_items_in_document != len(document):
            raise CampaignError("passage layout cannot contain context-only items")
    if observed_segments != expected_segments:
        raise CampaignError(
            f"expected {expected_segments} displayed Pearmut segments, observed {observed_segments}"
        )
    if observed_judged_items != expected_judged_items:
        raise CampaignError(
            f"expected {expected_judged_items} judged source items, observed {observed_judged_items}"
        )
    if observed_judgments != expected_judgments:
        raise CampaignError(
            f"expected {expected_judgments} candidate judgments, observed {observed_judgments}"
        )


def build_campaign_payload(
    dataset: dict[str, Any],
    mapping: dict[str, Any],
    campaign_id: str,
    pearmut: dict[str, Any],
    instructions: str,
    selected_passage_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build one Pearmut campaign, optionally for an assignment subset."""
    system_entries = dataset.get("systems")
    if not isinstance(system_entries, list) or not system_entries:
        raise CampaignError("candidate dataset has no systems")
    real_ids = [
        require_nonempty_string(entry.get("system_id"), "dataset system_id")
        for entry in system_entries
    ]
    real_to_anonymous = mapping_dict(mapping, campaign_id, real_ids)
    ordered_real_ids = [entry["real_system_id"] for entry in mapping["mappings"]]
    anonymous_ids = {real_to_anonymous[real] for real in real_ids}

    page_layout = pearmut.get("page_layout", PASSAGE_LAYOUT)
    if page_layout not in {PASSAGE_LAYOUT, RESUMABLE_LAYOUT}:
        raise CampaignError(
            f"pearmut.page_layout must be '{PASSAGE_LAYOUT}' or '{RESUMABLE_LAYOUT}'"
        )
    selected = None
    if selected_passage_ids is not None:
        if not selected_passage_ids or not all(
            isinstance(value, str) and value for value in selected_passage_ids
        ):
            raise CampaignError("assignment passage_ids must be a non-empty list of strings")
        if len(selected_passage_ids) != len(set(selected_passage_ids)):
            raise CampaignError("assignment passage_ids contain duplicates")
        selected = set(selected_passage_ids)

    documents: list[list[dict[str, Any]]] = []
    observed_segments = 0
    observed_judged_items = 0
    observed_judgments = 0
    observed_passages: set[str] = set()
    source_documents = dataset.get("documents")
    if not isinstance(source_documents, list) or not source_documents:
        raise CampaignError("candidate dataset has no documents")
    for document in source_documents:
        document_id = require_nonempty_string(document.get("document_id"), "dataset document_id")
        passages = document.get("passages")
        if not isinstance(passages, list) or not passages:
            raise CampaignError(f"document {document_id} has no passages")
        for passage in passages:
            passage_id = require_nonempty_string(
                passage.get("passage_id"), f"{document_id}.passage_id"
            )
            if selected is not None and passage_id not in selected:
                continue
            if passage_id in observed_passages:
                raise CampaignError(
                    f"passage_id must be globally unique in a campaign: {passage_id}"
                )
            observed_passages.add(passage_id)
            items = passage.get("items")
            if not isinstance(items, list) or not items:
                raise CampaignError(f"{document_id}/{passage_id} has no items")
            passage_segments: list[dict[str, Any]] = []
            previous_position = 0
            for item in items:
                position = item.get("position")
                if not isinstance(position, int) or position <= previous_position:
                    raise CampaignError(
                        f"{document_id}/{passage_id}: item positions are not strictly increasing"
                    )
                previous_position = position
                candidates = item.get("candidates")
                if not isinstance(candidates, dict) or set(candidates) != set(real_ids):
                    raise CampaignError(
                        f"{item.get('item_key')}: candidates do not match configured systems"
                    )
                targets = {
                    real_to_anonymous[real]: require_nonempty_string(
                        candidates[real], f"{item.get('item_key')}.{real}"
                    )
                    for real in ordered_real_ids
                }
                passage_segments.append(
                    {
                        "item_id": require_nonempty_string(
                            item.get("item_key"), "dataset item_key"
                        ),
                        "src": require_nonempty_string(
                            item.get("source_bp"), f"{item.get('item_key')}.source_bp"
                        ),
                        "tgt": targets,
                    }
                )
                observed_judged_items += 1
                observed_judgments += len(targets)

            if page_layout == PASSAGE_LAYOUT:
                documents.append(passage_segments)
                observed_segments += len(passage_segments)
            else:
                for focus_index, focus_segment in enumerate(passage_segments):
                    page: list[dict[str, Any]] = []
                    for context_index, context_segment in enumerate(passage_segments):
                        if context_index == focus_index:
                            page.append(
                                context_segment
                                | {
                                    "instructions": (
                                        "<strong>Current sentence — annotate the "
                                        "candidates below.</strong>"
                                    )
                                }
                            )
                        else:
                            page.append(
                                {
                                    "item_id": (
                                        f"context::{focus_segment['item_id']}::"
                                        f"{context_segment['item_id']}"
                                    ),
                                    "src": context_segment["src"],
                                    "tgt": {},
                                    "context_only": True,
                                }
                            )
                    documents.append(page)
                    observed_segments += len(page)

    if selected is not None and observed_passages != selected:
        raise CampaignError(
            f"assignment passage_ids not found: {sorted(selected - observed_passages)}"
        )
    if not documents:
        raise CampaignError("campaign selection contains no Pearmut documents")

    protocol = require_nonempty_string(pearmut.get("protocol"), "pearmut.protocol")
    if pearmut.get("show_model_names") is not False or pearmut.get("shuffle") is not False:
        raise CampaignError("pearmut.show_model_names and pearmut.shuffle must both be false")
    instructions_format = pearmut.get("instructions_format", "html")
    if instructions_format == "markdown":
        rendered_instructions = markdown_instructions_to_html(instructions)
    elif instructions_format == "html":
        rendered_instructions = instructions
    else:
        raise CampaignError("pearmut.instructions_format must be 'markdown' or 'html'")
    campaign = {
        "info": {
            "assignment": pearmut.get("assignment", "task-based"),
            "protocol": protocol,
            "instructions": rendered_instructions,
            "textfield": pearmut.get("textfield", "visible"),
            "show_model_names": False,
            "show_progress": pearmut.get("show_progress", True),
            "shuffle": False,
            "special_tokens": pearmut.get("special_tokens", ["[MISSING]"]),
        },
        "campaign_id": campaign_id,
        "metadata": {
            "page_layout": page_layout,
            "passage_ids": sorted(observed_passages),
            "displayed_segment_count": observed_segments,
            "judged_source_item_count": observed_judged_items,
            "candidate_judgment_count": observed_judgments,
        },
        "data": [documents],
    }
    validate_campaign(
        campaign,
        protocol,
        anonymous_ids,
        len(documents),
        observed_segments,
        observed_judged_items,
        observed_judgments,
        page_layout,
    )
    return campaign


def main() -> None:
    args = parse_args()
    try:
        config, path_root = load_config(args.config)
        campaign_id = require_nonempty_string(config.get("campaign_id"), "campaign_id")
        paths = config.get("paths")
        if not isinstance(paths, dict):
            raise CampaignError("config.paths must be an object")
        dataset_path = resolve_path(path_root, paths.get("candidate_dataset"), "paths.candidate_dataset")
        mapping_path = resolve_path(path_root, paths.get("private_mapping"), "paths.private_mapping")
        instructions_path = resolve_path(path_root, paths.get("pearmut_instructions"), "paths.pearmut_instructions")
        campaign_path = resolve_path(path_root, paths.get("pearmut_campaign"), "paths.pearmut_campaign")
        dataset, mapping = read_json(dataset_path), read_json(mapping_path)
        if dataset.get("campaign_id") != campaign_id:
            raise CampaignError("candidate dataset campaign_id does not match configuration")
        try:
            instructions = instructions_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CampaignError(f"could not read Pearmut instructions {instructions_path}: {exc}") from exc
        require_nonempty_string(instructions, "Pearmut instructions")

        pearmut = config.get("pearmut")
        if not isinstance(pearmut, dict):
            raise CampaignError("config.pearmut must be an object")
        campaign = build_campaign_payload(
            dataset, mapping, campaign_id, pearmut, instructions
        )
        atomic_write_json(campaign_path, campaign, args.overwrite)
        metadata = campaign["metadata"]
        print(
            f"Built and validated {campaign_path}: {len(campaign['data'][0])} document(s), "
            f"{metadata['displayed_segment_count']} displayed segment(s), "
            f"{metadata['judged_source_item_count']} judged source item(s), "
            f"{metadata['candidate_judgment_count']} candidate judgment(s), "
            f"{len(dataset['systems'])} anonymous systems"
        )
    except CampaignError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
