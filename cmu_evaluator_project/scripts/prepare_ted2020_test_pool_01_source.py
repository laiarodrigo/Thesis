#!/usr/bin/env python3
"""Prepare the four frozen TEST Pool 1 passages from authoritative TED XML."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from campaign_common import atomic_write_json, atomic_write_text


REPO_ROOT = Path(__file__).resolve().parents[2]
XML_ROOT = REPO_ROOT / "data/external/TED2020/TED2020/raw/pt_br"
OUTPUT_ROOT = REPO_ROOT / "cmu_evaluator_project/data/prepared/test_pool_01/source"
OUTPUT_JSON = OUTPUT_ROOT / "source.json"
OUTPUT_MARKDOWN = OUTPUT_ROOT / "source.md"
SELECTION_RECORD = (
    REPO_ROOT
    / "cmu_evaluator_project/data/passage_selection/ted2020_selected_candidates.md"
)
HISTORICAL_REVIEW = (
    REPO_ROOT
    / "cmu_evaluator_project/data/passage_selection/ted2020_manual_review_candidates.md"
)

SELECTIONS = [
    (
        "ted2020-1188_s9.2-s18",
        "ted2020-1188",
        ["9.2", "10", "11", "12", "13", "14", "15", "16", "17", "18"],
    ),
    (
        "ted2020-2279_s83-s91",
        "ted2020-2279",
        ["83", "84.1", "84.2", "85", "86", "87", "88", "89", "90", "91"],
    ),
    (
        "ted2020-1565_s7.2-s14",
        "ted2020-1565",
        ["7.2", "7.3", "8", "9", "10.1", "10.2", "11", "12", "13", "14"],
    ),
    (
        "ted2020-2568_s127-s135",
        "ted2020-2568",
        ["127", "128", "129", "130", "131", "132", "133", "134.1", "134.2", "135"],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def xml_sentences(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    return {
        str(node.attrib["id"]): "".join(node.itertext())
        for node in root.findall(".//s")
    }


def build() -> tuple[dict, str]:
    documents = []
    detail_lines = []
    for ordinal, (passage_id, document_id, sentence_ids) in enumerate(SELECTIONS, 1):
        source_xml = XML_ROOT / f"{document_id}.xml"
        available = xml_sentences(source_xml)
        missing = [sentence_id for sentence_id in sentence_ids if sentence_id not in available]
        if missing:
            raise ValueError(f"{passage_id}: missing XML sentence IDs {missing}")
        sentences = [
            {
                "sentence_id": sentence_id,
                "position": position,
                "source_bp": available[sentence_id],
            }
            for position, sentence_id in enumerate(sentence_ids, 1)
        ]
        documents.append(
            {
                "document_id": document_id,
                "source_xml": str(source_xml.relative_to(REPO_ROOT)),
                "passages": [
                    {
                        "passage_id": passage_id,
                        "sentence_ids": list(sentence_ids),
                        "sentence_count": len(sentence_ids),
                        "selection_label": "Strong",
                        "source_preparation": {"ad_hoc_changes": "none"},
                        "sentences": sentences,
                    }
                ],
            }
        )
        detail_lines.extend(
            [
                f"## {ordinal}. `{passage_id}`",
                "",
                f"- **Authoritative XML:** `{source_xml.relative_to(REPO_ROOT)}`",
                f"- **Sentence IDs:** {', '.join(f'`{value}`' for value in sentence_ids)}",
                "- **Source preparation:** none; exact XML text",
                "",
                *[
                    f"{position}. **{sentence['sentence_id']}.** {sentence['source_bp']}"
                    for position, sentence in enumerate(sentences, 1)
                ],
                "",
            ]
        )
    payload = {
        "schema_version": "1.0",
        "dataset_id": "ted2020_test_pool_01",
        "split": "TEST",
        "pool_id": "pool_01",
        "status": "provisional_external_annotator_collection",
        "final_benchmark_frozen": False,
        "source_language": "pt-BR",
        "target_language": "pt-PT",
        "passage_count": len(documents),
        "source_item_count": sum(
            passage["sentence_count"]
            for document in documents
            for passage in document["passages"]
        ),
        "selection_record": str(SELECTION_RECORD.relative_to(REPO_ROOT)),
        "historical_review_record": str(HISTORICAL_REVIEW.relative_to(REPO_ROOT)),
        "source_text_policy": "exact authoritative pt-BR XML; no cleanup or normalization",
        "documents": documents,
    }
    if payload["passage_count"] != 4 or payload["source_item_count"] != 40:
        raise ValueError("TEST Pool 1 must contain exactly 4 passages / 40 sentence units")
    markdown = "\n".join(
        [
            "# TED2020 TEST Pool 1 — canonical source",
            "",
            "These four frozen Pool 1 source passages are for provisional external-annotator TEST collection, not DEV. They remain sentence-level generation examples and are reconstructed only for annotation context.",
            "",
            "- **Passages:** 4",
            "- **Sentence units:** 40",
            "- **Selection label:** Strong",
            "- **Source text:** exact authoritative TED2020 pt-BR XML",
            "- **Ad-hoc cleanup:** none",
            "- **Final TEST benchmark:** not frozen",
            "",
            *detail_lines,
        ]
    ).rstrip() + "\n"
    return payload, markdown


def main() -> None:
    args = parse_args()
    payload, markdown = build()
    atomic_write_json(OUTPUT_JSON, payload, args.overwrite)
    atomic_write_text(OUTPUT_MARKDOWN, markdown, args.overwrite)
    print("Prepared 4 TEST Pool 1 passages / 40 exact XML sentence units")


if __name__ == "__main__":
    main()
