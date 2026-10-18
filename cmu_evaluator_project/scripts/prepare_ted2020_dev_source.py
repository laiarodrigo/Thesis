#!/usr/bin/env python3
"""Prepare the selected TED2020 DEV source from authoritative pt-BR XML."""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from campaign_common import atomic_write_json, atomic_write_text


REPO_ROOT = Path(__file__).resolve().parents[2]
XML_ROOT = REPO_ROOT / "data/external/TED2020/TED2020/raw/pt_br"
REVIEW_PATH = (
    REPO_ROOT
    / "cmu_evaluator_project/data/passage_selection/ted2020_manual_review_candidates.md"
)
OUTPUT_JSON = (
    REPO_ROOT / "cmu_evaluator_project/data/prepared/dev_01/source/source.json"
)
OUTPUT_SOURCE_MD = (
    REPO_ROOT / "cmu_evaluator_project/data/prepared/dev_01/source/source.md"
)
OUTPUT_SELECTION_MD = (
    REPO_ROOT
    / "cmu_evaluator_project/data/passage_selection/ted2020_selected_candidates.md"
)

SELECTIONS = [
    ("ted2020-18219_s30-s37", "ted2020-18219", ["30", "31.1", "31.2", "32", "33", "34", "35", "36", "37"]),
    ("ted2020-2203_s26-s32", "ted2020-2203", ["26", "27", "28.1", "28.2", "29.1", "29.2", "29.3", "30", "31", "32"]),
    ("ted2020-2433_s21-s30", "ted2020-2433", [str(value) for value in range(21, 31)]),
    ("ted2020-27951_s18-s27", "ted2020-27951", [str(value) for value in range(18, 28)]),
    ("ted2020-24360_s25-s33", "ted2020-24360", [str(value) for value in range(25, 34)]),
    ("ted2020-2609_s82-s91", "ted2020-2609", [str(value) for value in range(82, 92)]),
    ("ted2020-1663_s7-s16", "ted2020-1663", [str(value) for value in range(7, 17)]),
    ("ted2020-2860_s124-s133", "ted2020-2860", [str(value) for value in range(124, 134)]),
    ("ted2020-1886_s23-s32", "ted2020-1886", [str(value) for value in range(23, 33)]),
    ("ted2020-946_s43-s51", "ted2020-946", [str(value) for value in range(43, 52)]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def review_metadata(text: str, passage_id: str) -> dict[str, str | None]:
    start = text.find(f"### {passage_id}\n")
    if start < 0:
        raise ValueError(f"passage missing from manual review: {passage_id}")
    following = text.find("\n### ", start + 1)
    block = text[start : following if following >= 0 else len(text)]

    def value(label: str) -> str | None:
        match = re.search(rf"^- \*\*{re.escape(label)}:\*\* (.+)$", block, re.MULTILINE)
        return match.group(1) if match else None

    return {
        "manual_review_category": value("Manual-review category"),
        "origin": value("Origin"),
        "evidence_character": value("Evidence character"),
        "quality_concerns": value("Quality concerns"),
    }


def xml_sentences(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    return {
        str(node.attrib["id"]): "".join(node.itertext())
        for node in root.findall(".//s")
    }


def build() -> tuple[dict, str, str]:
    review_text = REVIEW_PATH.read_text(encoding="utf-8")
    documents = []
    table_rows = []
    detail_lines = []
    normalized_sentence_count = 0
    for ordinal, (passage_id, document_id, sentence_ids) in enumerate(SELECTIONS, 1):
        source_xml = XML_ROOT / f"{document_id}.xml"
        available = xml_sentences(source_xml)
        missing = [sentence_id for sentence_id in sentence_ids if sentence_id not in available]
        if missing:
            raise ValueError(f"{passage_id}: missing XML sentence IDs {missing}")
        metadata = review_metadata(review_text, passage_id)
        prepared_sentences = []
        normalized_ids = []
        for position, sentence_id in enumerate(sentence_ids, 1):
            raw_text = available[sentence_id]
            prepared_text = raw_text
            sentence_record = {
                "sentence_id": sentence_id,
                "position": position,
                "source_bp": prepared_text,
            }
            if passage_id == "ted2020-946_s43-s51" and raw_text.startswith("TD:"):
                prepared_text = re.sub(r"^TD:\s*", "", raw_text, count=1)
                sentence_record["source_bp"] = prepared_text
                sentence_record["raw_source_bp"] = raw_text
                sentence_record["source_preparation"] = {
                    "removed_leading_speaker_label": "TD:"
                }
                normalized_ids.append(sentence_id)
                normalized_sentence_count += 1
            prepared_sentences.append(sentence_record)
        if passage_id == "ted2020-946_s43-s51" and normalized_ids != ["48", "49"]:
            raise ValueError(
                f"expected TD: normalization in sentences 48 and 49, observed {normalized_ids}"
            )
        source_preparation = (
            {
                "removed_leading_speaker_label": "TD:",
                "sentence_ids": normalized_ids,
                "scope": "prepared DEV representation only",
            }
            if normalized_ids
            else {"ad_hoc_changes": "none"}
        )
        passage = {
            "passage_id": passage_id,
            "sentence_ids": sentence_ids,
            "sentence_count": len(sentence_ids),
            "previous_manual_review": metadata,
            "source_preparation": source_preparation,
            "sentences": prepared_sentences,
        }
        documents.append(
            {
                "document_id": document_id,
                "source_xml": str(source_xml.relative_to(REPO_ROOT)),
                "passages": [passage],
            }
        )
        category = metadata["manual_review_category"] or "not recorded"
        table_rows.append(
            f"| {ordinal} | `{passage_id}` | `{document_id}` | {len(sentence_ids)} | {category} |"
        )
        detail_lines.extend(
            [
                f"## {ordinal}. `{passage_id}`",
                "",
                f"- **TED document ID:** `{document_id}`",
                f"- **Sentence IDs:** {', '.join(f'`{value}`' for value in sentence_ids)}",
                f"- **Sentence count:** {len(sentence_ids)}",
                "- **Previous manual-review category:** "
                + (
                    metadata["manual_review_category"]
                    if metadata["manual_review_category"]
                    else "Not recorded in the current manual-review file; no category was inferred."
                ),
                f"- **Review-pool origin:** {metadata['origin']}",
                f"- **Evidence character:** {metadata['evidence_character']}",
                f"- **Recorded quality concerns:** {metadata['quality_concerns']}",
                f"- **Authoritative source:** `{source_xml.relative_to(REPO_ROOT)}`",
                "- **Source preparation:** "
                + (
                    "Removed only the leading `TD:` label and immediate whitespace from "
                    f"sentence(s) {', '.join(normalized_ids)} in this prepared DEV representation."
                    if normalized_ids
                    else "None; copied exactly from the authoritative pt-BR XML."
                ),
                "",
                "**Exact BP source used for DEV:**",
                "",
            ]
        )
        detail_lines.extend(
            f"{index}. **{row['sentence_id']}.** {row['source_bp']}"
            for index, row in enumerate(prepared_sentences, 1)
        )
        detail_lines.append("")

    payload = {
        "schema_version": "1.0",
        "dataset_id": "ted2020_dev_01",
        "split": "DEV",
        "status": "selected_dev_source",
        "source_language": "pt-BR",
        "target_language": "pt-PT",
        "passage_count": len(documents),
        "source_item_count": sum(
            passage["sentence_count"]
            for document in documents
            for passage in document["passages"]
        ),
        "selection_record": str(OUTPUT_SELECTION_MD.relative_to(REPO_ROOT)),
        "historical_review_record": str(REVIEW_PATH.relative_to(REPO_ROOT)),
        "test_passages_frozen": False,
        "documents": documents,
    }
    selection_md = "\n".join(
        [
            "# TED2020 selected DEV passages",
            "",
            "These 10 passages are selected for the DEV human-gold set and will be annotated by Rodrigo. Selection was intentional and was not restricted to passages previously labelled **Strong keep**. No TEST passages are frozen.",
            "",
            "The historical review file `ted2020_manual_review_candidates.md` remains unchanged. Its source-selection assessments are evidence, not hard exclusion rules. The current copy does not contain a Strong keep/Keep/Borderline/Drop field, so this file reports that category as **not recorded** rather than reconstructing or guessing it.",
            "",
            "| # | Passage ID | TED document ID | Sentence units | Previous review category |",
            "|---:|---|---|---:|---|",
            *table_rows,
            "",
            *detail_lines,
        ]
    ).rstrip() + "\n"
    source_md = "\n".join(
        [
            "# TED2020 DEV 01 — prepared source",
            "",
            "This directory is the canonical prepared source for the 10 selected DEV passages. Exact sentence text, IDs, review provenance, and passage-level notes are recorded in [`../../../passage_selection/ted2020_selected_candidates.md`](../../../passage_selection/ted2020_selected_candidates.md); machine-readable data is in `source.json`.",
            "",
            f"- **Passages:** {len(documents)}",
            f"- **Sentence units:** {payload['source_item_count']}",
            "- **Source:** authoritative TED2020 pt-BR XML files listed per document",
            "- **Generation unit:** one sentence at a time",
            "- **Annotation context:** reconstructed full passage",
            "- **TEST status:** no TEST passage is frozen",
            "",
            "## Source preparation",
            "",
            f"Exactly {normalized_sentence_count} sentence(s) were normalized: sentences `48` and `49` of `ted2020-946_s43-s51` had only the leading `TD:` label and its immediate whitespace removed. Their raw text is retained in `raw_source_bp` and the transformation is recorded in `source_preparation`.",
            "",
            "No other selected sentence received an ad-hoc correction, normalization, or rewrite. The raw XML and historical manual-review file were not modified.",
            "",
        ]
    )
    return payload, selection_md, source_md


def main() -> None:
    args = parse_args()
    payload, selection_md, source_md = build()
    atomic_write_json(OUTPUT_JSON, payload, args.overwrite)
    atomic_write_text(OUTPUT_SELECTION_MD, selection_md, args.overwrite)
    atomic_write_text(OUTPUT_SOURCE_MD, source_md, args.overwrite)
    print(
        f"Prepared {payload['passage_count']} DEV passages / "
        f"{payload['source_item_count']} sentence units"
    )


if __name__ == "__main__":
    main()
