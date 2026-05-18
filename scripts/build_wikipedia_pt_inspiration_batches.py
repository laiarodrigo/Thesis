#!/usr/bin/env python3
"""
Build small inspiration JSON files from Portuguese Wikipedia JSONL articles.

Expected input format matches NeMo Curator Wikipedia export JSONL, where each line
contains fields like: text, title, id, url, language, source_id.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+", flags=re.MULTILINE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“”«»A-ZÀ-ÖØ-Þ0-9])")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Split Portuguese Wikipedia JSONL into small JSON inspiration batches "
            "containing the first sentences of different articles."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSONL files exported from NeMo Curator Wikipedia extraction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "wikipedia_pt_inspiration",
        help="Directory where the batch JSON files will be written.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Maximum number of output JSON batch files to create. Use 0 to write all possible batches.",
    )
    parser.add_argument(
        "--articles-per-file",
        type=int,
        default=25,
        help="How many article excerpts to include in each output file.",
    )
    parser.add_argument(
        "--sentences-per-article",
        type=int,
        default=4,
        help="How many leading sentences to keep from each article.",
    )
    parser.add_argument(
        "--min-paragraph-words",
        type=int,
        default=25,
        help="Minimum words required for a source paragraph to be considered.",
    )
    parser.add_argument(
        "--min-text-words",
        type=int,
        default=60,
        help="Minimum total words required after paragraph extraction.",
    )
    parser.add_argument(
        "--language",
        default="pt",
        help="Language code expected in the source JSONL. Use empty string to disable filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed used before splitting into files.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def split_paragraphs(text: str) -> list[str]:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []
    parts = [normalize_space(part) for part in PARAGRAPH_SPLIT_RE.split(raw) if normalize_space(part)]
    if parts:
        return parts
    return [normalize_space(raw)] if normalize_space(raw) else []


def split_sentences(text: str) -> list[str]:
    raw = normalize_space(text)
    if not raw:
        return []
    parts = [normalize_space(part) for part in SENTENCE_SPLIT_RE.split(raw) if normalize_space(part)]
    return parts if parts else [raw]


def is_noisy_paragraph(text: str, *, min_paragraph_words: int) -> bool:
    compact = normalize_space(text)
    if count_words(compact) < min_paragraph_words:
        return True
    if not re.search(r"[.!?]", compact):
        return True
    if compact.count(":") >= 2 and compact.count(".") <= 1:
        return True
    return False


def select_leading_sentences(
    text: str,
    *,
    sentences_per_article: int,
    min_paragraph_words: int,
) -> list[str]:
    sentences: list[str] = []
    for part in split_paragraphs(text):
        if is_noisy_paragraph(part, min_paragraph_words=min_paragraph_words):
            continue
        for sentence in split_sentences(part):
            if count_words(sentence) < 5:
                continue
            sentences.append(sentence)
            if len(sentences) >= sentences_per_article:
                return sentences
    return sentences


def iter_jsonl_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    parsed["_source_path"] = str(path)
                    parsed["_line_number"] = line_number
                    records.append(parsed)
    return records


def record_identity_key(record: dict[str, Any]) -> str:
    article_id = normalize_space(str(record.get("id") or "")).strip()
    if article_id:
        return f"id:{article_id}"

    url = normalize_space(str(record.get("url") or "")).strip()
    if url:
        return f"url:{url}"

    title = normalize_space(str(record.get("title") or "")).strip().lower()
    if title:
        return f"title:{title}"

    text = normalize_space(str(record.get("text") or "")).strip()
    if text:
        return f"text:{text[:400]}"

    return f'fallback:{record.get("_source_path", "")}:{record.get("_line_number", "")}'


def record_quality_score(record: dict[str, Any]) -> tuple[int, int]:
    text = normalize_space(str(record.get("text") or "")).strip()
    title = normalize_space(str(record.get("title") or "")).strip()
    return (count_words(text), len(title))


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    best_by_key: dict[str, dict[str, Any]] = {}
    duplicate_count = 0

    for record in records:
        key = record_identity_key(record)
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = record
            continue

        duplicate_count += 1
        if record_quality_score(record) > record_quality_score(existing):
            best_by_key[key] = record

    return list(best_by_key.values()), duplicate_count


def build_examples(
    records: list[dict[str, Any]],
    *,
    language: str,
    sentences_per_article: int,
    min_paragraph_words: int,
    min_text_words: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []

    for record in records:
        record_language = normalize_space(str(record.get("language") or "")).lower()
        if language and record_language and record_language != language.lower():
            continue

        text = str(record.get("text") or "").strip()
        if not text:
            continue

        sentences = select_leading_sentences(
            text,
            sentences_per_article=sentences_per_article,
            min_paragraph_words=min_paragraph_words,
        )
        if not sentences:
            continue

        merged_text = " ".join(sentences)
        if count_words(merged_text) < min_text_words:
            continue

        article_id = normalize_space(str(record.get("id") or "")).strip()
        title = normalize_space(str(record.get("title") or "")).strip()

        examples.append(
            {
                "article_id": article_id,
                "title": title,
                "url": normalize_space(str(record.get("url") or "")).strip(),
                "language": record_language,
                "source_id": normalize_space(str(record.get("source_id") or "")).strip(),
                "text": merged_text,
                "sentences": sentences,
                "word_count": count_words(merged_text),
            }
        )

    return examples


def write_batches(
    examples: list[dict[str, Any]],
    *,
    output_dir: Path,
    num_files: int,
    articles_per_file: int,
    source_paths: list[Path],
    sentences_per_article: int,
    language: str,
    seed: int,
    raw_record_count: int,
    deduplicated_record_count: int,
    duplicate_record_count: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    if num_files == 0:
        num_files = max(1, math.ceil(len(examples) / articles_per_file))
    total_capacity = num_files * articles_per_file
    selected = list(examples[:total_capacity])

    written = 0
    for batch_index in range(num_files):
        start = batch_index * articles_per_file
        chunk = selected[start : start + articles_per_file]
        if not chunk:
            break

        payload = {
            "metadata": {
                "source": "wikipedia",
                "source_format": "nemo_curator_jsonl",
                "language": language or "",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sentences_per_article": sentences_per_article,
                "articles_in_batch": len(chunk),
                "raw_record_count": raw_record_count,
                "deduplicated_record_count": deduplicated_record_count,
                "duplicate_record_count": duplicate_record_count,
                "seed": seed,
                "input_files": [str(path) for path in source_paths],
            },
            "exemplos": chunk,
        }
        output_path = output_dir / f"ptwiki_inspiration_batch_{batch_index + 1:02d}.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written += 1

    return written


def main() -> None:
    args = parse_args()

    if args.num_files < 0:
        raise SystemExit("--num-files must be >= 0")
    if args.articles_per_file <= 0:
        raise SystemExit("--articles-per-file must be > 0")
    if args.sentences_per_article <= 0:
        raise SystemExit("--sentences-per-article must be > 0")
    if args.min_paragraph_words <= 0:
        raise SystemExit("--min-paragraph-words must be > 0")
    if args.min_text_words <= 0:
        raise SystemExit("--min-text-words must be > 0")

    missing = [path for path in args.input_jsonl if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing input JSONL file(s): " + ", ".join(str(path) for path in missing)
        )

    records = iter_jsonl_records(args.input_jsonl)
    raw_record_count = len(records)
    deduplicated_records, duplicate_record_count = deduplicate_records(records)
    rng = random.Random(args.seed)
    rng.shuffle(deduplicated_records)

    examples = build_examples(
        deduplicated_records,
        language=args.language,
        sentences_per_article=args.sentences_per_article,
        min_paragraph_words=args.min_paragraph_words,
        min_text_words=args.min_text_words,
    )
    if not examples:
        raise SystemExit("No suitable Wikipedia articles were found after filtering.")

    written = write_batches(
        examples,
        output_dir=args.output_dir,
        num_files=args.num_files,
        articles_per_file=args.articles_per_file,
        source_paths=args.input_jsonl,
        sentences_per_article=args.sentences_per_article,
        language=args.language,
        seed=args.seed,
        raw_record_count=raw_record_count,
        deduplicated_record_count=len(deduplicated_records),
        duplicate_record_count=duplicate_record_count,
    )

    print(
        f"Wrote {written} batch file(s) to {args.output_dir} "
        f"from {len(examples)} filtered Wikipedia article excerpts "
        f"(raw_records={raw_record_count}, deduplicated_records={len(deduplicated_records)}, "
        f"duplicates_removed={duplicate_record_count})."
    )


if __name__ == "__main__":
    main()
