#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a classification eval JSONL from a translation eval JSONL. "
            "Non-equal rows become pt-br/pt-pt classification examples based on the source variant; "
            "equal rows are deduplicated into a single 'equal' classification example."
        )
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def strip_encoder_prefix(text: str) -> str:
    return normalize_space(TASK_PREFIX_RE.sub("", text or ""))


def infer_source_label(row: dict[str, Any]) -> str | None:
    if bool(row.get("is_equal_pair")):
        return "equal"

    direction = normalize_space(str(row.get("direction") or row.get("task") or "")).casefold()
    if direction in {"br2pt", "translate_br2pt"}:
        return "pt-br"
    if direction in {"pt2br", "translate_pt2br"}:
        return "pt-pt"

    raw_input = str(row.get("input_text") or "")
    match = TASK_PREFIX_RE.match(raw_input)
    if not match:
        return None
    prefix = match.group(1).strip().casefold()
    if prefix == "br-pt":
        return "pt-br"
    if prefix == "pt-br":
        return "pt-pt"
    return None


def label_id(label: str) -> int:
    if label == "pt-br":
        return 0
    if label == "pt-pt":
        return 1
    if label == "equal":
        return 2
    raise ValueError(f"Unexpected label: {label!r}")


def key_for_row(source_id: Any, source_text: str, label: str) -> tuple[Any, str, str]:
    if label == "equal":
        return source_id, source_text, "equal"
    return source_id, source_text, label


def main() -> None:
    args = parse_args()
    if not args.input_path.is_file():
        raise SystemExit(f"Missing input JSONL: {args.input_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[Any, str, str]] = set()
    counts: Counter[str] = Counter()
    written = 0

    with args.input_path.open("r", encoding="utf-8") as in_fh, args.output_path.open("w", encoding="utf-8") as out_fh:
        for line_no, line in enumerate(in_fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            source_text = normalize_space(str(row.get("source_text") or strip_encoder_prefix(str(row.get("input_text") or ""))))
            source_id = row.get("source_id")
            label = infer_source_label(row)
            if not source_text or label is None:
                counts["skipped_invalid"] += 1
                continue

            dedupe_key = key_for_row(source_id, source_text, label)
            if dedupe_key in seen:
                counts["skipped_duplicate"] += 1
                continue
            seen.add(dedupe_key)

            out_row = {
                "source_id": source_id,
                "task": "classification",
                "text": source_text,
                "input_text": f"<id> {source_text}".strip(),
                "label": label,
                "label_id": label_id(label),
                "dataset": row.get("dataset", "UNKNOWN"),
                "bucket": row.get("bucket", "n/a"),
                "source": row.get("source"),
            }
            out_fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1
            counts[f"label:{label}"] += 1

    summary = {
        "input_path": args.input_path.as_posix(),
        "output_path": args.output_path.as_posix(),
        "written": written,
        "counts": dict(counts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
