#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator


MISSING = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-verify that a noequal_cls dataset is an exact ordered copy "
            "with only equal-pair classification rows removed."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--max-diffs", type=int, default=3)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def normalized_label(row: dict[str, Any]) -> str:
    raw = row.get("target_text", row.get("label", row.get("gold", "")))
    text = " ".join(str(raw or "").split()).casefold()
    if text.startswith("<pt-br>") or text in {"pt-br", "br"}:
        return "pt-br"
    if text.startswith("<pt-pt>") or text in {"pt-pt", "pt"}:
        return "pt-pt"
    if text in {"equal", "igual", "same", "shared"}:
        return "equal"
    return text


def is_equal_classification(row: dict[str, Any]) -> bool:
    return row.get("task") == "classification" and (
        bool(row.get("is_equal_pair")) or normalized_label(row) == "equal"
    )


def expected_rows(
    path: Path,
    *,
    dropped: Counter[str],
) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if is_equal_classification(row):
                dropped["equal_classification"] += 1
                continue
            yield line_no, row


def jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def canonical_bytes(row: dict[str, Any]) -> bytes:
    return json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compare_split(
    source_path: Path,
    filtered_path: Path,
    *,
    max_diffs: int,
) -> dict[str, Any]:
    dropped: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    expected_hash = hashlib.sha256()
    actual_hash = hashlib.sha256()
    differences: list[dict[str, Any]] = []

    for row_index, pair in enumerate(
        zip_longest(
            expected_rows(source_path, dropped=dropped),
            jsonl_rows(filtered_path),
            fillvalue=MISSING,
        ),
        start=1,
    ):
        expected_item, actual_item = pair
        counts["compared_positions"] += 1
        if expected_item is MISSING:
            counts["extra_filtered_rows"] += 1
            if len(differences) < max_diffs:
                differences.append({"row": row_index, "kind": "extra_filtered_row"})
            continue
        if actual_item is MISSING:
            counts["missing_filtered_rows"] += 1
            if len(differences) < max_diffs:
                differences.append({"row": row_index, "kind": "missing_filtered_row"})
            continue

        source_line, expected = expected_item
        filtered_line, actual = actual_item
        expected_bytes = canonical_bytes(expected)
        actual_bytes = canonical_bytes(actual)
        expected_hash.update(expected_bytes + b"\n")
        actual_hash.update(actual_bytes + b"\n")
        counts[f"task:{expected.get('task', 'missing')}"] += 1

        if expected != actual:
            counts["mismatched_rows"] += 1
            if len(differences) < max_diffs:
                differences.append(
                    {
                        "row": row_index,
                        "source_line": source_line,
                        "filtered_line": filtered_line,
                        "kind": "field_mismatch",
                        "differing_keys": sorted(
                            key
                            for key in set(expected) | set(actual)
                            if expected.get(key) != actual.get(key)
                        ),
                    }
                )

    failed = any(
        counts[key]
        for key in ("extra_filtered_rows", "missing_filtered_rows", "mismatched_rows")
    )
    return {
        "status": "failed" if failed else "passed",
        "source_path": source_path.as_posix(),
        "filtered_path": filtered_path.as_posix(),
        "counts": dict(sorted(counts.items())),
        "dropped_source_rows": dict(sorted(dropped.items())),
        "expected_sha256": expected_hash.hexdigest(),
        "actual_sha256": actual_hash.hexdigest(),
        "differences": differences,
    }


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {"splits": {}}
    failed = False
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        result = compare_split(
            args.source_dir / f"{split}.jsonl",
            args.filtered_dir / f"{split}.jsonl",
            max_diffs=args.max_diffs,
        )
        report["splits"][split] = result
        failed = failed or result["status"] != "passed"
        print(json.dumps({split: result}, ensure_ascii=False, indent=2), flush=True)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
