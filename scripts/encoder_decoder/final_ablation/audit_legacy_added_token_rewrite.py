#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator


LABEL_MAP = {
    "pt-br": "<pt-br>",
    "br": "<pt-br>",
    "pt-pt": "<pt-pt>",
    "pt": "<pt-pt>",
}
MISSING = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-compare legacy with-cls JSONL data with its added-token rewrite. "
            "All fields must match after rewriting classification targets."
        )
    )
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--added-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument("--drop-equal", action="store_true")
    parser.add_argument("--max-diffs", type=int, default=3)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def normalize(value: Any) -> str:
    return " ".join(str(value or "").split())


def classification_target(row: dict[str, Any]) -> str | None:
    raw = row.get("target_text", row.get("label", row.get("gold")))
    text = normalize(raw).lower()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    if text == "equal":
        return "equal"
    return None


def rewritten_legacy_rows(
    path: Path,
    *,
    drop_equal: bool,
    skipped: Counter[str],
) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = normalize(row.get("task")).lower()
            if task == "classification":
                mapped = classification_target(row)
                if mapped is None:
                    skipped["classification_unknown"] += 1
                    continue
                if mapped == "equal" and drop_equal:
                    skipped["classification_equal"] += 1
                    continue
                row["target_text"] = mapped
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
    legacy_path: Path,
    added_path: Path,
    *,
    drop_equal: bool,
    max_diffs: int,
) -> dict[str, Any]:
    skipped: Counter[str] = Counter()
    expected_hash = hashlib.sha256()
    actual_hash = hashlib.sha256()
    counts: Counter[str] = Counter()
    differences: list[dict[str, Any]] = []

    expected_iter = rewritten_legacy_rows(
        legacy_path,
        drop_equal=drop_equal,
        skipped=skipped,
    )
    actual_iter = jsonl_rows(added_path)

    for row_index, pair in enumerate(
        zip_longest(expected_iter, actual_iter, fillvalue=MISSING),
        start=1,
    ):
        expected_item, actual_item = pair
        counts["compared_positions"] += 1
        if expected_item is MISSING:
            counts["extra_added_rows"] += 1
            if len(differences) < max_diffs:
                differences.append({"row": row_index, "kind": "extra_added_row"})
            continue
        if actual_item is MISSING:
            counts["missing_added_rows"] += 1
            if len(differences) < max_diffs:
                differences.append({"row": row_index, "kind": "missing_added_row"})
            continue

        legacy_line, expected = expected_item
        added_line, actual = actual_item
        expected_bytes = canonical_bytes(expected)
        actual_bytes = canonical_bytes(actual)
        expected_hash.update(expected_bytes + b"\n")
        actual_hash.update(actual_bytes + b"\n")
        counts[f"task:{normalize(expected.get('task')).lower() or 'missing'}"] += 1

        if expected != actual:
            counts["mismatched_rows"] += 1
            if len(differences) < max_diffs:
                differing_keys = sorted(
                    key
                    for key in set(expected) | set(actual)
                    if expected.get(key) != actual.get(key)
                )
                differences.append(
                    {
                        "row": row_index,
                        "legacy_line": legacy_line,
                        "added_line": added_line,
                        "kind": "field_mismatch",
                        "differing_keys": differing_keys,
                    }
                )

    failed = any(
        counts[key]
        for key in ("extra_added_rows", "missing_added_rows", "mismatched_rows")
    )
    return {
        "status": "failed" if failed else "passed",
        "legacy_path": legacy_path.as_posix(),
        "added_path": added_path.as_posix(),
        "counts": dict(sorted(counts.items())),
        "skipped_legacy_rows": dict(sorted(skipped.items())),
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
            args.legacy_dir / f"{split}.jsonl",
            args.added_dir / f"{split}.jsonl",
            drop_equal=bool(args.drop_equal),
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
