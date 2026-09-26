#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PREFIX_RE = re.compile(
    r"^\s*(?:<br-pt>|<pt-br>|<pt-pt>|<id>|<cls>|BR\b|PT\b|CLS\b|pt-br\b|pt-pt\b)(?:\s*:\s*|\s+)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two dataset directories after normalizing legacy and final "
            "task-control strings. Use this to verify that old rendered datasets "
            "and rebuilt final source manifests contain the same examples."
        )
    )
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("translation", "classification"),
        default=["translation", "classification"],
        help="Tasks to compare after normalization.",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def normalize_space(text: object) -> str:
    return " ".join(str(text or "").split())


def strip_control_prefix(text: object) -> str:
    raw = normalize_space(text)
    while True:
        match = PREFIX_RE.match(raw)
        if not match:
            return raw
        raw = normalize_space(raw[match.end() :])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    return rows


def existing_split_files(dataset_dir: Path, split: str) -> list[Path]:
    candidates = [
        dataset_dir / f"{split}.jsonl",
        dataset_dir / f"translation_{split}.jsonl",
        dataset_dir / f"classification_{split}.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def load_split_rows(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    paths = existing_split_files(dataset_dir, split)
    if not paths:
        raise FileNotFoundError(f"No JSONL split files found for {split!r} in {dataset_dir}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        task_hint = "classification" if path.name.startswith("classification_") else "translation"
        for row in read_jsonl(path):
            row = dict(row)
            row.setdefault("_source_file", path.name)
            row.setdefault("_task_hint", task_hint)
            rows.append(row)
    return rows


def normalize_task(row: dict[str, Any]) -> str:
    raw = normalize_space(row.get("task") or row.get("direction") or row.get("_task_hint")).casefold()
    if raw in {"classification", "classify"}:
        return "classification"
    if raw in {"translate_br2pt", "translate_pt2br", "br2pt", "pt2br", "br-pt", "pt-br"}:
        return "translation"

    source_file = normalize_space(row.get("_source_file")).casefold()
    if source_file.startswith("classification_"):
        return "classification"
    if source_file.startswith("translation_"):
        return "translation"

    target = normalize_space(row.get("target_text")).casefold()
    stripped_target = strip_control_prefix(target)
    if stripped_target in {"pt-br", "pt-pt", "equal", "igual", "<pt-br>", "<pt-pt>"}:
        return "classification"
    return "translation"


def normalize_direction(row: dict[str, Any]) -> str:
    raw = normalize_space(row.get("direction") or row.get("task")).casefold()
    if raw in {"translate_br2pt", "br2pt", "br-pt"}:
        return "br2pt"
    if raw in {"translate_pt2br", "pt2br", "pt-br"}:
        return "pt2br"
    input_text = normalize_space(row.get("input_text") or row.get("source_text"))
    lowered = input_text.casefold()
    if lowered.startswith("<br-pt>"):
        return "br2pt"
    return "unknown"


def normalize_label(text: object) -> str:
    raw = strip_control_prefix(text).casefold()
    if raw in {"br", "pt-br", "<pt-br>", "brasil", "brasileiro", "brazilian"}:
        return "pt-br"
    if raw in {"pt", "pt-pt", "<pt-pt>", "portugal", "europeu", "european"}:
        return "pt-pt"
    if raw in {"equal", "igual", "same", "shared"}:
        return "equal"
    return raw or "unknown"


def row_source_text(row: dict[str, Any]) -> str:
    return strip_control_prefix(
        row.get("source_text")
        or row.get("text")
        or row.get("input_text")
        or ""
    )


def row_target_text(row: dict[str, Any]) -> str:
    return strip_control_prefix(row.get("target_text") or "")


def translation_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        normalize_direction(row),
        row_source_text(row),
        row_target_text(row),
        normalize_space(row.get("dataset")) or "unknown",
    )


def classification_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        row_source_text(row),
        normalize_label(row.get("target_text", row.get("label", row.get("gold", "")))),
        normalize_space(row.get("dataset")) or "unknown",
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = Counter(normalize_task(row) for row in rows)
    translations = [row for row in rows if normalize_task(row) == "translation"]
    classifications = [row for row in rows if normalize_task(row) == "classification"]
    return {
        "rows": len(rows),
        "task": dict(sorted(tasks.items())),
        "translation_rows": len(translations),
        "classification_rows": len(classifications),
        "translation_direction": dict(
            sorted(Counter(normalize_direction(row) for row in translations).items())
        ),
        "classification_label": dict(
            sorted(
                Counter(
                    normalize_label(row.get("target_text", row.get("label", row.get("gold", ""))))
                    for row in classifications
                ).items()
            )
        ),
        "translation_counter": Counter(translation_key(row) for row in translations),
        "classification_counter": Counter(classification_key(row) for row in classifications),
    }


def sample_diff(left: Counter, right: Counter, *, limit: int = 5) -> dict[str, list[str]]:
    return {
        "missing_from_right": [repr(item) for item in list((left - right).elements())[:limit]],
        "extra_in_right": [repr(item) for item in list((right - left).elements())[:limit]],
    }


def serializable_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"translation_counter", "classification_counter"}
    }


def compare_split(
    *,
    split: str,
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    tasks: set[str],
) -> dict[str, Any]:
    left = summarize(left_rows)
    right = summarize(right_rows)
    mismatches: list[str] = []
    if "translation" in tasks and left["translation_counter"] != right["translation_counter"]:
        mismatches.append("translation examples differ")
    if "classification" in tasks and left["classification_counter"] != right["classification_counter"]:
        mismatches.append("classification examples differ")
    return {
        "split": split,
        "status": "failed" if mismatches else "passed",
        "mismatches": mismatches,
        "left": serializable_summary(left),
        "right": serializable_summary(right),
        "translation_diff": sample_diff(left["translation_counter"], right["translation_counter"]),
        "classification_diff": sample_diff(left["classification_counter"], right["classification_counter"]),
    }


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "left_dir": args.left_dir.as_posix(),
        "right_dir": args.right_dir.as_posix(),
        "splits": {},
    }
    failed = False

    for split in args.splits:
        split_report = compare_split(
            split=split,
            left_rows=load_split_rows(args.left_dir, split),
            right_rows=load_split_rows(args.right_dir, split),
            tasks=set(args.tasks),
        )
        report["splits"][split] = split_report
        failed = failed or split_report["status"] != "passed"

        print(f"\n== {split}: {split_report['status']} ==")
        print(f"{args.left_name}: {split_report['left']}")
        print(f"{args.right_name}: {split_report['right']}")
        if split_report["mismatches"]:
            print("mismatches:", split_report["mismatches"])
            print("translation diff:", split_report["translation_diff"])
            print("classification diff:", split_report["classification_diff"])

    report["status"] = "failed" if failed else "passed"
    if args.out_report is not None:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote report: {args.out_report}")

    if failed and args.fail_on_mismatch:
        sys.exit(1)


if __name__ == "__main__":
    main()
