#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ("frmt", "golden")
DEFAULT_VIEWS = ("translation_only", "encoder_unified", "decoder_unified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build task-filtered final eval files. Translation rows are preserved. "
            "Classification_noequal rows remove all equal-pair/equal-label rows."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/encoder_decoder/t5gemma2/final_eval"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def normalized_label(row: dict[str, Any]) -> str:
    text = str(row.get("target_text") or row.get("label") or row.get("gold") or "")
    text = " ".join(text.split()).casefold()
    if text.startswith("<pt-br>") or text in {"pt-br", "br"}:
        return "pt-br"
    if text.startswith("<pt-pt>") or text in {"pt-pt", "pt"}:
        return "pt-pt"
    if text in {"equal", "igual", "same", "shared"}:
        return "equal"
    return text


def is_equal_classification(row: dict[str, Any]) -> bool:
    if row.get("task") != "classification":
        return False
    return bool(row.get("is_equal_pair")) or normalized_label(row) == "equal"


def build_view(input_path: Path) -> dict[str, Any]:
    out_dir = input_path.parent
    translation_path = out_dir / "translation_test.jsonl"
    classification_path = out_dir / "classification_test.jsonl"
    classification_noequal_path = out_dir / "classification_noequal_test.jsonl"

    counts: Counter[str] = Counter()
    with (
        translation_path.open("w", encoding="utf-8") as translation_fh,
        classification_path.open("w", encoding="utf-8") as classification_fh,
        classification_noequal_path.open("w", encoding="utf-8") as classification_noequal_fh,
    ):
        for row in iter_jsonl(input_path):
            task = row.get("task")
            if task == "translation":
                translation_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["translation"] += 1
            elif task == "classification":
                classification_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["classification"] += 1
                if is_equal_classification(row):
                    counts["classification_equal_removed"] += 1
                else:
                    classification_noequal_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts["classification_noequal"] += 1
            else:
                counts[f"skipped_unknown_task:{task}"] += 1

    return {
        "input": input_path.as_posix(),
        "translation_test": translation_path.as_posix(),
        "classification_test": classification_path.as_posix(),
        "classification_noequal_test": classification_noequal_path.as_posix(),
        "counts": dict(counts),
    }


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {"root": args.root.as_posix(), "views": {}}
    for dataset in args.datasets:
        for view in args.views:
            view_dir = args.root / dataset / view
            input_path = view_dir / f"{args.split}.jsonl"
            if not input_path.exists():
                fallback = view_dir / "valid.jsonl"
                if fallback.exists():
                    input_path = fallback
                else:
                    raise FileNotFoundError(input_path)
            key = f"{dataset}/{view}"
            report["views"][key] = build_view(input_path)
            print(json.dumps({key: report["views"][key]}, ensure_ascii=False))
    report_path = args.root / "task_filtered_eval_files_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
