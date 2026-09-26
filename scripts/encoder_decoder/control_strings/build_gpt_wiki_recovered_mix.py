#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate canonical GPT-Wikipedia translation/classification splits "
            "in recovered E order while dropping equal classification rows."
        )
    )
    parser.add_argument("--translation-train", type=Path, required=True)
    parser.add_argument("--translation-valid", type=Path, required=True)
    parser.add_argument("--classification-train", type=Path, required=True)
    parser.add_argument("--classification-valid", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-source-dir-name", default="stageB_gpt_wiki")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def normalize_label(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get("target_text", row.get("label", row.get("gold", "")))).split()
    ).casefold()


def write_split(
    translation_path: Path,
    classification_path: Path,
    output_path: Path,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as out_fh:
        for row in iter_jsonl(translation_path):
            out = dict(row)
            existing_dataset = str(out.get("dataset") or "").casefold()
            if "frmt" in existing_dataset:
                raise RuntimeError(f"FRMT row found in canonical GPT-Wikipedia file: {row}")
            out["dataset"] = out.get("dataset") or "GPT_WIKIPEDIA"
            out_fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            counts["translation"] += 1
        for row in iter_jsonl(classification_path):
            label = normalize_label(row)
            if label in {"equal", "igual", "same", "shared"}:
                counts["classification_equal_dropped"] += 1
                continue
            if label not in {"pt-br", "pt-pt", "<pt-br>", "<pt-pt>", "br", "pt"}:
                raise RuntimeError(f"Unexpected GPT-Wikipedia classification label: {row}")
            out = dict(row)
            existing_dataset = str(out.get("dataset") or "").casefold()
            if "frmt" in existing_dataset:
                raise RuntimeError(f"FRMT row found in canonical GPT-Wikipedia file: {row}")
            out["dataset"] = out.get("dataset") or "GPT_WIKIPEDIA"
            out_fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            counts["classification"] += 1
    if not counts["translation"] or not counts["classification"]:
        raise RuntimeError(f"Missing task in GPT-Wikipedia split: {dict(counts)}")
    temp_path.replace(output_path)
    return dict(counts)


def main() -> None:
    args = parse_args()
    inputs = (
        args.translation_train,
        args.translation_valid,
        args.classification_train,
        args.classification_valid,
    )
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.parent.name != args.expected_source_dir_name:
            raise RuntimeError(
                f"Expected canonical source directory {args.expected_source_dir_name!r}, "
                f"got {path.parent}"
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source_dataset": "GPT_WIKIPEDIA",
        "source_paths": [str(path) for path in inputs],
        "equal_classification_policy": "drop",
        "row_order": "all translation rows, then all classification rows",
        "train": write_split(
            args.translation_train,
            args.classification_train,
            args.out_dir / "train.jsonl",
        ),
        "valid": write_split(
            args.translation_valid,
            args.classification_valid,
            args.out_dir / "valid.jsonl",
        ),
    }
    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
