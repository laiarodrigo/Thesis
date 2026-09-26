#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create decoder-unified control-string data with equal translation "
            "rows removed. Classification rows are preserved unchanged."
        )
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("data/encoder_decoder/t5gemma2/control_string_protocol"),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["stageA_opensubs_only", "stageB_gpt_wiki", "stageB_gpt_wiki_frmt"],
    )
    parser.add_argument("--source-subdir", default="decoder_unified")
    parser.add_argument("--target-subdir", default="decoder_unified_noequal_translation")
    parser.add_argument("--report-name", default="decoder_noequal_translation_report.json")
    return parser.parse_args()


def normalize(value: object) -> str:
    return " ".join(str(value or "").split())


def coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return normalize(value).casefold() in {"1", "true", "yes", "y"}


def task_kind(row: dict[str, Any]) -> str:
    task = normalize(row.get("task")).casefold()
    if task in {"translation", "classification"}:
        return task
    raise ValueError(f"Unsupported task: {row}")


def target_without_label(row: dict[str, Any]) -> str:
    target = normalize(row.get("target_text"))
    label = normalize(row.get("source_variant_label"))
    if label and target.startswith(f"{label} "):
        return normalize(target[len(label) :])
    return target


def is_equal_translation(row: dict[str, Any]) -> bool:
    if task_kind(row) != "translation":
        return False
    if coerce_bool(row.get("is_equal_pair")):
        return True
    source = normalize(row.get("input_text"))
    target = target_without_label(row)
    return bool(source and target and source == target)


def rewrite_split(source: Path, target: Path) -> dict[str, int]:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    counts: Counter[str] = Counter()
    with source.open(encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = task_kind(row)
            counts[f"input_{task}"] += 1
            if is_equal_translation(row):
                counts["dropped_translation_equal"] += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[f"kept_{task}"] += 1
            if task == "translation" and coerce_bool(row.get("is_equal_pair")):
                raise AssertionError(
                    f"Equal translation row was not removed from {source}:{line_no}"
                )
    tmp.replace(target)
    counts["input_total"] = counts["input_translation"] + counts["input_classification"]
    counts["kept_total"] = counts["kept_translation"] + counts["kept_classification"]
    if counts["kept_translation"] == 0 or counts["kept_classification"] == 0:
        raise AssertionError(f"Missing task after filtering {source}: {dict(counts)}")
    return dict(counts)


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "status": "passed",
        "source_subdir": args.source_subdir,
        "target_subdir": args.target_subdir,
        "equal_translation_policy": "drop",
        "classification_policy": "preserve unchanged",
        "stages": {},
    }
    for stage in args.stages:
        stage_root = args.control_root / stage
        source_root = stage_root / args.source_subdir
        target_root = stage_root / args.target_subdir
        stage_report: dict[str, Any] = {}
        for split in ("train", "valid"):
            stage_report[split] = rewrite_split(
                source_root / f"{split}.jsonl",
                target_root / f"{split}.jsonl",
            )
        report["stages"][stage] = stage_report
        stage_report_path = target_root / "build_report.json"
        stage_report_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "stage": stage,
                    "source": str(source_root),
                    "target": str(target_root),
                    "splits": stage_report,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    report_path = args.control_root / args.report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
