#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_TOKENS = ("<pt-br>", "<pt-pt>")
PREFIX_RE = re.compile(r"^\s*<([^>]+)>\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare final-protocol rendered datasets against the source manifests "
            "used to build them. This checks row counts by task, direction, label, "
            "dataset, and equal-pair handling before launching training."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--translation-only-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--decoder-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--out-report", type=Path, default=None)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def normalize_space(text: object) -> str:
    return " ".join(str(text or "").split())


def strip_prefix(text: object) -> tuple[str | None, str]:
    raw = normalize_space(text)
    match = PREFIX_RE.match(raw)
    if not match:
        return None, raw
    return match.group(1).strip().casefold(), normalize_space(raw[match.end() :])


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


def source_split_path(source_dir: Path, split: str, task: str) -> Path:
    path = source_dir / f"{task}_{split}.jsonl"
    if path.exists():
        return path
    if split == "train":
        alt = source_dir / f"{task}_train.jsonl"
    elif split == "valid":
        alt = source_dir / f"{task}_valid.jsonl"
    else:
        alt = source_dir / f"{task}_{split}.jsonl"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Missing {task} {split} file in {source_dir}")


def final_split_path(view_dir: Path, split: str) -> Path:
    path = view_dir / f"{split}.jsonl"
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing final split {split!r} in {view_dir}")


def normalize_direction(row: dict[str, Any]) -> str:
    raw = normalize_space(row.get("direction") or row.get("task")).casefold()
    if raw in {"translate_br2pt", "br2pt", "br-pt"}:
        return "br2pt"
    if raw in {"translate_pt2br", "pt2br", "pt-br"}:
        return "pt2br"

    prefix, _ = strip_prefix(row.get("input_text") or row.get("source_text") or "")
    if prefix == "br-pt":
        return "br2pt"
    if prefix == "pt-br":
        return "pt2br"
    return raw or "unknown"


def clean_source_text(row: dict[str, Any]) -> str:
    _, clean_input = strip_prefix(row.get("input_text") or row.get("source_text") or "")
    return clean_input


def clean_target_text(row: dict[str, Any]) -> str:
    _, clean_target = strip_prefix(row.get("target_text") or "")
    return clean_target


def is_equal_translation_row(row: dict[str, Any]) -> bool:
    return bool(row.get("is_equal_pair", False)) or clean_source_text(row) == clean_target_text(row)


def source_token_from_translation(row: dict[str, Any]) -> str:
    direction = normalize_direction(row)
    if direction == "br2pt":
        return "<pt-br>"
    if direction == "pt2br":
        return "<pt-pt>"
    token = normalize_space(row.get("source_variant_label"))
    if token in SOURCE_TOKENS:
        return token
    return "unknown"


def normalize_class_label(row: dict[str, Any]) -> str:
    raw = normalize_space(
        row.get("target_text", row.get("label", row.get("gold", "")))
    ).casefold()
    if raw in {"<pt-br>", "pt-br", "br", "brasil", "brasileiro"}:
        return "<pt-br>"
    if raw in {"<pt-pt>", "pt-pt", "pt", "portugal", "europeu"}:
        return "<pt-pt>"
    if raw in {"equal", "igual", "same", "shared"}:
        return "equal"
    return raw or "unknown"


def final_task(row: dict[str, Any]) -> str:
    return normalize_space(row.get("task")).casefold() or "unknown"


def final_source_token(row: dict[str, Any]) -> str:
    explicit = normalize_space(row.get("source_variant_label"))
    if explicit in SOURCE_TOKENS:
        return explicit
    first = normalize_space(row.get("target_text")).split(" ", 1)[0]
    if first in SOURCE_TOKENS:
        return first
    first = normalize_space(row.get("input_text")).split(" ", 1)[0]
    if first in SOURCE_TOKENS:
        return first
    return "unknown"


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def summarize_source(
    translation_rows: list[dict[str, Any]],
    classification_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    translation_direction = Counter(normalize_direction(row) for row in translation_rows)
    translation_source = Counter(source_token_from_translation(row) for row in translation_rows)
    translation_dataset = Counter(normalize_space(row.get("dataset")) or "unknown" for row in translation_rows)
    translation_equal = sum(1 for row in translation_rows if is_equal_translation_row(row))

    raw_labels = Counter(normalize_class_label(row) for row in classification_rows)
    expected_labels: Counter[str] = Counter()
    expected_equal_rows = 0
    for row in classification_rows:
        label = normalize_class_label(row)
        if label == "equal":
            expected_labels["<pt-br>"] += 1
            expected_labels["<pt-pt>"] += 1
            expected_equal_rows += 2
        elif label in SOURCE_TOKENS:
            expected_labels[label] += 1

    classification_dataset = Counter(
        normalize_space(row.get("dataset")) or "unknown" for row in classification_rows
    )
    return {
        "translation": {
            "rows": len(translation_rows),
            "direction": counter_dict(translation_direction),
            "source_variant_label": counter_dict(translation_source),
            "dataset": counter_dict(translation_dataset),
            "equal_rows": translation_equal,
        },
        "classification_source": {
            "rows": len(classification_rows),
            "raw_label": counter_dict(raw_labels),
            "dataset": counter_dict(classification_dataset),
        },
        "classification_expected_final": {
            "rows": sum(expected_labels.values()),
            "target_text": counter_dict(expected_labels),
            "equal_rows": expected_equal_rows,
        },
    }


def summarize_final(rows: list[dict[str, Any]]) -> dict[str, Any]:
    translations = [row for row in rows if final_task(row) == "translation"]
    classifications = [row for row in rows if final_task(row) == "classification"]
    return {
        "rows": len(rows),
        "task": counter_dict(Counter(final_task(row) for row in rows)),
        "translation": {
            "rows": len(translations),
            "direction": counter_dict(Counter(normalize_direction(row) for row in translations)),
            "source_variant_label": counter_dict(
                Counter(final_source_token(row) for row in translations)
            ),
            "dataset": counter_dict(
                Counter(normalize_space(row.get("dataset")) or "unknown" for row in translations)
            ),
            "equal_rows": sum(1 for row in translations if bool(row.get("is_equal_pair", False))),
        },
        "classification": {
            "rows": len(classifications),
            "target_text": counter_dict(
                Counter(normalize_space(row.get("target_text")) for row in classifications)
            ),
            "dataset": counter_dict(
                Counter(normalize_space(row.get("dataset")) or "unknown" for row in classifications)
            ),
            "equal_rows": sum(1 for row in classifications if bool(row.get("is_equal_pair", False))),
        },
    }


def compare_split(source: dict[str, Any], final_views: dict[str, dict[str, Any]]) -> list[str]:
    mismatches: list[str] = []
    expected_translation = source["translation"]
    expected_classification = source["classification_expected_final"]

    for view_name in ("translation_only", "encoder", "decoder"):
        actual_translation = final_views[view_name]["translation"]
        for key in ("rows", "direction", "source_variant_label", "dataset", "equal_rows"):
            if actual_translation[key] != expected_translation[key]:
                mismatches.append(
                    f"{view_name}: translation {key} expected={expected_translation[key]} "
                    f"actual={actual_translation[key]}"
                )

    if final_views["translation_only"]["classification"]["rows"] != 0:
        mismatches.append("translation_only: contains classification rows")

    for view_name in ("encoder", "decoder"):
        actual_classification = final_views[view_name]["classification"]
        for key in ("rows", "target_text", "equal_rows"):
            if actual_classification[key] != expected_classification[key]:
                mismatches.append(
                    f"{view_name}: classification {key} expected={expected_classification[key]} "
                    f"actual={actual_classification[key]}"
                )

    return mismatches


def print_human_summary(split: str, split_report: dict[str, Any]) -> None:
    source = split_report["source"]
    print(f"\n== {split} ==")
    print("source translation:", source["translation"])
    print("source classification raw:", source["classification_source"])
    print("expected final classification:", source["classification_expected_final"])
    for view_name, view in split_report["final_views"].items():
        print(f"{view_name} final tasks:", view["task"])
        print(f"{view_name} final translation:", view["translation"])
        if view_name != "translation_only":
            print(f"{view_name} final classification:", view["classification"])
    if split_report["mismatches"]:
        print("mismatches:")
        for item in split_report["mismatches"]:
            print(f"  - {item}")
    else:
        print("mismatches: none")


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "source_dir": args.source_dir.as_posix(),
        "translation_only_dir": args.translation_only_dir.as_posix(),
        "encoder_dir": args.encoder_dir.as_posix(),
        "decoder_dir": args.decoder_dir.as_posix(),
        "splits": {},
    }
    all_mismatches: list[str] = []

    for split in args.splits:
        translation_rows = read_jsonl(source_split_path(args.source_dir, split, "translation"))
        classification_rows = read_jsonl(source_split_path(args.source_dir, split, "classification"))
        source_summary = summarize_source(translation_rows, classification_rows)
        final_views = {
            "translation_only": summarize_final(read_jsonl(final_split_path(args.translation_only_dir, split))),
            "encoder": summarize_final(read_jsonl(final_split_path(args.encoder_dir, split))),
            "decoder": summarize_final(read_jsonl(final_split_path(args.decoder_dir, split))),
        }
        mismatches = compare_split(source_summary, final_views)
        all_mismatches.extend(f"{split}: {item}" for item in mismatches)
        split_report = {
            "source": source_summary,
            "final_views": final_views,
            "mismatches": mismatches,
        }
        report["splits"][split] = split_report
        print_human_summary(split, split_report)

    report["status"] = "failed" if all_mismatches else "passed"
    report["mismatches"] = all_mismatches

    if args.out_report is not None:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote report: {args.out_report}")

    if all_mismatches and args.fail_on_mismatch:
        sys.exit(1)


if __name__ == "__main__":
    main()
