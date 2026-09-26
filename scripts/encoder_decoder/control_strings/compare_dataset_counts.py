#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare control-string dataset counts with recovered E/D datasets."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
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


def normalized_task(row: dict[str, Any]) -> str:
    task = str(row.get("task") or "").strip().casefold()
    if task in {"translation", "translate_br2pt", "translate_pt2br"}:
        return "translation"
    if task in {"classification", "classify"}:
        return "classification"
    raise ValueError(f"Unsupported task: {task!r}")


def is_equal(row: dict[str, Any], task: str) -> bool:
    if bool(row.get("is_equal_pair")):
        return True
    if task == "classification":
        label = str(row.get("target_text", row.get("label", row.get("gold", "")))).strip().casefold()
        return label in {"equal", "igual", "same", "shared"}
    return False


def count_file(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    for row in iter_jsonl(path):
        task = normalized_task(row)
        counts["total"] += 1
        counts[task] += 1
        if is_equal(row, task):
            counts[f"{task}_equal"] += 1
        dataset = str(row.get("dataset") or "UNKNOWN").strip() or "UNKNOWN"
        datasets[dataset] += 1
    return {"path": str(path), "counts": dict(counts), "datasets": dict(datasets)}


def count_pair(root: Path) -> dict[str, Any]:
    return {
        "train": count_file(root / "train.jsonl"),
        "valid": count_file(root / "valid.jsonl"),
    }


def core_counts(payload: dict[str, Any], split: str) -> dict[str, int]:
    counts = payload[split]["counts"]
    return {
        key: int(counts.get(key, 0))
        for key in (
            "total",
            "translation",
            "classification",
            "translation_equal",
            "classification_equal",
        )
    }


def delta(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: left[key] - right[key] for key in left}


def has_blocking_delta(
    values: dict[str, int],
    *,
    ignore_keys: set[str] | None = None,
) -> bool:
    ignore_keys = ignore_keys or set()
    return any(value for key, value in values.items() if key not in ignore_keys)


def main() -> None:
    args = parse_args()
    specifications = {
        "gpt_wiki": {
            "new_e": args.control_root / "stageB_gpt_wiki" / "encoder_unified",
            "new_d": args.control_root / "stageB_gpt_wiki" / "decoder_unified",
            "recovered_e": args.data_root / "stageB_gpt_wiki_translation_plus_cls_noequal",
            "recovered_d": args.data_root / "stageB_gpt_wiki_label_first_with_cls_noequal",
        },
        "gpt_wiki_frmt": {
            "new_e": args.control_root / "stageB_gpt_wiki_frmt" / "encoder_unified",
            "new_d": args.control_root / "stageB_gpt_wiki_frmt" / "decoder_unified",
            "recovered_e": args.data_root / "stageB_gpt_wiki_frmt_translation_plus_cls_noequal",
            "recovered_d": args.data_root / "stageB_gpt_wiki_frmt_label_first_noequal",
        },
    }
    report: dict[str, Any] = {"status": "passed", "variants": {}}
    csv_rows: list[dict[str, Any]] = []
    for variant, paths in specifications.items():
        payloads = {name: count_pair(path) for name, path in paths.items()}
        comparisons: dict[str, Any] = {}
        for split in ("train", "valid"):
            new_e = core_counts(payloads["new_e"], split)
            new_d = core_counts(payloads["new_d"], split)
            recovered_e = core_counts(payloads["recovered_e"], split)
            recovered_d = core_counts(payloads["recovered_d"], split)
            comparisons[split] = {
                "new_e_minus_new_d": delta(new_e, new_d),
                "new_e_minus_recovered_e": delta(new_e, recovered_e),
                "new_d_minus_recovered_d": delta(new_d, recovered_d),
            }
            if has_blocking_delta(comparisons[split]["new_e_minus_new_d"]):
                report["status"] = "review_required"
            if has_blocking_delta(
                comparisons[split]["new_e_minus_recovered_e"],
                ignore_keys={"translation_equal"},
            ):
                report["status"] = "review_required"
        report["variants"][variant] = {
            "datasets": payloads,
            "comparisons": comparisons,
            "note": (
                "A non-zero new D minus recovered D delta may be expected because "
                "the new D retains equal translation rows. A non-zero "
                "new E minus recovered E translation_equal delta is diagnostic "
                "only because recovered E files do not consistently preserve "
                "the equal-pair metadata flag."
            ),
        }
        for split in ("train", "valid"):
            baselines = {"new_e": "recovered_e", "new_d": "recovered_d"}
            for name in ("new_e", "new_d", "recovered_e", "recovered_d"):
                counts = core_counts(payloads[name], split)
                baseline_name = baselines.get(name, name)
                baseline = core_counts(payloads[baseline_name], split)
                row: dict[str, Any] = {
                    "variant": variant,
                    "split": split,
                    "model_data": name,
                    **counts,
                }
                row.update({f"delta_{key}": counts[key] - baseline[key] for key in counts})
                csv_rows.append(row)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.report.with_suffix(".csv")
    fieldnames = [
        "variant",
        "split",
        "model_data",
        "total",
        "translation",
        "classification",
        "translation_equal",
        "classification_equal",
        "delta_total",
        "delta_translation",
        "delta_classification",
        "delta_translation_equal",
        "delta_classification_equal",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved compact comparison: {csv_path}")


if __name__ == "__main__":
    main()
