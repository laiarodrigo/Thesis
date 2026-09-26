#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit decoder-unified control-string data where equal translation "
            "rows are removed and classification rows are preserved."
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
    parser.add_argument("--report-name", default="decoder_noequal_translation_audit.json")
    parser.add_argument("--br-control", default="<pt-br>")
    parser.add_argument("--pt-control", default="<pt-pt>")
    return parser.parse_args()


def normalize(value: object) -> str:
    return " ".join(str(value or "").split())


def coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return normalize(value).casefold() in {"1", "true", "yes", "y"}


def task_kind(row: dict[str, Any]) -> str:
    task = normalize(row.get("task")).casefold()
    if task not in {"translation", "classification"}:
        raise AssertionError(f"Unsupported task: {task!r}")
    return task


def label(row: dict[str, Any], *, labels: set[str]) -> str:
    value = normalize(row.get("source_variant_label"))
    if value not in labels:
        raise AssertionError(f"Unsupported source_variant_label: {value!r}")
    return value


def target_without_label(row: dict[str, Any]) -> str:
    target = normalize(row.get("target_text"))
    row_label = normalize(row.get("source_variant_label"))
    if row_label and target.startswith(f"{row_label} "):
        return normalize(target[len(row_label) :])
    return target


def is_equal_translation(row: dict[str, Any]) -> bool:
    if task_kind(row) != "translation":
        return False
    if coerce_bool(row.get("is_equal_pair")):
        return True
    source = normalize(row.get("input_text"))
    target = target_without_label(row)
    return bool(source and target and source == target)


def canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def update_digest(digest: hashlib._Hash, row: dict[str, Any]) -> None:
    digest.update(canonical_row(row).encode("utf-8"))
    digest.update(b"\n")


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def audit_split(source: Path, target: Path, *, labels: set[str]) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    if not target.exists():
        raise FileNotFoundError(target)

    counts: Counter[str] = Counter()
    expected_digest = hashlib.sha256()
    actual_digest = hashlib.sha256()
    actual_iter = iter_jsonl(target)
    next_actual: tuple[int, dict[str, Any]] | None = next(actual_iter, None)

    for source_line, row in iter_jsonl(source):
        task = task_kind(row)
        label(row, labels=labels)
        counts[f"source_{task}"] += 1

        if task == "classification":
            if coerce_bool(row.get("is_equal_pair")):
                raise AssertionError(f"Equal classification row in source {source}:{source_line}")
            if row.get("loss_on_first_token_only") is not True:
                raise AssertionError(
                    f"D classification is not first-token-only in source {source}:{source_line}"
                )

        if is_equal_translation(row):
            counts["dropped_translation_equal"] += 1
            continue

        update_digest(expected_digest, row)
        if next_actual is None:
            raise AssertionError(f"Target ended early while expecting source row {source_line}")

        target_line, target_row = next_actual
        if canonical_row(target_row) != canonical_row(row):
            raise AssertionError(
                "Target row mismatch after filtering "
                f"{source}:{source_line} -> {target}:{target_line}"
            )
        if is_equal_translation(target_row):
            raise AssertionError(f"Equal translation row kept in target {target}:{target_line}")
        if task_kind(target_row) == "classification" and target_row.get("loss_on_first_token_only") is not True:
            raise AssertionError(
                f"D classification is not first-token-only in target {target}:{target_line}"
            )

        update_digest(actual_digest, target_row)
        counts[f"kept_{task}"] += 1
        next_actual = next(actual_iter, None)

    if next_actual is not None:
        target_line, _ = next_actual
        raise AssertionError(f"Target has extra row at {target}:{target_line}")

    counts["source_total"] = counts["source_translation"] + counts["source_classification"]
    counts["kept_total"] = counts["kept_translation"] + counts["kept_classification"]
    if counts["kept_translation"] == 0 or counts["kept_classification"] == 0:
        raise AssertionError(f"Missing task after filtering {target}: {dict(counts)}")
    if counts["kept_translation"] + counts["dropped_translation_equal"] != counts["source_translation"]:
        raise AssertionError(f"Translation accounting mismatch for {target}: {dict(counts)}")
    if counts["kept_classification"] != counts["source_classification"]:
        raise AssertionError(f"Classification accounting mismatch for {target}: {dict(counts)}")

    expected_hash = expected_digest.hexdigest()
    actual_hash = actual_digest.hexdigest()
    if expected_hash != actual_hash:
        raise AssertionError(f"Filtered row hash mismatch for {target}")

    return {
        **dict(counts),
        "filtered_row_hash": actual_hash,
        "classification_preserved": True,
        "equal_translation_removed": True,
        "decoder_classification_first_token_only": True,
    }


def main() -> None:
    args = parse_args()
    labels = {args.br_control, args.pt_control}
    report: dict[str, Any] = {
        "status": "passed",
        "source_subdir": args.source_subdir,
        "target_subdir": args.target_subdir,
        "br_control": args.br_control,
        "pt_control": args.pt_control,
        "stages": {},
    }
    for stage in args.stages:
        stage_root = args.control_root / stage
        source_root = stage_root / args.source_subdir
        target_root = stage_root / args.target_subdir
        stage_report: dict[str, Any] = {}
        for split in ("train", "valid"):
            stage_report[split] = audit_split(
                source_root / f"{split}.jsonl",
                target_root / f"{split}.jsonl",
                labels=labels,
            )
        report["stages"][stage] = stage_report
        (target_root / "audit_report.json").write_text(
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
