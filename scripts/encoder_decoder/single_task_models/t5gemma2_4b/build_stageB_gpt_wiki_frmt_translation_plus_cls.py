#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


TASK_PREFIX_RE = re.compile(r"^\s*<([^>]+)>\s*", flags=re.IGNORECASE)
VALID_MIN_ROWS = 200
VALID_TOPUP_SEED = 42


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description=(
            "Build a single-task Stage B dataset that mixes translation rows with "
            "classification rows for the standard seq2seq trainer."
        )
    )
    parser.add_argument(
        "--translation-train",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageB_gpt_wiki_frmt_mix" / "translation_train.jsonl",
    )
    parser.add_argument(
        "--translation-valid",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageB_gpt_wiki_frmt_mix" / "translation_valid.jsonl",
    )
    parser.add_argument(
        "--classification-train",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageB_gpt_wiki_frmt_mix" / "classification_train.jsonl",
    )
    parser.add_argument(
        "--classification-valid",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageB_gpt_wiki_frmt_mix" / "classification_valid.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root
        / "data"
        / "encoder_decoder"
        / "t5gemma2"
        / "compare_staged_v2"
        / "stageB_gpt_wiki_frmt_translation_plus_cls_noequal",
    )
    parser.add_argument(
        "--cls-prefix",
        default="<cls>",
        help="Encoder-side prefix used to mark classification examples.",
    )
    parser.add_argument(
        "--keep-equal",
        action="store_true",
        help="Keep equal classification rows. Default is to drop them.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def strip_encoder_prefix(text: str) -> str:
    return normalize_space(TASK_PREFIX_RE.sub("", text or ""))


def normalize_label(raw: Any) -> str | None:
    text = normalize_space(str(raw or "")).lower()
    if text in {"pt-br", "br"}:
        return "pt-br"
    if text in {"pt-pt", "pt"}:
        return "pt-pt"
    if text == "equal":
        return "equal"
    if "brasil" in text:
        return "pt-br"
    if "europeu" in text or "portugal" in text:
        return "pt-pt"
    return None


def normalize_direction(row: dict[str, Any]) -> str:
    for key in ("direction", "task"):
        value = normalize_space(str(row.get(key) or ""))
        if value:
            return value
    return "translation"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def convert_translation_row(row: dict[str, Any]) -> dict[str, Any] | None:
    input_text = normalize_space(str(row.get("input_text") or ""))
    target_text = normalize_space(str(row.get("target_text") or ""))
    if not input_text or not target_text:
        return None
    return {
        "input_text": input_text,
        "target_text": target_text,
        "task": "translation",
        "dataset": row.get("dataset"),
        "bucket": row.get("bucket"),
        "direction": normalize_direction(row),
    }


def convert_classification_row(
    row: dict[str, Any],
    *,
    cls_prefix: str,
    keep_equal: bool,
) -> dict[str, Any] | None:
    source = strip_encoder_prefix(str(row.get("input_text") or row.get("source_text") or ""))
    raw_label = row.get("target_text", row.get("label", row.get("gold")))
    label = normalize_label(raw_label)
    if not source or label is None:
        return None
    if label == "equal" and not keep_equal:
        return None
    return {
        "input_text": f"{cls_prefix} {source}".strip(),
        "target_text": label,
        "task": "classification",
        "dataset": row.get("dataset"),
        "bucket": row.get("bucket"),
        "direction": "classification",
    }


def write_mixed_split(
    *,
    translation_path: Path,
    classification_path: Path,
    out_path: Path,
    cls_prefix: str,
    keep_equal: bool,
) -> dict[str, Any]:
    counts = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_fh:
        for row in iter_jsonl(translation_path):
            converted = convert_translation_row(row)
            if converted is None:
                counts["translation_skipped"] += 1
                continue
            out_fh.write(json.dumps(converted, ensure_ascii=False) + "\n")
            counts["translation_written"] += 1
            counts[f"translation_dataset:{normalize_space(str(converted.get('dataset') or 'UNKNOWN'))}"] += 1

        for row in iter_jsonl(classification_path):
            converted = convert_classification_row(
                row,
                cls_prefix=cls_prefix,
                keep_equal=keep_equal,
            )
            if converted is None:
                raw_label = normalize_label(row.get("target_text", row.get("label", row.get("gold"))))
                if raw_label == "equal":
                    counts["classification_skipped_equal"] += 1
                else:
                    counts["classification_skipped_invalid"] += 1
                continue
            out_fh.write(json.dumps(converted, ensure_ascii=False) + "\n")
            counts["classification_written"] += 1
            counts[f"classification_dataset:{normalize_space(str(converted.get('dataset') or 'UNKNOWN'))}"] += 1
            counts[f"classification_label:{converted['target_text']}"] += 1

    counts["total_written"] = counts["translation_written"] + counts["classification_written"]
    return dict(counts)


def top_up_validation_from_train(
    *,
    train_path: Path,
    valid_path: Path,
    existing_count: int,
    min_rows: int,
    seed: int,
) -> dict[str, Any]:
    if min_rows <= 0 or existing_count >= min_rows:
        return {"added_rows": 0}

    sample_size = min_rows - existing_count
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for row in iter_jsonl(train_path):
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(row)
            continue
        idx = rng.randrange(seen)
        if idx < sample_size:
            reservoir[idx] = row

    counts: Counter[str] = Counter()
    with valid_path.open("a", encoding="utf-8") as out_fh:
        for row in reservoir:
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["added_rows"] += 1
            task = normalize_space(str(row.get("task") or "")).casefold()
            dataset = normalize_space(str(row.get("dataset") or "UNKNOWN"))
            if task == "classification":
                counts["classification_written"] += 1
                counts[f"classification_dataset:{dataset}"] += 1
                counts[f"classification_label:{normalize_space(str(row.get('target_text') or ''))}"] += 1
            else:
                counts["translation_written"] += 1
                counts[f"translation_dataset:{dataset}"] += 1

    counts["total_written"] = counts["translation_written"] + counts["classification_written"]
    return dict(counts)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_stats = write_mixed_split(
        translation_path=args.translation_train,
        classification_path=args.classification_train,
        out_path=args.out_dir / "train.jsonl",
        cls_prefix=args.cls_prefix,
        keep_equal=bool(args.keep_equal),
    )
    valid_stats = write_mixed_split(
        translation_path=args.translation_valid,
        classification_path=args.classification_valid,
        out_path=args.out_dir / "valid.jsonl",
        cls_prefix=args.cls_prefix,
        keep_equal=bool(args.keep_equal),
    )
    valid_topup = top_up_validation_from_train(
        train_path=args.out_dir / "train.jsonl",
        valid_path=args.out_dir / "valid.jsonl",
        existing_count=int(valid_stats.get("total_written", 0)),
        min_rows=VALID_MIN_ROWS,
        seed=VALID_TOPUP_SEED,
    )
    if valid_topup.get("added_rows", 0):
        for key, value in valid_topup.items():
            valid_stats[key] = int(valid_stats.get(key, 0)) + int(value)

    report = {
        "train": train_stats,
        "valid": valid_stats,
        "valid_topup_from_train": valid_topup,
        "cls_prefix": args.cls_prefix,
        "keep_equal": bool(args.keep_equal),
        "translation_train": args.translation_train.as_posix(),
        "translation_valid": args.translation_valid.as_posix(),
        "classification_train": args.classification_train.as_posix(),
        "classification_valid": args.classification_valid.as_posix(),
        "out_dir": args.out_dir.as_posix(),
    }
    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
