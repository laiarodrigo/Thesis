#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)
TRANSLATION_VALID_MIN_ROWS = 200
CLASSIFICATION_VALID_MIN_ROWS = 0
VALID_TOPUP_SEED = 42


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage A OpenSubs+FRMT data from existing exported JSONL files, "
            "avoiding direct DuckDB source access."
        )
    )
    parser.add_argument(
        "--opensubs-train",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageA_opensubs_only" / "translation_train.jsonl",
    )
    parser.add_argument(
        "--opensubs-valid",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageA_opensubs_only" / "translation_valid.jsonl",
    )
    parser.add_argument(
        "--frmt-train",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "frmt_only" / "translation_train.jsonl",
    )
    parser.add_argument(
        "--frmt-valid",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "frmt_only" / "translation_valid.jsonl",
    )
    parser.add_argument(
        "--opensubs-cls-train",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageA_opensubs_only" / "classification_train.jsonl",
    )
    parser.add_argument(
        "--opensubs-cls-valid",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageA_opensubs_only" / "classification_valid.jsonl",
    )
    parser.add_argument(
        "--frmt-cls-train",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "frmt_only" / "classification_train.jsonl",
    )
    parser.add_argument(
        "--frmt-cls-valid",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "frmt_only" / "classification_valid.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageA_opensubs_frmt",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def extract_task_prefix(text: str) -> str | None:
    match = TASK_PREFIX_RE.match(text or "")
    if not match:
        return None
    return match.group(1).strip().lower()


def infer_direction(row: dict) -> str:
    for key in ("task", "direction"):
        value = normalize_space(str(row.get(key) or "")).casefold()
        if value in {"translate_br2pt", "br2pt"}:
            return "translate_br2pt"
        if value in {"translate_pt2br", "pt2br"}:
            return "translate_pt2br"
    prefix = extract_task_prefix(str(row.get("input_text") or ""))
    if prefix == "br-pt":
        return "translate_br2pt"
    if prefix == "pt-br":
        return "translate_pt2br"
    return "translation"


def infer_task(row: dict, *, default_task: str) -> str:
    value = normalize_space(str(row.get("task") or row.get("direction") or "")).casefold()
    if value in {"translate_br2pt", "br2pt"}:
        return "translate_br2pt"
    if value in {"translate_pt2br", "pt2br"}:
        return "translate_pt2br"
    if value in {"classification", "classify"}:
        return "classification"
    return default_task


def normalize_dataset(raw: object, *, fallback: str) -> str:
    text = normalize_space(str(raw or "")).casefold()
    if text == "frmt":
        return "FRMT"
    if text in {"opensubs", "open_subs", "open subtitles"}:
        return "OpenSubs"
    if text in {"unknown", ""}:
        return fallback
    return str(raw)


def normalize_row(row: dict, *, fallback_dataset: str, default_task: str) -> dict:
    out = dict(row)
    out["dataset"] = normalize_dataset(row.get("dataset"), fallback=fallback_dataset)
    out["direction"] = infer_direction(row)
    out["task"] = infer_task(row, default_task=default_task)
    return out


def iter_normalized_rows(
    src_paths: list[tuple[Path, str]],
    *,
    default_task: str,
):
    for path, fallback_dataset in src_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        with path.open("r", encoding="utf-8") as in_fh:
            for line in in_fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield path, normalize_row(
                    row,
                    fallback_dataset=fallback_dataset,
                    default_task=default_task,
                )


def write_mix(src_paths: list[tuple[Path, str]], out_path: Path, *, default_task: str) -> dict:
    counts = Counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_fh:
        for path, normalized in iter_normalized_rows(src_paths, default_task=default_task):
            out_fh.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            counts["rows"] += 1
            counts[f"dataset:{normalized['dataset']}"] += 1
            counts[f"task:{normalized['task']}"] += 1
            counts[f"direction:{normalized['direction']}"] += 1
            counts[f"source:{path.as_posix()}"] += 1
    return dict(counts)


def top_up_validation_from_train(
    *,
    train_path: Path,
    out_path: Path,
    existing_count: int,
    min_rows: int,
    default_task: str,
    fallback_dataset: str,
    seed: int,
) -> dict:
    if min_rows <= 0 or existing_count >= min_rows:
        return {"added_rows": 0}

    sample_size = min_rows - existing_count
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    for _, normalized in iter_normalized_rows([(train_path, fallback_dataset)], default_task=default_task):
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(normalized)
            continue
        idx = rng.randrange(seen)
        if idx < sample_size:
            reservoir[idx] = normalized

    with out_path.open("a", encoding="utf-8") as out_fh:
        for row in reservoir:
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter()
    counts["added_rows"] = len(reservoir)
    for row in reservoir:
        counts[f"dataset:{row['dataset']}"] += 1
        counts[f"task:{row['task']}"] += 1
        counts[f"direction:{row['direction']}"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    translation_train_counts = write_mix(
        [(args.opensubs_train, "OpenSubs"), (args.frmt_train, "FRMT")],
        args.out_dir / "translation_train.jsonl",
        default_task="translation",
    )
    translation_valid_counts = write_mix(
        [(args.opensubs_valid, "OpenSubs"), (args.frmt_valid, "FRMT")],
        args.out_dir / "translation_valid.jsonl",
        default_task="translation",
    )
    translation_valid_topup = top_up_validation_from_train(
        train_path=args.opensubs_train,
        out_path=args.out_dir / "translation_valid.jsonl",
        existing_count=int(translation_valid_counts.get("rows", 0)),
        min_rows=TRANSLATION_VALID_MIN_ROWS,
        default_task="translation",
        fallback_dataset="OpenSubs",
        seed=VALID_TOPUP_SEED,
    )
    classification_train_counts = write_mix(
        [(args.opensubs_cls_train, "OpenSubs"), (args.frmt_cls_train, "FRMT")],
        args.out_dir / "classification_train.jsonl",
        default_task="classification",
    )
    classification_valid_counts = write_mix(
        [(args.opensubs_cls_valid, "OpenSubs"), (args.frmt_cls_valid, "FRMT")],
        args.out_dir / "classification_valid.jsonl",
        default_task="classification",
    )
    classification_valid_topup = top_up_validation_from_train(
        train_path=args.opensubs_cls_train,
        out_path=args.out_dir / "classification_valid.jsonl",
        existing_count=int(classification_valid_counts.get("rows", 0)),
        min_rows=CLASSIFICATION_VALID_MIN_ROWS,
        default_task="classification",
        fallback_dataset="OpenSubs",
        seed=VALID_TOPUP_SEED + 1,
    )

    if translation_valid_topup.get("added_rows", 0):
        translation_valid_counts["rows"] = int(translation_valid_counts.get("rows", 0)) + int(
            translation_valid_topup["added_rows"]
        )
    if classification_valid_topup.get("added_rows", 0):
        classification_valid_counts["rows"] = int(classification_valid_counts.get("rows", 0)) + int(
            classification_valid_topup["added_rows"]
        )

    report = {
        "translation_train": translation_train_counts,
        "translation_valid": translation_valid_counts,
        "translation_valid_topup_from_train": translation_valid_topup,
        "classification_train": classification_train_counts,
        "classification_valid": classification_valid_counts,
        "classification_valid_topup_from_train": classification_valid_topup,
        "out_dir": args.out_dir.as_posix(),
    }
    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote: {args.out_dir / 'translation_train.jsonl'}")
    print(f"Wrote: {args.out_dir / 'translation_valid.jsonl'}")
    print(f"Wrote: {args.out_dir / 'classification_train.jsonl'}")
    print(f"Wrote: {args.out_dir / 'classification_valid.jsonl'}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
