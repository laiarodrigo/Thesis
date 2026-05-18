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
            "Build a Stage B mixed dataset in the label-first decoder format: "
            "translation rows become '<LABEL> target sentence' on the decoder side, "
            "classification rows become just 'BR', 'PT', or the configured equal token. "
            "Equal translation rows are dropped by default; equal classification rows "
            "are dropped unless --keep-equal-classification is enabled."
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
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageB_gpt_wiki_frmt_label_first_with_cls_noequal",
    )
    parser.add_argument("--br-token", default="BR")
    parser.add_argument("--pt-token", default="PT")
    parser.add_argument(
        "--equal-token",
        default="igual",
        help="Decoder token used for equal classification rows when they are kept.",
    )
    parser.add_argument(
        "--keep-equal",
        action="store_true",
        help="Keep equal translation rows.",
    )
    parser.add_argument(
        "--keep-equal-classification",
        action="store_true",
        help="Keep equal classification rows using --equal-token on the decoder side.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def strip_encoder_prefix(text: str) -> str:
    return normalize_space(TASK_PREFIX_RE.sub("", text or ""))


def infer_source_variant(row: dict[str, Any]) -> str | None:
    for key in ("direction", "task"):
        value = normalize_space(str(row.get(key) or "")).casefold()
        if value in {"translate_br2pt", "br2pt"}:
            return "br"
        if value in {"translate_pt2br", "pt2br"}:
            return "pt"
    raw_input = str(row.get("input_text") or "")
    match = TASK_PREFIX_RE.match(raw_input)
    if not match:
        return None
    prefix = match.group(1).strip().casefold()
    if prefix == "br-pt":
        return "br"
    if prefix == "pt-br":
        return "pt"
    return None


def normalize_class_label(
    raw: Any,
    *,
    br_token: str,
    pt_token: str,
    equal_token: str,
) -> str | None:
    text = normalize_space(str(raw or "")).lower()
    if text in {"pt-br", "br"}:
        return br_token
    if text in {"pt-pt", "pt"}:
        return pt_token
    if text in {"equal", "igual", normalize_space(equal_token).lower()}:
        return "equal"
    if "brasil" in text:
        return br_token
    if "europeu" in text or "portugal" in text:
        return pt_token
    return None


def convert_translation_row(
    row: dict[str, Any],
    *,
    br_token: str,
    pt_token: str,
    keep_equal: bool,
) -> tuple[dict[str, Any] | None, str]:
    source_text = strip_encoder_prefix(str(row.get("input_text") or row.get("source_text") or ""))
    target_text = normalize_space(str(row.get("target_text") or ""))
    if not source_text or not target_text:
        return None, "invalid"

    source_variant = infer_source_variant(row)
    if source_variant is None:
        return None, "invalid"

    is_equal = bool(row.get("is_equal_pair")) or source_text == target_text
    if is_equal and not keep_equal:
        return None, "equal"

    decoder_label = br_token if source_variant == "br" else pt_token
    return {
        "input_text": source_text,
        "target_text": f"{decoder_label} {target_text}".strip(),
        "task": "translation",
        "dataset": row.get("dataset"),
        "bucket": row.get("bucket"),
        "direction": normalize_space(str(row.get("direction") or row.get("task") or "translation")),
        "source_variant_label": decoder_label,
        "is_equal_pair": is_equal,
        "loss_on_first_token_only": False,
        # Equal rows keep the standard target format but skip loss on the BR/PT prefix.
        "loss_mask_prefix_tokens": 1 if is_equal else 0,
    }, "ok"


def convert_classification_row(
    row: dict[str, Any],
    *,
    br_token: str,
    pt_token: str,
    equal_token: str,
    keep_equal_classification: bool,
) -> tuple[dict[str, Any] | None, str]:
    source = strip_encoder_prefix(str(row.get("input_text") or row.get("source_text") or row.get("text") or ""))
    label = normalize_class_label(
        row.get("target_text", row.get("label", row.get("gold"))),
        br_token=br_token,
        pt_token=pt_token,
        equal_token=equal_token,
    )
    if not source or label is None:
        return None, "invalid"
    if label == "equal":
        if not keep_equal_classification:
            return None, "equal"
        target_text = equal_token
        is_equal = True
    else:
        target_text = label
        is_equal = False
    return {
        "input_text": source,
        "target_text": target_text,
        "task": "classification",
        "dataset": row.get("dataset"),
        "bucket": row.get("bucket"),
        "direction": "classification",
        "source_variant_label": "",
        "is_equal_pair": is_equal,
        "loss_on_first_token_only": True,
    }, "ok"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_mixed_split(
    *,
    translation_path: Path,
    classification_path: Path,
    output_path: Path,
    br_token: str,
    pt_token: str,
    equal_token: str,
    keep_equal: bool,
    keep_equal_classification: bool,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_fh:
        for row in iter_jsonl(translation_path):
            converted, status = convert_translation_row(
                row,
                br_token=br_token,
                pt_token=pt_token,
                keep_equal=keep_equal,
            )
            if converted is None:
                counts[f"translation_skipped_{status}"] += 1
                continue
            out_fh.write(json.dumps(converted, ensure_ascii=False) + "\n")
            counts["translation_written"] += 1
            counts[f"translation_dataset:{normalize_space(str(converted.get('dataset') or 'UNKNOWN'))}"] += 1
            counts[f"translation_source_variant:{converted['source_variant_label']}"] += 1

        for row in iter_jsonl(classification_path):
            converted, status = convert_classification_row(
                row,
                br_token=br_token,
                pt_token=pt_token,
                equal_token=equal_token,
                keep_equal_classification=keep_equal_classification,
            )
            if converted is None:
                counts[f"classification_skipped_{status}"] += 1
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
) -> dict[str, int]:
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
                counts[f"translation_source_variant:{normalize_space(str(row.get('source_variant_label') or ''))}"] += 1

    counts["total_written"] = counts["translation_written"] + counts["classification_written"]
    return dict(counts)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_stats = write_mixed_split(
        translation_path=args.translation_train,
        classification_path=args.classification_train,
        output_path=args.out_dir / "train.jsonl",
        br_token=args.br_token,
        pt_token=args.pt_token,
        equal_token=args.equal_token,
        keep_equal=bool(args.keep_equal),
        keep_equal_classification=bool(args.keep_equal_classification),
    )
    valid_stats = write_mixed_split(
        translation_path=args.translation_valid,
        classification_path=args.classification_valid,
        output_path=args.out_dir / "valid.jsonl",
        br_token=args.br_token,
        pt_token=args.pt_token,
        equal_token=args.equal_token,
        keep_equal=bool(args.keep_equal),
        keep_equal_classification=bool(args.keep_equal_classification),
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
        "translation_train": args.translation_train.as_posix(),
        "translation_valid": args.translation_valid.as_posix(),
        "classification_train": args.classification_train.as_posix(),
        "classification_valid": args.classification_valid.as_posix(),
        "out_dir": args.out_dir.as_posix(),
        "br_token": args.br_token,
        "pt_token": args.pt_token,
        "equal_token": args.equal_token,
        "keep_equal": bool(args.keep_equal),
        "keep_equal_classification": bool(args.keep_equal_classification),
    }
    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
