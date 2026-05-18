#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TASK_PREFIX_RE = re.compile(r"^\s*<([^>]+)>\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite Stage B translation JSONL into the label-first decoder format: "
            "encoder gets only the source sentence, decoder target starts with BR/PT "
            "followed by the translated sentence. Equal pairs are dropped by default."
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
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageB_gpt_wiki_frmt_label_first_noequal",
    )
    parser.add_argument("--br-token", default="BR")
    parser.add_argument("--pt-token", default="PT")
    parser.add_argument("--keep-equal", action="store_true")
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
    out = {
        "input_text": source_text,
        "target_text": f"{decoder_label} {target_text}".strip(),
        "task": "translation",
        "dataset": row.get("dataset"),
        "bucket": row.get("bucket"),
        "direction": normalize_space(str(row.get("direction") or row.get("task") or "translation")),
        "source_variant_label": decoder_label,
        "is_equal_pair": is_equal,
        # Equal rows keep the standard target format but skip loss on the BR/PT prefix.
        "loss_mask_prefix_tokens": 1 if is_equal else 0,
    }
    return out, "ok"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_split(
    *,
    input_path: Path,
    output_path: Path,
    br_token: str,
    pt_token: str,
    keep_equal: bool,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_fh:
        for row in iter_jsonl(input_path):
            converted, status = convert_translation_row(
                row,
                br_token=br_token,
                pt_token=pt_token,
                keep_equal=keep_equal,
            )
            if converted is None:
                counts[f"skipped_{status}"] += 1
                continue
            out_fh.write(json.dumps(converted, ensure_ascii=False) + "\n")
            counts["written"] += 1
            counts[f"dataset:{normalize_space(str(converted.get('dataset') or 'UNKNOWN'))}"] += 1
            counts[f"source_variant:{converted['source_variant_label']}"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_stats = write_split(
        input_path=args.translation_train,
        output_path=args.out_dir / "train.jsonl",
        br_token=args.br_token,
        pt_token=args.pt_token,
        keep_equal=bool(args.keep_equal),
    )
    valid_stats = write_split(
        input_path=args.translation_valid,
        output_path=args.out_dir / "valid.jsonl",
        br_token=args.br_token,
        pt_token=args.pt_token,
        keep_equal=bool(args.keep_equal),
    )

    report = {
        "train": train_stats,
        "valid": valid_stats,
        "translation_train": args.translation_train.as_posix(),
        "translation_valid": args.translation_valid.as_posix(),
        "out_dir": args.out_dir.as_posix(),
        "br_token": args.br_token,
        "pt_token": args.pt_token,
        "keep_equal": bool(args.keep_equal),
    }
    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
