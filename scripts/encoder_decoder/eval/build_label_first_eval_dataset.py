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
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite evaluation JSONL into the label-first decoder format. "
            "Translation rows become target='BR/PT <sentence>'; classification rows become "
            "target='BR/PT/<equal-token>'."
        )
    )
    parser.add_argument("--task", choices=["translation", "classification"], required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--br-token", default="BR")
    parser.add_argument("--pt-token", default="PT")
    parser.add_argument(
        "--equal-token",
        default="equal",
        help="Decoder token used for equal classification rows when they are kept.",
    )
    parser.add_argument("--drop-equal", action="store_true")
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
    drop_equal: bool,
) -> tuple[dict[str, Any] | None, str]:
    source_text = strip_encoder_prefix(str(row.get("input_text") or row.get("source_text") or ""))
    target_text = normalize_space(str(row.get("target_text") or row.get("gold") or ""))
    if not source_text or not target_text:
        return None, "invalid"

    source_variant = infer_source_variant(row)
    if source_variant is None:
        return None, "invalid"

    is_equal = bool(row.get("is_equal_pair")) or source_text == target_text
    if is_equal and drop_equal:
        return None, "equal"

    label = br_token if source_variant == "br" else pt_token
    out = dict(row)
    out["input_text"] = source_text
    out["target_text"] = f"{label} {target_text}".strip()
    out["task"] = "translation"
    out["source_variant_label"] = label
    out["is_equal_pair"] = is_equal
    return out, "ok"


def convert_classification_row(
    row: dict[str, Any],
    *,
    br_token: str,
    pt_token: str,
    equal_token: str,
    drop_equal: bool,
) -> tuple[dict[str, Any] | None, str]:
    source_text = strip_encoder_prefix(str(row.get("input_text") or row.get("source_text") or row.get("text") or ""))
    label = normalize_class_label(
        row.get("target_text", row.get("label", row.get("gold"))),
        br_token=br_token,
        pt_token=pt_token,
        equal_token=equal_token,
    )
    if not source_text or label is None:
        return None, "invalid"
    if label == "equal" and drop_equal:
        return None, "equal"

    out = dict(row)
    out["input_text"] = source_text
    out["target_text"] = equal_token if label == "equal" else label
    out["task"] = "classification"
    return out, "ok"


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    with args.input_path.open("r", encoding="utf-8") as in_fh, args.output_path.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.task == "translation":
                converted, status = convert_translation_row(
                    row,
                    br_token=args.br_token,
                    pt_token=args.pt_token,
                    drop_equal=bool(args.drop_equal),
                )
            else:
                converted, status = convert_classification_row(
                    row,
                    br_token=args.br_token,
                    pt_token=args.pt_token,
                    equal_token=args.equal_token,
                    drop_equal=bool(args.drop_equal),
                )
            if converted is None:
                counts[f"skipped_{status}"] += 1
                continue
            out_fh.write(json.dumps(converted, ensure_ascii=False) + "\n")
            counts["written"] += 1
            counts[f"dataset:{normalize_space(str(converted.get('dataset') or 'UNKNOWN'))}"] += 1
            if args.task == "translation":
                counts[f"source_variant:{converted['source_variant_label']}"] += 1
            else:
                counts[f"classification_label:{converted['target_text']}"] += 1

    print(
        json.dumps(
            {
                "task": args.task,
                "input_path": args.input_path.as_posix(),
                "output_path": args.output_path.as_posix(),
                "br_token": args.br_token,
                "pt_token": args.pt_token,
                "equal_token": args.equal_token,
                "drop_equal": bool(args.drop_equal),
                "counts": dict(counts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
