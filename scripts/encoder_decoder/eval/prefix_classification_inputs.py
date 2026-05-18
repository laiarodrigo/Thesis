#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefix classification input_text rows for seq2seq classification evaluation."
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--prefix", default="<cls>")
    parser.add_argument("--drop-equal", action="store_true")
    return parser.parse_args()


def normalize_label(raw: object) -> str | None:
    text = str(raw or "").strip().lower()
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


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_equal = 0
    skipped_invalid = 0
    with args.input_path.open("r", encoding="utf-8") as in_fh, args.output_path.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = normalize_label(row.get("target_text", row.get("label", row.get("gold"))))
            if label is None:
                skipped_invalid += 1
                continue
            if label == "equal" and args.drop_equal:
                skipped_equal += 1
                continue
            src = str(row.get("input_text") or "").strip()
            row["input_text"] = f"{args.prefix} {src}".strip()
            row["target_text"] = label
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "input_path": args.input_path.as_posix(),
                "output_path": args.output_path.as_posix(),
                "prefix": args.prefix,
                "drop_equal": bool(args.drop_equal),
                "written": written,
                "skipped_equal": skipped_equal,
                "skipped_invalid": skipped_invalid,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
