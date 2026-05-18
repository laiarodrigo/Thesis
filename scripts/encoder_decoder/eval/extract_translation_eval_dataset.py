#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a canonical translation eval dataset from an existing JSONL. "
            "Expected input columns: input_text plus gold or target_text."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise SystemExit(f"Missing input JSONL: {args.input_jsonl}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = 0

    with args.input_jsonl.open("r", encoding="utf-8") as src, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            input_text = row.get("input_text")
            gold = row.get("gold", row.get("target_text"))
            if not input_text or not gold:
                raise SystemExit(
                    f"{args.input_jsonl}:{line_no} missing required input_text/gold fields"
                )
            out_row = {
                "id": row.get("id", rows),
                "input_text": str(input_text),
                "gold": str(gold),
            }
            dst.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            rows += 1

    print(f"Wrote {rows} rows -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
