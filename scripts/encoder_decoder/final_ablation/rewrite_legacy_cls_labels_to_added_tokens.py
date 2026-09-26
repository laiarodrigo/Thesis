#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_MAP = {
    "pt-br": "<pt-br>",
    "br": "<pt-br>",
    "pt-pt": "<pt-pt>",
    "pt": "<pt-pt>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite legacy with-cls seq2seq datasets so classification targets "
            "use added-token labels <pt-br>/<pt-pt>, while translation rows stay unchanged."
        )
    )
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,valid")
    parser.add_argument(
        "--drop-equal",
        action="store_true",
        help="Drop classification rows whose target/label is equal.",
    )
    return parser.parse_args()


def normalize(value: Any) -> str:
    return " ".join(str(value or "").split())


def label_to_token(value: Any) -> str | None:
    text = normalize(value).lower()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    if text == "equal":
        return "equal"
    return None


def rewrite_split(src_path: Path, dst_path: Path, *, drop_equal: bool) -> Counter:
    counts: Counter[str] = Counter()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with src_path.open("r", encoding="utf-8") as in_fh, dst_path.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            if not line.strip():
                continue
            row = json.loads(line)
            task = normalize(row.get("task")).lower()

            if task == "classification":
                mapped = label_to_token(row.get("target_text", row.get("label", row.get("gold"))))
                if mapped is None:
                    counts["classification_skipped_unknown_label"] += 1
                    continue
                if mapped == "equal" and drop_equal:
                    counts["classification_skipped_equal"] += 1
                    continue
                row["target_text"] = mapped
                counts[f"classification_label:{mapped}"] += 1
            elif task == "translation":
                counts["translation"] += 1
            else:
                counts[f"other_task:{task or 'missing'}"] += 1

            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["written"] += 1

    return counts


def main() -> None:
    args = parse_args()
    report = {
        "src_dir": args.src_dir.as_posix(),
        "dst_dir": args.dst_dir.as_posix(),
        "drop_equal": bool(args.drop_equal),
        "splits": {},
    }

    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        src_path = args.src_dir / f"{split}.jsonl"
        dst_path = args.dst_dir / f"{split}.jsonl"
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        counts = rewrite_split(src_path, dst_path, drop_equal=bool(args.drop_equal))
        report["splits"][split] = dict(counts)
        print(dst_path, dict(counts), flush=True)

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    (args.dst_dir / "rewrite_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
