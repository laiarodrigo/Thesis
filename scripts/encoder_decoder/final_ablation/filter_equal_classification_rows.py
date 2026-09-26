#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_GROUPS = (
    "stageA_opensubs_only",
    "stageB_gpt_wiki",
    "stageB_gpt_wiki_frmt_mix",
)
DEFAULT_VIEWS = ("encoder_unified", "decoder_unified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create final-protocol ablation views where equal-pair classification "
            "rows are removed, while all translation rows are preserved."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/encoder_decoder/t5gemma2/final_protocol"),
    )
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    parser.add_argument("--suffix", default="noequal_cls")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def is_equal_classification(row: dict[str, Any]) -> bool:
    return row.get("task") == "classification" and bool(row.get("is_equal_pair"))


def filter_split(src: Path, dst: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as out_fh:
        for row in iter_jsonl(src):
            task = str(row.get("task") or "unknown")
            if is_equal_classification(row):
                counts["dropped_equal_classification"] += 1
                continue
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["kept_total"] += 1
            counts[f"kept_{task}"] += 1
            if row.get("is_equal_pair"):
                counts[f"kept_{task}_equal"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    root = args.root
    report: dict[str, Any] = {
        "root": root.as_posix(),
        "suffix": args.suffix,
        "groups": {},
    }

    for group in args.groups:
        group_report: dict[str, Any] = {}
        for view in args.views:
            src_dir = root / group / view
            dst_dir = root / group / f"{view}_{args.suffix}"
            view_report: dict[str, Any] = {
                "source": src_dir.as_posix(),
                "target": dst_dir.as_posix(),
                "splits": {},
            }
            for split in ("train", "valid"):
                src = src_dir / f"{split}.jsonl"
                if not src.exists():
                    raise FileNotFoundError(src)
                dst = dst_dir / f"{split}.jsonl"
                view_report["splits"][split] = filter_split(src, dst)
            (dst_dir / "build_report.json").write_text(
                json.dumps(view_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            group_report[view] = view_report
            print(json.dumps({"group": group, "view": view, **view_report}, ensure_ascii=False))
        report["groups"][group] = group_report

    report_path = root / f"build_{args.suffix}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
