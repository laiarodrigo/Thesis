#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PREFIX_RE = re.compile(r"^\s*<([^>]+)>\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E/D evaluation views for control-string models.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--br-control", default="<pt-br>")
    parser.add_argument("--pt-control", default="<pt-pt>")
    parser.add_argument("--classification-control", default="<cls>")
    return parser.parse_args()


def normalize_space(value: object) -> str:
    return " ".join(str(value or "").split())


def strip_prefix(value: object) -> str:
    text = str(value or "")
    match = PREFIX_RE.match(text)
    if match:
        text = text[match.end() :]
    return normalize_space(text)


def normalize_label(value: object, *, br_control: str, pt_control: str) -> str | None:
    text = normalize_space(value).casefold()
    if text in {"pt-br", "<pt-br>", "br", "brasil", "brasileiro"}:
        return br_control
    if text in {"pt-pt", "<pt-pt>", "pt", "portugal", "europeu"}:
        return pt_control
    return None


def copy_jsonl(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_classification(
    source: Path,
    target: Path,
    *,
    br_control: str,
    pt_control: str,
    classification_control: str | None,
) -> dict[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            raw_label = row.get("target_text", row.get("label", row.get("gold")))
            label = normalize_label(raw_label, br_control=br_control, pt_control=pt_control)
            if label is None:
                counts["rows_dropped_nonbinary"] += 1
                continue
            source_text = strip_prefix(row.get("input_text") or row.get("source_text"))
            if not source_text:
                raise ValueError(f"Empty classification input in {source}:{line_no}")
            out = dict(row)
            out["input_text"] = (
                f"{classification_control} {source_text}"
                if classification_control
                else source_text
            )
            out["target_text"] = label
            out["task"] = "classification"
            out["source_variant_label"] = label
            out["is_equal_pair"] = False
            out["loss_on_first_token_only"] = classification_control is None
            out["control_representation"] = "string"
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            counts["rows_written"] += 1
            counts[f"label:{label}"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {
        "source_root": str(args.source_root),
        "out_root": str(args.out_root),
        "br_control": args.br_control,
        "pt_control": args.pt_control,
        "classification_control": args.classification_control,
        "datasets": {},
    }
    for dataset in ("frmt", "golden"):
        dataset_report: dict[str, object] = {}
        for mode in ("encoder_unified", "decoder_unified"):
            source_dir = args.source_root / dataset / mode
            out_dir = args.out_root / dataset / mode
            translation_source = source_dir / "translation_test.jsonl"
            classification_source = source_dir / "classification_noequal_test.jsonl"
            if not translation_source.exists() or not classification_source.exists():
                raise FileNotFoundError(
                    f"Missing final-eval source files under {source_dir}"
                )
            translation_count = copy_jsonl(
                translation_source,
                out_dir / "translation_test.jsonl",
            )
            classification_counts = build_classification(
                classification_source,
                out_dir / "classification_test.jsonl",
                br_control=args.br_control,
                pt_control=args.pt_control,
                classification_control=(
                    args.classification_control if mode == "encoder_unified" else None
                ),
            )
            dataset_report[mode] = {
                "translation_rows": translation_count,
                "classification": classification_counts,
            }
        report["datasets"][dataset] = dataset_report

    args.out_root.mkdir(parents=True, exist_ok=True)
    report_path = args.out_root / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
