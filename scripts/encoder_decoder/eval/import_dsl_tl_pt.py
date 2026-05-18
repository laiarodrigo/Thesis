#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


LABEL_MAP = {
    "PT-BR": ("pt-br", 0),
    "PT-PT": ("pt-pt", 1),
    "PT": ("equal", 2),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Convert the Portuguese DSL-TL split into the repo's classification JSONL format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "external" / "DSL-TL" / "DSL-TL-Corpus" / "PT-DSL-TL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "dsl_tl_pt",
    )
    return parser.parse_args()


def build_row(source_id: str, text: str, raw_label: str) -> dict:
    label, label_id = LABEL_MAP[raw_label]
    clean_text = " ".join((text or "").split())
    return {
        "source_id": int(source_id) if source_id.isdigit() else source_id,
        "task": "classification",
        "text": clean_text,
        "input_text": f"<id> {clean_text}".strip(),
        "label": label,
        "label_id": label_id,
        "raw_label": raw_label,
        "dataset": "DSL-TL",
        "bucket": "n/a",
        "source": "LanguageTechnologyLab/DSL-TL",
    }


def convert_tsv(input_path: Path, output_path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with input_path.open("r", encoding="utf-8", newline="") as in_fh, output_path.open(
        "w", encoding="utf-8"
    ) as out_fh:
        reader = csv.reader(in_fh, delimiter="\t", quotechar='"')
        for row_idx, parts in enumerate(reader, start=1):
            if len(parts) != 3:
                raise ValueError(f"{input_path}:{row_idx} expected 3 TSV columns, got {len(parts)}")
            source_id, text, raw_label = parts
            raw_label = raw_label.strip()
            if raw_label not in LABEL_MAP:
                raise ValueError(f"{input_path}:{row_idx} unexpected label {raw_label!r}")
            out_row = build_row(source_id=source_id.strip(), text=text, raw_label=raw_label)
            out_fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            counts[raw_label] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_in = args.input_dir / "PT_train.tsv"
    dev_in = args.input_dir / "PT_dev.tsv"
    train_out = args.output_dir / "classification_train.jsonl"
    valid_out = args.output_dir / "classification_valid.jsonl"
    test_out = args.output_dir / "classification_test.jsonl"
    meta_out = args.output_dir / "build_report.json"

    train_counts = convert_tsv(train_in, train_out)
    dev_counts = convert_tsv(dev_in, valid_out)
    valid_out.replace(test_out)
    dev_counts = convert_tsv(dev_in, valid_out)

    summary = {
        "input_dir": args.input_dir.as_posix(),
        "output_dir": args.output_dir.as_posix(),
        "label_map": {
            raw: {"label": label, "label_id": label_id}
            for raw, (label, label_id) in LABEL_MAP.items()
        },
        "splits": {
            "classification_train.jsonl": {
                "source_tsv": train_in.as_posix(),
                "label_counts": train_counts,
                "n": sum(train_counts.values()),
            },
            "classification_valid.jsonl": {
                "source_tsv": dev_in.as_posix(),
                "label_counts": dev_counts,
                "n": sum(dev_counts.values()),
            },
            "classification_test.jsonl": {
                "source_tsv": dev_in.as_posix(),
                "note": "Copied from PT_dev.tsv because the official DSL-TL test labels are hidden.",
                "label_counts": dev_counts,
                "n": sum(dev_counts.values()),
            },
        },
    }

    with meta_out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
