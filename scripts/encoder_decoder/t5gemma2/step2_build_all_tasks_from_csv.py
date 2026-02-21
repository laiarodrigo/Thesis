#!/usr/bin/env python3
"""
Step 2: Build full (unsplit) translation + classification datasets from GPT CSV.
Uses all rows for both tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Build full task datasets (unsplit) from pt_PT/pt_BR CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "data" / "pt_variant_prompts_500.csv",
        help="Input CSV with pt_PT and pt_BR columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2",
        help="Output directory.",
    )
    parser.add_argument(
        "--pairs-file",
        default="pairs_all.jsonl",
        help="Output file with canonical pair rows.",
    )
    parser.add_argument(
        "--translation-file",
        default="translation_all.jsonl",
        help="Output file for translation examples.",
    )
    parser.add_argument(
        "--classification-file",
        default="classification_all.jsonl",
        help="Output file for classification examples.",
    )
    parser.add_argument(
        "--br2pt-token",
        default="<br-pt>",
        help="Prefix token for BR->PT translation inputs.",
    )
    parser.add_argument(
        "--pt2br-token",
        default="<pt-br>",
        help="Prefix token for PT->BR translation inputs.",
    )
    parser.add_argument(
        "--classification-token",
        default="<id>",
        help="Prefix token for classification text inputs (optional helper field).",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(text.split())


def to_int(value: Any, fallback: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return fallback


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = args.out_dir / args.pairs_file
    translation_path = args.out_dir / args.translation_file
    classification_path = args.out_dir / args.classification_file

    label_to_id = {"pt-br": 0, "pt-pt": 1, "equal": 2}

    pair_count = 0
    trans_count = 0
    cls_count = 0
    cls_label_counts = Counter()

    with (
        args.input_csv.open("r", encoding="utf-8", newline="") as in_fh,
        pairs_path.open("w", encoding="utf-8") as pairs_fh,
        translation_path.open("w", encoding="utf-8") as trans_fh,
        classification_path.open("w", encoding="utf-8") as cls_fh,
    ):
        reader = csv.DictReader(in_fh)
        missing = [c for c in ["pt_PT", "pt_BR"] if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Input CSV missing required columns: {', '.join(missing)}")

        for idx, row in enumerate(reader, start=1):
            pt_pt = normalize_space(str(row.get("pt_PT", "")).strip())
            pt_br = normalize_space(str(row.get("pt_BR", "")).strip())
            if not pt_pt or not pt_br:
                continue

            source_id = to_int(row.get("id"), idx)
            is_equal = pt_pt == pt_br

            pair_record = {
                "source_id": source_id,
                "pt_pt": pt_pt,
                "pt_br": pt_br,
                "is_equal": is_equal,
                "pt_pt_words": word_count(pt_pt),
                "pt_br_words": word_count(pt_br),
            }
            pairs_fh.write(json.dumps(pair_record, ensure_ascii=False) + "\n")
            pair_count += 1

            # Translation task: use all rows, both directions.
            t1 = {
                "source_id": source_id,
                "task": "translation",
                "direction": "br2pt",
                "input_text": f"{args.br2pt_token} {pt_br}",
                "target_text": pt_pt,
                "source_text": pt_br,
                "target_variant": "pt-pt",
                "is_equal_pair": is_equal,
            }
            t2 = {
                "source_id": source_id,
                "task": "translation",
                "direction": "pt2br",
                "input_text": f"{args.pt2br_token} {pt_pt}",
                "target_text": pt_br,
                "source_text": pt_pt,
                "target_variant": "pt-br",
                "is_equal_pair": is_equal,
            }
            trans_fh.write(json.dumps(t1, ensure_ascii=False) + "\n")
            trans_fh.write(json.dumps(t2, ensure_ascii=False) + "\n")
            trans_count += 2

            # Classification task (for classification head):
            # equal -> one row; non-equal -> one pt-pt + one pt-br row.
            if is_equal:
                crows = [
                    {
                        "source_id": source_id,
                        "task": "classification",
                        "text": pt_pt,
                        "input_text": f"{args.classification_token} {pt_pt}",
                        "label": "equal",
                        "label_id": label_to_id["equal"],
                    }
                ]
            else:
                crows = [
                    {
                        "source_id": source_id,
                        "task": "classification",
                        "text": pt_pt,
                        "input_text": f"{args.classification_token} {pt_pt}",
                        "label": "pt-pt",
                        "label_id": label_to_id["pt-pt"],
                    },
                    {
                        "source_id": source_id,
                        "task": "classification",
                        "text": pt_br,
                        "input_text": f"{args.classification_token} {pt_br}",
                        "label": "pt-br",
                        "label_id": label_to_id["pt-br"],
                    },
                ]

            for c in crows:
                cls_fh.write(json.dumps(c, ensure_ascii=False) + "\n")
                cls_count += 1
                cls_label_counts[c["label"]] += 1

    print("Build complete (ALL DATA):")
    print(f"  pairs: {pair_count} -> {pairs_path}")
    print(f"  translation examples: {trans_count} -> {translation_path}")
    print(f"  classification examples: {cls_count} -> {classification_path}")
    print(f"  classification label counts: {dict(cls_label_counts)}")


if __name__ == "__main__":
    main()
