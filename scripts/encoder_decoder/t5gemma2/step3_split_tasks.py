#!/usr/bin/env python3
"""
Step 3: Strict train/valid/test split by source pair id, then expand to tasks.
Translation and classification are exported separately.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_TO_ID = {"pt-br": 0, "pt-pt": 1, "equal": 2}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Split pair dataset strictly by source_id, then export task datasets."
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "pairs_all.jsonl",
        help="Input canonical pairs JSONL (from Step 2).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2",
        help="Output directory.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--br2pt-token", default="<br-pt>")
    parser.add_argument("--pt2br-token", default="<pt-br>")
    parser.add_argument("--classification-token", default="<id>")
    parser.add_argument(
        "--export-pair-splits",
        action="store_true",
        default=True,
        help="Also export pair-level split files (pairs_train/valid/test.jsonl).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_pairs(
    pairs: list[dict[str, Any]], *, train_ratio: float, valid_ratio: float, seed: int
) -> dict[str, list[dict[str, Any]]]:
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid

    train = pairs[:n_train]
    valid = pairs[n_train : n_train + n_valid]
    test = pairs[n_train + n_valid : n_train + n_valid + n_test]

    return {"train": train, "valid": valid, "test": test}


def build_translation_examples(pair: dict[str, Any], br2pt_token: str, pt2br_token: str) -> list[dict[str, Any]]:
    source_id = int(pair["source_id"])
    pt_pt = str(pair["pt_pt"])
    pt_br = str(pair["pt_br"])
    is_equal = bool(pair.get("is_equal", pt_pt == pt_br))

    return [
        {
            "source_id": source_id,
            "task": "translation",
            "direction": "br2pt",
            "input_text": f"{br2pt_token} {pt_br}",
            "target_text": pt_pt,
            "source_text": pt_br,
            "target_variant": "pt-pt",
            "is_equal_pair": is_equal,
        },
        {
            "source_id": source_id,
            "task": "translation",
            "direction": "pt2br",
            "input_text": f"{pt2br_token} {pt_pt}",
            "target_text": pt_br,
            "source_text": pt_pt,
            "target_variant": "pt-br",
            "is_equal_pair": is_equal,
        },
    ]


def build_classification_examples(pair: dict[str, Any], classification_token: str) -> list[dict[str, Any]]:
    source_id = int(pair["source_id"])
    pt_pt = str(pair["pt_pt"])
    pt_br = str(pair["pt_br"])
    is_equal = bool(pair.get("is_equal", pt_pt == pt_br))

    if is_equal:
        return [
            {
                "source_id": source_id,
                "task": "classification",
                "text": pt_pt,
                "input_text": f"{classification_token} {pt_pt}",
                "label": "equal",
                "label_id": LABEL_TO_ID["equal"],
            }
        ]

    return [
        {
            "source_id": source_id,
            "task": "classification",
            "text": pt_pt,
            "input_text": f"{classification_token} {pt_pt}",
            "label": "pt-pt",
            "label_id": LABEL_TO_ID["pt-pt"],
        },
        {
            "source_id": source_id,
            "task": "classification",
            "text": pt_br,
            "input_text": f"{classification_token} {pt_br}",
            "label": "pt-br",
            "label_id": LABEL_TO_ID["pt-br"],
        },
    ]


def main() -> None:
    args = parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("--train-ratio must be in (0, 1)")
    if not (0.0 <= args.valid_ratio < 1.0):
        raise SystemExit("--valid-ratio must be in [0, 1)")
    if args.train_ratio + args.valid_ratio >= 1.0:
        raise SystemExit("--train-ratio + --valid-ratio must be < 1")

    pairs = load_jsonl(args.pairs_file)
    if not pairs:
        raise SystemExit(f"No rows found in {args.pairs_file}")

    split_map = split_pairs(
        pairs,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    for split_name, split_pairs_rows in split_map.items():
        trans_rows: list[dict[str, Any]] = []
        cls_rows: list[dict[str, Any]] = []

        for pair in split_pairs_rows:
            trans_rows.extend(
                build_translation_examples(
                    pair,
                    br2pt_token=args.br2pt_token,
                    pt2br_token=args.pt2br_token,
                )
            )
            cls_rows.extend(build_classification_examples(pair, classification_token=args.classification_token))

        trans_path = args.out_dir / f"translation_{split_name}.jsonl"
        cls_path = args.out_dir / f"classification_{split_name}.jsonl"
        write_jsonl(trans_path, trans_rows)
        write_jsonl(cls_path, cls_rows)

        label_counts = Counter(row["label"] for row in cls_rows)
        print(
            f"[{split_name}] pairs={len(split_pairs_rows)} "
            f"translation_examples={len(trans_rows)} classification_examples={len(cls_rows)} "
            f"classification_labels={dict(label_counts)}"
        )

        if args.export_pair_splits:
            pair_path = args.out_dir / f"pairs_{split_name}.jsonl"
            write_jsonl(pair_path, split_pairs_rows)

    print("\nStep 3 completed.")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
