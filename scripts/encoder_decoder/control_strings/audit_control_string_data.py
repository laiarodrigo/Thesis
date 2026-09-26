#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paired E/D datasets produced by the control-string protocol."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--br-control", default="<pt-br>")
    parser.add_argument("--pt-control", default="<pt-pt>")
    parser.add_argument("--classification-control", default="<cls>")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-train-valid-overlap",
        action="store_true",
        help="Report, rather than reject, overlap inherited from recovered splits.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def remove_prefix(text: object, prefix: str) -> str:
    value = " ".join(str(text or "").split())
    expected = f"{prefix} "
    if not value.startswith(expected):
        raise AssertionError(f"Expected prefix {prefix!r} in {value!r}")
    return value[len(expected) :]


def pair_hash(row: dict[str, Any], source: str, target: str) -> str:
    dataset = str(row.get("dataset") or "").strip().casefold()
    left, right = sorted((source.casefold(), target.casefold()))
    payload = "\x1f".join((dataset, left, right)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_split(
    root: Path,
    split: str,
    *,
    classification_control: str,
    labels: set[str],
    collect_pairs: bool = False,
    forbidden_pairs: set[str] | None = None,
    allow_forbidden_pairs: bool = False,
) -> tuple[dict[str, int], set[str]]:
    encoder_path = root / "encoder_unified" / f"{split}.jsonl"
    decoder_path = root / "decoder_unified" / f"{split}.jsonl"
    if not encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(f"Missing paired {split} files under {root}")

    counts: Counter[str] = Counter()
    translation_pairs: set[str] = set()

    encoder_rows = iter_jsonl(encoder_path)
    decoder_rows = iter_jsonl(decoder_path)
    for position, pair in enumerate(
        itertools.zip_longest(encoder_rows, decoder_rows), start=1
    ):
        encoder_item, decoder_item = pair
        if encoder_item is None or decoder_item is None:
            raise AssertionError(f"E/D row-count mismatch in {split} at position {position}")
        encoder_line, encoder = encoder_item
        decoder_line, decoder = decoder_item
        if encoder_line != decoder_line:
            raise AssertionError(f"E/D line mismatch at position {position}")

        task = str(encoder.get("task") or "")
        if task != decoder.get("task") or task not in {"translation", "classification"}:
            raise AssertionError(f"Task mismatch at {split}:{position}")
        label = str(encoder.get("source_variant_label") or "")
        if label != decoder.get("source_variant_label") or label not in labels:
            raise AssertionError(f"Source-label mismatch at {split}:{position}")

        for key in ("dataset", "bucket", "direction", "id", "source_id", "is_equal_pair"):
            if encoder.get(key) != decoder.get(key):
                raise AssertionError(f"Metadata mismatch for {key!r} at {split}:{position}")

        encoder_input = " ".join(str(encoder.get("input_text") or "").split())
        encoder_target = " ".join(str(encoder.get("target_text") or "").split())
        decoder_input = " ".join(str(decoder.get("input_text") or "").split())
        decoder_target = " ".join(str(decoder.get("target_text") or "").split())
        if not all((encoder_input, encoder_target, decoder_input, decoder_target)):
            raise AssertionError(f"Empty text field at {split}:{position}")

        counts[task] += 1
        if task == "translation":
            clean_encoder_input = remove_prefix(encoder_input, label)
            clean_decoder_target = remove_prefix(decoder_target, label)
            if clean_encoder_input != decoder_input or clean_decoder_target != encoder_target:
                raise AssertionError(f"E/D translation payload mismatch at {split}:{position}")
            is_equal = bool(encoder.get("is_equal_pair"))
            if clean_encoder_input == encoder_target and not is_equal:
                raise AssertionError(f"Exact-copy translation is not marked equal at {split}:{position}")
            counts["translation_equal" if is_equal else "translation_non_equal"] += 1
            current_pair = pair_hash(encoder, clean_encoder_input, encoder_target)
            if forbidden_pairs is not None and current_pair in forbidden_pairs:
                counts["train_validation_pair_overlap_rows"] += 1
                if not allow_forbidden_pairs:
                    raise AssertionError(f"Train/validation pair leakage at {split}:{position}")
            if collect_pairs:
                translation_pairs.add(current_pair)
        else:
            if bool(encoder.get("is_equal_pair")) or bool(decoder.get("is_equal_pair")):
                raise AssertionError(f"Equal classification row at {split}:{position}")
            clean_encoder_input = remove_prefix(encoder_input, classification_control)
            if decoder_input.startswith(f"{classification_control} "):
                raise AssertionError(f"Decoder row contains {classification_control} at {split}:{position}")
            if clean_encoder_input != decoder_input:
                raise AssertionError(f"E/D classification input mismatch at {split}:{position}")
            if encoder_target != label or decoder_target != label:
                raise AssertionError(f"Classification target mismatch at {split}:{position}")
            if encoder.get("loss_on_first_token_only") is not False:
                raise AssertionError(f"E classification is not full-sequence at {split}:{position}")
            if decoder.get("loss_on_first_token_only") is not True:
                raise AssertionError(f"D classification is not first-token-only at {split}:{position}")
    if counts["translation"] == 0 or counts["classification"] == 0:
        raise AssertionError(f"Missing task in {split}: {dict(counts)}")
    return dict(counts), translation_pairs


def main() -> None:
    args = parse_args()
    valid_counts, valid_pairs = audit_split(
        args.root,
        "valid",
        classification_control=args.classification_control,
        labels={args.br_control, args.pt_control},
        collect_pairs=True,
    )
    train_counts, _ = audit_split(
        args.root,
        "train",
        classification_control=args.classification_control,
        labels={args.br_control, args.pt_control},
        forbidden_pairs=valid_pairs,
        allow_forbidden_pairs=args.allow_train_valid_overlap,
    )

    report = {
        "status": "passed",
        "root": str(args.root),
        "br_control": args.br_control,
        "pt_control": args.pt_control,
        "classification_control": args.classification_control,
        "encoder_classification_prefix": args.classification_control,
        "decoder_classification_prefix": None,
        "train": train_counts,
        "valid": valid_counts,
        "train_validation_pair_overlap_rows": train_counts.get(
            "train_validation_pair_overlap_rows", 0
        ),
        "train_validation_overlap_allowed": args.allow_train_valid_overlap,
    }
    report_path = args.report or args.root / "audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
