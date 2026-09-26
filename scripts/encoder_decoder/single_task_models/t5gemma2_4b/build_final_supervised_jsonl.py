#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TASK_PREFIX_RE = re.compile(r"^\s*<([^>]+)>\s*", flags=re.IGNORECASE)
VALID_TOPUP_SEED = 42


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description=(
            "Render final-protocol supervised JSONL files for the translation-only, "
            "tokens-in-encoder, and labels-in-decoder families."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("translation_only", "encoder_unified", "decoder_unified"),
        required=True,
    )
    parser.add_argument("--translation-train", type=Path, required=True)
    parser.add_argument("--translation-valid", type=Path, required=True)
    parser.add_argument("--classification-train", type=Path, default=None)
    parser.add_argument("--classification-valid", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--br-source-token", default="<pt-br>")
    parser.add_argument("--pt-source-token", default="<pt-pt>")
    parser.add_argument("--cls-prefix", default="<cls>")
    parser.add_argument("--valid-min-rows", type=int, default=200)
    parser.add_argument("--valid-topup-seed", type=int, default=VALID_TOPUP_SEED)
    parser.add_argument(
        "--no-duplicate-equal-classification",
        action="store_true",
        help=(
            "Disable final-protocol duplication of equal classification rows. "
            "This should normally remain unset for final reruns."
        ),
    )
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def strip_encoder_prefix(text: str) -> tuple[str | None, str]:
    match = TASK_PREFIX_RE.match(text or "")
    if not match:
        return None, normalize_space(text)
    prefix = match.group(1).strip().casefold()
    return prefix, normalize_space((text or "")[match.end() :])


def infer_source_label(
    row: dict[str, Any],
    *,
    br_source_token: str,
    pt_source_token: str,
) -> str | None:
    for key in ("direction", "task"):
        value = normalize_space(str(row.get(key) or "")).casefold()
        if value in {"translate_br2pt", "br2pt", "br-pt"}:
            return br_source_token
        if value in {"translate_pt2br", "pt2br", "pt-br"}:
            return pt_source_token

    prefix, _ = strip_encoder_prefix(str(row.get("input_text") or ""))
    if prefix == "br-pt":
        return br_source_token
    if prefix == "pt-br":
        return pt_source_token
    if prefix in {"pt-br", "pt-pt"}:
        return f"<{prefix}>"
    return None


def normalize_class_label(
    raw: Any,
    *,
    br_source_token: str,
    pt_source_token: str,
) -> str | None:
    text = normalize_space(str(raw or "")).casefold()
    if text in {"pt-br", "<pt-br>", "br", "brasil", "brasileiro", "brazilian"}:
        return br_source_token
    if text in {"pt-pt", "<pt-pt>", "pt", "portugal", "europeu", "european"}:
        return pt_source_token
    if text in {"equal", "igual", "same", "shared"}:
        return "equal"
    if "brasil" in text:
        return br_source_token
    if "europeu" in text or "portugal" in text:
        return pt_source_token
    return None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            yield row


def base_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset"),
        "bucket": row.get("bucket"),
        "direction": normalize_space(str(row.get("direction") or row.get("task") or "")),
        "id": row.get("id"),
        "is_equal_pair": bool(row.get("is_equal_pair", False)),
    }


def convert_translation_row(
    row: dict[str, Any],
    *,
    mode: str,
    br_source_token: str,
    pt_source_token: str,
) -> tuple[dict[str, Any] | None, str]:
    _, clean_input = strip_encoder_prefix(
        str(row.get("input_text") or row.get("source_text") or "")
    )
    target_text = normalize_space(str(row.get("target_text") or ""))
    if not clean_input or not target_text:
        return None, "invalid"
    source_label = infer_source_label(
        row,
        br_source_token=br_source_token,
        pt_source_token=pt_source_token,
    )
    if source_label is None:
        return None, "unknown_source_label"

    out = base_metadata(row)
    out["task"] = "translation"
    out["source_variant_label"] = source_label
    out["is_equal_pair"] = bool(out["is_equal_pair"]) or clean_input == target_text
    out["loss_on_first_token_only"] = False

    if mode in {"translation_only", "encoder_unified"}:
        out["input_text"] = f"{source_label} {clean_input}".strip()
        out["target_text"] = target_text
    elif mode == "decoder_unified":
        out["input_text"] = clean_input
        out["target_text"] = f"{source_label} {target_text}".strip()
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return out, "ok"


def classification_targets(
    label: str,
    *,
    br_source_token: str,
    pt_source_token: str,
    duplicate_equal_classification: bool,
) -> list[str]:
    if label == "equal":
        if not duplicate_equal_classification:
            return []
        return [br_source_token, pt_source_token]
    return [label]


def convert_classification_row(
    row: dict[str, Any],
    *,
    mode: str,
    br_source_token: str,
    pt_source_token: str,
    cls_prefix: str,
    duplicate_equal_classification: bool,
) -> tuple[list[dict[str, Any]], str]:
    if mode == "translation_only":
        return [], "not_applicable"

    _, clean_input = strip_encoder_prefix(
        str(row.get("input_text") or row.get("source_text") or row.get("text") or "")
    )
    label = normalize_class_label(
        row.get("target_text", row.get("label", row.get("gold"))),
        br_source_token=br_source_token,
        pt_source_token=pt_source_token,
    )
    if not clean_input or label is None:
        return [], "invalid"

    targets = classification_targets(
        label,
        br_source_token=br_source_token,
        pt_source_token=pt_source_token,
        duplicate_equal_classification=duplicate_equal_classification,
    )
    if not targets:
        return [], "equal_skipped"

    rows: list[dict[str, Any]] = []
    for target in targets:
        out = base_metadata(row)
        out["task"] = "classification"
        out["direction"] = "classification"
        out["source_variant_label"] = target
        out["is_equal_pair"] = label == "equal" or bool(out["is_equal_pair"])
        out["loss_on_first_token_only"] = True
        out["input_text"] = (
            f"{cls_prefix} {clean_input}".strip()
            if mode == "encoder_unified"
            else clean_input
        )
        out["target_text"] = target
        rows.append(out)
    return rows, "ok"


def write_split(
    *,
    mode: str,
    translation_path: Path,
    classification_path: Path | None,
    output_path: Path,
    br_source_token: str,
    pt_source_token: str,
    cls_prefix: str,
    duplicate_equal_classification: bool,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_fh:
        for row in iter_jsonl(translation_path):
            converted, status = convert_translation_row(
                row,
                mode=mode,
                br_source_token=br_source_token,
                pt_source_token=pt_source_token,
            )
            if converted is None:
                counts[f"translation_skipped_{status}"] += 1
                continue
            out_fh.write(json.dumps(converted, ensure_ascii=False) + "\n")
            counts["translation_written"] += 1
            counts[f"translation_source:{converted['source_variant_label']}"] += 1
            if converted.get("is_equal_pair"):
                counts["translation_equal_written"] += 1

        if mode != "translation_only":
            if classification_path is None:
                raise ValueError(f"{mode} requires a classification JSONL path")
            for row in iter_jsonl(classification_path):
                converted_rows, status = convert_classification_row(
                    row,
                    mode=mode,
                    br_source_token=br_source_token,
                    pt_source_token=pt_source_token,
                    cls_prefix=cls_prefix,
                    duplicate_equal_classification=duplicate_equal_classification,
                )
                if not converted_rows:
                    counts[f"classification_skipped_{status}"] += 1
                    continue
                for converted in converted_rows:
                    out_fh.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    counts["classification_written"] += 1
                    counts[f"classification_label:{converted['target_text']}"] += 1
                    if converted.get("is_equal_pair"):
                        counts["classification_equal_written"] += 1

    counts["total_written"] = counts["translation_written"] + counts["classification_written"]
    return dict(counts)


def top_up_validation_from_train(
    *,
    train_path: Path,
    valid_path: Path,
    existing_count: int,
    min_rows: int,
    seed: int,
) -> dict[str, int]:
    if min_rows <= 0 or existing_count >= min_rows:
        return {"added_rows": 0}

    sample_size = min_rows - existing_count
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for row in iter_jsonl(train_path):
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(row)
            continue
        idx = rng.randrange(seen)
        if idx < sample_size:
            reservoir[idx] = row

    counts: Counter[str] = Counter()
    with valid_path.open("a", encoding="utf-8") as out_fh:
        for row in reservoir:
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["added_rows"] += 1
            task = normalize_space(str(row.get("task") or "")).casefold()
            counts[f"{task or 'unknown'}_written"] += 1
            if row.get("is_equal_pair"):
                counts[f"{task or 'unknown'}_equal_written"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    duplicate_equal_classification = not bool(args.no_duplicate_equal_classification)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_stats = write_split(
        mode=args.mode,
        translation_path=args.translation_train,
        classification_path=args.classification_train,
        output_path=args.out_dir / "train.jsonl",
        br_source_token=args.br_source_token,
        pt_source_token=args.pt_source_token,
        cls_prefix=args.cls_prefix,
        duplicate_equal_classification=duplicate_equal_classification,
    )
    valid_stats = write_split(
        mode=args.mode,
        translation_path=args.translation_valid,
        classification_path=args.classification_valid,
        output_path=args.out_dir / "valid.jsonl",
        br_source_token=args.br_source_token,
        pt_source_token=args.pt_source_token,
        cls_prefix=args.cls_prefix,
        duplicate_equal_classification=duplicate_equal_classification,
    )
    valid_topup = top_up_validation_from_train(
        train_path=args.out_dir / "train.jsonl",
        valid_path=args.out_dir / "valid.jsonl",
        existing_count=int(valid_stats.get("total_written", 0)),
        min_rows=int(args.valid_min_rows),
        seed=int(args.valid_topup_seed),
    )
    if valid_topup.get("added_rows", 0):
        valid_stats["total_written"] = int(valid_stats.get("total_written", 0)) + int(
            valid_topup["added_rows"]
        )
        for key, value in valid_topup.items():
            if key == "added_rows":
                continue
            valid_stats[key] = int(valid_stats.get(key, 0)) + int(value)

    report = {
        "mode": args.mode,
        "br_source_token": args.br_source_token,
        "pt_source_token": args.pt_source_token,
        "cls_prefix": args.cls_prefix,
        "duplicate_equal_classification": duplicate_equal_classification,
        "translation_train": args.translation_train.as_posix(),
        "translation_valid": args.translation_valid.as_posix(),
        "classification_train": args.classification_train.as_posix()
        if args.classification_train
        else None,
        "classification_valid": args.classification_valid.as_posix()
        if args.classification_valid
        else None,
        "train": train_stats,
        "valid": valid_stats,
        "valid_topup_from_train": valid_topup,
        "out_dir": args.out_dir.as_posix(),
    }
    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
