#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Split PtBrVId translated pair CSV into canonical pair splits and "
            "encoder-decoder translation/classification JSONLs."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "data" / "ptbrvarid" / "translated_stageb_pairs" / "translated_pairs.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "ptbrvarid_translated_stageB",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument(
        "--test-per-group",
        type=int,
        default=0,
        help=(
            "If > 0, reserve exactly this many canonical pairs per (domain, source_label) "
            "for the test split before allocating the remainder to train/valid."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--br2pt-token", default="<br-pt>")
    parser.add_argument("--pt2br-token", default="<pt-br>")
    parser.add_argument("--classification-token", default="<id>")
    parser.add_argument(
        "--exclude-domains",
        default="",
        help="Comma-separated domain names to exclude, e.g. 'social_media,web'.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def parse_excluded_domains(raw: str) -> set[str]:
    return {
        normalize_space(part).casefold()
        for part in str(raw or "").split(",")
        if normalize_space(part)
    }


def read_rows(path: Path, *, excluded_domains: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"pt_BR", "pt_PT"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Input CSV missing required columns: {', '.join(sorted(missing))}")
        rows: list[dict[str, str]] = []
        for idx, row in enumerate(reader, start=1):
            pt_br = normalize_space(str(row.get("pt_BR") or ""))
            pt_pt = normalize_space(str(row.get("pt_PT") or ""))
            if not pt_br or not pt_pt:
                continue
            item = {key: str(value or "") for key, value in row.items()}
            item["pt_BR"] = pt_br
            item["pt_PT"] = pt_pt
            item["sample_id"] = normalize_space(item.get("sample_id") or str(idx))
            item["domain"] = normalize_space(item.get("domain") or "unknown")
            item["source_label"] = normalize_space(item.get("source_label") or item.get("label") or "unknown")
            if item["domain"].casefold() in excluded_domains:
                continue
            rows.append(item)
    if not rows:
        raise SystemExit(f"No usable translated rows found in {path}")
    return rows


def split_group(
    rows: list[dict[str, str]],
    *,
    train_ratio: float,
    valid_ratio: float,
    test_per_group: int,
    seed: int,
    domain: str,
    source_label: str,
) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(f"{seed}:{domain}:{source_label}")
    items = list(rows)
    rng.shuffle(items)
    n = len(items)
    if test_per_group > 0:
        if n <= test_per_group:
            raise SystemExit(
                f"Group ({domain}, {source_label}) has only {n} rows, cannot reserve "
                f"{test_per_group} for test."
            )
        n_test = test_per_group
        remaining = n - n_test
        n_valid = int(n * valid_ratio)
        n_valid = min(n_valid, remaining)
        n_train = remaining - n_valid
    else:
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        n_test = n - n_train - n_valid
    return {
        "train": items[:n_train],
        "valid": items[n_train : n_train + n_valid],
        "test": items[n_train + n_valid : n_train + n_valid + n_test],
    }


def build_pair_record(row: dict[str, str]) -> dict[str, Any]:
    pt_br = row["pt_BR"]
    pt_pt = row["pt_PT"]
    return {
        "source_id": row["sample_id"],
        "dataset": "PtBrVId",
        "domain": row["domain"],
        "source_label": row["source_label"],
        "pt_br": pt_br,
        "pt_pt": pt_pt,
        "is_equal": pt_br == pt_pt,
    }


def build_translation_examples(
    pair: dict[str, Any],
    *,
    br2pt_token: str,
    pt2br_token: str,
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": pair["source_id"],
            "dataset": pair["dataset"],
            "domain": pair["domain"],
            "task": "translation",
            "direction": "br2pt",
            "input_text": f"{br2pt_token} {pair['pt_br']}",
            "target_text": pair["pt_pt"],
            "source_text": pair["pt_br"],
            "target_variant": "pt-pt",
            "is_equal_pair": bool(pair["is_equal"]),
        },
        {
            "source_id": pair["source_id"],
            "dataset": pair["dataset"],
            "domain": pair["domain"],
            "task": "translation",
            "direction": "pt2br",
            "input_text": f"{pt2br_token} {pair['pt_pt']}",
            "target_text": pair["pt_br"],
            "source_text": pair["pt_pt"],
            "target_variant": "pt-br",
            "is_equal_pair": bool(pair["is_equal"]),
        },
    ]


def build_classification_examples(pair: dict[str, Any], *, classification_token: str) -> list[dict[str, Any]]:
    if bool(pair["is_equal"]):
        return []
    return [
        {
            "source_id": pair["source_id"],
            "dataset": pair["dataset"],
            "domain": pair["domain"],
            "task": "classify",
            "input_text": f"{classification_token} {pair['pt_br']}",
            "target_text": "pt-br",
            "text": pair["pt_br"],
        },
        {
            "source_id": pair["source_id"],
            "dataset": pair["dataset"],
            "domain": pair["domain"],
            "task": "classify",
            "input_text": f"{classification_token} {pair['pt_pt']}",
            "target_text": "pt-pt",
            "text": pair["pt_pt"],
        },
    ]


def build_source_classification_example(
    pair: dict[str, Any],
    *,
    classification_token: str,
) -> dict[str, Any]:
    source_label = normalize_space(str(pair.get("source_label") or "")).casefold()
    if source_label == "pt-br":
        source_text = normalize_space(str(pair.get("pt_br") or ""))
        target_text = "pt-br"
    elif source_label == "pt-pt":
        source_text = normalize_space(str(pair.get("pt_pt") or ""))
        target_text = "pt-pt"
    else:
        raise ValueError(f"Unsupported source_label in pair: {pair.get('source_label')!r}")
    return {
        "source_id": pair["source_id"],
        "dataset": pair["dataset"],
        "domain": pair["domain"],
        "task": "classify",
        "input_text": f"{classification_token} {source_text}",
        "target_text": target_text,
        "text": source_text,
        "source_label": target_text,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("--train-ratio must be in (0, 1)")
    if not (0.0 <= args.valid_ratio < 1.0):
        raise SystemExit("--valid-ratio must be in [0, 1)")
    if args.train_ratio + args.valid_ratio >= 1.0:
        raise SystemExit("--train-ratio + --valid-ratio must be < 1")
    if args.test_per_group < 0:
        raise SystemExit("--test-per-group must be >= 0")

    excluded_domains = parse_excluded_domains(args.exclude_domains)
    rows = read_rows(args.input_csv, excluded_domains=excluded_domains)
    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row["domain"], row["source_label"])].append(row)

    split_pairs: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    group_report: list[dict[str, Any]] = []
    for (domain, source_label), group_items in sorted(grouped_rows.items()):
        group_splits = split_group(
            group_items,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_per_group=args.test_per_group,
            seed=args.seed,
            domain=domain,
            source_label=source_label,
        )
        group_report.append(
            {
                "domain": domain,
                "source_label": source_label,
                "pairs_total": len(group_items),
                "pairs_train": len(group_splits["train"]),
                "pairs_valid": len(group_splits["valid"]),
                "pairs_test": len(group_splits["test"]),
            }
        )
        for split_name, split_items in group_splits.items():
            split_pairs[split_name].extend(build_pair_record(row) for row in split_items)

    report: dict[str, Any] = {
        "input_csv": args.input_csv.as_posix(),
        "out_dir": args.out_dir.as_posix(),
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_per_group": args.test_per_group,
        "seed": args.seed,
        "excluded_domains": sorted(excluded_domains),
        "group_report": group_report,
        "splits": {},
    }

    for split_name, pairs in split_pairs.items():
        translation_rows: list[dict[str, Any]] = []
        classification_rows: list[dict[str, Any]] = []
        classification_source_rows: list[dict[str, Any]] = []
        for pair in pairs:
            translation_rows.extend(
                build_translation_examples(
                    pair,
                    br2pt_token=args.br2pt_token,
                    pt2br_token=args.pt2br_token,
                )
            )
            classification_rows.extend(
                build_classification_examples(
                    pair,
                    classification_token=args.classification_token,
                )
            )
            classification_source_rows.append(
                build_source_classification_example(
                    pair,
                    classification_token=args.classification_token,
                )
            )

        write_jsonl(args.out_dir / f"pairs_{split_name}.jsonl", pairs)
        write_jsonl(args.out_dir / f"translation_{split_name}.jsonl", translation_rows)
        write_jsonl(args.out_dir / f"classification_{split_name}.jsonl", classification_rows)
        write_jsonl(
            args.out_dir / f"classification_source_{split_name}.jsonl",
            classification_source_rows,
        )

        translation_directions = Counter(row["direction"] for row in translation_rows)
        classification_labels = Counter(row["target_text"] for row in classification_rows)
        classification_source_labels = Counter(row["target_text"] for row in classification_source_rows)
        report["splits"][split_name] = {
            "pairs": len(pairs),
            "translation_examples": len(translation_rows),
            "classification_examples": len(classification_rows),
            "classification_source_examples": len(classification_source_rows),
            "translation_directions": dict(sorted(translation_directions.items())),
            "classification_labels": dict(sorted(classification_labels.items())),
            "classification_source_labels": dict(sorted(classification_source_labels.items())),
        }

    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
