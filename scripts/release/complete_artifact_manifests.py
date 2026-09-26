#!/usr/bin/env python3
"""Complete the dissertation model and data manifests from repository artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

CHUNK_SIZE = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_manifest(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def complete_model_rows(root: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        config_path = root / row["config_path"]
        weight_path = root / row["output_dir"] / row["weight_file"]
        if not config_path.is_file():
            row["status"] = "missing_config"
            continue
        if not weight_path.is_file():
            row["status"] = "missing_weight"
            continue
        row["size_bytes"] = str(weight_path.stat().st_size)
        row["sha256"] = sha256_file(weight_path)
        row["status"] = "ready"


def complete_data_rows(root: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        data_path = root / row["path"]
        if not data_path.is_file():
            row["status"] = "missing_data"
            continue
        row["size_bytes"] = str(data_path.stat().st_size)
        row["sha256"] = sha256_file(data_path)
        row["status"] = "ready"


def report(label: str, rows: list[dict[str, str]]) -> int:
    counts = Counter(row["status"] for row in rows)
    ready_bytes = sum(
        int(row["size_bytes"])
        for row in rows
        if row["status"] == "ready" and row["size_bytes"]
    )
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(f"{label}: rows={len(rows)}, {summary}, ready_bytes={ready_bytes}")
    return sum(count for status, count in counts.items() if status != "ready")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Calculate sizes and SHA-256 hashes for dissertation artifacts."
    )
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("results/dissertation/model_manifest.csv"),
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=Path("data/manifests/dissertation_data_manifest.csv"),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write completed fields back to the two CSV files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a nonzero status if any artifact or config is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    model_path = root / args.model_manifest
    data_path = root / args.data_manifest

    model_fields, model_rows = read_manifest(model_path)
    data_fields, data_rows = read_manifest(data_path)
    if len(model_rows) != 34:
        raise ValueError(f"Expected 34 model rows, found {len(model_rows)}")
    if len(data_rows) != 26:
        raise ValueError(f"Expected 26 data rows, found {len(data_rows)}")

    complete_model_rows(root, model_rows)
    complete_data_rows(root, data_rows)
    missing = report("models", model_rows) + report("data", data_rows)

    if args.write:
        write_manifest(model_path, model_fields, model_rows)
        write_manifest(data_path, data_fields, data_rows)
        print("Updated both manifests.")

    return 1 if args.strict and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
