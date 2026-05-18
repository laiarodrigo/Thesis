#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Any

from translate_ptbrvarid_batches import (
    canonical_label,
    infer_direction,
    infer_target_variant,
    maybe_resolve_db_path,
    normalize_space,
    source_text_from_row,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Replace unresolved PtBrVId sampled rows with fresh samples from the same "
            "(domain, label) groups, preserving the rest of the sampled manifest."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=repo_root / "data" / "duckdb" / "subs_ptbr_filtered.duckdb",
        help="DuckDB file containing table ptbrvarid.",
    )
    parser.add_argument(
        "--dataset",
        default="PtBrVId",
        help="Dataset tag inside ptbrvarid (default: PtBrVId).",
    )
    parser.add_argument(
        "--sampled-rows-csv",
        type=Path,
        required=True,
        help="Existing sampled_rows.csv file.",
    )
    parser.add_argument(
        "--failures-csv",
        type=Path,
        required=True,
        help="Existing translation_failures.csv file.",
    )
    parser.add_argument(
        "--out-sampled-csv",
        type=Path,
        required=True,
        help="Output sampled_rows.csv with failed rows replaced by fresh samples.",
    )
    parser.add_argument(
        "--out-replacements-csv",
        type=Path,
        default=None,
        help="Optional CSV containing only the replacement rows.",
    )
    parser.add_argument(
        "--out-report-json",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=48,
        help="Sampling seed for replacement selection.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_candidate_rows(
    *,
    db_path: Path,
    dataset: str,
) -> dict[tuple[str, str], list[dict[str, str]]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency 'duckdb'. Use an environment with duckdb installed.") from exc

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {name for (name,) in con.execute("SHOW TABLES").fetchall()}
        if "ptbrvarid" not in tables:
            raise SystemExit(f"Table 'ptbrvarid' not found in {db_path}")

        column_names = {
            str(name).casefold()
            for _, name, *_ in con.execute("PRAGMA table_info('ptbrvarid')").fetchall()
        }
        has_dataset_column = "dataset" in column_names

        query = """
            SELECT
              COALESCE(split, '') AS split,
              lower(trim(COALESCE(domain, ''))) AS domain,
              label,
              text_pt_br,
              text_pt_pt
            FROM ptbrvarid
            WHERE lower(trim(COALESCE(domain, ''))) <> ''
              AND label IN (?, ?)
        """
        params: list[Any] = ["pt-BR", "pt-PT"]
        if has_dataset_column:
            query = query.replace(
                "WHERE lower(trim(COALESCE(domain, ''))) <> ''",
                "WHERE dataset = ?\n              AND lower(trim(COALESCE(domain, ''))) <> ''",
            )
            params.insert(0, dataset)
        rows = con.execute(query, params).fetchall()
    finally:
        con.close()

    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for split, domain, label, text_pt_br, text_pt_pt in rows:
        row = {
            "split": normalize_space(str(split or "")),
            "domain": normalize_space(str(domain or "")).casefold(),
            "label": canonical_label(str(label or "")),
            "text_pt_br": normalize_space(str(text_pt_br or "")),
            "text_pt_pt": normalize_space(str(text_pt_pt or "")),
        }
        if not row["domain"] or row["label"] not in {"pt-BR", "pt-PT"}:
            continue
        source_text = source_text_from_row(row)
        if not source_text:
            continue
        grouped[(row["domain"], row["label"])][source_text.casefold()] = row

    return {key: list(inner.values()) for key, inner in grouped.items()}


def next_sample_id(existing_rows: list[dict[str, str]]) -> int:
    max_id = 0
    for row in existing_rows:
        sample_id = normalize_space(str(row.get("sample_id") or ""))
        if not sample_id.startswith("ptbrvid_"):
            continue
        try:
            max_id = max(max_id, int(sample_id.split("_", 1)[1]))
        except ValueError:
            continue
    return max_id + 1


def main() -> None:
    args = parse_args()
    args.db = maybe_resolve_db_path(args.db)
    if not args.db.exists():
        raise SystemExit(f"DuckDB file not found: {args.db}")

    sampled_rows = read_csv_rows(args.sampled_rows_csv)
    failure_rows = read_csv_rows(args.failures_csv)
    if not sampled_rows:
        raise SystemExit(f"Empty sampled_rows CSV: {args.sampled_rows_csv}")
    if not failure_rows:
        raise SystemExit(f"Empty failures CSV: {args.failures_csv}")

    failed_ids = {normalize_space(row["sample_id"]) for row in failure_rows}
    failure_counts = Counter(
        (
            normalize_space(str(row.get("domain") or "")).casefold(),
            canonical_label(str(row.get("source_label") or "")),
        )
        for row in failure_rows
    )
    used_source_texts = {
        normalize_space(str(row.get("source_text") or "")).casefold()
        for row in sampled_rows
        if normalize_space(str(row.get("source_text") or ""))
    }

    grouped_candidates = load_candidate_rows(
        db_path=args.db,
        dataset=args.dataset,
    )

    replacement_rows: list[dict[str, str]] = []
    next_id = next_sample_id(sampled_rows)

    for group_key, needed in sorted(failure_counts.items()):
        domain, label = group_key
        pool = [
            row for row in grouped_candidates.get(group_key, [])
            if source_text_from_row(row).casefold() not in used_source_texts
        ]
        rng = Random(f"{args.seed}:{domain}:{label}:{needed}")
        rng.shuffle(pool)
        if len(pool) < needed:
            raise SystemExit(
                f"Not enough replacement rows for group {domain}/{label}. "
                f"Needed {needed}, found {len(pool)} after exclusions."
            )
        for row in pool[:needed]:
            source_text = source_text_from_row(row)
            replacement = {
                "sample_id": f"ptbrvid_{next_id:06d}",
                "dataset": args.dataset,
                "split": row["split"],
                "domain": domain,
                "label": label,
                "target_variant": infer_target_variant(label),
                "direction": infer_direction(label),
                "source_text": source_text,
                "text_pt_br": row["text_pt_br"],
                "text_pt_pt": row["text_pt_pt"],
            }
            replacement_rows.append(replacement)
            used_source_texts.add(source_text.casefold())
            next_id += 1

    kept_rows = [
        row for row in sampled_rows
        if normalize_space(str(row.get("sample_id") or "")) not in failed_ids
    ]
    merged_rows = kept_rows + replacement_rows
    merged_rows.sort(key=lambda row: normalize_space(str(row.get("sample_id") or "")))

    fieldnames = [
        "sample_id",
        "dataset",
        "split",
        "domain",
        "label",
        "target_variant",
        "direction",
        "source_text",
        "text_pt_br",
        "text_pt_pt",
    ]
    write_csv(args.out_sampled_csv, merged_rows, fieldnames)

    if args.out_replacements_csv is not None:
        write_csv(args.out_replacements_csv, replacement_rows, fieldnames)

    if args.out_report_json is not None:
        args.out_report_json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "db": args.db.as_posix(),
            "dataset": args.dataset,
            "sampled_rows_csv": args.sampled_rows_csv.as_posix(),
            "failures_csv": args.failures_csv.as_posix(),
            "out_sampled_csv": args.out_sampled_csv.as_posix(),
            "out_replacements_csv": args.out_replacements_csv.as_posix() if args.out_replacements_csv else None,
            "failed_count": len(failure_rows),
            "replacement_count": len(replacement_rows),
            "groups": [
                {"domain": domain, "label": label, "count": count}
                for (domain, label), count in sorted(failure_counts.items())
            ],
        }
        args.out_report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "failed_count": len(failure_rows),
                "replacement_count": len(replacement_rows),
                "out_sampled_csv": args.out_sampled_csv.as_posix(),
                "out_replacements_csv": args.out_replacements_csv.as_posix() if args.out_replacements_csv else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
