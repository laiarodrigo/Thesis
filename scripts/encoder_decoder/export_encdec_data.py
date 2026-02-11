#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional

import duckdb


SELECT_COLUMNS = [
    "dataset",
    "split",
    "source",
    "bucket",
    "theme",
    "domain",
    "label",
    "text_pt_br",
    "text_pt_pt",
    "ref_pt_pt_manual",
    "ref_pt_pt_deepl",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_project_db = repo_root / "data" / "duckdb" / "subs_project.duckdb"
    default_source_db = repo_root / "data" / "duckdb" / "subs.duckdb"
    default_out_dir = repo_root / "data" / "encoder_decoder"

    parser = argparse.ArgumentParser(
        description=(
            "Export encoder-decoder datasets for translation and 3-way classification "
            "(pt-br, pt-pt, equal) from DuckDB views."
        )
    )
    parser.add_argument("--project-db", type=Path, default=default_project_db)
    parser.add_argument("--source-db", type=Path, default=default_source_db)
    parser.add_argument("--train-view", default="train_data")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-view", default="train_data")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--translation-train-file", default="translation_train.jsonl")
    parser.add_argument("--translation-valid-file", default="translation_valid.jsonl")
    parser.add_argument("--classification-train-file", default="classification_train.jsonl")
    parser.add_argument("--classification-valid-file", default="classification_valid.jsonl")
    parser.add_argument(
        "--classification-token",
        default="<id>",
        help="Task token prefix used for classification inputs.",
    )
    parser.add_argument(
        "--br2pt-token",
        default="<br-pt>",
        help="Task token prefix used for BR->PT translation inputs.",
    )
    parser.add_argument(
        "--pt2br-token",
        default="<pt-br>",
        help="Task token prefix used for PT->BR translation inputs.",
    )
    return parser.parse_args()


def as_clean_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def canonical_label(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    t = normalize_space(raw).lower()
    t = t.replace("_", "-")

    if t in {"pt-br", "ptbr", "portugues do brasil", "português do brasil"}:
        return "pt-br"
    if t in {"pt-pt", "ptpt", "portugues europeu", "português europeu", "portugal"}:
        return "pt-pt"
    if t in {"equal", "shared", "same"}:
        return "equal"
    return None


def connect_project(project_db: Path, source_db: Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(project_db.as_posix(), read_only=read_only)
    dbl = con.execute("PRAGMA database_list").fetchall()
    names = {row[1] for row in dbl}
    if "src" not in names:
        con.execute(f"ATTACH '{source_db.as_posix()}' AS src")
    return con


def count_rows_for_split(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str,
    split: Optional[str],
) -> int:
    query = f"SELECT COUNT(*) FROM {view_name}"
    params = []
    if split is not None:
        query += " WHERE lower(split) = ?"
        params.append(split.lower())
    return int(con.execute(query, params).fetchone()[0])


def stream_from_view(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    split: Optional[str] = None,
    batch_size: int = 10_000,
    limit: Optional[int] = None,
) -> Iterator[Dict]:
    offset = 0
    selected = ", ".join(SELECT_COLUMNS)

    base_query = f"SELECT {selected} FROM {view_name}"
    params = []
    if split is not None:
        base_query += " WHERE lower(split) = ?"
        params.append(split.lower())

    while True:
        if limit is not None:
            remaining = limit - offset
            if remaining <= 0:
                break
            cur_batch = min(batch_size, remaining)
        else:
            cur_batch = batch_size

        rows = con.execute(base_query + " LIMIT ? OFFSET ?", [*params, cur_batch, offset]).fetchall()
        if not rows:
            break

        for row in rows:
            yield dict(zip(SELECT_COLUMNS, row))

        offset += len(rows)


def build_translation_examples(row: Dict, br2pt_token: str, pt2br_token: str) -> list[Dict]:
    examples = []
    src_br = as_clean_text(row.get("text_pt_br"))
    src_pt = as_clean_text(row.get("text_pt_pt"))
    if not src_br or not src_pt:
        return examples

    examples.append(
        {
            "input_text": f"{br2pt_token} {src_br}",
            "target_text": src_pt,
            "task": "translate_br2pt",
            "dataset": row.get("dataset"),
            "bucket": row.get("bucket"),
        }
    )
    examples.append(
        {
            "input_text": f"{pt2br_token} {src_pt}",
            "target_text": src_br,
            "task": "translate_pt2br",
            "dataset": row.get("dataset"),
            "bucket": row.get("bucket"),
        }
    )
    return examples


def build_classification_examples(row: Dict, classification_token: str) -> list[Dict]:
    examples = []
    src_br = as_clean_text(row.get("text_pt_br"))
    src_pt = as_clean_text(row.get("text_pt_pt"))
    label = canonical_label(as_clean_text(row.get("label")))

    # If an explicit dataset label exists, trust it.
    if label in {"pt-br", "pt-pt", "equal"}:
        text = src_br or src_pt
        if text:
            examples.append(
                {
                    "input_text": f"{classification_token} {text}",
                    "target_text": label,
                    "task": "classify",
                    "dataset": row.get("dataset"),
                    "bucket": row.get("bucket"),
                }
            )
        return examples

    has_br = bool(src_br)
    has_pt = bool(src_pt)

    if has_br and has_pt:
        if normalize_space(src_br) == normalize_space(src_pt):
            examples.append(
                {
                    "input_text": f"{classification_token} {src_br}",
                    "target_text": "equal",
                    "task": "classify",
                    "dataset": row.get("dataset"),
                    "bucket": row.get("bucket"),
                }
            )
            return examples

        examples.append(
            {
                "input_text": f"{classification_token} {src_br}",
                "target_text": "pt-br",
                "task": "classify",
                "dataset": row.get("dataset"),
                "bucket": row.get("bucket"),
            }
        )
        examples.append(
            {
                "input_text": f"{classification_token} {src_pt}",
                "target_text": "pt-pt",
                "task": "classify",
                "dataset": row.get("dataset"),
                "bucket": row.get("bucket"),
            }
        )
        return examples

    if has_br:
        examples.append(
            {
                "input_text": f"{classification_token} {src_br}",
                "target_text": "pt-br",
                "task": "classify",
                "dataset": row.get("dataset"),
                "bucket": row.get("bucket"),
            }
        )
    if has_pt:
        examples.append(
            {
                "input_text": f"{classification_token} {src_pt}",
                "target_text": "pt-pt",
                "task": "classify",
                "dataset": row.get("dataset"),
                "bucket": row.get("bucket"),
            }
        )
    return examples


def export_split(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str,
    split: str,
    limit: Optional[int],
    batch_size: int,
    translation_out_path: Path,
    classification_out_path: Path,
    br2pt_token: str,
    pt2br_token: str,
    classification_token: str,
) -> Dict:
    row_count = 0
    trans_count = 0
    cls_count = 0
    trans_task_counts = Counter()
    cls_label_counts = Counter()

    with (
        translation_out_path.open("w", encoding="utf-8") as trans_fh,
        classification_out_path.open("w", encoding="utf-8") as cls_fh,
    ):
        for row in stream_from_view(
            con=con,
            view_name=view_name,
            split=split,
            batch_size=batch_size,
            limit=limit,
        ):
            row_count += 1

            for ex in build_translation_examples(row, br2pt_token=br2pt_token, pt2br_token=pt2br_token):
                trans_fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                trans_count += 1
                trans_task_counts[ex["task"]] += 1

            for ex in build_classification_examples(row, classification_token=classification_token):
                cls_fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                cls_count += 1
                cls_label_counts[ex["target_text"]] += 1

            if row_count % 100_000 == 0:
                print(
                    f"[{split}] processed rows={row_count} "
                    f"translation_examples={trans_count} classification_examples={cls_count}"
                )

    return {
        "rows": row_count,
        "translation_examples": trans_count,
        "classification_examples": cls_count,
        "translation_tasks": dict(trans_task_counts),
        "classification_labels": dict(cls_label_counts),
    }


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    con = connect_project(args.project_db, args.source_db, read_only=True)
    try:
        print("Sanity check (source rows from DuckDB views):")
        train_source_rows = count_rows_for_split(con, view_name=args.train_view, split=args.train_split)
        valid_source_rows = count_rows_for_split(con, view_name=args.valid_view, split=args.valid_split)
        print(
            f"  train source: view={args.train_view} split={args.train_split} rows={train_source_rows}"
        )
        print(
            f"  valid source: view={args.valid_view} split={args.valid_split} rows={valid_source_rows}"
        )

        translation_train_path = args.out_dir / args.translation_train_file
        translation_valid_path = args.out_dir / args.translation_valid_file
        classification_train_path = args.out_dir / args.classification_train_file
        classification_valid_path = args.out_dir / args.classification_valid_file

        train_stats = export_split(
            con,
            view_name=args.train_view,
            split=args.train_split,
            limit=args.train_limit,
            batch_size=args.batch_size,
            translation_out_path=translation_train_path,
            classification_out_path=classification_train_path,
            br2pt_token=args.br2pt_token,
            pt2br_token=args.pt2br_token,
            classification_token=args.classification_token,
        )
        valid_stats = export_split(
            con,
            view_name=args.valid_view,
            split=args.valid_split,
            limit=args.valid_limit,
            batch_size=args.batch_size,
            translation_out_path=translation_valid_path,
            classification_out_path=classification_valid_path,
            br2pt_token=args.br2pt_token,
            pt2br_token=args.pt2br_token,
            classification_token=args.classification_token,
        )
    finally:
        con.close()

    print("\nExport completed.")
    print("Train split:")
    print(
        f"  rows={train_stats['rows']} translation_examples={train_stats['translation_examples']} "
        f"classification_examples={train_stats['classification_examples']}"
    )
    print(f"  translation_tasks={train_stats['translation_tasks']}")
    print(f"  classification_labels={train_stats['classification_labels']}")
    print("Valid split:")
    print(
        f"  rows={valid_stats['rows']} translation_examples={valid_stats['translation_examples']} "
        f"classification_examples={valid_stats['classification_examples']}"
    )
    print(f"  translation_tasks={valid_stats['translation_tasks']}")
    print(f"  classification_labels={valid_stats['classification_labels']}")

    print("\nSanity check (encdec JSONL inputs):")
    print(f"  {translation_train_path}: lines={count_lines(translation_train_path)}")
    print(f"  {translation_valid_path}: lines={count_lines(translation_valid_path)}")
    print(f"  {classification_train_path}: lines={count_lines(classification_train_path)}")
    print(f"  {classification_valid_path}: lines={count_lines(classification_valid_path)}")


if __name__ == "__main__":
    main()
