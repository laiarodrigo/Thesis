#!/usr/bin/env python3
"""
Export Golden Collection test rows from DuckDB into encoder-decoder eval JSONL files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import duckdb


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Export Golden Collection eval datasets from DuckDB.")
    parser.add_argument(
        "--project-db",
        type=Path,
        default=repo_root / "data" / "duckdb" / "subs_project.duckdb",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=repo_root / "data" / "duckdb" / "subs.duckdb",
    )
    parser.add_argument("--view", default="test_data")
    parser.add_argument("--dataset", default="Gold")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "golden_collection",
    )
    parser.add_argument("--translation-file", default="translation_test.jsonl")
    parser.add_argument("--classification-file", default="classification_test.jsonl")
    parser.add_argument("--br2pt-token", default="<br-pt>")
    parser.add_argument("--pt2br-token", default="<pt-br>")
    parser.add_argument("--classification-token", default="<id>")
    return parser.parse_args()


def as_clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def connect_project(project_db: Path, source_db: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(project_db.as_posix(), read_only=True)
    dbs = {row[1] for row in con.execute("PRAGMA database_list").fetchall()}
    if "src" not in dbs:
        con.execute(f"ATTACH '{source_db.as_posix()}' AS src")
    return con


def stream_golden_rows(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str,
    dataset_name: str,
    split_name: str | None,
    batch_size: int,
) -> Iterator[dict[str, object]]:
    query = f"""
        SELECT
            ROW_NUMBER() OVER () AS source_id,
            dataset,
            split,
            source,
            bucket,
            theme,
            label,
            text_pt_br,
            text_pt_pt,
            ref_pt_pt_manual,
            ref_pt_pt_deepl
        FROM {view_name}
        WHERE lower(dataset) = lower(?)
    """
    params: list[object] = [dataset_name]
    if split_name:
        query += " AND lower(split) = lower(?)"
        params.append(split_name)
    query += " ORDER BY source_id"

    cur = con.execute(query, params)
    columns = [d[0] for d in cur.description]
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            yield dict(zip(columns, row))


def to_label_id(label: str) -> int:
    if label == "pt-br":
        return 0
    if label == "pt-pt":
        return 1
    return 2


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    translation_out = args.out_dir / args.translation_file
    classification_out = args.out_dir / args.classification_file

    con = connect_project(args.project_db, args.source_db)
    source_rows = 0
    translation_examples = 0
    classification_examples = 0

    with (
        translation_out.open("w", encoding="utf-8") as trans_fh,
        classification_out.open("w", encoding="utf-8") as cls_fh,
    ):
        for row in stream_golden_rows(
            con,
            view_name=args.view,
            dataset_name=args.dataset,
            split_name=args.split,
            batch_size=args.batch_size,
        ):
            source_rows += 1
            source_id = int(row["source_id"])
            text_br = as_clean_text(row.get("text_pt_br"))
            text_pt = (
                as_clean_text(row.get("text_pt_pt"))
                or as_clean_text(row.get("ref_pt_pt_manual"))
                or as_clean_text(row.get("ref_pt_pt_deepl"))
            )

            if text_br and text_pt:
                is_equal = normalize_space(text_br) == normalize_space(text_pt)

                trans_fh.write(
                    json.dumps(
                        {
                            "id": translation_examples,
                            "source_id": source_id,
                            "task": "translation",
                            "direction": "br2pt",
                            "input_text": f"{args.br2pt_token} {text_br}",
                            "target_text": text_pt,
                            "source_text": text_br,
                            "target_variant": "pt-pt",
                            "is_equal_pair": is_equal,
                            "dataset": row.get("dataset"),
                            "bucket": row.get("bucket"),
                            "source": row.get("source"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                translation_examples += 1

                trans_fh.write(
                    json.dumps(
                        {
                            "id": translation_examples,
                            "source_id": source_id,
                            "task": "translation",
                            "direction": "pt2br",
                            "input_text": f"{args.pt2br_token} {text_pt}",
                            "target_text": text_br,
                            "source_text": text_pt,
                            "target_variant": "pt-br",
                            "is_equal_pair": is_equal,
                            "dataset": row.get("dataset"),
                            "bucket": row.get("bucket"),
                            "source": row.get("source"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                translation_examples += 1

                if is_equal:
                    label = "equal"
                    text = text_br
                    cls_fh.write(
                        json.dumps(
                            {
                                "source_id": source_id,
                                "task": "classification",
                                "text": text,
                                "input_text": f"{args.classification_token} {text}",
                                "label": label,
                                "label_id": to_label_id(label),
                                "dataset": row.get("dataset"),
                                "bucket": row.get("bucket"),
                                "source": row.get("source"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    classification_examples += 1
                else:
                    for text, label in ((text_br, "pt-br"), (text_pt, "pt-pt")):
                        cls_fh.write(
                            json.dumps(
                                {
                                    "source_id": source_id,
                                    "task": "classification",
                                    "text": text,
                                    "input_text": f"{args.classification_token} {text}",
                                    "label": label,
                                    "label_id": to_label_id(label),
                                    "dataset": row.get("dataset"),
                                    "bucket": row.get("bucket"),
                                    "source": row.get("source"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        classification_examples += 1
                continue

            if text_br:
                label = "pt-br"
                cls_fh.write(
                    json.dumps(
                        {
                            "source_id": source_id,
                            "task": "classification",
                            "text": text_br,
                            "input_text": f"{args.classification_token} {text_br}",
                            "label": label,
                            "label_id": to_label_id(label),
                            "dataset": row.get("dataset"),
                            "bucket": row.get("bucket"),
                            "source": row.get("source"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                classification_examples += 1

            if text_pt:
                label = "pt-pt"
                cls_fh.write(
                    json.dumps(
                        {
                            "source_id": source_id,
                            "task": "classification",
                            "text": text_pt,
                            "input_text": f"{args.classification_token} {text_pt}",
                            "label": label,
                            "label_id": to_label_id(label),
                            "dataset": row.get("dataset"),
                            "bucket": row.get("bucket"),
                            "source": row.get("source"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                classification_examples += 1

    con.close()

    print("Golden export completed")
    print(f"  source rows: {source_rows}")
    print(f"  translation examples: {translation_examples}")
    print(f"  classification examples: {classification_examples}")
    print(f"  translation file: {translation_out}")
    print(f"  classification file: {classification_out}")


if __name__ == "__main__":
    main()
