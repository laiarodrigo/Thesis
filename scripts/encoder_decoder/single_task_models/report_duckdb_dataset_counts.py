#!/usr/bin/env python3
import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Report row counts by split/dataset from DuckDB view/table.")
    parser.add_argument("--project-db", type=Path, default=repo_root / "data" / "duckdb" / "subs_project.duckdb")
    parser.add_argument("--source-db", type=Path, default=repo_root / "data" / "duckdb" / "subs.duckdb")
    parser.add_argument("--view", required=True, help="View/table name, e.g. train_data or src.train_data")
    parser.add_argument("--split", default=None, help="Optional split filter (train/valid/test)")
    parser.add_argument("--top-buckets", type=int, default=12, help="How many buckets to print if bucket column exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    con = duckdb.connect(args.project_db.as_posix(), read_only=True)
    try:
        attached_src = False
        if args.view.lower().startswith("src."):
            con.execute(f"ATTACH '{args.source_db.as_posix()}' AS src (READ_ONLY)")
            attached_src = True

        def get_cols_with_retry():
            nonlocal attached_src
            try:
                return [r[0].lower() for r in con.execute(f"SELECT * FROM {args.view} LIMIT 0").description]
            except duckdb.BinderException as e:
                msg = str(e).lower()
                if attached_src or "catalog \"src\" does not exist" not in msg:
                    raise
                con.execute(f"ATTACH '{args.source_db.as_posix()}' AS src (READ_ONLY)")
                attached_src = True
                return [r[0].lower() for r in con.execute(f"SELECT * FROM {args.view} LIMIT 0").description]

        cols = get_cols_with_retry()
        if "dataset" not in cols:
            raise SystemExit(f"Column `dataset` not found in {args.view}")

        where = ""
        params = []
        if args.split:
            if "split" not in cols:
                raise SystemExit(f"Column `split` not found in {args.view}, cannot filter by split")
            where = " WHERE lower(split)=?"
            params = [args.split.lower()]

        print(f"View: {args.view}")
        if args.split:
            print(f"Split filter: {args.split}")

        q = (
            f"SELECT coalesce(lower(split),'(null)') AS split, dataset, count(*) AS n "
            f"FROM {args.view}{where} GROUP BY 1,2 ORDER BY 1,3 DESC"
        )
        rows = con.execute(q, params).fetchall()
        if not rows:
            print("No rows found.")
            return

        print("\nCounts by split x dataset:")
        for split, dataset, n in rows:
            print(f"  split={split:>6} dataset={dataset:<20} n={n}")

        total = sum(r[2] for r in rows)
        print(f"\nTotal rows: {total}")

        if "bucket" in cols:
            q2 = (
                f"SELECT dataset, coalesce(bucket,'(null)') AS bucket, count(*) AS n "
                f"FROM {args.view}{where} GROUP BY 1,2 ORDER BY n DESC LIMIT {int(args.top_buckets)}"
            )
            print("\nTop buckets:")
            for dataset, bucket, n in con.execute(q2, params).fetchall():
                print(f"  dataset={dataset:<20} bucket={bucket:<20} n={n}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
