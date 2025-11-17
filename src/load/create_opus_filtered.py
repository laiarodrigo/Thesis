#!/usr/bin/env python3
# src/load/create_opus_filtered.py
import os
import sys
import argparse
import duckdb

def main():
    ap = argparse.ArgumentParser(description="Create opus_moses_filtered from flagged Parquet shards.")
    ap.add_argument("--db", required=True, help="Path to DuckDB file (e.g., ../data/duckdb/subs.duckdb)")
    ap.add_argument("--shards_glob", required=True,
                    help="Glob to flagged Parquet shards (e.g., ~/repos/Thesis/data/filter_shards/shard_*.parquet)")
    ap.add_argument("--persist-flags", action="store_true",
                    help="Also create/replace opus_filter_simple table from shards")
    args = ap.parse_args()

    db_path = os.path.expanduser(args.db)
    shards  = os.path.expanduser(args.shards_glob)

    # Quick existence checks
    if not os.path.exists(db_path):
        print(f"[error] DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Connect once
    con = duckdb.connect(db_path)
    try:
        # Sanity: opus_moses present?
        try:
            con.execute("SELECT 1 FROM opus_moses LIMIT 1").fetchone()
        except Exception:
            print("[error] opus_moses table not found in DB.", file=sys.stderr)
            sys.exit(1)

        # Sanity: shards are visible?
        try:
            cnt = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{shards}')").fetchone()[0]
        except Exception as e:
            print(f"[error] Could not read shards via parquet_scan('{shards}'): {e}", file=sys.stderr)
            sys.exit(1)
        if cnt == 0:
            print(f"[warn] No rows found in shards glob: {shards}")

        if args.persist_flags:
            # Materialize a clean, deduped flags table
            con.execute("""
                CREATE TABLE IF NOT EXISTS opus_filter_simple(
                    line_no BIGINT PRIMARY KEY,
                    pair_id BIGINT,
                    reason  TEXT,
                    base_sim DOUBLE,
                    thr     DOUBLE,
                    L       INTEGER
                )
            """)
            con.execute(f"""
                CREATE OR REPLACE TABLE opus_filter_simple AS
                SELECT line_no, pair_id, reason, base_sim, thr, L
                FROM (
                  SELECT *,
                         ROW_NUMBER() OVER (PARTITION BY line_no ORDER BY base_sim ASC) AS rn
                  FROM parquet_scan('{shards}')
                )
                WHERE rn = 1
            """)
            flagged = con.execute("SELECT COUNT(*) FROM opus_filter_simple").fetchone()[0]
            print(f"[flags] loaded into opus_filter_simple: {flagged:,}")

        # Counts before
        total_before = con.execute("SELECT COUNT(*) FROM opus_moses").fetchone()[0]
        print(f"[opus_moses] rows before: {total_before:,}")

        # Build filtered table via anti-join (streaming from Parquet)
        con.execute("DROP TABLE IF EXISTS opus_moses_filtered")
        con.execute(f"""
            CREATE TABLE opus_moses_filtered AS
            SELECT o.*
            FROM opus_moses o
            LEFT JOIN parquet_scan('{shards}') f USING (line_no)
            WHERE f.line_no IS NULL
            ORDER BY o.line_no
        """)

        # Indexes for downstream speed
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS opus_moses_filtered_line_pk ON opus_moses_filtered(line_no)")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS opus_moses_filtered_pair_uq  ON opus_moses_filtered(pair_id)")

        con.execute("FORCE CHECKPOINT")

        kept_after = con.execute("SELECT COUNT(*) FROM opus_moses_filtered").fetchone()[0]
        removed    = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{shards}')").fetchone()[0]
        print(f"[opus_moses_filtered] rows: {kept_after:,}")
        print(f"[removed by filter ] rows: {removed:,}")

        # Optional consistency hint (won't always match if opus_moses wasn’t the exact pre-filter total)
        if total_before is not None and kept_after is not None and removed is not None:
            approx_total = kept_after + removed
            print(f"[check] kept + removed = {approx_total:,}  (original opus_moses: {total_before:,})")

        print("[ok] opus_moses_filtered created.")
    finally:
        con.close()

if __name__ == "__main__":
    main()
