#!/usr/bin/env python3
# scripts/pass_1_opus.py
import sys, pathlib, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
# make both repo root and src importable
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# try imports from your modules (opus_pipeline.py first)
try:
    from opus_pipeline import (
        plan_ops_over_corpus,
        apply_ops_ctas_swap,
        apply_neighbor_moves_corpus_inplace,
        run_repeated_prefix_cleaner_chunked,
    )
except Exception as e:
    print("Cannot import pipeline functions:", e)
    sys.exit(2)

DB = "/cfs/home/u036584/projects/thesis/data/duckdb/subs.duckdb"

def main():
    t0 = time.time()
    print(f"PASS 1 starting on {DB}")

    # 1) merge + swap
    plan_ops_over_corpus(block_size=50_000, reset=False, db_path=DB)
    apply_ops_ctas_swap(db_path=DB)

    # 2) neighbor clause moves (in-place)
    apply_neighbor_moves_corpus_inplace(
        block_size=50_000, overlap=3, margin=0.04, max_clause_chars=60, max_iters=5
    )

    # 3) repeated-prefix cleaner (in-place)
    run_repeated_prefix_cleaner_chunked(
        db_path=DB,
        prev_window=6, min_prefix_tokens=6, coverage_thresh=0.92,
        require_both=False, collapse_adjacent_dups=True,
        delete_on_trigger=True, delete_if_empty_only=False,
        chunk_size=50_000,
        start_line=None, end_line=None,
        apply_changes=True, print_updates=False, trace_lines=None
    )
    print(f"PASS 1 done in {(time.time()-t0)/3600:.2f} h")

if __name__ == "__main__":
    main()
