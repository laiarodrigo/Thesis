# scripts/pass_2_cpu.py
import os, json, subprocess, sys
import duckdb

DB      = "/cfs/home/u036584/projects/thesis/data/duckdb/subs.duckdb"
OUT_DIR = "/cfs/home/u036584/repos/Thesis/data/filter_shards"
SHARDER = "/cfs/home/u036584/repos/Thesis/src/shard_filter.py"

os.makedirs(OUT_DIR, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# how many worker processes and threads per worker?
WORKERS          = int(os.environ.get("WORKERS", "12"))        # processes
THREADS_PER_PROC = int(os.environ.get("THREADS_PER_PROC", "2"))# BLAS/torch threads each

THR = {"base_min_sim": 0.30, "long_min_sim": 0.70, "short_len": 8, "long_len": 28}

def die(msg):
    print("FATAL:", msg, file=sys.stderr)
    sys.exit(2)

def main():
    if not os.path.exists(DB):
        die(f"DB not found: {DB}")
    con = duckdb.connect(DB, read_only=True)
    try:
        mn, mx = con.execute("SELECT MIN(line_no), MAX(line_no) FROM opus_moses").fetchone()
    except Exception as e:
        con.close()
        die(f"opus_moses missing? {e}")
    con.close()
    mn, mx = int(mn), int(mx)
    total = mx - mn + 1

    # sanity vs available cores
    try:
        import os as _os
        cores = _os.cpu_count() or 1
    except Exception:
        cores = 1
    max_workers = max(1, cores // max(1, THREADS_PER_PROC))
    workers = min(WORKERS, max_workers)

    print(f"Cores={cores}  workers={workers}  threads/worker={THREADS_PER_PROC}", flush=True)

    base = total // workers
    rem  = total % workers
    shards, cur = [], mn
    for i in range(workers):
        size  = base + (1 if i < rem else 0)
        start = cur
        end   = cur + size - 1
        shards.append((start, end))
        cur = end + 1

    print("Shards:", shards, flush=True)

    procs = []
    for rank, (start, end) in enumerate(shards):
        outp = f"{OUT_DIR}/shard_{start}_{end}.parquet"
        cmd = ["python", "-u", SHARDER, DB, str(start), str(end), outp, json.dumps(THR)]
        env = os.environ.copy()
        # force CPU and set per-process threads
        env["FORCE_CPU"] = "1"
        env.pop("CUDA_VISIBLE_DEVICES", None)
        for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            env[var] = str(THREADS_PER_PROC)
        env["TORCH_CPU_THREADS"] = str(THREADS_PER_PROC)

        print(f"Launch: {' '.join(cmd)}  (CPU worker {rank})", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    rc = 0
    for p in procs:
        p.wait()
        rc |= p.returncode
    if rc != 0:
        die("At least one shard failed; check logs above.")

    # merge shards into DuckDB and apply the filter
    con = duckdb.connect(DB)
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
    con.execute("""
        INSERT INTO opus_filter_simple (line_no, pair_id, reason, base_sim, thr, L)
        SELECT line_no, pair_id, reason, base_sim, thr, L
        FROM read_parquet(?)
        ON CONFLICT(line_no) DO UPDATE SET
          pair_id=EXCLUDED.pair_id,
          reason=EXCLUDED.reason,
          base_sim=EXCLUDED.base_sim,
          thr=EXCLUDED.thr,
          L=EXCLUDED.L
    """, [os.path.join(OUT_DIR, "shard_*.parquet")])

    con.execute("DROP TABLE IF EXISTS opus_moses_new")
    con.execute("""
        CREATE TABLE opus_moses_new AS
        SELECT o.*
        FROM opus_moses o
        LEFT JOIN opus_filter_simple f USING (line_no)
        WHERE f.line_no IS NULL
        ORDER BY o.line_no
    """)
    con.execute("DROP TABLE opus_moses")
    con.execute("ALTER TABLE opus_moses_new RENAME TO opus_moses")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS opus_moses_line_pk ON opus_moses(line_no)")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS opus_moses_pair_uq  ON opus_moses(pair_id)")
    con.execute("FORCE CHECKPOINT")
    con.close()
    print("PASS 2 CPU complete: shards merged, filter applied.", flush=True)

if __name__ == "__main__":
    main()
