# export_duckdb_streaming.py
import duckdb, os, io, re, json
from pathlib import Path

DB = "data/duckdb/subs.duckdb"
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# choose your ordering key — use line_no if it’s contiguous/stable; otherwise pair_id
ORDER_COL = "line_no"   # or "pair_id"

# sanitize so every row becomes exactly one line in the .raw files
CLEAN_RE = re.compile(r"[\r\n\t]+")
def clean(s: str) -> str:
    return CLEAN_RE.sub(" ", (s or "")).strip()

# stream rows in big chunks
CHUNK = 200_000

con = duckdb.connect(DB)
con.execute("PRAGMA threads=4")

query = f"""
SELECT pair_id, sent_pt_br, sent_pt_pt, {ORDER_COL} AS ord
FROM opus_moses
ORDER BY ord
"""

# outputs
fbr  = io.open(OUT_DIR / "ptBR.raw", "w", encoding="utf8", newline="\n")
fpt  = io.open(OUT_DIR / "ptPT.raw", "w", encoding="utf8", newline="\n")
fmap = io.open(OUT_DIR / "pairid_to_idx.tsv", "w", encoding="utf8", newline="\n")
# (optional) if ORDER_COL is already a 0-based contiguous index, you can skip fmap entirely

rows = 0
idx  = 0

# try fast Arrow batches; fall back to fetchmany() if Arrow isn't available
try:
    it = con.execute(query).fetch_record_batch(rows_per_batch=CHUNK)
    for batch in it:
        pid_col = batch.column(0)  # pair_id
        br_col  = batch.column(1)  # sent_pt_br
        pt_col  = batch.column(2)  # sent_pt_pt
        for i in range(batch.num_rows):
            pid = pid_col[i].as_py()
            br  = clean(br_col[i].as_py())
            pt  = clean(pt_col[i].as_py())
            fbr.write(br + "\n"); fpt.write(pt + "\n")
            fmap.write(f"{pid}\t{idx}\n")
            idx += 1; rows += 1
except Exception:
    cur = con.execute(query)
    while True:
        chunk = cur.fetchmany(CHUNK)
        if not chunk: break
        for pid, br, pt, _ord in chunk:
            br  = clean(br)
            pt  = clean(pt)
            fbr.write(br + "\n"); fpt.write(pt + "\n")
            fmap.write(f"{pid}\t{idx}\n")
            idx += 1; rows += 1

fbr.close(); fpt.close(); fmap.close()
print(f"wrote {rows:,} rows to data/ptBR.raw + data/ptPT.raw")
print("pairid_to_idx.tsv has: <pair_id>\\t<0-based-line-index>")
