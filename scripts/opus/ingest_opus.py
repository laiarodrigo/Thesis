#!/usr/bin/env python3
# scripts/ingest_opus.py
import argparse
import sys
from pathlib import Path
import duckdb

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(p: str | None) -> Path | None:
    if p is None:
        return None
    return Path(p).expanduser().resolve()


def find_files(cache_dir: Path, br_file: Path | None, pt_file: Path | None) -> tuple[Path, Path]:
    if br_file and pt_file:
        return br_file, pt_file

    # Prefer explicit OpenSubtitles OPUS names if present
    prefer_br = cache_dir / "OpenSubtitles.pt-pt_BR.pt_BR"
    prefer_pt = cache_dir / "OpenSubtitles.pt-pt_BR.pt"
    if not br_file and not pt_file and prefer_br.exists() and prefer_pt.exists():
        return prefer_br, prefer_pt

    br_candidates = sorted(p for p in cache_dir.iterdir() if p.is_file() and p.name.endswith(".pt_BR"))
    pt_candidates = sorted(
        p for p in cache_dir.iterdir()
        if p.is_file() and p.name.endswith(".pt") and not p.name.endswith(".pt_BR")
    )

    if br_file:
        br_candidates = [br_file]
    if pt_file:
        pt_candidates = [pt_file]

    if len(br_candidates) != 1 or len(pt_candidates) != 1:
        msg = ["Could not uniquely identify the two OPUS files."]
        msg.append(f"cache_dir: {cache_dir}")
        msg.append(f".pt_BR candidates: {[p.name for p in br_candidates]}")
        msg.append(f".pt candidates: {[p.name for p in pt_candidates]}")
        msg.append("Provide --br-file and --pt-file to disambiguate.")
        raise RuntimeError("\n".join(msg))

    return br_candidates[0], pt_candidates[0]


def count_lines(fp: Path) -> int:
    import mmap
    with fp.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return mm.read().count(b"\n")


def ensure_schema(con: duckdb.DuckDBPyConnection, *, dedupe: bool) -> None:
    con.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_opus_moses START 1;

        CREATE TABLE IF NOT EXISTS opus_moses (
            line_no     BIGINT PRIMARY KEY,                 -- 1-based position
            pair_id     BIGINT DEFAULT nextval('seq_opus_moses'),
            sent_pt_br  TEXT,
            sent_pt_pt  TEXT
        );
    """)

    cols = {r[0] for r in con.execute("PRAGMA table_info('opus_moses')").fetchall()}
    required = {"line_no", "pair_id", "sent_pt_br", "sent_pt_pt"}
    if not required.issubset(cols):
        raise RuntimeError(
            "opus_moses exists but does not match the expected schema. "
            "Expected columns: line_no, pair_id, sent_pt_br, sent_pt_pt. "
            "Consider recreating the table or using a fresh DB."
        )

    if dedupe:
        con.execute("""
            DELETE FROM opus_moses
            USING (
                SELECT line_no,
                       ROW_NUMBER() OVER (
                           PARTITION BY sent_pt_br, sent_pt_pt
                           ORDER BY line_no
                       ) AS dup
                FROM opus_moses
            ) AS tmp
            WHERE opus_moses.line_no = tmp.line_no
              AND tmp.dup > 1;
        """)
        con.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_text_pair
                ON opus_moses(sent_pt_br, sent_pt_pt);
        """)


def iter_pairs(br_path: Path, pt_path: Path, start_at: int):
    with br_path.open("r", encoding="utf-8") as br_f, pt_path.open("r", encoding="utf-8") as pt_f:
        for _ in range(start_at):
            next(br_f)
            next(pt_f)

        for ln, (br, pt) in enumerate(zip(br_f, pt_f), start_at + 1):
            yield ln, br.rstrip("\n"), pt.rstrip("\n")


def main() -> int:
    root = _repo_root()

    ap = argparse.ArgumentParser(
        description="Ingest OPUS OpenSubtitles pairs into DuckDB (opus_moses)."
    )
    ap.add_argument("--cache-dir", default=str(root / "data/opus_cache"),
                    help="Directory containing OPUS .pt/.pt_BR files (default: data/opus_cache)")
    ap.add_argument("--db", default=str(root / "data/duckdb/subs.duckdb"),
                    help="DuckDB path (default: data/duckdb/subs.duckdb)")
    ap.add_argument("--br-file", default=None,
                    help="Explicit PT-BR file (ends with .pt_BR)")
    ap.add_argument("--pt-file", default=None,
                    help="Explicit PT-PT file (ends with .pt, but not .pt_BR)")
    ap.add_argument("--batch-size", type=int, default=50_000,
                    help="Insert batch size (default: 50000)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Do not resume; start from line 1")
    ap.add_argument("--skip-dedupe", action="store_true",
                    help="Skip de-dup + unique text index creation")
    ap.add_argument("--checkpoint-threshold", default="100MB",
                    help="DuckDB checkpoint_threshold (default: 100MB)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only resolve paths and exit")
    args = ap.parse_args()

    cache_dir = _resolve_path(args.cache_dir) or (root / "data/opus_cache")
    db_path = _resolve_path(args.db) or (root / "data/duckdb/subs.duckdb")
    br_file = _resolve_path(args.br_file)
    pt_file = _resolve_path(args.pt_file)

    if not cache_dir.exists():
        print(f"[error] cache dir not found: {cache_dir}", file=sys.stderr)
        return 1

    try:
        br_path, pt_path = find_files(cache_dir, br_file, pt_file)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    print(f"BR file: {br_path}")
    print(f"PT file: {pt_path}")
    print(f"DB: {db_path}")

    if args.dry_run:
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("Counting lines...", flush=True)
    total_lines = count_lines(br_path)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"SET checkpoint_threshold='{args.checkpoint_threshold}'")
        con.execute("SET enable_progress_bar=true")
        ensure_schema(con, dedupe=(not args.skip_dedupe))

        done = 0
        if not args.no_resume:
            done = con.execute("SELECT MAX(line_no) FROM opus_moses").fetchone()[0] or 0
        print(f"Resume line: {done:,}")

        batch = []
        total = max(0, total_lines - done)
        iterator = iter_pairs(br_path, pt_path, done)
        if tqdm is not None:
            iterator = tqdm(iterator, total=total, desc="Importing", unit="pairs")

        for ln, br, pt in iterator:
            batch.append((ln, br, pt))
            if len(batch) >= args.batch_size:
                con.executemany(
                    "INSERT OR IGNORE INTO opus_moses (line_no, sent_pt_br, sent_pt_pt) VALUES (?, ?, ?)",
                    batch,
                )
                batch.clear()

        if batch:
            con.executemany(
                "INSERT OR IGNORE INTO opus_moses (line_no, sent_pt_br, sent_pt_pt) VALUES (?, ?, ?)",
                batch,
            )

        con.execute("PRAGMA force_checkpoint")
        con.execute("CHECKPOINT")
    finally:
        con.close()

    print("[ok] ingest complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
