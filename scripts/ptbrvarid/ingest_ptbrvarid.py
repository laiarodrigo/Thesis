#!/usr/bin/env python3
# scripts/ingest_ptbrvarid.py
import argparse
from pathlib import Path
import time
import duckdb
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data/duckdb/subs.duckdb"

DDL = """
CREATE TABLE IF NOT EXISTS ptbrvarid (
  dataset     TEXT,
  domain      TEXT,
  split       TEXT,
  label       TEXT,
  text_pt_br  TEXT,
  text_pt_pt  TEXT
);
"""


def _label_to_lang(label):
    # PtBrVId-Raw uses label 1 for pt-BR, 0 for pt-PT
    try:
        if int(label) == 1:
            return "pt-BR"
        return "pt-PT"
    except Exception:
        # fallback if label is already a string
        s = str(label).strip()
        if s in ("pt-BR", "pt-PT"):
            return s
        s = s.lower().replace("_", "-")
        if s == "pt-br":
            return "pt-BR"
        if s == "pt-pt":
            return "pt-PT"
        return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest PtBrVId into DuckDB (ptbrvarid table).")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="DuckDB path (default: data/duckdb/subs.duckdb)")
    ap.add_argument("--dataset", default="liaad/PtBrVId-Raw",
                    help="HF dataset id (default: liaad/PtBrVId-Raw)")
    ap.add_argument("--batch-size", type=int, default=50_000, help="Insert batch size")
    ap.add_argument("--progress-every", type=int, default=100_000,
                    help="Print heartbeat every N streamed rows per split (0 disables)")
    ap.add_argument("--start-domain", default=None, help="Start from this domain (skip earlier domains)")
    ap.add_argument("--start-split", default=None, help="Start from this split within start-domain (skip earlier splits)")
    ap.add_argument("--keep-existing", action="store_true",
                    help="Do not delete existing PtBrVId-Raw rows (useful for resume)")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ingest_ptbrvarid] db={db_path} dataset={args.dataset} batch_size={args.batch_size}")
    print(f"[ingest_ptbrvarid] progress_every={args.progress_every}")
    if args.start_domain:
        print(f"[ingest_ptbrvarid] start_domain={args.start_domain}")
    if args.start_split:
        print(f"[ingest_ptbrvarid] start_split={args.start_split}")
    if args.keep_existing:
        print("[ingest_ptbrvarid] keep_existing=True (no delete)")

    con = duckdb.connect(db_path.as_posix())
    try:
        con.execute(DDL)
        # Ensure required columns exist (older tables might miss 'domain')
        cols = con.execute("PRAGMA table_info('ptbrvarid')").df()["name"].str.lower().tolist()
        for col in ["dataset", "domain", "split", "label", "text_pt_br", "text_pt_pt"]:
            if col not in cols:
                con.execute(f"ALTER TABLE ptbrvarid ADD COLUMN {col} TEXT")
        # reset only the RAW tag to avoid mixing runs
        if not args.keep_existing:
            con.execute("DELETE FROM ptbrvarid WHERE dataset='PtBrVId-Raw'")

        domains = get_dataset_config_names(args.dataset)
        print(f"[ingest_ptbrvarid] found {len(domains)} domains")
        total_written = 0

        started = args.start_domain is None
        for domain in domains:
            if not started:
                if domain != args.start_domain:
                    continue
                started = True
            try:
                splits = get_dataset_split_names(args.dataset, domain)
            except Exception as e:
                print(f"[ingest_ptbrvarid] ERROR listing splits for domain={domain}: {e}")
                raise
            print(f"[ingest_ptbrvarid] domain={domain} splits={splits}")
            for split in splits:
                if args.start_domain and args.start_split and domain == args.start_domain:
                    if split != args.start_split:
                        continue
                    # only skip within the start domain once
                    args.start_split = None
                print(f"[ingest_ptbrvarid] start domain={domain} split={split}")
                try:
                    ds = load_dataset(args.dataset, domain, split=split, streaming=True)
                except Exception as e:
                    print(f"[ingest_ptbrvarid] ERROR loading dataset domain={domain} split={split}: {e}")
                    raise

                buf = []
                split_streamed = 0
                split_inserted = 0
                t0 = time.time()
                next_progress = args.progress_every if args.progress_every > 0 else None
                for ex in ds:
                    split_streamed += 1
                    text = (ex.get("text") or "").strip()
                    if not text:
                        if next_progress is not None and split_streamed >= next_progress:
                            elapsed = max(time.time() - t0, 1e-9)
                            print(
                                f"[ingest_ptbrvarid] progress {domain}/{split}: "
                                f"streamed={split_streamed:,} inserted={split_inserted + len(buf):,} "
                                f"rate={split_streamed / elapsed:,.1f} rows/s",
                                flush=True,
                            )
                            next_progress += args.progress_every
                        continue
                    lang = _label_to_lang(ex.get("label"))
                    br = text if lang == "pt-BR" else None
                    pt = text if lang == "pt-PT" else None
                    buf.append(("PtBrVId-Raw", domain, split, lang, br, pt))

                    if len(buf) >= args.batch_size:
                        con.executemany(
                            "INSERT INTO ptbrvarid (dataset, domain, split, label, text_pt_br, text_pt_pt) VALUES (?, ?, ?, ?, ?, ?)",
                            buf,
                        )
                        total_written += len(buf)
                        split_inserted += len(buf)
                        buf.clear()

                    if next_progress is not None and split_streamed >= next_progress:
                        elapsed = max(time.time() - t0, 1e-9)
                        print(
                            f"[ingest_ptbrvarid] progress {domain}/{split}: "
                            f"streamed={split_streamed:,} inserted={split_inserted + len(buf):,} "
                            f"rate={split_streamed / elapsed:,.1f} rows/s",
                            flush=True,
                        )
                        next_progress += args.progress_every

                if buf:
                    con.executemany(
                        "INSERT INTO ptbrvarid (dataset, domain, split, label, text_pt_br, text_pt_pt) VALUES (?, ?, ?, ?, ?, ?)",
                        buf,
                    )
                    total_written += len(buf)
                    split_inserted += len(buf)

                elapsed = max(time.time() - t0, 1e-9)
                print(
                    f"✓ {domain}/{split}: streamed={split_streamed:,} inserted={split_inserted:,} "
                    f"elapsed={elapsed/60:.1f} min total_written={total_written:,}",
                    flush=True,
                )

        print(f"[ok] PtBrVId-Raw rows inserted: {total_written:,}")
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
