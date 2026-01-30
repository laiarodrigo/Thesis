#!/usr/bin/env python3
# scripts/ingest_ptbrvarid.py
import argparse
from pathlib import Path
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
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path.as_posix())
    try:
        con.execute(DDL)
        # reset only the RAW tag to avoid mixing runs
        con.execute("DELETE FROM ptbrvarid WHERE dataset='PtBrVId-Raw'")

        domains = get_dataset_config_names(args.dataset)
        total_written = 0

        for domain in domains:
            splits = get_dataset_split_names(args.dataset, domain)
            for split in splits:
                ds = load_dataset(args.dataset, domain, split=split, streaming=True)

                buf = []
                for ex in ds:
                    text = (ex.get("text") or "").strip()
                    if not text:
                        continue
                    lang = _label_to_lang(ex.get("label"))
                    br = text if lang == "pt-BR" else None
                    pt = text if lang == "pt-PT" else None
                    buf.append(("PtBrVId-Raw", domain, split, lang, br, pt))

                    if len(buf) >= args.batch_size:
                        con.executemany(
                            "INSERT INTO ptbrvarid VALUES (?, ?, ?, ?, ?, ?)",
                            buf,
                        )
                        total_written += len(buf)
                        buf.clear()

                if buf:
                    con.executemany(
                        "INSERT INTO ptbrvarid VALUES (?, ?, ?, ?, ?, ?)",
                        buf,
                    )
                    total_written += len(buf)

                print(f"✓ {domain}/{split}: wrote {total_written:,} total rows", flush=True)

        print(f"[ok] PtBrVId-Raw rows inserted: {total_written:,}")
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
