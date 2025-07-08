from pathlib import Path
import duckdb

DB_PATH = Path("../data/duckdb/subs.duckdb")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DDL = """
-- 1) Create sequences (if they don't already exist)
CREATE SEQUENCE IF NOT EXISTS seq_movies START 1;
CREATE SEQUENCE IF NOT EXISTS seq_subtitle_pairs START 1;

-- 2) Use those sequences as the DEFAULT for your PK columns
CREATE TABLE IF NOT EXISTS movies (
  movie_id BIGINT PRIMARY KEY DEFAULT nextval('seq_movies'),
  imdb_id  TEXT    UNIQUE,
  title    TEXT
);

CREATE TABLE IF NOT EXISTS subtitle_pairs (
  pair_id     BIGINT PRIMARY KEY DEFAULT nextval('seq_subtitle_pairs'),
  movie_id    BIGINT,
  pair_no     INT,
  start_pt_ms INT,
  end_pt_ms   INT,
  text_pt     TEXT,
  start_br_ms INT,
  end_br_ms   INT,
  text_br     TEXT,
  score       REAL
);
"""




def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB_PATH))
    con.execute(DDL)               
    return con
