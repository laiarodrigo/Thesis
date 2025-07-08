from __future__ import annotations
import time, csv, gzip, requests, random
from pathlib import Path
import pandas as pd
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
from db import connect
from extract.access_open_subtitles import API_BASE, BASE_HEADERS   # you already have these

RAW_CATALOG_DIR = Path("../data/raw/catalogs")
RAW_CATALOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_REQ_PER_10S = 40
PAUSE_PER_REQ   = 10 / MAX_REQ_PER_10S

@retry(
    wait=wait_exponential_jitter(initial=1, max=600, jitter=0.3),
    stop=stop_after_attempt(9999),
    reraise=True,
)
def _page(params: dict) -> dict:
    r = requests.get(f"{API_BASE}/subtitles", headers=BASE_HEADERS, params=params, timeout=15)
    if 500 <= r.status_code < 600:
        r.raise_for_status()
    return r.json()

def _catalog_one(lang: str, year: int) -> Path:
    """Download / resume one (lang, year) catalogue and store as CSV."""
    out_csv = RAW_CATALOG_DIR / f"subs_{lang}_{year}.csv"
    seen_ids = set()

    # resume support
    if out_csv.exists():
        df_prev  = pd.read_csv(out_csv)
        seen_ids = set(df_prev["imdb_id"])
        start_p  = df_prev["page"].max() + 1
    else:
        with out_csv.open("w", newline="") as f:
            csv.DictWriter(f, ["page", "imdb_id", "title"]).writeheader()
        start_p = 1

    page = start_p
    while True:
        payload = _page({
            "languages": lang,
            "year": year,
            "type": "movie",
            "page": page
        })
        data = payload["data"]
        if not data:
            break

        rows = []
        for hit in data:
            feat = hit["attributes"]["feature_details"]
            imdb = feat["imdb_id"]
            if imdb in seen_ids:
                continue
            rows.append({"page": page, "imdb_id": imdb, "title": feat["title"]})
            seen_ids.add(imdb)

        with out_csv.open("a", newline="") as f:
            csv.DictWriter(f, ["page", "imdb_id", "title"]).writerows(rows)

        page += 1
        time.sleep(PAUSE_PER_REQ)
        if page > payload["total_pages"]:
            break

    return out_csv

# ------------ public API ------------------------------------------------
def collect_movies(years: range, *, langs=("pt-br", "pt-pt")) -> int:
    """
    Ensure the DuckDB `movies` table contains every film that has
    subtitles in *all* requested `langs` during the given `years`.
    Returns the number of rows after the upsert.
    """
    csv_paths = [_catalog_one(lang, y) for lang in langs for y in years]
    dfs       = [pd.read_csv(p, usecols=["imdb_id", "title"]) for p in csv_paths]

    # inner-join across all languages to keep only titles present in each list
    df_final = dfs[0]
    for df in dfs[1:]:
        df_final = df_final.merge(df, on="imdb_id", suffixes=("", "_dup"))
    df_final = df_final[["imdb_id", "title"]]          # drop dup columns
    df_final.drop_duplicates(inplace=True)

    con = connect()
    # bulk upsert
    con.execute("""
        INSERT INTO movies (imdb_id, title)
        SELECT * FROM df_final
        ON CONFLICT (imdb_id) DO UPDATE SET title = EXCLUDED.title
    """)
    total = con.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    con.close()
    return total



def load_intersection_from_catalog(year: int, n: int = 10) -> pd.DataFrame:
    """
    Return a DataFrame with up to *n* movies that have BOTH pt-BR and pt-PT
    subtitles for the given year, based on the CSVs already on disk.
    """
    br_csv = RAW_CATALOG_DIR / f"subs_pt-br_{year}.csv"
    pt_csv = RAW_CATALOG_DIR / f"subs_pt-pt_{year}.csv"
    if not (br_csv.exists() and pt_csv.exists()):
        raise FileNotFoundError("Run collect_movies() first to create catalog CSVs.")

    df_br = pd.read_csv(br_csv, usecols=["imdb_id", "title"])
    df_pt = pd.read_csv(pt_csv, usecols=["imdb_id", "title"])

    df = df_br.merge(df_pt, on="imdb_id", suffixes=("_br", "_pt"))
    df = df.rename(columns={"title_br": "title"})   # keep one title column
    return df.head(n)