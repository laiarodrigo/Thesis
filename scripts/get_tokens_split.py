import argparse
import pathlib
from collections import defaultdict

import duckdb
import pandas as pd
from transformers import AutoTokenizer


# -------------------
# CONFIG
# -------------------
PROJECT_DB_PATH = pathlib.Path("data/duckdb/subs_project.duckdb")
SOURCE_DB_PATH  = pathlib.Path("data/duckdb/subs.duckdb")

MODEL_NAME = "Qwen/Qwen3-0.6B"  # make sure this matches your actual HF model id


# -------------------
# DB + TOKENIZER SETUP
# -------------------
def get_connection() -> duckdb.DuckDBPyConnection:
    # read_only=True prevents the locking issue
    con = duckdb.connect(PROJECT_DB_PATH.as_posix(), read_only=True)

    con.execute("PRAGMA threads=1;")
    con.execute("PRAGMA preserve_insertion_order=false;")
    con.execute("PRAGMA memory_limit='4GB';")
    con.execute("PRAGMA temp_directory='/tmp/duckdb_tmp';")

    dbl = con.execute("PRAGMA database_list").df()
    if not (dbl["name"] == "src").any():
        # src is also just read; this is fine in a read-only connection
        con.execute(f"ATTACH '{SOURCE_DB_PATH.as_posix()}' AS src;")

    return con


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def count_tokens(text: str | None) -> int:
        if text is None:
            return 0
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    return count_tokens


# -------------------
# TOKEN STATS (PER SPLIT)
# -------------------
def token_stats_for_table(
    con: duckdb.DuckDBPyConnection,
    count_tokens,
    table: str,
    has_split_col: bool,
    batch_size: int = 50_000,
) -> pd.DataFrame:
    """
    Compute token stats per split for text_pt_br and text_pt_pt in `table`.

    table: 'train_data' or 'test_data'
    has_split_col: True if table has a 'split' column (train_data),
                   False if it doesn't (test_data -> inject 'test').
    """

    if has_split_col:
        sql = f"""
            SELECT split, text_pt_br, text_pt_pt
            FROM {table}
            WHERE text_pt_br IS NOT NULL OR text_pt_pt IS NOT NULL
        """
    else:
        # test_data usually has no split column; inject 'test'
        sql = f"""
            SELECT 'test' AS split, text_pt_br, text_pt_pt
            FROM {table}
            WHERE text_pt_br IS NOT NULL OR text_pt_pt IS NOT NULL
        """

    rel = con.execute(sql)

    # stats[split] = dict with counters
    stats = defaultdict(lambda: {
        "n_rows": 0,
        "tokens_br": 0,
        "tokens_pt": 0,
    })

    iteration = 0

    while True:
        rows = rel.fetchmany(batch_size)  # list of tuples (split, text_pt_br, text_pt_pt)
        if not rows:
            break
        iteration += 1
        print(f"  Processing batch {iteration} ({len(rows)} rows)...")

        for split, text_pt_br, text_pt_pt in rows:
            s = stats[split]
            s["n_rows"] += 1
            if text_pt_br is not None:
                s["tokens_br"] += count_tokens(text_pt_br)
            if text_pt_pt is not None:
                s["tokens_pt"] += count_tokens(text_pt_pt)

    # convert to DataFrame
    out_rows = []
    for split, s in stats.items():
        n = s["n_rows"] or 1  # avoid division by zero
        out_rows.append({
            "split": split,
            "n_rows": s["n_rows"],
            "total_tokens_pt_br": s["tokens_br"],
            "mean_tokens_pt_br": s["tokens_br"] / n,
            "total_tokens_pt_pt": s["tokens_pt"],
            "mean_tokens_pt_pt": s["tokens_pt"] / n,
        })

    return pd.DataFrame(out_rows)


# -------------------
# MAIN
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-csv",
        type=str,
        default="token_len_stats.csv",
        help="Where to save token statistics CSV",
    )
    args = parser.parse_args()

    print("Connecting to DuckDB...")
    con = get_connection()

    print("Loading tokenizer...")
    count_tokens = get_tokenizer()

    print("Computing train/valid token stats...")
    train_valid_stats = token_stats_for_table(
        con=con,
        count_tokens=count_tokens,
        table="train_data",
        has_split_col=True,
    )

    print("Computing test token stats...")
    test_stats = token_stats_for_table(
        con=con,
        count_tokens=count_tokens,
        table="test_data",
        has_split_col=False,
    )

    token_len_stats = (
        pd.concat([train_valid_stats, test_stats], ignore_index=True)
          .sort_values(["split"])
    )

    # Preview in stdout (small)
    print("\n=== Token length stats (head) ===")
    print(token_len_stats.head())

    # Save full CSV
    out_path = pathlib.Path(args.out_csv).resolve()
    token_len_stats.to_csv(out_path, index=False)
    print(f"\nSaved token stats to: {out_path}")


if __name__ == "__main__":
    main()
