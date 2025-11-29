import duckdb
import pathlib
import pandas as pd

pd.set_option("display.max_colwidth", 180)

# Paths must match your training script
BASE_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DB_PATH = BASE_DIR / "data" / "duckdb" / "subs_project.duckdb"
SOURCE_DB_PATH  = BASE_DIR / "data" / "duckdb" / "subs.duckdb"

PROJECT_DB_STR = PROJECT_DB_PATH.as_posix()
SOURCE_DB_STR  = SOURCE_DB_PATH.as_posix()


def connect_project(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(PROJECT_DB_STR, read_only=read_only)
    dbl = con.execute("PRAGMA database_list").df()
    if not (dbl["name"] == "src").any():
        con.execute(f"ATTACH '{SOURCE_DB_STR}' AS src")
    return con


def main():
    con = connect_project(read_only=True)
    print("DB:", PROJECT_DB_STR)

    print("\nDESCRIBE train_data:")
    print(con.execute("DESCRIBE train_data").df())

    print("\ntrain_data by dataset/split:")
    print(con.execute("""
        SELECT dataset, split, COUNT(*) AS n
        FROM train_data
        GROUP BY dataset, split
        ORDER BY dataset, split
    """).df())

    print("\ntest_data by dataset/split:")
    print(con.execute("""
        SELECT dataset, split, COUNT(*) AS n
        FROM test_data
        GROUP BY dataset, split
        ORDER BY dataset, split
    """).df())

    con.close()


if __name__ == "__main__":
    main()
