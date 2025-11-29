# sanity_check_train_data.py
import duckdb, pathlib

PROJECT_DB_PATH = pathlib.Path("data/duckdb/subs_project.duckdb")
con = duckdb.connect(PROJECT_DB_PATH.as_posix(), read_only=True)

print("DB:", PROJECT_DB_PATH)
print("\nDESCRIBE train_data:")
print(con.execute("DESCRIBE train_data").df())

print("\ntrain_data by dataset/split:")
print(con.execute("""
    SELECT dataset, split, COUNT(*) AS n
    FROM train_data
    GROUP BY dataset, split
    ORDER BY dataset, split
""").df())

con.close()
