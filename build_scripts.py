# ===========================
#  IMPORTS & COMMON SETUP
# ===========================
import duckdb
import pathlib
import pandas as pd
from datasets import load_dataset

pd.set_option("display.max_colwidth", 180)

PROJECT_DB_PATH = pathlib.Path("data/duckdb/subs_project.duckdb")
SOURCE_DB_PATH  = pathlib.Path("data/duckdb/subs.duckdb")

PROJECT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Connect and attach src
con = duckdb.connect(PROJECT_DB_PATH.as_posix(), read_only=False)
try:
    con.execute(f"ATTACH '{SOURCE_DB_PATH.as_posix()}' AS src")
except duckdb.BinderException:
    pass

def tbl_exists(q: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {q} LIMIT 1")
        return True
    except Exception:
        return False

def lo(c: str) -> str:
    return f"lower(coalesce({c},''))"

print("✓ Connected. src attached:", SOURCE_DB_PATH.exists())


def peek(qname: str, where: str = "", n: int = 5):
    w = f"WHERE {where}" if where else ""
    print(f"\n--- PEEK {qname} {w} LIMIT {n} ---")
    try:
        df = con.execute(f"SELECT * FROM {qname} {w} LIMIT {n}").df()
        print(df)
    except Exception as e:
        print("PEEK failed:", e)

def domain_stats(qname: str, domain_col: str = "domain"):
    print(f"\n--- DOMAIN STATS {qname}.{domain_col} ---")
    try:
        df = con.execute(f"""
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN {domain_col} IS NULL OR length(trim({domain_col}))=0 THEN 1 ELSE 0 END) AS domain_missing,
              SUM(CASE WHEN {domain_col} IS NOT NULL AND length(trim({domain_col}))>0 THEN 1 ELSE 0 END) AS domain_present
            FROM {qname}
        """).df()
        print(df)
        top = con.execute(f"""
            SELECT lower(trim({domain_col})) AS domain, COUNT(*) AS n
            FROM {qname}
            WHERE {domain_col} IS NOT NULL AND length(trim({domain_col}))>0
            GROUP BY 1
            ORDER BY n DESC
            LIMIT 20
        """).df()
        print(top)
    except Exception as e:
        print("DOMAIN STATS failed:", e)

# ===========================
#  GOLD COLLECTION (test only)
# ===========================
# HF: joaosanches/golden_collection
bp     = load_dataset("joaosanches/golden_collection", split="gold_collection")
manual = load_dataset("joaosanches/golden_collection", split="referencia_manual")
deepl  = load_dataset("joaosanches/golden_collection", split="referencia_DeepL")
assert len(bp)==len(manual)==len(deepl)==500, (len(bp),len(manual),len(deepl))

con.execute("""
CREATE TABLE IF NOT EXISTS gold_test (
  bucket TEXT,
  theme  TEXT,
  text_pt_br TEXT,
  ref_pt_pt_manual TEXT,
  ref_pt_pt_deepl  TEXT
);
""")
con.execute("DELETE FROM gold_test;")

def _clean(s: str) -> str:
    if s is None: return None
    return " ".join(str(s).replace("\r"," ").replace("\n"," ").split()).strip()

rows = [
    ("n/a", "n/a",
     _clean(bp[i]["text"]),
     _clean(manual[i]["text"]),
     _clean(deepl[i]["text"]))
    for i in range(500)
]
con.executemany(
    "INSERT INTO gold_test (bucket, theme, text_pt_br, ref_pt_pt_manual, ref_pt_pt_deepl) VALUES (?,?,?,?,?)",
    rows
)

print("Gold rows:", con.execute("SELECT COUNT(*) FROM gold_test").fetchone()[0])

# ===========================
#  PtBrVarId  (train / valid / test via hash)
# ===========================
# Discover the PtBrVarId source table inside src
src_ptbr = None
for cand in ["src.main.ptbrvarid", "src.ptbrvarid", "src.main.PtBrVId", "src.PtBrVId"]:
    if tbl_exists(cand):
        src_ptbr = cand
        break
if not src_ptbr:
    raise RuntimeError("Cannot find ptbrvarid in subs.duckdb (tried src.main.ptbrvarid, src.ptbrvarid, ...).")

print("Using ptbrvarid source:", src_ptbr)

# What dataset tags exist?
print("\n--- DATASET TAGS in source table ---")
print(con.execute(f"""
    SELECT dataset, COUNT(*) AS n
    FROM {src_ptbr}
    GROUP BY 1
    ORDER BY n DESC
""").df())

# Peek a few source rows (important: include only columns of interest if the table is wide)
print("\n--- SOURCE SAMPLE (columns of interest) ---")
print(con.execute(f"""
    SELECT dataset, domain, split, label, text_pt_br, text_pt_pt
    FROM {src_ptbr}
    LIMIT 5
""").df())

domain_stats(src_ptbr, "domain")

# 1) Repair noisy rows (decode label + text)
KNOWN_DOMAINS = ("journalistic","legal","web","literature","politics","social_media")

con.execute("DROP VIEW IF EXISTS ptbrvid_repaired_v;")
con.execute(f"""
CREATE VIEW ptbrvid_repaired_v AS
WITH raw AS (
  SELECT dataset, domain, split, label, text_pt_br, text_pt_pt
  FROM {src_ptbr}
  WHERE dataset='PtBrVId'
),
norm AS (
  SELECT
    -- CLEAN DOMAIN: only keep true subset labels; discard garbage text
    CASE
      WHEN domain IS NOT NULL AND lower(trim(domain)) IN {KNOWN_DOMAINS}
        THEN lower(trim(domain))
      ELSE NULL
    END AS domain,

    -- normalize label to pt-BR / pt-PT
    CASE
      WHEN lower(trim(label)) IN ('pt-br','pt_br','ptbr') THEN 'pt-BR'
      WHEN lower(trim(label)) IN ('pt-pt','pt_pt','ptpt') THEN 'pt-PT'
      WHEN label IN ('pt-BR','pt-PT') THEN label
      ELSE label
    END AS lang,

    -- TEXT: use the proper columns only (NO domain fallback)
    CASE
      WHEN text_pt_br IS NOT NULL AND length(trim(text_pt_br))>0 THEN text_pt_br
      WHEN text_pt_pt IS NOT NULL AND length(trim(text_pt_pt))>0 THEN text_pt_pt
      ELSE NULL
    END AS text
  FROM raw
)
SELECT
  'PtBrVId' AS dataset,
  domain,
  lang AS label,
  CASE WHEN lang='pt-BR' THEN text END AS text_pt_br,
  CASE WHEN lang='pt-PT' THEN text END AS text_pt_pt
FROM norm
WHERE lang IN ('pt-BR','pt-PT')
  AND text IS NOT NULL
  AND length(trim(text)) > 0;
""")

print(con.execute("""
SELECT domain, COUNT(*) AS n
FROM ptbrvid_repaired_v
GROUP BY 1
ORDER BY n DESC;
""").df().head(20))


print("PtBrVId repaired rows:",
      con.execute("SELECT COUNT(*) FROM ptbrvid_repaired_v").fetchone()[0])

peek("ptbrvid_repaired_v", n=5)
domain_stats("ptbrvid_repaired_v", "domain")

# 2) Deduplicate text (per label)
con.execute("""
DROP VIEW IF EXISTS ptbr_unique_v;
CREATE VIEW ptbr_unique_v AS
WITH keyed AS (
  SELECT
    label,
    domain,
    NULLIF(TRIM(text_pt_br),'') AS text_pt_br,
    NULLIF(TRIM(text_pt_pt),'') AS text_pt_pt,
    lower(coalesce(text_pt_br, text_pt_pt, '')) AS k_txt
  FROM ptbrvid_repaired_v
),
clean AS (
  SELECT label, domain, text_pt_br, text_pt_pt, k_txt
  FROM keyed
  WHERE k_txt <> ''
),
agg AS (
  SELECT
    label, domain, k_txt,
    arg_max(text_pt_br, length(coalesce(text_pt_br,''))) AS text_pt_br,
    arg_max(text_pt_pt, length(coalesce(text_pt_pt,''))) AS text_pt_pt
  FROM clean
  GROUP BY label, domain, k_txt
)
SELECT label, domain, text_pt_br, text_pt_pt
FROM agg;
""")

peek("ptbr_unique_v", n=5)
domain_stats("ptbr_unique_v", "domain")

print("PtBrVId unique rows:",
      con.execute("SELECT COUNT(*) FROM ptbr_unique_v").fetchone()[0])

# 3) Deterministic split: train / valid / test
#    - valid: 1%   (~0.01)
#    - test:  0.05% (~0.0005)
con.execute("""
DROP VIEW IF EXISTS ptbr_split_assign_v;
CREATE VIEW ptbr_split_assign_v AS
WITH base AS (
  SELECT
    label,
    domain,
    text_pt_br,
    text_pt_pt,
    lower(coalesce(text_pt_br, text_pt_pt, '')) AS k_txt,
    hash(lower(coalesce(text_pt_br, text_pt_pt, '')), coalesce(label,'')) AS h
  FROM ptbr_unique_v
)
SELECT
  CASE
    WHEN (h % 10000) < 5   THEN 'test'
    WHEN (h % 10000) < 105 THEN 'valid'
    ELSE 'train'
  END AS split,
  label,
  domain,
  text_pt_br,
  text_pt_pt
FROM base;
""")

peek("ptbr_split_assign_v", n=5)
domain_stats("ptbr_split_assign_v", "domain")

print("PtBrVId split counts:")
print(con.execute("""
    SELECT split, COUNT(*) AS n
    FROM ptbr_split_assign_v
    GROUP BY 1
    ORDER BY n DESC
""").df())

# ===========================
#  FRMT (hugosousa/frmt)
#        dev: 1% valid, 99% train
#        test: all test
# ===========================
# ===========================
#  FRMT (hugosousa/frmt) — CLEAN IMPORT
# ===========================
ds_dev  = load_dataset("hugosousa/frmt", split="dev")
ds_test = load_dataset("hugosousa/frmt", split="test")

def _nonempty_pair(x):
    br = x.get("br")
    pt = x.get("pt")
    if br is None or pt is None:
        return False
    # also reject whitespace-only
    return bool(str(br).strip()) and bool(str(pt).strip())

# Filter at source (prevents writing null rows into DuckDB)
ds_dev  = ds_dev.filter(_nonempty_pair)
ds_test = ds_test.filter(_nonempty_pair)

df_dev  = ds_dev.to_pandas()
df_test = ds_test.to_pandas()

# Rename and keep only what you actually use downstream
df_dev  = df_dev.rename(columns={"br": "text_pt_br", "pt": "text_pt_pt"})
df_test = df_test.rename(columns={"br": "text_pt_br", "pt": "text_pt_pt"})

keep_cols = [c for c in ["bucket", "text_pt_br", "text_pt_pt"] if c in df_dev.columns]
df_dev  = df_dev[keep_cols].copy()
df_test = df_test[keep_cols].copy()

# Drop old tables and write clean ones
con.execute("DROP TABLE IF EXISTS frmt_dev;")
con.execute("DROP TABLE IF EXISTS frmt_test;")

con.register("df_dev_clean", df_dev)
con.register("df_test_clean", df_test)

con.execute("CREATE TABLE frmt_dev  AS SELECT * FROM df_dev_clean;")
con.execute("CREATE TABLE frmt_test AS SELECT * FROM df_test_clean;")

print("FRMT dev rows (clean): ", con.execute("SELECT COUNT(*) FROM frmt_dev").fetchone()[0])
print("FRMT test rows (clean):", con.execute("SELECT COUNT(*) FROM frmt_test").fetchone()[0])

con.execute("""
DROP VIEW IF EXISTS frmt_dev_clean_v;
CREATE VIEW frmt_dev_clean_v AS
SELECT *
FROM frmt_dev
WHERE text_pt_br IS NOT NULL AND text_pt_pt IS NOT NULL
  AND length(trim(text_pt_br)) > 0
  AND length(trim(text_pt_pt)) > 0;

DROP VIEW IF EXISTS frmt_test_clean_v;
CREATE VIEW frmt_test_clean_v AS
SELECT *
FROM frmt_test
WHERE text_pt_br IS NOT NULL AND text_pt_pt IS NOT NULL
  AND length(trim(text_pt_br)) > 0
  AND length(trim(text_pt_pt)) > 0;
""")

con.execute("""
DROP VIEW IF EXISTS frmt_dev_split_v;
CREATE VIEW frmt_dev_split_v AS
WITH base AS (
  SELECT
    *,
    hash(lower(trim(text_pt_br)), lower(trim(text_pt_pt))) AS h
  FROM frmt_dev_clean_v
)
SELECT
  CASE WHEN (h % 100) = 0 THEN 'valid' ELSE 'train' END AS split,
  *
FROM base;
""")

print("FRMT dev split counts:")
print(con.execute("""
    SELECT split, COUNT(*) AS n
    FROM frmt_dev_split_v
    GROUP BY 1
    ORDER BY n DESC
""").df())

# ===========================
#  OPUS / OpenSubtitles (train only)
# ===========================
def has_cols(q: str, want=("sent_pt_br","sent_pt_pt")) -> bool:
    try:
        cols = con.execute(f"DESCRIBE {q}").df()["column_name"].str.lower().tolist()
        return all(w in cols for w in want)
    except Exception:
        return False

candidates = [
    "src.main.opus_moses_filtered", "main.opus_moses_filtered",
    "src.main.opus_filtered",       "main.opus_filtered",
    "src.main.opus_filter_simple",  "main.opus_filter_simple",
    "src.main.opus_moses",          "main.opus_moses",
]
chosen = None
for q in candidates:
    if tbl_exists(q) and has_cols(q):
        chosen = q
        break

con.execute("DROP VIEW IF EXISTS opus_source;")
if chosen:
    con.execute(f"CREATE VIEW opus_source AS SELECT * FROM {chosen};")
    print(f"✓ opus_source -> {chosen}")
else:
    con.execute("CREATE VIEW opus_source AS SELECT NULL::TEXT AS sent_pt_br, NULL::TEXT AS sent_pt_pt WHERE 1=0;")
    print("! No OPUS table found with (sent_pt_br, sent_pt_pt)")

# ===========================
#  TEST PAIRS GUARD (FRMT test + Gold)
# ===========================
con.execute("DROP VIEW IF EXISTS test_pairs_guard;")
con.execute("""
CREATE VIEW test_pairs_guard AS
SELECT DISTINCT n_br, n_pt
FROM (
  SELECT lower(trim(coalesce(text_pt_br,''))) AS n_br,
         lower(trim(coalesce(text_pt_pt,''))) AS n_pt
  FROM frmt_test_clean_v
  UNION ALL
  SELECT lower(trim(coalesce(text_pt_br,''))) AS n_br,
         lower(trim(coalesce(ref_pt_pt_manual,''))) AS n_pt
  FROM gold_test
) t
WHERE n_br <> '' AND n_pt <> '';
""")

print("test_pairs_guard rows:",
      con.execute("SELECT COUNT(*) FROM test_pairs_guard").fetchone()[0])

# ===========================
#  UNIFIED VIEWS: train_data / test_data
#  Columns:
#    dataset, split, source, bucket, theme, label,
#    text_pt_br, text_pt_pt, ref_pt_pt_manual, ref_pt_pt_deepl
# ===========================

con.execute("DROP VIEW IF EXISTS train_data;")
con.execute("""
CREATE VIEW train_data AS

-- OPUS (train only, anti-join against known test pairs)
SELECT
  'OpenSubs' AS dataset,
  'train'    AS split,
  'opus_source' AS source,
  'n/a' AS bucket,
  'n/a' AS theme,
  CAST(NULL AS TEXT) AS label,
  o.sent_pt_br AS text_pt_br,
  o.sent_pt_pt AS text_pt_pt,
  CAST(NULL AS TEXT) AS ref_pt_pt_manual,
  CAST(NULL AS TEXT) AS ref_pt_pt_deepl
FROM opus_source o
LEFT JOIN test_pairs_guard g
  ON lower(coalesce(o.sent_pt_br,'')) = g.n_br
 AND lower(coalesce(o.sent_pt_pt,'')) = g.n_pt
WHERE g.n_br IS NULL

UNION ALL

-- PtBrVarId (train + valid)  (INSERTED BLOCK: domain -> theme)
SELECT
  'PtBrVId' AS dataset,
  p.split     AS split,
  'liaad/PtBrVId' AS source,
  'n/a' AS bucket,
  COALESCE(p.domain,'n/a') AS theme,
  p.label,
  p.text_pt_br,
  p.text_pt_pt,
  CAST(NULL AS TEXT) AS ref_pt_pt_manual,
  CAST(NULL AS TEXT) AS ref_pt_pt_deepl
FROM ptbr_split_assign_v p
WHERE lower(p.split) IN ('train','valid')

UNION ALL

-- FRMT dev: train + valid
SELECT
  'FRMT' AS dataset,
  d.split AS split,
  'hugosousa/frmt' AS source,
  d.bucket,
  'n/a' AS theme,
  CAST(NULL AS TEXT) AS label,
  d.text_pt_br,
  d.text_pt_pt,
  CAST(NULL AS TEXT) AS ref_pt_pt_manual,
  CAST(NULL AS TEXT) AS ref_pt_pt_deepl
FROM frmt_dev_split_v d;
""")

con.execute("DROP VIEW IF EXISTS test_data;")
con.execute("""
CREATE VIEW test_data AS

-- PtBrVarId test (INSERTED BLOCK: domain -> theme)
SELECT
  'PtBrVId' AS dataset,
  'test'      AS split,
  'liaad/PtBrVId' AS source,
  'n/a' AS bucket,
  COALESCE(p.domain,'n/a') AS theme,
  p.label,
  p.text_pt_br,
  p.text_pt_pt,
  CAST(NULL AS TEXT) AS ref_pt_pt_manual,
  CAST(NULL AS TEXT) AS ref_pt_pt_deepl
FROM ptbr_split_assign_v p
WHERE lower(p.split) = 'test'

UNION ALL

-- FRMT test (recommended to use clean view so null/empty never leaks)
SELECT
  'FRMT' AS dataset,
  'test' AS split,
  'hugosousa/frmt' AS source,
  f.bucket,
  'n/a' AS theme,
  CAST(NULL AS TEXT) AS label,
  f.text_pt_br,
  f.text_pt_pt,
  CAST(NULL AS TEXT) AS ref_pt_pt_manual,
  CAST(NULL AS TEXT) AS ref_pt_pt_deepl
FROM frmt_test_clean_v f

UNION ALL

-- Gold Collection (test only)
SELECT
  'Gold' AS dataset,
  'test' AS split,
  'joaosanches/golden_collection' AS source,
  COALESCE(g.bucket,'n/a'),
  COALESCE(g.theme,'n/a'),
  'n/a' AS label,
  g.text_pt_br,
  CAST(NULL AS TEXT) AS text_pt_pt,
  g.ref_pt_pt_manual,
  g.ref_pt_pt_deepl
FROM gold_test g;
""")

# Quick summary
print("\n=== Unified view sizes ===")
print("train_data rows:",
      con.execute("SELECT COUNT(*) FROM train_data").fetchone()[0])
print("test_data rows :",
      con.execute("SELECT COUNT(*) FROM test_data").fetchone()[0])

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


print("\nFRMT test null/empty check:")
print(con.execute("""
SELECT
  SUM(CASE WHEN text_pt_br IS NULL OR length(trim(text_pt_br))=0 THEN 1 ELSE 0 END) AS bad_br,
  SUM(CASE WHEN text_pt_pt IS NULL OR length(trim(text_pt_pt))=0 THEN 1 ELSE 0 END) AS bad_pt
FROM frmt_test;
""").df())

print(con.execute("""
SELECT theme, COUNT(*) AS n
FROM test_data
WHERE dataset='PtBrVId'
GROUP BY 1
ORDER BY n DESC
LIMIT 30;
""").df())
