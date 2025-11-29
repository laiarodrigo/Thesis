# ===========================
#  IMPORTS & COMMON SETUP
# ===========================
import duckdb
import pathlib
import pandas as pd
from datasets import load_dataset

pd.set_option("display.max_colwidth", 180)

PROJECT_DB_PATH = pathlib.Path("../data/duckdb/subs_project.duckdb")
SOURCE_DB_PATH  = pathlib.Path("../data/duckdb/subs.duckdb")

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

# 1) Repair noisy rows (decode label + text)
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
    -- language
    CASE
      WHEN lower(label) IN ('pt-br','pt-pt')
           THEN CASE WHEN lower(label)='pt-br' THEN 'pt-BR' ELSE 'pt-PT' END
      WHEN text_pt_br IN ('pt-BR','pt-PT')
           THEN text_pt_br
      ELSE NULL
    END AS lang,

    -- text (prefer proper columns; fall back to domain if it really looks like text)
    CASE
      WHEN text_pt_br IS NOT NULL AND text_pt_br NOT IN ('pt-BR','pt-PT') THEN text_pt_br
      WHEN text_pt_pt IS NOT NULL AND text_pt_pt NOT IN ('pt-BR','pt-PT') THEN text_pt_pt
      WHEN domain IS NOT NULL
           AND lower(domain) NOT IN ('journalistic','legal','web','literature','politics','social_media')
           AND length(domain) > 40 THEN domain
      ELSE NULL
    END AS text
  FROM raw
)
SELECT
  'PtBrVId' AS dataset,
  lang  AS label,
  CASE WHEN lang='pt-BR' THEN text END AS text_pt_br,
  CASE WHEN lang='pt-PT' THEN text END AS text_pt_pt
FROM norm
WHERE lang IS NOT NULL AND text IS NOT NULL;
""")

print("PtBrVId repaired rows:",
      con.execute("SELECT COUNT(*) FROM ptbrvid_repaired_v").fetchone()[0])

# 2) Deduplicate text (per label)
con.execute("DROP VIEW IF EXISTS ptbr_unique_v;")
con.execute("""
CREATE VIEW ptbr_unique_v AS
WITH keyed AS (
  SELECT
    label,
    NULLIF(TRIM(text_pt_br),'') AS text_pt_br,
    NULLIF(TRIM(text_pt_pt),'') AS text_pt_pt,
    lower(coalesce(text_pt_br, text_pt_pt, '')) AS k_txt
  FROM ptbrvid_repaired_v
),
clean AS (
  SELECT label, text_pt_br, text_pt_pt, k_txt
  FROM keyed
  WHERE k_txt <> ''
),
agg AS (
  SELECT
    label, k_txt,
    arg_max(text_pt_br, length(coalesce(text_pt_br,''))) AS text_pt_br,
    arg_max(text_pt_pt, length(coalesce(text_pt_pt,''))) AS text_pt_pt
  FROM clean
  GROUP BY label, k_txt
)
SELECT label, text_pt_br, text_pt_pt
FROM agg;
""")

print("PtBrVId unique rows:",
      con.execute("SELECT COUNT(*) FROM ptbr_unique_v").fetchone()[0])

# 3) Deterministic split: train / valid / test
#    - valid: 1%   (~0.01)
#    - test:  0.05% (~0.0005)
con.execute("DROP VIEW IF EXISTS ptbr_split_assign_v;")
con.execute("""
CREATE VIEW ptbr_split_assign_v AS
WITH base AS (
  SELECT
    label,
    text_pt_br,
    text_pt_pt,
    lower(coalesce(text_pt_br, text_pt_pt, '')) AS k_txt,
    hash(lower(coalesce(text_pt_br, text_pt_pt, '')), coalesce(label,'')) AS h
  FROM ptbr_unique_v
)
SELECT
  CASE
    WHEN (h % 10000) < 5   THEN 'test'   -- 5 / 10000 = 0.05%
    WHEN (h % 10000) < 105 THEN 'valid'  -- next 100 / 10000 = 1%
    ELSE 'train'
  END AS split,
  label,
  text_pt_br,
  text_pt_pt
FROM base;
""")

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
# Load from HF
ds_dev  = load_dataset("hugosousa/frmt", split="dev")
ds_test = load_dataset("hugosousa/frmt", split="test")

df_dev  = ds_dev.to_pandas()
df_test = ds_test.to_pandas()

# Ensure column names are text_pt_br / text_pt_pt.
# (hugosousa/frmt already uses 'br' and 'pt' columns; rename them.)
if "br" in df_dev.columns and "pt" in df_dev.columns:
    df_dev  = df_dev.rename(columns={"br": "text_pt_br", "pt": "text_pt_pt"})
    df_test = df_test.rename(columns={"br": "text_pt_br", "pt": "text_pt_pt"})

# Drop old tables and create new ones
con.execute("DROP TABLE IF EXISTS frmt_dev;")
con.execute("DROP TABLE IF EXISTS frmt_test;")

con.execute("CREATE TABLE frmt_dev  AS SELECT * FROM df_dev;")
con.execute("CREATE TABLE frmt_test AS SELECT * FROM df_test;")

print("FRMT dev rows:",  con.execute("SELECT COUNT(*) FROM frmt_dev").fetchone()[0])
print("FRMT test rows:", con.execute("SELECT COUNT(*) FROM frmt_test").fetchone()[0])

# Split FRMT dev into train/valid (1% valid)
con.execute("DROP VIEW IF EXISTS frmt_dev_split_v;")
con.execute("""
CREATE VIEW frmt_dev_split_v AS
WITH base AS (
  SELECT
    *,
    hash(lower(coalesce(text_pt_br,'')), lower(coalesce(text_pt_pt,''))) AS h
  FROM frmt_dev
)
SELECT
  CASE
    WHEN (h % 100) = 0 THEN 'valid'  -- 1% of dev
    ELSE 'train'
  END AS split,
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
have_frmt = tbl_exists("frmt_test")
have_gold = tbl_exists("gold_test")

if have_frmt and have_gold:
    con.execute(f"""
      CREATE VIEW test_pairs_guard AS
      SELECT DISTINCT n_br, n_pt FROM (
        SELECT {lo('text_pt_br')} AS n_br, {lo('text_pt_pt')} AS n_pt FROM frmt_test
        UNION ALL
        SELECT {lo('text_pt_br')} AS n_br, {lo('ref_pt_pt_manual')} AS n_pt FROM gold_test
      );
    """)
elif have_frmt:
    con.execute(f"""
      CREATE VIEW test_pairs_guard AS
      SELECT DISTINCT {lo('text_pt_br')} AS n_br, {lo('text_pt_pt')} AS n_pt FROM frmt_test;
    """)
elif have_gold:
    con.execute(f"""
      CREATE VIEW test_pairs_guard AS
      SELECT DISTINCT {lo('text_pt_br')} AS n_br, {lo('ref_pt_pt_manual')} AS n_pt FROM gold_test;
    """)
else:
    con.execute("CREATE VIEW test_pairs_guard AS SELECT ''::TEXT AS n_br, ''::TEXT AS n_pt WHERE 1=0;")

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

-- PtBrVarId (train + valid)
SELECT
  'PtBrVarId' AS dataset,
  p.split     AS split,      -- 'train' or 'valid'
  'liaad/PtBrVId' AS source,
  'n/a' AS bucket,
  'n/a' AS theme,
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
  d.split AS split,          -- 'train' or 'valid'
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

-- PtBrVarId test
SELECT
  'PtBrVarId' AS dataset,
  'test'      AS split,
  'liaad/PtBrVId' AS source,
  'n/a' AS bucket,
  'n/a' AS theme,
  p.label,
  p.text_pt_br,
  p.text_pt_pt,
  CAST(NULL AS TEXT) AS ref_pt_pt_manual,
  CAST(NULL AS TEXT) AS ref_pt_pt_deepl
FROM ptbr_split_assign_v p
WHERE lower(p.split) = 'test'

UNION ALL

-- FRMT test
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
FROM frmt_test f

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
