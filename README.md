# Thesis

ja consigo extrair ficheiros pt e br do open subtitles.
agr falta adicionalos a uma base de dados (em principio nao importa o filme em si, ou seja só precisamos de dar parse nos ficheiros e cada linha da base de dados é uma fala e a sua comparação)

fazer funcao para dar parse
fazer funcao de upload na base de dados
colocar o codigo já existente dos notebooks em funcoes python e nos respetivos ficheiros
fazer requirements.txt

## Docs

- EDA inventory and pipeline diagrams: `docs/eda_map.md`

## OPUS pipeline (exact run order)

1) Ingest OPUS/OpenSubtitles text files into DuckDB (`opus_moses`):
```
python scripts/opus/ingest_opus.py
```

2) Pass 1 cleaning on `opus_moses`:
```
python scripts/opus/pass_1_opus.py
```

3) Pass 2 filtering (SimAlign + length-adaptive thresholding + length-ratio outliers) and apply it to `opus_moses`:
```
python scripts/opus/pass_2_opus.py
```

Output DB: `data/duckdb/subs.duckdb` (table `opus_moses` is updated in-place).

## PtBrVarId pipeline (exact run order)

1) Ingest PtBrVId (raw) from HuggingFace into `subs.duckdb` (`ptbrvarid`, dataset tag `PtBrVId-Raw`):
```
python scripts/ptbrvarid/ingest_ptbrvarid.py
```

2) Run PtBrVId filtering pipeline (jusText + author transforms + dedup + IQR):
```
python scripts/ptbrvarid/filter_ptbrvarid.py
```

3) Build project views (splits + unified train/test views) into `subs_project.duckdb`:
```
python scripts/project/build_project_db.py
```

Output DB: `data/duckdb/subs_project.duckdb` (views: `train_data`, `test_data`, etc.).

## Where the filters live (SimAlign + length-adaptive thresholding + length-ratio outliers)

- **Current pipeline (used by `pass_2_opus.py`)**: `src/shard_filter.py`
  - `new_sim(...)` builds SimAlign similarity with `SentenceAligner`.
  - `length_adaptive_flag(...)` is the length-adaptive thresholding (based on sentence length, not ratio).
  - `length_ratio_flag(...)` flags length-ratio outliers (`len_pt / len_br`, default bounds 0.5–2.0).
- `scripts/opus/pass_2_opus.py` shards the data, runs `src/shard_filter.py`, and then removes flagged rows from `opus_moses`.
  - You can override ratio bounds with `RATIO_LOW` and `RATIO_HIGH`.

- **Alternate/monolithic filter inside `src/opus_pipeline.py`** (only used if you call it directly):
  - `alignment_quality_features(...)` + `alignment_similarity_only_flag(...)`
  - `build_simple_filter_flags_chunked(...)` and `apply_simple_filter_ctas_swap(...)`

## Length ratio outlier analysis (scatter_outliers.pdf)

The length-ratio filter is based on `len_pt / len_br` and is used to
generate `eda's/images/scatter_outliers.pdf`. That logic originates in
`eda's/10_opus_statistics.ipynb` (e.g., `plot_scatter_with_outliers(...)` and
`outlier_report(...)`) and is now **part of pass_2** via `src/shard_filter.py`.

## Tuning (recommended bounds + examples)

Recommended defaults:
- `RATIO_LOW=0.5`, `RATIO_HIGH=2.0` (length ratio `len_pt/len_br`)
- `base_min_sim=0.30`, `long_min_sim=0.70`, `short_len=8`, `long_len=28`

Examples:
```
RATIO_LOW=0.6 RATIO_HIGH=1.9 python scripts/opus/pass_2_opus.py
```
```
WORKERS=1 RATIO_LOW=0.5 RATIO_HIGH=2.2 python scripts/opus/pass_2_opus.py
```

## What is `src/load/create_opus_filtered.py`? Am I using it?

It is a utility that builds a **separate** filtered table (`opus_moses_filtered`) from the **Parquet shards**
produced by `pass_2_opus.py`, and can optionally materialize `opus_filter_simple`. It is **not** called by the
main pipeline above, so you are not using it unless you run it manually.

The shards exist to parallelize the filter step across multiple CPU processes. If you run with a single worker
(`WORKERS=1`), `pass_2_opus.py` simply creates **one shard** covering the full range; the results are identical,
just slower.
