# EDA Map and Pipelines

This file documents what each notebook in `eda's/` does and how it fits in the pipelines.
It is meant as a quick reference when moving notebook logic into scripts.

## OPUS pipeline (OpenSubtitles via OPUS dataset)

```
OPUS files
  data/opus_cache/OpenSubtitles.pt-pt_BR.pt_BR
  data/opus_cache/OpenSubtitles.pt-pt_BR.pt
        |
        v
scripts/opus/ingest_opus.py (extracted from eda's/05_access_OPUS.ipynb)
  - loads OPUS files
  - writes/updates DuckDB table: opus_moses in data/duckdb/subs.duckdb
        |
        v
scripts/opus/pass_1_opus.py  (uses src/opus_pipeline.py)
  - plan_ops_over_corpus -> apply_ops_ctas_swap
  - apply_neighbor_moves_corpus_inplace
  - run_repeated_prefix_cleaner_chunked
        |
        v
scripts/opus/pass_2_opus.py  (uses src/shard_filter.py)
  - shard filter on opus_moses
  - writes filter shards to data/filter_shards/*.parquet
  - applies filter back into opus_moses
        |
        v
data/duckdb/subs.duckdb  (cleaned opus_moses)
```

## OpenSubtitles (non-OPUS) pipeline

```
TMDB API -> tmdb_movies_2024_with_imdb.csv
        |
        v
eda's/02_acessing_opensubtitles.ipynb
  - lists/filters movies with both pt-br and pt-pt subtitles
  - downloads .srt files
        |
        v
eda's/03_aligning_subtitles.ipynb
  - tests subtitle alignment on sample .srt pairs
  - uses src/transform/align_subtitles.py
        |
        v
eda's/04_ingest_align_store.ipynb
  - end-to-end ingest, align, and store in data/duckdb/subs.duckdb
```

## PtBrVarId pipeline

```
HuggingFace: liaad/PtBrVId-Raw
        |
        v
scripts/ptbrvarid/ingest_ptbrvarid.py
  - ingests PtBrVId-Raw into data/duckdb/subs.duckdb (table: ptbrvarid)
        |
        v
scripts/ptbrvarid/filter_ptbrvarid.py
  - applies jusText + author filters + IQR length filter
  - writes processed PtBrVId into data/duckdb/subs.duckdb (table: ptbrvarid)
        |
        v
scripts/project/build_project_db.py
  - builds ptbrvarid splits + unified views in data/duckdb/subs_project.duckdb
```

## Notebook inventory

| Notebook | Purpose | Key inputs | Outputs | Notes |
|---|---|---|---|---|
| `01_accessing_tmdb.ipynb` | Pulls movies from TMDB and writes IMDb ids | TMDB API | `tmdb_movies_2024_with_imdb.csv` | Left as-is (no changes planned) |
| `02_acessing_opensubtitles.ipynb` | Finds/downloads PT-BR and PT-PT subtitles | `tmdb_movies_2024_with_imdb.csv`, OpenSubtitles API | `.srt` files in `data/raw/...` | Left as-is (no changes planned) |
| `03_aligning_subtitles.ipynb` | Aligns PT-BR/PT-PT .srt pairs; sanity checks | sample `.srt` files | alignment diagnostics | Uses `src/transform/align_subtitles.py` and `src/load/load_subtitles.py`; no unique functions | 
| `04_ingest_align_store.ipynb` | Ingests + aligns subtitles and stores in DuckDB | OpenSubtitles API + `.srt` | `data/duckdb/subs.duckdb` | Defines helpers `td_to_srt`, `would_merge`, `calculate_sentence_difference` (not in .py yet) |
| `05_access_OPUS.ipynb` | Loads OPUS files into `opus_moses` | OPUS files in `data/opus_cache/` | `data/duckdb/subs.duckdb` (table `opus_moses`) | Core OPUS ingestion logic lives here |
| `06_accessing_PtBrVarId.ipynb` | Ingests PtBrVarId dataset | HuggingFace dataset | `ptbrvarid` table in `subs.duckdb` | Non-OPUS dataset ingestion |
| `07_splits.ipynb` | Builds `subs_project.duckdb` with train/test/gold splits | `subs.duckdb`, FRMT | `subs_project.duckdb` | Dataset split logic |
| `08_preprocessing.ipynb` | OPUS cleaning + filtering (moves, dedup, simalign) | `subs.duckdb` | updated `opus_moses` | Most code is in `src/opus_pipeline.py`; notebook includes debug helpers |
| `09_model_Qwen0.6B.ipynb` | Qwen LoRA training + streaming | `subs_project.duckdb` | model checkpoints | Training logic |
| `10_opus_statistics.ipynb` | OPUS stats + outlier reports | `subs.duckdb` | `word_diff_top.csv`, `outliers_hard.csv` | Stats utilities exist only in notebook |
| `11_all_data_statistics.ipynb` | Token/char stats across datasets | `subs_project.duckdb` | tables/plots | Some helpers already in `scripts/get_tokens_split.py` |
| `12_test.ipynb` | Eval helpers for latest jsonl results | `eval_results/*.jsonl` | ad-hoc metrics | Notebook-only utilities |
