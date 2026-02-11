# Thesis

ja consigo extrair ficheiros pt e br do open subtitles.
agr falta adicionalos a uma base de dados (em principio nao importa o filme em si, ou seja só precisamos de dar parse nos ficheiros e cada linha da base de dados é uma fala e a sua comparação)

fazer funcao para dar parse
fazer funcao de upload na base de dados
colocar o codigo já existente dos notebooks em funcoes python e nos respetivos ficheiros
fazer requirements.txt

## Docs

- EDA inventory and pipeline diagrams: `docs/eda_map.md`

## Decoder-Only (Qwen3 + Axolotl, LoRA)

1) Export training/validation JSONL for Axolotl:
```
python scripts/decoder_only/axolotl/export_qwen_chat_data.py
```

2) Preprocess + train with Axolotl:
```
scripts/decoder_only/axolotl/run_qwen3_lora.sh
```

3) Evaluate the trained adapter:
```
python scripts/test_qwen_lora.py \
  --use-adapter \
  --adapter-dir outputs/decoder_only/qwen3-0_6b-ptbr-ptpt-lora
```

Main config: `configs/decoder_only/axolotl/qwen3_lora.yaml`

## Encoder-Decoder (HF + PEFT LoRA, 3-way classification)

This path is organized under `scripts/encoder_decoder/` and `configs/encoder_decoder/`.
Classification labels are: `pt-br`, `pt-pt`, `equal`.
It uses `transformers` + `peft` training scripts under `scripts/encoder_decoder/` (not the Axolotl runner).

1) Export encoder-decoder datasets (translation + classification):
```
scripts/encoder_decoder/run_export.sh
```

This writes:
- `data/encoder_decoder/translation_train.jsonl`
- `data/encoder_decoder/translation_valid.jsonl`
- `data/encoder_decoder/classification_train.jsonl`
- `data/encoder_decoder/classification_valid.jsonl`

2) Train translation LoRA:
```
scripts/encoder_decoder/run_train_translation_lora.sh
```

3) Train classification LoRA:
```
scripts/encoder_decoder/run_train_classification_lora.sh
```

4) Evaluate translation:
```
scripts/encoder_decoder/eval/run_eval_translation.sh \
  --dataset-path data/encoder_decoder/translation_valid.jsonl \
  --model-id google/flan-t5-base \
  --adapter-dir outputs/encoder_decoder/flan_t5_base_translation_lora
```

5) Evaluate classification:
```
scripts/encoder_decoder/eval/run_eval_classification.sh \
  --dataset-path data/encoder_decoder/classification_valid.jsonl \
  --model-id google/flan-t5-base \
  --adapter-dir outputs/encoder_decoder/flan_t5_base_classification_lora
```

## HPO Pipelines

### Encoder-Decoder (Optuna)

1) Define search spaces:
- `hpo/encoder_decoder/translation_search_space.yaml`
- `hpo/encoder_decoder/classification_search_space.yaml`

2) Run studies:
```
python scripts/hpo/encoder_decoder/run_optuna_translation.py
python scripts/hpo/encoder_decoder/run_optuna_classification.py
```

3) Collect results:
```
python scripts/hpo/encoder_decoder/collect_optuna_results.py
```

The runner scripts call `scripts/encoder_decoder/train_encdec_lora.py` for each trial, write trial configs under `hpo/encoder_decoder/generated_*`, train outputs under `outputs/hpo/encoder_decoder/*`, and metrics are read from `trainer_state.json` (`eval_loss`).

### Decoder-Only (Axolotl)

1) Define search space:
- `hpo/decoder_only/qwen3_search_space.yaml`

2) Generate trial configs:
```
python scripts/hpo/decoder_only/generate_axolotl_trials.py
```

3) Run trials:
```
scripts/hpo/decoder_only/run_trials.sh
```

4) Collect results:
```
python scripts/hpo/decoder_only/collect_axolotl_metrics.py
```

The generator creates per-trial Axolotl configs and a manifest CSV, the runner executes `axolotl preprocess/train` per manifest row, and metrics are collected from `trainer_state.json` under each trial output directory.

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
