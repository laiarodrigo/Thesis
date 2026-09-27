# Thesis Dataset Statistics Summary

## Resolved Inputs
- Repository root: `/cfs/home/u036584/repos/Thesis`
- Reference config: `configs/encoder_decoder/t5gemma2/comparison_staged/translation_fullft_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls_from_stageA_opensubs_plus_ptbrvarid.yaml`
- Tokenizer source: `/cfs/home/u036584/repos/Thesis/outputs/encoder_decoder/compare_staged/t5gemma2_270m_fullft_stageA_opensubs_plus_ptbrvarid_label_first_with_cls`
- Project DB: `data/duckdb/subs_project.duckdb`
- Source DB: `data/duckdb/subs.duckdb`
- PtBrVId DB used for preprocessing/raw stats: `data/duckdb/subs_ptbr_filtered.duckdb`

## Key Findings
- OpenSubs contributes train rows only in the project training view. The source valid count for dataset=OpenSubs is zero.
- Validation top-up to 200 rows is implemented in data-building scripts, not in the trainer. The training scripts consume already-built JSONL files.
- Golden Collection is wired as a test-only evaluation dataset in the current repository layout.
- PtBrVId is loaded directly from a DuckDB table in the Stage A augmentation code, not from the unified subs_project training view.
- The script distinguishes the older unfiltered GPT+FRMT family from the staged GPT-Wikipedia+FRMT family that applies FRMT filtering.
- For thesis-facing Stage A summaries, OpenSubs+FRMT training can also be interpreted with an effective-exposure assumption where train FRMT is ignored relative to OpenSubs.

## Dataset Counts
- OpenSubs train source pairs: 10,347,883 (translation examples: 20,695,766; classification examples: 17,698,911).
- FRMT clean pairs: train=2,502, valid=20, test=2,611.
- GPT-Wikipedia pairs: train=4,519, valid=564, test=566.
- Golden Collection test pairs: 500 (translation examples: 1,000; classification examples: 773).
- PtBrVId filtered raw texts: 2,990,995 train rows (pt-BR=337,146, pt-PT=2,653,849).
- PtBrVId translated canonical pairs: train=3,600, valid=0, test=400.

## Stage Highlights
- Stage A OpenSubs-only translation: train OpenSubs=20,695,766, valid OpenSubs=0.
- Stage A clean OpenSubs+PtBrVId classification source mix: train OpenSubs=5,583,112, train PtBrVId=2,749,891.
- Stage A OpenSubs+FRMT thesis assumption: effective train OpenSubs=20,695,766, effective train FRMT=0, valid FRMT=40, valid OpenSubs top-up=160.
- Legacy GPT+FRMT unfiltered translation: train GPT=n/a, train FRMT=5,004, valid GPT=n/a, valid FRMT=40.
- Stage B GPT+FRMT filtered translation mix: train GPT=n/a, train FRMT=5,004, valid GPT=n/a, valid FRMT=40.
- Stage B label-first translation-only: train GPT=n/a, train FRMT=4,964, valid GPT=n/a, valid FRMT=40.

## Preprocessing
- Total jusText removals recorded in ptbrvarid_metrics: 4,785,486.
- Preprocessing summary rows written: 42; jusText domain rows written: 6.

## Tokenization
- Token statistic rows written: 22.

## Config Anomalies
- configs/encoder_decoder/t5gemma2/comparison_staged/classification_head_fullft_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/classification_train.jsonl.
- configs/encoder_decoder/t5gemma2/comparison_staged/translation_fullft_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/translation_train.jsonl.
- configs/encoder_decoder/t5gemma2/comparison_staged/translation_fullft_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls.yaml is named '*with_cls*' but points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_noequal/train.jsonl, which is the translation-only label-first path.
- configs/encoder_decoder/t5gemma2_4b/comparison_staged/classification_head_r24_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/classification_train.jsonl.
- configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r16_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/translation_train.jsonl.
- configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/translation_train.jsonl.
- configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_frmt.yaml is named '*opensubs_frmt*' but currently points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/translation_train.jsonl.
- configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls.yaml is named '*with_cls*' but points to data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_noequal/train.jsonl, which is the translation-only label-first path.
