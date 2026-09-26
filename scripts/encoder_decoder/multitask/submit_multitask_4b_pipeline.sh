#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
PARTITION="${PARTITION:-hlt_msc}"
GPU_GRES="${GPU_GRES:-gpu:RTX_3090_24GB:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-30G}"
BUILD_TIME="${BUILD_TIME:-14-00:00:00}"
TRAIN_TIME="${TRAIN_TIME:-14-00:00:00}"
EVAL_TIME="${EVAL_TIME:-14-00:00:00}"

PROJECT_DB_PATH="${PROJECT_DB_PATH:-$REPO_ROOT/data/duckdb/subs_project.duckdb}"
SOURCE_DB_PATH="${SOURCE_DB_PATH:-$REPO_ROOT/data/duckdb/subs.duckdb}"

STAGEA_EXPORT_OUT_DIR="${STAGEA_EXPORT_OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only}"
STAGEA_MULTITASK_OUT_DIR="${STAGEA_MULTITASK_OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/multitask_stageA}"

FRMT_TRANSLATION_TRAIN="${FRMT_TRANSLATION_TRAIN:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only/translation_train.jsonl}"
FRMT_TRANSLATION_VALID="${FRMT_TRANSLATION_VALID:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only/translation_valid.jsonl}"
FRMT_CLASSIFICATION_TRAIN="${FRMT_CLASSIFICATION_TRAIN:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only/classification_train.jsonl}"
FRMT_CLASSIFICATION_VALID="${FRMT_CLASSIFICATION_VALID:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only/classification_valid.jsonl}"

GPT_TRANSLATION_TRAIN="${GPT_TRANSLATION_TRAIN:-$REPO_ROOT/data/encoder_decoder/t5gemma2/gpt_refresh_2st_20260312_2109/translation_train.jsonl}"
GPT_TRANSLATION_VALID="${GPT_TRANSLATION_VALID:-$REPO_ROOT/data/encoder_decoder/t5gemma2/gpt_refresh_2st_20260312_2109/translation_valid.jsonl}"
GPT_CLASSIFICATION_TRAIN="${GPT_CLASSIFICATION_TRAIN:-$REPO_ROOT/data/encoder_decoder/t5gemma2/gpt_refresh_2st_20260312_2109/classification_train.jsonl}"
GPT_CLASSIFICATION_VALID="${GPT_CLASSIFICATION_VALID:-$REPO_ROOT/data/encoder_decoder/t5gemma2/gpt_refresh_2st_20260312_2109/classification_valid.jsonl}"

STAGEB_MIX_OUT_DIR="${STAGEB_MIX_OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_refresh2_frmt_mix}"
STAGEB_MULTITASK_OUT_DIR="${STAGEB_MULTITASK_OUT_DIR:-$REPO_ROOT/data/encoder_decoder/t5gemma2/compare_staged_v2/multitask_stageB}"

STAGEA_CONFIG="${STAGEA_CONFIG:-configs/encoder_decoder/multitask_4b/stageA_compare_staged_v2.yaml}"
STAGEB_CONFIG="${STAGEB_CONFIG:-configs/encoder_decoder/multitask_4b/stageB_gpt_refresh2_frmt.yaml}"
STAGEA_OUT="${STAGEA_OUT:-outputs/encoder_decoder/multitask_compare/t5gemma2_4b_multitask_r8_stageA_compare_staged_v2}"
STAGEB_OUT="${STAGEB_OUT:-outputs/encoder_decoder/multitask_compare/t5gemma2_4b_multitask_r8_stageB_gpt_refresh2_frmt}"

GOLDEN_TRANSLATION_DATASET="${GOLDEN_TRANSLATION_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl}"
GOLDEN_CLASSIFICATION_DATASET="${GOLDEN_CLASSIFICATION_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/classification_test.jsonl}"
GOLDEN_EVAL_OUT="${GOLDEN_EVAL_OUT:-eval_results/encoder_decoder/multitask_compare/t5gemma2_4b_multitask_r8_stageB_golden}"

FRMT_TRANSLATION_DATASET="${FRMT_TRANSLATION_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only/translation_test.jsonl}"
FRMT_CLASSIFICATION_DATASET="${FRMT_CLASSIFICATION_DATASET:-$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only/classification_test.jsonl}"
FRMT_EVAL_OUT="${FRMT_EVAL_OUT:-eval_results/encoder_decoder/multitask_compare/t5gemma2_4b_multitask_r8_stageB_frmt}"

BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-384}"
MAX_NEW_TOKENS_TRANSLATION="${MAX_NEW_TOKENS_TRANSLATION:-192}"
MAX_NEW_TOKENS_CLASSIFICATION="${MAX_NEW_TOKENS_CLASSIFICATION:-2}"
ADAPTIVE_MAX_NEW_TOKENS="${ADAPTIVE_MAX_NEW_TOKENS:-1}"
ADAPTIVE_RATIO="${ADAPTIVE_RATIO:-1.15}"
ADAPTIVE_MARGIN="${ADAPTIVE_MARGIN:-6}"
ADAPTIVE_MIN_NEW_TOKENS="${ADAPTIVE_MIN_NEW_TOKENS:-6}"
ADAPTIVE_MAX_NEW_TOKENS_CEILING="${ADAPTIVE_MAX_NEW_TOKENS_CEILING:-192}"

cd "$REPO_ROOT"

for req in \
  "$STAGEA_CONFIG" \
  "$STAGEB_CONFIG" \
  "scripts/encoder_decoder/multitask/train_multitask_seq2seq.py" \
  "scripts/encoder_decoder/multitask/run_build_multitask_stageA.sh" \
  "scripts/encoder_decoder/multitask/run_build_multitask_stageB_gpt_refresh2_frmt.sh" \
  "scripts/encoder_decoder/multitask/run_eval_multitask_4b.sh" \
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageA_opensubs_frmt.sh" \
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_build_translation_gpt_refresh2_frmt_mix.sh"
do
  [[ -e "$req" ]] || { echo "ERROR: missing file $req" >&2; exit 2; }
done

for dataset in \
  "$GOLDEN_TRANSLATION_DATASET" \
  "$GOLDEN_CLASSIFICATION_DATASET"
do
  [[ -e "$dataset" ]] || { echo "ERROR: missing dataset $dataset" >&2; exit 2; }
done

export_stagea_wrap() {
  printf "set -euxo pipefail; cd '%s'; source '%s/bin/activate'; PROJECT_DB_PATH='%s' SOURCE_DB_PATH='%s' OUT_DIR='%s' bash scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageA_opensubs_frmt.sh" \
    "$REPO_ROOT" \
    "$VENV_PATH" \
    "$PROJECT_DB_PATH" \
    "$SOURCE_DB_PATH" \
    "$STAGEA_EXPORT_OUT_DIR"
}

build_stagea_multitask_wrap() {
  printf "set -euxo pipefail; cd '%s'; source '%s/bin/activate'; TRANSLATION_TRAIN_IN='%s/translation_train.jsonl' TRANSLATION_VALID_IN='%s/translation_valid.jsonl' CLASSIFICATION_TRAIN_IN='%s/classification_train.jsonl' CLASSIFICATION_VALID_IN='%s/classification_valid.jsonl' OUT_DIR='%s' bash scripts/encoder_decoder/multitask/run_build_multitask_stageA.sh" \
    "$REPO_ROOT" \
    "$VENV_PATH" \
    "$STAGEA_EXPORT_OUT_DIR" \
    "$STAGEA_EXPORT_OUT_DIR" \
    "$STAGEA_EXPORT_OUT_DIR" \
    "$STAGEA_EXPORT_OUT_DIR" \
    "$STAGEA_MULTITASK_OUT_DIR"
}

build_stageb_mix_wrap() {
  printf "set -euxo pipefail; cd '%s'; source '%s/bin/activate'; bash scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_build_translation_gpt_refresh2_frmt_mix.sh --frmt-train '%s' --frmt-valid '%s' --gpt-train '%s' --gpt-valid '%s' --frmt-cls-train '%s' --frmt-cls-valid '%s' --gpt-cls-train '%s' --gpt-cls-valid '%s' --out-dir '%s'" \
    "$REPO_ROOT" \
    "$VENV_PATH" \
    "$FRMT_TRANSLATION_TRAIN" \
    "$FRMT_TRANSLATION_VALID" \
    "$GPT_TRANSLATION_TRAIN" \
    "$GPT_TRANSLATION_VALID" \
    "$FRMT_CLASSIFICATION_TRAIN" \
    "$FRMT_CLASSIFICATION_VALID" \
    "$GPT_CLASSIFICATION_TRAIN" \
    "$GPT_CLASSIFICATION_VALID" \
    "$STAGEB_MIX_OUT_DIR"
}

build_stageb_multitask_wrap() {
  printf "set -euxo pipefail; cd '%s'; source '%s/bin/activate'; TRANSLATION_TRAIN_IN='%s/translation_train.jsonl' TRANSLATION_VALID_IN='%s/translation_valid.jsonl' CLASSIFICATION_TRAIN_IN='%s/classification_train.jsonl' CLASSIFICATION_VALID_IN='%s/classification_valid.jsonl' OUT_DIR='%s' bash scripts/encoder_decoder/multitask/run_build_multitask_stageB_gpt_refresh2_frmt.sh" \
    "$REPO_ROOT" \
    "$VENV_PATH" \
    "$STAGEB_MIX_OUT_DIR" \
    "$STAGEB_MIX_OUT_DIR" \
    "$STAGEB_MIX_OUT_DIR" \
    "$STAGEB_MIX_OUT_DIR" \
    "$STAGEB_MULTITASK_OUT_DIR"
}

eval_wrap() {
  local out_dir="$1"
  local translation_dataset="$2"
  local classification_dataset="$3"
  local extra_adaptive=""
  if [[ "$ADAPTIVE_MAX_NEW_TOKENS" == "1" ]]; then
    extra_adaptive="ADAPTIVE_MAX_NEW_TOKENS=1"
  else
    extra_adaptive="ADAPTIVE_MAX_NEW_TOKENS=0"
  fi
  printf "set -euxo pipefail; cd '%s'; source '%s/bin/activate'; MODEL_ID='%s/%s' OUTPUT_DIR='%s/%s' TRANSLATION_DATASET='%s' CLASSIFICATION_DATASET='%s' BATCH_SIZE='%s' MAX_SOURCE_LENGTH='%s' MAX_NEW_TOKENS_TRANSLATION='%s' MAX_NEW_TOKENS_CLASSIFICATION='%s' ADAPTIVE_RATIO='%s' ADAPTIVE_MARGIN='%s' ADAPTIVE_MIN_NEW_TOKENS='%s' ADAPTIVE_MAX_NEW_TOKENS_CEILING='%s' %s bash scripts/encoder_decoder/multitask/run_eval_multitask_4b.sh" \
    "$REPO_ROOT" \
    "$VENV_PATH" \
    "$REPO_ROOT" \
    "$STAGEB_OUT" \
    "$REPO_ROOT" \
    "$out_dir" \
    "$translation_dataset" \
    "$classification_dataset" \
    "$BATCH_SIZE" \
    "$MAX_SOURCE_LENGTH" \
    "$MAX_NEW_TOKENS_TRANSLATION" \
    "$MAX_NEW_TOKENS_CLASSIFICATION" \
    "$ADAPTIVE_RATIO" \
    "$ADAPTIVE_MARGIN" \
    "$ADAPTIVE_MIN_NEW_TOKENS" \
    "$ADAPTIVE_MAX_NEW_TOKENS_CEILING" \
    "$extra_adaptive"
}

J_PREP_A="$(
  sbatch --parsable \
    --job-name=mt4b_prep_a \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$BUILD_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="$(export_stagea_wrap)"
)"

J_BUILD_A="$(
  sbatch --parsable \
    --dependency=afterok:"$J_PREP_A" \
    --job-name=mt4b_data_a \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$BUILD_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="$(build_stagea_multitask_wrap)"
)"

J_PREP_B="$(
  sbatch --parsable \
    --job-name=mt4b_prep_b \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$BUILD_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="$(build_stageb_mix_wrap)"
)"

J_BUILD_B="$(
  sbatch --parsable \
    --dependency=afterok:"$J_PREP_B" \
    --job-name=mt4b_data_b \
    --partition="$PARTITION" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$BUILD_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="$(build_stageb_multitask_wrap)"
)"

J_A="$(
  sbatch --parsable \
    --dependency=afterok:"$J_BUILD_A" \
    --job-name=mt4b_a \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TRAIN_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="set -euxo pipefail; cd '$REPO_ROOT'; source '$VENV_PATH/bin/activate'; python -u scripts/encoder_decoder/multitask/train_multitask_seq2seq.py --config '$STAGEA_CONFIG' --execute"
)"

J_B="$(
  sbatch --parsable \
    --dependency=afterok:"$J_A":"$J_BUILD_B" \
    --job-name=mt4b_b \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TRAIN_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="set -euxo pipefail; cd '$REPO_ROOT'; source '$VENV_PATH/bin/activate'; python -u scripts/encoder_decoder/multitask/train_multitask_seq2seq.py --config '$STAGEB_CONFIG' --execute"
)"

J_E_GOLD="$(
  sbatch --parsable \
    --dependency=afterok:"$J_B" \
    --job-name=mt4b_eval_gold \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$EVAL_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="$(eval_wrap "$GOLDEN_EVAL_OUT" "$GOLDEN_TRANSLATION_DATASET" "$GOLDEN_CLASSIFICATION_DATASET")"
)"

J_E_FRMT="$(
  sbatch --parsable \
    --dependency=afterok:"$J_B" \
    --job-name=mt4b_eval_frmt \
    --partition="$PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$EVAL_TIME" \
    --output=slurm-%x-%j.out \
    --error=slurm-%x-%j.err \
    --wrap="$(eval_wrap "$FRMT_EVAL_OUT" "$FRMT_TRANSLATION_DATASET" "$FRMT_CLASSIFICATION_DATASET")"
)"

echo "Submitted multitask 4B chain:"
echo "  prep_stageA = $J_PREP_A"
echo "  data_stageA = $J_BUILD_A"
echo "  prep_stageB = $J_PREP_B"
echo "  data_stageB = $J_BUILD_B"
echo "  stageA      = $J_A"
echo "  stageB      = $J_B"
echo "  eval_golden = $J_E_GOLD"
echo "  eval_frmt   = $J_E_FRMT"
echo
echo "Monitor:"
echo "  squeue -j $J_PREP_A,$J_BUILD_A,$J_PREP_B,$J_BUILD_B,$J_A,$J_B,$J_E_GOLD,$J_E_FRMT"
