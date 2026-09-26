#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_PATH="${VENV_PATH:-$HOME/thesis_t5}"
PARTITION="${PARTITION:-hlt_msc}"
GPU_GRES="${GPU_GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
GPU_MEM="${GPU_MEM:-48G}"
GPU_TIME="${GPU_TIME:-24:00:00}"
CPU_MEM="${CPU_MEM:-24G}"
CPU_TIME="${CPU_TIME:-08:00:00}"
TOP_SUSPECTS="${TOP_SUSPECTS:-50}"
BEAMS="${BEAMS:-1 4}"

EVAL_SBATCH="${EVAL_SBATCH:-scripts/slurm/eval_t5gemma2_4b_translation_r24_stageB_gpt_wiki.sbatch}"
DIAG_SBATCH="${DIAG_SBATCH:-scripts/slurm/eval_prediction_diagnostics.sbatch}"
EVAL_SCRIPT="${EVAL_SCRIPT:-scripts/encoder_decoder/eval/run_eval_translation_gemma4b_r24_adaptive.sh}"

for req in \
  "$EVAL_SBATCH" \
  "$DIAG_SBATCH" \
  "$EVAL_SCRIPT" \
  "scripts/encoder_decoder/eval/diagnose_prediction_rewards.py"
do
  [[ -e "$req" ]] || { echo "ERROR: missing file $req" >&2; exit 2; }
done

declare -A ADAPTERS=(
  [stageB_gpt_wiki]="$REPO_ROOT/outputs/encoder_decoder/compare_staged/t5gemma2_4b_translation_r24_stageB_gpt_wiki"
  [stageC_bleu]="$REPO_ROOT/outputs/encoder_decoder/compare_staged/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki"
  [stageC_wer]="$REPO_ROOT/outputs/encoder_decoder/compare_staged/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki_wer"
  [stageC_bleu_wer]="$REPO_ROOT/outputs/encoder_decoder/compare_staged/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki_bleu_wer"
)

declare -A DATASETS=(
  [golden]="$REPO_ROOT/data/encoder_decoder/t5gemma2/golden_collection/translation_test.jsonl"
  [frmt]="$REPO_ROOT/data/encoder_decoder/t5gemma2/frmt_only/translation_test.jsonl"
)

declare -A GOLDEN_RUNS=(
  [stageB]="$REPO_ROOT/eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageB_gpt_wiki/20260330_170813_translation_predictions.jsonl"
  [stageC_bleu]="$REPO_ROOT/eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki/20260331_124432_translation_predictions.jsonl"
  [stageC_wer]="$REPO_ROOT/eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki_wer/20260402_104555_translation_predictions.jsonl"
  [stageC_bleu_wer]="$REPO_ROOT/eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki_bleu_wer/20260412_034835_translation_predictions.jsonl"
)

declare -A FRMT_MATCHED_RUNS=(
  [stageB]="$REPO_ROOT/eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r24_stageB_gpt_wiki/20260330_170813_translation_predictions.jsonl"
  [stageC_bleu]="$REPO_ROOT/eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki/20260331_151506_translation_predictions.jsonl"
  [stageC_wer]="$REPO_ROOT/eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki_wer/20260402_104555_translation_predictions.jsonl"
)

beam_jobs=()
echo "Submitting beam diagnostics"
for model_name in "${!ADAPTERS[@]}"; do
  adapter_dir="${ADAPTERS[$model_name]}"
  if [[ ! -d "$adapter_dir" ]]; then
    echo "  skip missing adapter: $model_name -> $adapter_dir"
    continue
  fi
  for dataset_name in "${!DATASETS[@]}"; do
    dataset_path="${DATASETS[$dataset_name]}"
    if [[ ! -f "$dataset_path" ]]; then
      echo "  skip missing dataset: $dataset_name -> $dataset_path"
      continue
    fi
    for beam in $BEAMS; do
      output_dir="$REPO_ROOT/eval_results/encoder_decoder/diagnostics/${model_name}/${dataset_name}/beam${beam}"
      job_name="diag_${model_name}_${dataset_name}_b${beam}"
      job_id="$(
        VENV_PATH="$VENV_PATH" \
        EVAL_SCRIPT="$EVAL_SCRIPT" \
        ADAPTER_DIR="$adapter_dir" \
        OUTPUT_DIR="$output_dir" \
        DATASET_PATH="$dataset_path" \
        NUM_BEAMS="$beam" \
        sbatch --parsable \
          --job-name="$job_name" \
          --partition="$PARTITION" \
          --cpus-per-task="$CPUS_PER_TASK" \
          --mem="$GPU_MEM" \
          --time="$GPU_TIME" \
          --gres="$GPU_GRES" \
          "$EVAL_SBATCH"
      )"
      beam_jobs+=("$job_id:$job_name")
      echo "  submitted $job_name -> $job_id"
    done
  done
done

format_run_specs() {
  local -n runs_ref=$1
  local pieces=()
  local key
  for key in "${!runs_ref[@]}"; do
    local path="${runs_ref[$key]}"
    [[ -f "$path" ]] || continue
    pieces+=("${key}=${path}")
  done
  local joined=""
  local piece
  for piece in "${pieces[@]}"; do
    if [[ -n "$joined" ]]; then
      joined+=";"
    fi
    joined+="$piece"
  done
  printf '%s' "$joined"
}

diag_jobs=()

golden_specs="$(format_run_specs GOLDEN_RUNS)"
if [[ -n "$golden_specs" ]]; then
  golden_output="$REPO_ROOT/eval_results/encoder_decoder/diagnostics/offline_reward_compare/golden_r24_gpt_wiki"
  job_id="$(
    VENV_PATH="$VENV_PATH" \
    OUTPUT_DIR="$golden_output" \
    BASELINE_NAME="stageB" \
    PREDICTION_RUNS="$golden_specs" \
    TOP_SUSPECTS="$TOP_SUSPECTS" \
    sbatch --parsable \
      --job-name="preddiag_golden" \
      --partition="$PARTITION" \
      --cpus-per-task=4 \
      --mem="$CPU_MEM" \
      --time="$CPU_TIME" \
      "$DIAG_SBATCH"
  )"
  diag_jobs+=("$job_id:preddiag_golden")
  echo "Submitted offline golden diagnostics -> $job_id"
else
  echo "Skip golden offline diagnostics: no prediction JSONLs found"
fi

frmt_specs="$(format_run_specs FRMT_MATCHED_RUNS)"
if [[ -n "$frmt_specs" ]]; then
  frmt_output="$REPO_ROOT/eval_results/encoder_decoder/diagnostics/offline_reward_compare/frmt_r24_gpt_wiki_matched"
  job_id="$(
    VENV_PATH="$VENV_PATH" \
    OUTPUT_DIR="$frmt_output" \
    BASELINE_NAME="stageB" \
    PREDICTION_RUNS="$frmt_specs" \
    TOP_SUSPECTS="$TOP_SUSPECTS" \
    sbatch --parsable \
      --job-name="preddiag_frmt" \
      --partition="$PARTITION" \
      --cpus-per-task=4 \
      --mem="$CPU_MEM" \
      --time="$CPU_TIME" \
      "$DIAG_SBATCH"
  )"
  diag_jobs+=("$job_id:preddiag_frmt")
  echo "Submitted offline FRMT diagnostics -> $job_id"
else
  echo "Skip FRMT offline diagnostics: no matched prediction JSONLs found"
fi

cat <<EOF
Submitted Stage C eval diagnostics.

Beam jobs:
$(printf '  %s\n' "${beam_jobs[@]:-none}")
Offline diagnostic jobs:
$(printf '  %s\n' "${diag_jobs[@]:-none}")
EOF
