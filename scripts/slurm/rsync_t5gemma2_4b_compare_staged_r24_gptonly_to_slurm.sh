#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REMOTE_HOST="${REMOTE_HOST:-u036584@slurm.hlt.inesc-id.pt}"
REMOTE_REPO="${REMOTE_REPO:-~/repos/Thesis}"
DRY_RUN="${DRY_RUN:-0}"

FILES=(
  "data/wikipedia_pt_variant_csv/pt_variant_prompts_wikipedia_merged.csv"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_only.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_plus_frmt.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageB_gpt_wikipedia_plus_frmt.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageB_gpt_wikipedia_only.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/classification_head_r24_stageA_opensubs_plus_frmt.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/classification_head_r24_stageB_gpt_wikipedia_only.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gpt_wiki.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gpt_wiki_wer.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gpt_wiki_bleu_wer.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gpt_wiki_frmt.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gpt_wiki_frmt_wer.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gpt_wiki_frmt_bleu_wer.yaml"
  "configs/encoder_decoder/multitask_4b/stageB_gpt_wiki_frmt_from_translation_stageA_r48_noequal.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_only.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_plus_frmt.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_plus_frmt_g07_bf16.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_plus_frmt_label_first.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt_with_cls.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_only.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_only_g07_bf16.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageC_drgrpo_frmt_gpt_wiki_legit_bleu_wer.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageC_drgrpo_frmt_gpt_wiki_legit_bleu_wer_g07_bf16.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_plus_ptbrvarid_label_first_with_cls_equal_legit.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_plus_ptbrvarid_label_first_with_cls_equal_legit_g07_bf16.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls_equal_legit.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls_equal_legit_g07_bf16.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageC_drgrpo_gpt_wikipedia_plus_frmt_label_first_with_cls_equal_bleu_wer_first_token_gold_legit.yaml"
  "configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageC_drgrpo_gpt_wikipedia_plus_frmt_label_first_with_cls_equal_bleu_wer_first_token_gold_legit_g07_bf16.yaml"
  "scripts/encoder_decoder/single_task_models/export_encdec_data.py"
  "scripts/encoder_decoder/single_task_models/t5gemma2/step2_build_all_tasks_from_csv.py"
  "scripts/encoder_decoder/single_task_models/t5gemma2/step3_split_tasks.py"
  "scripts/encoder_decoder/single_task_models/t5gemma2/step4a_translation_train_boilerplate.py"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_translation_gpt_wiki_frmt_mix.py"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_stageB_gpt_wiki_frmt_translation_plus_cls.py"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_stageB_gpt_wiki_frmt_label_first_with_cls.py"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_final_supervised_jsonl.py"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_translation_stageA_opensubs_frmt.py"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/build_translation_stageA_opensubs_frmt_label_first.py"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageA_opensubs_plus_frmt.sh"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageA_opensubs_plus_frmt_label_first.sh"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageB_gpt_wikipedia_plus_frmt.sh"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageB_gpt_wikipedia_plus_frmt_with_cls.sh"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls.sh"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_export_stageB_gpt_wikipedia_only.sh"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_classification_compare_staged.sh"
  "scripts/encoder_decoder/single_task_models/t5gemma2_4b/run_translation_compare_staged.sh"
  "scripts/encoder_decoder/train_encdec_lora.py"
  "scripts/encoder_decoder/task_balanced_sampler.py"
  "scripts/encoder_decoder/smoke_final_protocol.py"
  "scripts/encoder_decoder/audit_final_supervised_jsonl.py"
  "scripts/encoder_decoder/eval/evaluate_encdec.py"
  "scripts/encoder_decoder/eval/evaluate_classification_head.py"
  "scripts/encoder_decoder/eval/upsert_classification_report_row.py"
  "scripts/encoder_decoder/eval/build_label_first_eval_dataset.py"
  "scripts/encoder_decoder/eval/extract_translation_eval_dataset.py"
  "scripts/encoder_decoder/eval/prefix_classification_inputs.py"
  "scripts/encoder_decoder/eval/diagnose_prediction_rewards.py"
  "scripts/encoder_decoder/eval/run_eval_translation_gemma4b_r24_adaptive.sh"
  "scripts/encoder_decoder/stage_c/build_stage_c_subset.py"
  "scripts/encoder_decoder/stage_c/summarize_candidate_debug.py"
  "scripts/encoder_decoder/stage_c/train_stage_c_seq2seq_grpo.py"
  "scripts/encoder_decoder/multitask/build_multitask_jsonl.py"
  "scripts/encoder_decoder/multitask/task_protocol.py"
  "scripts/encoder_decoder/multitask/train_multitask_seq2seq.py"
  "scripts/encoder_decoder/multitask/eval_multitask_seq2seq_skeleton.py"
  "scripts/encoder_decoder/multitask/run_build_multitask_stageB_gpt_wiki_frmt_noequal.sh"
  "scripts/encoder_decoder/multitask/run_train_multitask_4b_stageB_gpt_wiki_frmt_from_translation_stageA_r48.sh"
  "scripts/encoder_decoder/multitask/run_eval_multitask_4b.sh"
  "scripts/slurm/build_stage_c_subset.sbatch"
  "scripts/slurm/train_t5gemma2_4b_translation_stageC_grpo.sbatch"
  "scripts/slurm/submit_t5gemma2_4b_stageC_gptwiki_dual.sh"
  "scripts/slurm/train_t5gemma2_4b_translation_compare_staged.sbatch"
  "scripts/slurm/train_t5gemma2_classification_compare_staged.sbatch"
  "scripts/slurm/eval_t5gemma2_4b_translation_r24_stageB_gpt_wiki.sbatch"
  "scripts/slurm/eval_t5gemma2_seq2seq_classification.sbatch"
  "scripts/slurm/eval_prediction_diagnostics.sbatch"
  "scripts/slurm/submit_stagec_eval_diagnostics.sh"
  "scripts/slurm/submit_t5gemma2_4b_compare_staged_r24_gptonly_pipeline.sh"
)

RSYNC_FLAGS=(-avP --relative)
if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_FLAGS+=(-n)
fi

for path in "${FILES[@]}"; do
  [[ -e "$path" ]] || { echo "ERROR: missing file $path" >&2; exit 2; }
done

ssh "$REMOTE_HOST" "mkdir -p $REMOTE_REPO"
rsync "${RSYNC_FLAGS[@]}" "${FILES[@]}" "$REMOTE_HOST:$REMOTE_REPO/"

cat <<EOF
Sync complete.

Remote repo: $REMOTE_HOST:$REMOTE_REPO

Next:
  ssh $REMOTE_HOST
  cd $REMOTE_REPO
  bash scripts/slurm/submit_t5gemma2_4b_compare_staged_r24_gptonly_pipeline.sh
EOF
