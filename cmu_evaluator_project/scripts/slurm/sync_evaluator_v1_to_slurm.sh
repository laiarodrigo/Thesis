#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
REMOTE_HOST="${REMOTE_HOST:-u036584@slurm.hlt.inesc-id.pt}"
REMOTE_REPO="${REMOTE_REPO:-/cfs/home/u036584/repos/Thesis}"

cd "$REPO_ROOT"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO/cmu_evaluator_project/logs/slurm' '$REMOTE_REPO/cmu_evaluator_project/logs/model_reasoning' '$REMOTE_REPO/cmu_evaluator_project/runtime/evaluator_v1' '$REMOTE_REPO/cmu_evaluator_project/runtime/evaluator_model_reasoning'"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO/cmu_evaluator_project/logs/gemma31b' '$REMOTE_REPO/cmu_evaluator_project/runtime/evaluator_gemma31b'"
rsync -avP --relative \
  ./cmu_evaluator_project/data/benchmark/v1_37_input.jsonl \
  ./cmu_evaluator_project/data/benchmark/gemma31b_g0_g1_eval_96.jsonl \
  ./cmu_evaluator_project/data/benchmark/gemma31b_g0_g1_manifest.json \
  ./cmu_evaluator_project/data/benchmark/gemma31b_g1_eight_shot_demonstrations.json \
  ./cmu_evaluator_project/guidelines/v1/stage1_A.md \
  ./cmu_evaluator_project/guidelines/v1/stage1_B.md \
  ./cmu_evaluator_project/guidelines/v1/stage1_C.md \
  ./cmu_evaluator_project/guidelines/v1/stage1_E_accuracy.md \
  ./cmu_evaluator_project/guidelines/v1/stage1_E_fluency.md \
  ./cmu_evaluator_project/guidelines/v1/stage2_score.md \
  ./cmu_evaluator_project/scripts/requirements_gemma4.txt \
  ./cmu_evaluator_project/scripts/run_evaluator_v0.py \
  ./cmu_evaluator_project/scripts/run_evaluator_v1.py \
  ./cmu_evaluator_project/scripts/compare_v0_passage.py \
  ./cmu_evaluator_project/scripts/build_gemma31b_g0_g1.py \
  ./cmu_evaluator_project/scripts/run_evaluator_f2_hlt.sh \
  ./cmu_evaluator_project/scripts/run_evaluator_g_hlt.sh \
  ./cmu_evaluator_project/scripts/slurm/run_evaluator_f1.sbatch \
  ./cmu_evaluator_project/scripts/slurm/run_evaluator_v1.sbatch \
  "$REMOTE_HOST:$REMOTE_REPO/"

echo "Synced blind V1/G0/G1 inputs, frozen G1 demonstration payload, manifests, unchanged prompts, and launch code; no evaluation gold or held-out data were transferred."
