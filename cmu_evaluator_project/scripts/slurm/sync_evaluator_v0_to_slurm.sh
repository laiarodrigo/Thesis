#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
REMOTE_HOST="${REMOTE_HOST:-u036584@slurm.hlt.inesc-id.pt}"
REMOTE_REPO="${REMOTE_REPO:-/cfs/home/u036584/repos/Thesis}"

cd "$REPO_ROOT"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO/cmu_evaluator_project/logs/slurm' '$REMOTE_REPO/cmu_evaluator_project/runtime/evaluator_dev'"
rsync -avP --relative \
  ./cmu_evaluator_project/data/benchmark/dev_01_v0_input.jsonl \
  ./cmu_evaluator_project/guidelines/CMU_BP_EP_evaluator_v0_instructions.md \
  ./cmu_evaluator_project/scripts/requirements_gemma4.txt \
  ./cmu_evaluator_project/scripts/run_evaluator_v0.py \
  ./cmu_evaluator_project/scripts/slurm/run_evaluator_v0.sbatch \
  "$REMOTE_HOST:$REMOTE_REPO/"

echo "Synced label-free V0 inputs and runner only; no annotations or TEST data were transferred."
