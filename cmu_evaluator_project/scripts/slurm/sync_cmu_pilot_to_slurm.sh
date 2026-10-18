#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
REMOTE_HOST="${REMOTE_HOST:-u036584@slurm.hlt.inesc-id.pt}"
REMOTE_REPO="${REMOTE_REPO:-/cfs/home/u036584/repos/Thesis}"

cd "$REPO_ROOT"
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO/cmu_evaluator_project/logs/slurm'"
rsync -avP \
  --exclude 'logs/slurm/*.out' \
  --exclude 'logs/slurm/*.err' \
  --exclude 'data/pearmut/' \
  --exclude 'data/annotations/' \
  cmu_evaluator_project/ \
  "$REMOTE_HOST:$REMOTE_REPO/cmu_evaluator_project/"

echo "Synced CMU generation inputs (excluding Pearmut and annotations) to $REMOTE_HOST:$REMOTE_REPO/cmu_evaluator_project/"
