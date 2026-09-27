#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REMOTE_HOST="${REMOTE_HOST:-u036584@slurm.hlt.inesc-id.pt}"
REMOTE_REPO="${REMOTE_REPO:-~/repos/Thesis}"
DRY_RUN="${DRY_RUN:-0}"

FILES=(
  "scripts/analysis/thesis_dataset_statistics.py"
  "scripts/analysis/run_thesis_dataset_statistics.sh"
  "scripts/slurm/run_thesis_dataset_statistics.sbatch"
  "scripts/slurm/submit_thesis_dataset_statistics.sh"
  "scripts/slurm/rsync_thesis_dataset_statistics_to_slurm.sh"
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

Remote repo:
  $REMOTE_HOST:$REMOTE_REPO

Next:
  ssh $REMOTE_HOST
  cd $REMOTE_REPO
  bash scripts/slurm/submit_thesis_dataset_statistics.sh
EOF
