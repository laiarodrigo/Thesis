#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Prefer the repo-local environment because the system python in this workspace
# does not necessarily include duckdb/transformers.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x "$REPO_ROOT/thesis/bin/python" ]]; then
  python_bin="$REPO_ROOT/thesis/bin/python"
elif [[ -x "$REPO_ROOT/.venv-axolotl/bin/python" ]]; then
  python_bin="$REPO_ROOT/.venv-axolotl/bin/python"
else
  python_bin="python3"
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "ERROR: python interpreter not found: $python_bin" >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT}"

for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  if [[ "$arg" == "--output-dir" ]]; then
    next_index=$((i + 1))
    if (( next_index <= $# )); then
      OUTPUT_DIR="${!next_index}"
    fi
  elif [[ "$arg" == --output-dir=* ]]; then
    OUTPUT_DIR="${arg#--output-dir=}"
  fi
done

if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
fi

echo "Running thesis dataset statistics"
echo "  repo root:   $REPO_ROOT"
echo "  python:      $python_bin"
echo "  output dir:  $OUTPUT_DIR"

"$python_bin" "$REPO_ROOT/scripts/analysis/thesis_dataset_statistics.py" \
  --repo-root "$REPO_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
