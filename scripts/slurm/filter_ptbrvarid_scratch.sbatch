#!/usr/bin/env bash
#SBATCH --job-name=ptbr_filter
#SBATCH --partition=cpu-cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV_NAME"
fi

REPO_ROOT="${REPO_ROOT:-$PWD}"
SOURCE_DB="${SOURCE_DB:-$REPO_ROOT/data/duckdb/subs.duckdb}"
OUTPUT_DB="${OUTPUT_DB:-$REPO_ROOT/data/duckdb/subs_ptbr_filtered.duckdb}"
DOMAIN="${DOMAIN:-}"
SPLIT="${SPLIT:-train}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20000}"
WORK_ROOT="${SLURM_TMPDIR:-/tmp/$USER/ptbr_filter_${SLURM_JOB_ID:-manual}}"

cleanup() {
  if [[ -n "${WORK_ROOT:-}" && -d "${WORK_ROOT:-}" ]]; then
    rm -rf "$WORK_ROOT"
  fi
}

trap cleanup EXIT

if [[ -z "$DOMAIN" ]]; then
  echo "ERROR: set DOMAIN (e.g. DOMAIN=journalistic)." >&2
  exit 2
fi

rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"
mkdir -p "$(dirname "$OUTPUT_DB")"

if [[ -f "$OUTPUT_DB" ]]; then
  BASE_DB="$OUTPUT_DB"
else
  BASE_DB="$SOURCE_DB"
fi

WORK_DB="$WORK_ROOT/$(basename "$OUTPUT_DB")"

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
echo "[$(date)] domain=$DOMAIN split=$SPLIT"
echo "[$(date)] source_db=$SOURCE_DB"
echo "[$(date)] output_db=$OUTPUT_DB"
echo "[$(date)] base_db=$BASE_DB"
echo "[$(date)] work_root=$WORK_ROOT"

if [[ ! -f "$BASE_DB" ]]; then
  echo "ERROR: base DB not found: $BASE_DB" >&2
  exit 2
fi

echo "[$(date)] copying DB to local scratch..."
cp "$BASE_DB" "$WORK_DB"
if [[ -f "${BASE_DB}.wal" ]]; then
  cp "${BASE_DB}.wal" "${WORK_DB}.wal"
fi

python3 - <<PY
from pathlib import Path
import duckdb

db = Path("$WORK_DB")
con = duckdb.connect(db.as_posix(), read_only=True)
print("scratch_raw_total:", con.execute(
    "SELECT COUNT(*) FROM ptbrvarid WHERE dataset='PtBrVId-Raw'"
).fetchone()[0])
print("scratch_existing_filtered:", con.execute(
    "SELECT COUNT(*) FROM ptbrvarid WHERE dataset='PtBrVId'"
).fetchone()[0])
con.close()
PY

CMD=(
  python3
  -u
  scripts/ptbrvarid/filter_ptbrvarid.py
  --db "$WORK_DB"
  --domain "$DOMAIN"
  --split "$SPLIT"
  --progress-every "$PROGRESS_EVERY"
)

printf '[%s] command:' "$(date)"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

python3 - <<PY
from pathlib import Path
import duckdb

db = Path("$WORK_DB")
con = duckdb.connect(db.as_posix())
con.execute("CHECKPOINT")
metrics = con.execute(
    """
    SELECT domain, split, raw, after_nonempty, after_jusText, after_author,
           after_clean, after_dedup, after_filters, after_IQR
    FROM ptbrvarid_metrics
    WHERE dataset='PtBrVId' AND domain=? AND split=?
    """,
    ["$DOMAIN", "$SPLIT"],
).fetchall()
rows = con.execute(
    """
    SELECT COUNT(*)
    FROM ptbrvarid
    WHERE dataset='PtBrVId' AND domain=? AND split=?
    """,
    ["$DOMAIN", "$SPLIT"],
).fetchone()[0]
con.close()
print("filtered_rows:", rows)
print("metrics:", metrics)
PY

TMP_OUT="${OUTPUT_DB}.job${SLURM_JOB_ID:-manual}.tmp"
echo "[$(date)] copying scratch DB back to $TMP_OUT"
cp "$WORK_DB" "$TMP_OUT"
mv "$TMP_OUT" "$OUTPUT_DB"

if [[ -f "${WORK_DB}.wal" ]]; then
  rm -f "${WORK_DB}.wal"
fi
rm -f "${OUTPUT_DB}.wal" "${TMP_OUT}.wal"

echo "[$(date)] done"
