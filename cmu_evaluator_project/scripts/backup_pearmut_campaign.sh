#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <PEARMUT_ROOT> <CAMPAIGN_ID>" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
PEARMUT_ROOT_INPUT="$1"
CAMPAIGN_ID="$2"
[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: invalid campaign ID" >&2
  exit 2
}
[[ -d "$PEARMUT_ROOT_INPUT" ]] || {
  echo "ERROR: Pearmut root does not exist: $PEARMUT_ROOT_INPUT" >&2
  exit 2
}
PEARMUT_ROOT_ABS="$(cd "$PEARMUT_ROOT_INPUT" && pwd)"
CAMPAIGN_PATH="$PEARMUT_ROOT_ABS/data/campaigns/$CAMPAIGN_ID.json"
OUTPUT_PATH="$PEARMUT_ROOT_ABS/data/outputs/$CAMPAIGN_ID.jsonl"
PROGRESS_PATH="$PEARMUT_ROOT_ABS/data/progress.json"
[[ -f "$CAMPAIGN_PATH" ]] || { echo "ERROR: missing $CAMPAIGN_PATH" >&2; exit 2; }
[[ -f "$PROGRESS_PATH" ]] || { echo "ERROR: missing $PROGRESS_PATH" >&2; exit 2; }

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive_dir="$PEARMUT_ROOT_ABS/backups"
archive_path="$archive_dir/${CAMPAIGN_ID}_${timestamp}.tar.gz"
staging="$(mktemp -d)"
cleanup() { rm -rf "$staging"; }
trap cleanup EXIT
mkdir -p "$archive_dir" "$staging/provenance"
umask 077

cp "$CAMPAIGN_PATH" "$staging/campaign.json"
if [[ -f "$OUTPUT_PATH" ]]; then
  cp "$OUTPUT_PATH" "$staging/annotations.jsonl"
else
  : >"$staging/annotations.jsonl"
fi

python - "$PROGRESS_PATH" "$CAMPAIGN_ID" "$staging/progress.json" <<'PY'
import json
import sys
from pathlib import Path

source, campaign_id, destination = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
progress = json.loads(source.read_text(encoding="utf-8"))
if campaign_id not in progress:
    raise SystemExit(f"ERROR: {campaign_id} is absent from {source}")
destination.write_text(
    json.dumps({campaign_id: progress[campaign_id]}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

python - "$REPO_ROOT/cmu_evaluator_project/data/pearmut" "$CAMPAIGN_ID" "$staging/provenance" <<'PY'
import json
import shutil
import sys
from pathlib import Path

root, campaign_id, destination = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
for path in root.rglob("*.json"):
    if "private" not in {part.lower() for part in path.parts}:
        continue
    if "roster" in path.name.lower():
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if campaign_id not in json.dumps(payload, ensure_ascii=False):
        continue
    relative = path.relative_to(root)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
PY

cat >"$staging/README.txt" <<EOF
Pearmut campaign backup
campaign_id=$CAMPAIGN_ID
created_utc=$timestamp
annotations_authoritative=true
private_real_name_roster_included=false
EOF

tar -C "$staging" -czf "$archive_path" .
sha256sum "$archive_path" >"$archive_path.sha256"
echo "$archive_path"
echo "$archive_path.sha256"
