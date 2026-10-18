#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || "$3" != "--confirm-service-stopped" ]]; then
  echo "Usage: $0 <BACKUP.tar.gz> <CAMPAIGN_ID> --confirm-service-stopped" >&2
  echo "Stop/scale the Railway service to zero before running this command." >&2
  exit 2
fi

ARCHIVE="$1"
CAMPAIGN_ID="$2"
[[ -f "$ARCHIVE" ]] || { echo "ERROR: backup not found: $ARCHIVE" >&2; exit 2; }
[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: invalid campaign ID" >&2
  exit 2
}
command -v railway >/dev/null || { echo "ERROR: Railway CLI is not installed" >&2; exit 2; }

if [[ -f "$ARCHIVE.sha256" ]]; then
  expected_checksum="$(awk 'NR == 1 {print $1}' "$ARCHIVE.sha256")"
  actual_checksum="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
  [[ -n "$expected_checksum" && "$actual_checksum" == "$expected_checksum" ]] || {
    echo "ERROR: backup SHA-256 check failed: $ARCHIVE" >&2
    exit 2
  }
fi

staging="$(mktemp -d)"
cleanup() { rm -rf "$staging"; }
trap cleanup EXIT

python - "$ARCHIVE" "$staging" "$CAMPAIGN_ID" <<'PY'
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

archive, destination, campaign_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
with tarfile.open(archive, "r:gz") as handle:
    for member in handle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not member.isfile():
            if member.isdir():
                continue
            raise SystemExit(f"ERROR: unsafe backup member: {member.name}")
    handle.extractall(destination, filter="data")

campaign = json.loads((destination / "campaign.json").read_text(encoding="utf-8"))
if campaign.get("campaign_id") != campaign_id:
    raise SystemExit("ERROR: backup campaign ID does not match requested campaign")
progress = json.loads((destination / "progress.json").read_text(encoding="utf-8"))
if set(progress) != {campaign_id}:
    raise SystemExit("ERROR: backup progress is not isolated to the requested campaign")
for required in ("annotations.jsonl", "README.txt"):
    if not (destination / required).is_file():
        raise SystemExit(f"ERROR: backup is missing {required}")
PY

echo "Downloading current global progress so other campaigns remain untouched..."
railway volume files download "/pearmut/data/progress.json" "$staging/live-progress.json"

python - "$staging/live-progress.json" "$staging/progress.json" \
  "$CAMPAIGN_ID" "$staging/merged-progress.json" <<'PY'
import json
import sys
from pathlib import Path

live_path = Path(sys.argv[1])
backup_path = Path(sys.argv[2])
campaign_id = sys.argv[3]
output_path = Path(sys.argv[4])
live = json.loads(live_path.read_text(encoding="utf-8"))
backup = json.loads(backup_path.read_text(encoding="utf-8"))
live[campaign_id] = backup[campaign_id]
output_path.write_text(
    json.dumps(live, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

echo "Restoring $CAMPAIGN_ID to the stopped Railway service volume..."
railway volume files upload "$staging/campaign.json" \
  "/pearmut/data/campaigns/$CAMPAIGN_ID.json" --overwrite
railway volume files upload "$staging/annotations.jsonl" \
  "/pearmut/data/outputs/$CAMPAIGN_ID.jsonl" --overwrite
# Upload global progress last, after its campaign and annotation files are present.
railway volume files upload "$staging/merged-progress.json" \
  "/pearmut/data/progress.json" --overwrite

echo "Restore complete. Start/scale the Railway service back to one replica, then check /healthz."
