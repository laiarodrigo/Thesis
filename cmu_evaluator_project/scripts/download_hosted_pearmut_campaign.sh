#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <CAMPAIGN_ID>" >&2
  exit 2
fi

CAMPAIGN_ID="$1"
[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: invalid campaign ID" >&2
  exit 2
}
command -v railway >/dev/null || {
  echo "ERROR: Railway CLI is not installed. See deployment/railway/README.md." >&2
  exit 2
}

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
DOWNLOAD_ROOT="${PEARMUT_HOSTED_DOWNLOAD_ROOT:-$REPO_ROOT/cmu_evaluator_project/runtime/hosted_downloads}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
snapshot="$DOWNLOAD_ROOT/${CAMPAIGN_ID}_$timestamp"
state="$snapshot/state"

mkdir -p "$DOWNLOAD_ROOT"
if [[ -e "$snapshot" ]]; then
  echo "ERROR: snapshot destination already exists; refusing overwrite: $snapshot" >&2
  exit 2
fi
mkdir -p "$state/data/campaigns" "$state/data/outputs"

echo "Downloading live campaign state from the linked Railway volume..."
railway volume files download \
  "/pearmut/data/campaigns/$CAMPAIGN_ID.json" \
  "$state/data/campaigns/$CAMPAIGN_ID.json"
railway volume files download \
  "/pearmut/data/outputs/$CAMPAIGN_ID.jsonl" \
  "$state/data/outputs/$CAMPAIGN_ID.jsonl"
railway volume files download \
  "/pearmut/data/progress.json" \
  "$state/data/progress.json"

python - "$state" "$CAMPAIGN_ID" <<'PY'
import json
import sys
from pathlib import Path

root, campaign_id = Path(sys.argv[1]), sys.argv[2]
campaign = json.loads(
    (root / "data" / "campaigns" / f"{campaign_id}.json").read_text(encoding="utf-8")
)
if campaign.get("campaign_id") != campaign_id:
    raise SystemExit("ERROR: downloaded campaign ID does not match the requested ID")
progress = json.loads((root / "data" / "progress.json").read_text(encoding="utf-8"))
if campaign_id not in progress:
    raise SystemExit("ERROR: requested campaign is absent from downloaded progress.json")
output = root / "data" / "outputs" / f"{campaign_id}.jsonl"
for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), 1):
    try:
        json.loads(line)
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"ERROR: invalid annotation JSONL at line {line_number}: {error}"
        ) from error
PY

backup_output="$(
  REPO_ROOT="$REPO_ROOT" bash \
    "$REPO_ROOT/cmu_evaluator_project/scripts/backup_pearmut_campaign.sh" \
    "$state" "$CAMPAIGN_ID"
)"
archive="$(printf '%s\n' "$backup_output" | sed -n '1p')"
checksum="$(printf '%s\n' "$backup_output" | sed -n '2p')"

echo "Snapshot directory: $snapshot"
echo "Raw annotations: $state/data/outputs/$CAMPAIGN_ID.jsonl"
echo "Campaign backup: $archive"
echo "SHA-256 file: $checksum"
echo "No curated/final research artifact was overwritten."
