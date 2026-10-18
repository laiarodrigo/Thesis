#!/usr/bin/env bash
set -euo pipefail

EXPECTED_VOLUME_PATH="/pearmut-volume"
EXPECTED_PEARMUT_ROOT="$EXPECTED_VOLUME_PATH/pearmut"
CAMPAIGN_SOURCE_DIR="/app/campaigns"
PEARMUT_PORT=8001

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${RAILWAY_VOLUME_MOUNT_PATH:-}" == "$EXPECTED_VOLUME_PATH" ]] || fail \
  "Railway volume is missing or mounted incorrectly. Attach one at $EXPECTED_VOLUME_PATH; RAILWAY_VOLUME_MOUNT_PATH='${RAILWAY_VOLUME_MOUNT_PATH:-unset}'."
[[ "${PEARMUT_ROOT:-}" == "$EXPECTED_PEARMUT_ROOT" ]] || fail \
  "PEARMUT_ROOT must be exactly $EXPECTED_PEARMUT_ROOT, inside the persistent volume."
[[ -n "${PEARMUT_TOKEN_MAIN:-}" ]] || fail \
  "PEARMUT_TOKEN_MAIN must be set to a long random Railway secret."
[[ -n "${PEARMUT_BACKUP_SIGNING_KEY:-}" ]] || fail \
  "PEARMUT_BACKUP_SIGNING_KEY must be set to a retained secret."
[[ "${#PEARMUT_BACKUP_SIGNING_KEY}" -ge 32 ]] || fail \
  "PEARMUT_BACKUP_SIGNING_KEY must be set to a retained secret of at least 32 characters."
[[ -n "${PEARMUT_CAMPAIGN_IDS:-}" ]] || fail \
  "PEARMUT_CAMPAIGN_IDS must explicitly name the campaigns to load."
[[ "${PORT:-}" =~ ^[0-9]+$ ]] || fail \
  "Railway did not provide a numeric PORT for the public proxy."
[[ "$PORT" -ne "$PEARMUT_PORT" ]] || fail \
  "Railway PORT must differ from Pearmut's internal port $PEARMUT_PORT."
[[ -d "$EXPECTED_VOLUME_PATH" ]] || fail \
  "Expected Railway volume directory does not exist: $EXPECTED_VOLUME_PATH"

python - "$EXPECTED_VOLUME_PATH" <<'PY'
import os
import sys

expected = os.path.realpath(sys.argv[1])

def unescape_mount_path(value: str) -> str:
    return (
        value.replace(r"\040", " ")
        .replace(r"\011", "\t")
        .replace(r"\012", "\n")
        .replace(r"\134", "\\")
    )

mounts = set()
with open("/proc/self/mountinfo", encoding="utf-8") as handle:
    for line in handle:
        fields = line.split(" - ", 1)[0].split()
        if len(fields) >= 5:
            mounts.add(os.path.realpath(unescape_mount_path(fields[4])))
if expected not in mounts:
    raise SystemExit(
        f"ERROR: {expected} exists but is not a mounted filesystem; refusing "
        "to place mutable Pearmut state in the disposable container layer."
    )
PY

umask 077
mkdir -p "$PEARMUT_ROOT/data/campaigns" "$PEARMUT_ROOT/data/outputs"
probe="$(mktemp "$EXPECTED_VOLUME_PATH/.pearmut-write-test.XXXXXX")" || fail \
  "Railway volume is not writable: $EXPECTED_VOLUME_PATH"
rm -f "$probe"

python - "$CAMPAIGN_SOURCE_DIR" "$PEARMUT_ROOT" "$PEARMUT_CAMPAIGN_IDS" <<'PY'
import json
import os
import re
import subprocess
import sys
from pathlib import Path

source_dir = Path(sys.argv[1])
root = Path(sys.argv[2])
campaign_ids = [value.strip() for value in sys.argv[3].split(",") if value.strip()]
if not campaign_ids:
    raise SystemExit("ERROR: PEARMUT_CAMPAIGN_IDS did not contain a campaign ID")
if len(campaign_ids) != len(set(campaign_ids)):
    raise SystemExit("ERROR: PEARMUT_CAMPAIGN_IDS contains a duplicate")

safe_id = re.compile(r"^[A-Za-z0-9._-]+$")
progress_path = root / "data" / "progress.json"

for campaign_id in campaign_ids:
    if not safe_id.fullmatch(campaign_id):
        raise SystemExit(f"ERROR: unsafe campaign ID: {campaign_id!r}")
    source = source_dir / f"{campaign_id}.json"
    runtime_campaign = root / "data" / "campaigns" / f"{campaign_id}.json"
    runtime_output = root / "data" / "outputs" / f"{campaign_id}.jsonl"
    if not source.is_file():
        raise SystemExit(
            f"ERROR: campaign {campaign_id!r} is not packaged in this image; "
            "build/deploy again after creating its public campaign JSON"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("campaign_id") != campaign_id:
        raise SystemExit(f"ERROR: campaign ID mismatch in {source}")
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        progress = {}
    except json.JSONDecodeError as error:
        raise SystemExit(f"ERROR: invalid persistent progress file: {error}") from error

    in_progress = campaign_id in progress
    has_campaign = runtime_campaign.is_file()
    if in_progress and has_campaign:
        print(f"Campaign {campaign_id} already loaded; preserving all runtime state.")
        continue
    if in_progress != has_campaign:
        raise SystemExit(
            f"ERROR: inconsistent persistent state for {campaign_id}: "
            f"progress={in_progress}, campaign_file={has_campaign}; refusing overwrite"
        )
    if runtime_output.exists():
        raise SystemExit(
            f"ERROR: orphan annotation output exists for absent campaign {campaign_id}; "
            "refusing overwrite"
        )

    print(f"Loading absent campaign {campaign_id} into the persistent volume.")
    subprocess.run(
        [
            "pearmut",
            "add",
            str(source),
            "--url",
            "http://127.0.0.1",
        ],
        check=True,
        env=os.environ,
        stdout=subprocess.DEVNULL,
    )
PY

python -m uvicorn pearmut.app:app \
  --host 127.0.0.1 --port "$PEARMUT_PORT" --no-access-log &
pearmut_pid=$!

python /app/scripts/pearmut_public_proxy.py \
  --bind 0.0.0.0 --port "$PORT" --allow-public-bind \
  --upstream-host 127.0.0.1 --upstream-port "$PEARMUT_PORT" &
proxy_pid=$!

cleanup() {
  trap - EXIT INT TERM
  kill "$proxy_pid" "$pearmut_pid" 2>/dev/null || true
  wait "$proxy_pid" "$pearmut_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python - "$PORT" "$pearmut_pid" "$proxy_pid" <<'PY'
import os
import sys
import time
import urllib.request

port, pearmut_pid, proxy_pid = map(int, sys.argv[1:])
url = f"http://127.0.0.1:{port}/healthz"
for _ in range(100):
    for pid, name in ((pearmut_pid, "Pearmut"), (proxy_pid, "protective proxy")):
        try:
            os.kill(pid, 0)
        except OSError as error:
            raise SystemExit(f"ERROR: {name} exited during startup") from error
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status == 200:
                print(f"Protected Pearmut is ready on Railway PORT {port}.")
                raise SystemExit(0)
    except Exception:
        time.sleep(0.1)
raise SystemExit("ERROR: protected Pearmut did not become ready")
PY

set +e
wait -n "$pearmut_pid" "$proxy_pid"
status=$?
set -e
echo "ERROR: a hosted Pearmut process exited with status $status" >&2
exit "$status"
