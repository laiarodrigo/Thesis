#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
RUNTIME_ROOT="${PEARMUT_ROOT:-$REPO_ROOT/cmu_evaluator_project/runtime/external_smoke_01}"
CAMPAIGN_ID="ted2020_pilot_external_smoke_01_cesa"
CAMPAIGN_JSON="$REPO_ROOT/cmu_evaluator_project/data/pearmut/external_smoke_01/campaigns/$CAMPAIGN_ID.json"
PEARMUT_PORT="${PEARMUT_LOCAL_PORT:-8001}"
PROXY_PORT="${PEARMUT_PUBLIC_PROXY_PORT:-8765}"
PYTHON="$REPO_ROOT/thesis/bin/python"
PEARMUT="$REPO_ROOT/thesis/bin/pearmut"

[[ -f "$CAMPAIGN_JSON" ]] || {
  echo "ERROR: missing smoke campaign: $CAMPAIGN_JSON" >&2
  echo "Build it with configs/external_smoke_01.json first." >&2
  exit 2
}

umask 077
mkdir -p "$RUNTIME_ROOT/data/campaigns" "$RUNTIME_ROOT/data/outputs" "$RUNTIME_ROOT/logs"

if [[ ! -f "$RUNTIME_ROOT/data/progress.json" ]] || ! \
  "$PYTHON" - "$RUNTIME_ROOT/data/progress.json" "$CAMPAIGN_ID" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
campaign_id = sys.argv[2]
try:
    progress = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if campaign_id in progress else 1)
PY
then
  PEARMUT_ROOT="$RUNTIME_ROOT" "$PEARMUT" add "$CAMPAIGN_JSON" \
    --url "http://127.0.0.1:$PROXY_PORT"
fi

PEARMUT_ROOT="$RUNTIME_ROOT" "$PYTHON" -m uvicorn pearmut.app:app \
  --host 127.0.0.1 --port "$PEARMUT_PORT" \
  >"$RUNTIME_ROOT/logs/pearmut.log" 2>&1 &
pearmut_pid=$!

"$PYTHON" "$REPO_ROOT/cmu_evaluator_project/scripts/pearmut_public_proxy.py" \
  --bind 127.0.0.1 --port "$PROXY_PORT" \
  --upstream-host 127.0.0.1 --upstream-port "$PEARMUT_PORT" \
  >"$RUNTIME_ROOT/logs/public_proxy.log" 2>&1 &
proxy_pid=$!

cleanup() {
  kill "$proxy_pid" "$pearmut_pid" 2>/dev/null || true
  wait "$proxy_pid" "$pearmut_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 50); do
  if curl --fail --silent "http://127.0.0.1:$PROXY_PORT/annotate" >/dev/null; then
    break
  fi
  if ! kill -0 "$pearmut_pid" 2>/dev/null || ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "ERROR: Pearmut or its protective proxy exited during startup." >&2
    echo "Inspect $RUNTIME_ROOT/logs/" >&2
    exit 1
  fi
  sleep 0.1
done

curl --fail --silent "http://127.0.0.1:$PROXY_PORT/annotate" >/dev/null || {
  echo "ERROR: protected Pearmut did not become ready." >&2
  exit 1
}

echo "Protected Pearmut is ready at http://127.0.0.1:$PROXY_PORT"
echo "Persistent root: $RUNTIME_ROOT"
echo "Start the HTTPS tunnel in a second terminal. Press Ctrl+C here to stop Pearmut."
wait -n "$pearmut_pid" "$proxy_pid"
