#!/usr/bin/env bash
set -euo pipefail

: "${SMOKE_ALLOWED_EMAIL:?Set SMOKE_ALLOWED_EMAIL to the friend's email address}"
PROXY_PORT="${PEARMUT_PUBLIC_PROXY_PORT:-8765}"
ORIGIN="http://127.0.0.1:$PROXY_PORT"

if command -v cloudflared >/dev/null 2>&1; then
  exec cloudflared tunnel --url "$ORIGIN" --allowed-mail "$SMOKE_ALLOWED_EMAIL"
fi

if command -v docker >/dev/null 2>&1; then
  exec docker run --rm --network host cloudflare/cloudflared:latest \
    tunnel --no-autoupdate --url "$ORIGIN" --allowed-mail "$SMOKE_ALLOWED_EMAIL"
fi

echo "ERROR: neither cloudflared nor Docker is available." >&2
echo "Install cloudflared, then rerun this command." >&2
exit 2
