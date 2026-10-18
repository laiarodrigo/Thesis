#!/usr/bin/env python3
"""Local allow-listing reverse proxy for public Pearmut annotation links."""

from __future__ import annotations

import argparse
import http.client
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pearmut_user_backup import (
    UserBackupAuthorizationError,
    UserBackupIncompleteError,
    UserBackupNotFoundError,
    UserBackupStateError,
    build_user_backup,
)


ANNOTATION_GET_ROUTES = {
    "/",
    "/annotate",
    "/annotate.html",
    "/annotate.bundle.js",
    "/annotator-backup.js",
    "/style.css",
    "/favicon.svg",
}
ANNOTATION_POST_ROUTES = {
    "/get-next-item",
    "/get-i-item",
    "/log-response",
    "/annotator-backup",
}
HEALTH_ROUTE = "/healthz"
BLOCKED_RESEARCH_ROUTES = {
    "/add-campaign",
    "/dashboard",
    "/dashboard-data",
    "/dashboard-results",
    "/download-annotations",
    "/download-campaigns",
    "/download-progress",
    "/export-results",
    "/purge-campaign",
    "/reset-task",
}

BACKUP_SCRIPT_ROUTE = "/annotator-backup.js"
BACKUP_EXPORT_ROUTE = "/annotator-backup"
BACKUP_SCRIPT_TAG = b'<script defer src="/annotator-backup.js"></script>'
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def route_allowed(method: str, path: str) -> bool:
    """Return whether a public request is required by the annotation UI."""
    clean_path = urlsplit(path).path
    if method in {"GET", "HEAD"}:
        return (
            clean_path in ANNOTATION_GET_ROUTES
            or clean_path == HEALTH_ROUTE
            or clean_path.startswith("/assets/")
        )
    if method == "POST":
        return clean_path in ANNOTATION_POST_ROUTES
    return False


def inject_backup_script(response_body: bytes) -> bytes:
    """Add the completion-only backup UI without modifying Pearmut itself."""
    if BACKUP_SCRIPT_TAG in response_body:
        return response_body
    marker = b"</body>"
    if marker not in response_body:
        return response_body
    return response_body.replace(marker, BACKUP_SCRIPT_TAG + marker, 1)


def upstream_request_path(path: str) -> str:
    """Never forward the browser's scoped backup capability to Pearmut logs."""
    parsed = urlsplit(path)
    if parsed.path not in {"/annotate", "/annotate.html"}:
        return path
    query = urlencode(
        [(key, value) for key, value in parse_qsl(parsed.query) if key != "backup_token"]
    )
    return urlunsplit(("", "", parsed.path, query, ""))


def make_handler(
    upstream_host: str,
    upstream_port: int,
    *,
    pearmut_root: str | Path | None = None,
    backup_signing_key: str | None = None,
    backup_script_path: str | Path | None = None,
):
    runtime_root = Path(pearmut_root or os.environ.get("PEARMUT_ROOT", "."))
    signing_key = backup_signing_key or os.environ.get(
        "PEARMUT_BACKUP_SIGNING_KEY", ""
    )
    script_path = Path(backup_script_path or Path(__file__).with_name("annotator_backup.js"))

    class PearmutPublicHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "PearmutPublicProxy/1.0"

        def do_GET(self) -> None:  # noqa: N802
            self._forward_if_allowed()

        def do_HEAD(self) -> None:  # noqa: N802
            self._forward_if_allowed()

        def do_POST(self) -> None:  # noqa: N802
            self._forward_if_allowed()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._deny()

        def _deny(self) -> None:
            payload = b"Not available on the public annotation service.\n"
            self.send_response(403)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _forward_if_allowed(self) -> None:
            if not route_allowed(self.command, self.path):
                self._deny()
                return
            clean_path = urlsplit(self.path).path
            if clean_path == HEALTH_ROUTE:
                self._health()
                return
            if clean_path == BACKUP_SCRIPT_ROUTE:
                self._backup_script()
                return
            if clean_path == BACKUP_EXPORT_ROUTE:
                self._user_backup()
                return
            raw_length = self.headers.get("Content-Length", "0")
            try:
                content_length = int(raw_length)
            except ValueError:
                self.send_error(400, "Invalid Content-Length")
                return
            if content_length < 0 or content_length > 10 * 1024 * 1024:
                self.send_error(413, "Request body too large")
                return
            body = self.rfile.read(content_length) if content_length else None
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
                and key.lower() not in {"host", "content-length", "accept-encoding"}
            }
            headers["Host"] = f"{upstream_host}:{upstream_port}"
            headers["X-Forwarded-Proto"] = "https"
            headers["X-Forwarded-Host"] = self.headers.get("Host", "")
            if body is not None:
                headers["Content-Length"] = str(len(body))
            connection = http.client.HTTPConnection(
                upstream_host, upstream_port, timeout=30
            )
            try:
                connection.request(
                    self.command,
                    upstream_request_path(self.path),
                    body=body,
                    headers=headers,
                )
                response = connection.getresponse()
                response_body = response.read()
                response_headers = response.getheaders()
                response_status = response.status
            except (OSError, http.client.HTTPException):
                self.send_error(502, "Pearmut upstream unavailable")
                return
            finally:
                connection.close()
            if (
                self.command == "GET"
                and clean_path in {"/annotate", "/annotate.html"}
                and response_status == 200
            ):
                response_body = inject_backup_script(response_body)
            self.send_response(response_status)
            for key, value in response_headers:
                if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() not in {
                    "content-length",
                    "content-encoding",
                }:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response_body)

        def _backup_script(self) -> None:
            try:
                payload = script_path.read_bytes()
            except OSError:
                self.send_error(500, "Backup interface is unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "public, max-age=300")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _backup_error(self, status: int, message: str) -> None:
            payload = (message + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _user_backup(self) -> None:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                content_length = int(raw_length)
            except ValueError:
                self._backup_error(400, "Invalid request length")
                return
            if content_length <= 0 or content_length > 64 * 1024:
                self._backup_error(413, "Invalid backup request size")
                return
            try:
                request = json.loads(self.rfile.read(content_length))
                campaign_id = request["campaign_id"]
                user_id = request["user_id"]
                supplied_token = request["backup_token"]
            except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
                self._backup_error(400, "Invalid backup request")
                return
            if not all(
                isinstance(value, str)
                for value in (campaign_id, user_id, supplied_token)
            ):
                self._backup_error(400, "Invalid backup request")
                return
            try:
                backup = build_user_backup(
                    runtime_root,
                    campaign_id,
                    user_id,
                    supplied_token,
                    signing_key,
                )
            except UserBackupAuthorizationError:
                self._backup_error(403, "Backup capability is not valid for this user")
                return
            except UserBackupNotFoundError:
                self._backup_error(404, "Campaign assignment was not found")
                return
            except UserBackupIncompleteError as error:
                self._backup_error(409, str(error))
                return
            except UserBackupStateError:
                self._backup_error(500, "Persistent backup state is unavailable")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header(
                "Content-Disposition", f'attachment; filename="{backup.filename}"'
            )
            self.send_header("Content-Length", str(len(backup.content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(backup.content)

        def _health(self) -> None:
            """Report ready only when the localhost-only Pearmut process responds."""
            connection = http.client.HTTPConnection(
                upstream_host, upstream_port, timeout=5
            )
            try:
                connection.request("GET", "/annotate")
                response = connection.getresponse()
                response.read()
                healthy = 200 <= response.status < 300
            except (OSError, http.client.HTTPException):
                healthy = False
            finally:
                connection.close()
            payload = b"ok\n" if healthy else b"Pearmut upstream unavailable\n"
            self.send_response(200 if healthy else 503)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            # The public launcher is a static page on another origin. This
            # endpoint contains no campaign/user data, so a wildcard is safe
            # here and deliberately nowhere else on the proxy.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def log_message(self, format_string: str, *args: object) -> None:
            # Deliberately omit query strings; only the route path is logged.
            clean_path = urlsplit(self.path).path
            print(f'{self.client_address[0]} "{self.command} {clean_path}"', flush=True)

    return PearmutPublicHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=8001)
    parser.add_argument(
        "--allow-public-bind",
        action="store_true",
        help="Permit 0.0.0.0/:: for a hosted edge; Pearmut remains localhost-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loopback_hosts = {"127.0.0.1", "::1", "localhost"}
    public_hosts = {"0.0.0.0", "::"}
    if args.bind not in loopback_hosts and not (
        args.allow_public_bind and args.bind in public_hosts
    ):
        raise SystemExit(
            "ERROR: non-loopback proxy binding requires --allow-public-bind "
            "and must use 0.0.0.0 or ::"
        )
    if args.upstream_host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit("ERROR: Pearmut upstream must be localhost-only")
    backup_signing_key = os.environ.get("PEARMUT_BACKUP_SIGNING_KEY", "")
    if len(backup_signing_key) < 32:
        raise SystemExit(
            "ERROR: PEARMUT_BACKUP_SIGNING_KEY must contain at least 32 characters"
        )
    server = ThreadingHTTPServer(
        (args.bind, args.port),
        make_handler(
            args.upstream_host,
            args.upstream_port,
            backup_signing_key=backup_signing_key,
        ),
    )
    print(
        f"Protected Pearmut proxy listening on http://{args.bind}:{args.port}; "
        f"upstream http://{args.upstream_host}:{args.upstream_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
