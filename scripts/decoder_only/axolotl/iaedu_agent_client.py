#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

UUID_SUFFIX_RE = re.compile(
    r"\s*[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\s*$"
)
PROCESSING_PREFIX_RE = re.compile(r"^\s*(?:Processing)+", flags=re.IGNORECASE)


def debug_log(message: str) -> None:
    if os.getenv("IAEDU_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[iaedu-debug] {message}", flush=True)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def parse_json_string(name: str, value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be valid JSON. Got: {value}") from exc
    return json.dumps(parsed, ensure_ascii=False)


def rotated_thread_id(base_thread_id: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]", "", base_thread_id).strip("-_")
    while True:
        trimmed = re.sub(r"-[0-9a-fA-F]{8}$", "", base)
        if trimmed == base:
            break
        base = trimmed
    if not base:
        base = "thread"
    if len(base) > 64:
        base = base[:64].rstrip("-_")
    suffix = uuid.uuid4().hex[:8]
    return f"{base}-{suffix}"


def is_retryable_backend_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    retryable_markers = (
        "backend processing error",
        "500 server error",
        "502 server error",
        "503 server error",
        "504 server error",
        "429",
        "too many requests",
        "read timed out",
        "timed out",
        "connection reset",
        "connection aborted",
        "temporary failure",
    )
    return any(marker in msg for marker in retryable_markers)


def is_retryable_backend_response(text: str) -> bool:
    msg = (text or "").lower()
    retryable_markers = (
        "rate limit reached",
        "processingrate limit reached",
        "(429)",
        "too many requests",
        "backend processing error",
        "temporary failure",
    )
    return any(marker in msg for marker in retryable_markers)


def resolve_api_config(
    *,
    env_file: Path,
    endpoint: str | None,
    api_key: str | None,
    channel_id: str | None,
    thread_id: str | None,
    short_thread_id: str | None,
    user_info: str,
    user_id: str | None,
    user_context: str | None,
    request_timeout: int,
) -> dict[str, Any]:
    load_env_file(env_file)

    resolved_endpoint = endpoint or os.getenv("IAEDU_ENDPOINT")
    resolved_api_key = api_key or os.getenv("IAEDU_API_KEY")
    resolved_channel_id = channel_id or os.getenv("IAEDU_CHANNEL_ID")
    resolved_thread_id = thread_id or os.getenv("IAEDU_THREAD_ID")
    resolved_short_thread_id = short_thread_id or os.getenv("IAEDU_SHORT_THREAD_ID")

    missing = []
    if not resolved_endpoint:
        missing.append("IAEDU_ENDPOINT")
    if not resolved_api_key:
        missing.append("IAEDU_API_KEY")
    if not resolved_channel_id:
        missing.append("IAEDU_CHANNEL_ID")
    if not resolved_thread_id:
        missing.append("IAEDU_THREAD_ID")
    if missing:
        raise SystemExit(
            "Missing IAEDU config. Set these variables in env or --env-file: "
            + ", ".join(missing)
        )

    return {
        "endpoint": resolved_endpoint.replace("/agent-chat//api/", "/agent-chat/api/"),
        "api_key": resolved_api_key,
        "channel_id": resolved_channel_id,
        "thread_id": resolved_thread_id,
        "short_thread_id": resolved_short_thread_id,
        "user_info": parse_json_string("--user-info", user_info),
        "user_id": user_id,
        "user_context": parse_json_string("--user-context", user_context) if user_context else None,
        "request_timeout": request_timeout,
    }


def collect_text_fragments(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(collect_text_fragments(item))
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in {
                "content",
                "text",
                "answer",
                "response",
                "message",
                "output",
                "output_text",
                "delta",
            }:
                out.extend(collect_text_fragments(item))
            elif isinstance(item, (dict, list)):
                out.extend(collect_text_fragments(item))
    return out


def collapse_exact_duplicate(text: str) -> str:
    stripped = text.strip()
    if len(stripped) < 2 or len(stripped) % 2:
        return text
    midpoint = len(stripped) // 2
    if stripped[:midpoint] == stripped[midpoint:]:
        return stripped[:midpoint]
    return text


def clean_iaedu_response_text(text: str) -> str:
    """Remove IAEDU stream status/metadata fragments from the model answer."""

    cleaned = text or ""
    cleaned = PROCESSING_PREFIX_RE.sub("", cleaned).strip()
    cleaned = UUID_SUFFIX_RE.sub("", cleaned).strip()
    cleaned = collapse_exact_duplicate(cleaned)
    return cleaned.strip()


def send_agent_message(config: dict[str, Any], message: str, *, thread_id: str) -> str:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency 'requests'. Install it with: pip install requests") from exc

    form_data: dict[str, str] = {
        "channel_id": config["channel_id"],
        "thread_id": thread_id,
        "user_info": config["user_info"],
        "message": message,
    }
    if config.get("user_id"):
        form_data["user_id"] = config["user_id"]
    if config.get("user_context"):
        form_data["user_context"] = config["user_context"]

    request_timeout = float(config["request_timeout"])
    connect_timeout = float(os.getenv("IAEDU_CONNECT_TIMEOUT", min(30.0, request_timeout)))
    read_timeout = float(os.getenv("IAEDU_READ_TIMEOUT", min(30.0, request_timeout)))
    debug_log(
        "request-start"
        f" thread_id={thread_id}"
        f" message_chars={len(message)}"
        f" connect_timeout={connect_timeout}"
        f" read_timeout={read_timeout}"
        f" wall_timeout={request_timeout}"
    )

    response = requests.post(
        config["endpoint"],
        headers={"x-api-key": config["api_key"]},
        data=form_data,
        stream=True,
        timeout=(connect_timeout, read_timeout),
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    debug_log(f"response-open content_type={content_type!r} status={response.status_code}")
    if "text/event-stream" not in content_type:
        debug_log(f"response-non-stream chars={len(response.text)}")
        return clean_iaedu_response_text(response.text)

    chunks: list[str] = []
    started_at = time.monotonic()
    try:
        for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
            elapsed = time.monotonic() - started_at
            if elapsed > request_timeout:
                raise TimeoutError(
                    f"IAEDU stream exceeded wall timeout "
                    f"({elapsed:.1f}s > {config['request_timeout']}s)"
                )
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("event:") or line.startswith("id:") or line.startswith(":"):
                continue

            payload = line[5:].strip() if line.startswith("data:") else line
            if payload == "[DONE]":
                debug_log(f"response-done chunks={len(chunks)} chars={sum(len(c) for c in chunks)}")
                break

            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                chunks.append(payload)
                debug_log(f"response-text-chunk chars={len(payload)} total_chunks={len(chunks)}")
                continue

            extracted = collect_text_fragments(parsed)
            if extracted:
                chunks.extend(extracted)
                debug_log(
                    "response-json-text"
                    f" fragments={len(extracted)}"
                    f" total_chunks={len(chunks)}"
                    f" total_chars={sum(len(c) for c in chunks)}"
                )
            else:
                serialized = json.dumps(parsed, ensure_ascii=False)
                chunks.append(serialized)
                debug_log(f"response-json-raw chars={len(serialized)} total_chunks={len(chunks)}")
    except requests.exceptions.ReadTimeout:
        if chunks:
            debug_log(
                "response-read-timeout-returning-partial"
                f" chunks={len(chunks)}"
                f" chars={sum(len(c) for c in chunks)}"
            )
            return clean_iaedu_response_text("".join(chunks))
        raise

    debug_log(f"response-return chars={sum(len(c) for c in chunks)} chunks={len(chunks)}")
    return clean_iaedu_response_text("".join(chunks))


def request_with_retries(
    config: dict[str, Any],
    message: str,
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[str, str]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        thread_id = rotated_thread_id(str(config["thread_id"]))
        try:
            debug_log(f"attempt={attempt + 1}/{max_retries + 1}")
            raw = send_agent_message(config, message, thread_id=thread_id)
            if is_retryable_backend_response(raw):
                raise RuntimeError(raw)
            return raw, thread_id
        except Exception as exc:
            last_error = exc
            debug_log(f"request-error attempt={attempt + 1} error={exc!r}")
            if attempt >= max_retries or not is_retryable_backend_error(exc):
                break
            sleep_seconds = retry_backoff_seconds * (attempt + 1)
            debug_log(f"retry-sleep seconds={sleep_seconds}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"IAEDU request failed after retries: {last_error}") from last_error
