#!/usr/bin/env python3
"""
Generate pt-PT / pt-BR sentence pairs via IAEDU agent API and save to CSV.

Default behavior:
- 200 generated examples
- 80% long sentences, 10% short sentences, 10% equal sentences
- long sentences: 31-60 words
- style guided by exemplos.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable


PAIR_REGEX = re.compile(
    r'"pt_PT"\s*:\s*"(?P<ptpt>[^"]+)"\s*,\s*"pt_BR"\s*:\s*"(?P<ptbr>[^"]+)"'
)
FLEX_PAIR_REGEX_A = re.compile(
    r'"(?:pt_PT|pt-PT|ptPT|pt_pt)"\s*:\s*"([^"]+)"\s*,\s*"'
    r'(?:pt_BR|pt-BR|ptBR|pt_br)"\s*:\s*"([^"]+)"',
    flags=re.IGNORECASE,
)
FLEX_PAIR_REGEX_B = re.compile(
    r'"(?:pt_BR|pt-BR|ptBR|pt_br)"\s*:\s*"([^"]+)"\s*,\s*"'
    r'(?:pt_PT|pt-PT|ptPT|pt_pt)"\s*:\s*"([^"]+)"',
    flags=re.IGNORECASE,
)


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def mask_secret(value: str, keep_start: int = 4, keep_end: int = 3) -> str:
    if len(value) <= keep_start + keep_end:
        return "*" * len(value)
    return value[:keep_start] + ("*" * (len(value) - keep_start - keep_end)) + value[-keep_end:]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate pt-PT/pt-BR sentence pairs via IAEDU agent API and export CSV. "
            "Defaults to 200 examples with 80% long, 10% short, 10% equal."
        )
    )
    parser.add_argument(
        "--examples-file",
        type=Path,
        default=repo_root / "exemplos.txt",
        help="Reference examples file used to mimic style.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "data" / "pt_variant_prompts_500.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=repo_root / ".env",
        help="Optional .env-style file with IAEDU credentials.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="IAEDU API endpoint. If omitted, uses IAEDU_ENDPOINT from env.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="IAEDU API key. If omitted, uses IAEDU_API_KEY from env.",
    )
    parser.add_argument(
        "--channel-id",
        default=None,
        help="IAEDU channel id. If omitted, uses IAEDU_CHANNEL_ID from env.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="IAEDU thread id. If omitted, uses IAEDU_THREAD_ID from env.",
    )
    parser.add_argument(
        "--short-thread-id",
        default=None,
        help="Optional separate IAEDU thread id for short examples (or IAEDU_SHORT_THREAD_ID).",
    )
    parser.add_argument(
        "--user-info",
        default="{}",
        help="Mandatory IAEDU user_info field as JSON string.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Optional IAEDU user_id.",
    )
    parser.add_argument(
        "--user-context",
        default=None,
        help="Optional IAEDU user_context JSON string.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="HTTP timeout per request in seconds.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.0,
        help="Pause between requests to avoid rate spikes.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=200,
        help="Total number of examples to generate.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append generated rows to existing output CSV instead of overwriting.",
    )
    parser.add_argument(
        "--long-ratio",
        type=float,
        default=0.8,
        help="Share of long sentences in total generated rows.",
    )
    parser.add_argument(
        "--equal-ratio",
        type=float,
        default=0.1,
        help="Share of equal sentences where pt_PT == pt_BR.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="How many long examples to request per API call.",
    )
    parser.add_argument(
        "--short-batch-size",
        type=int,
        default=12,
        help="How many short examples to request per API call.",
    )
    parser.add_argument(
        "--equal-batch-size",
        type=int,
        default=8,
        help="How many equal examples to request per API call.",
    )
    parser.add_argument(
        "--long-min-words",
        type=int,
        default=31,
        help="Minimum words for long sentences.",
    )
    parser.add_argument(
        "--long-max-words",
        type=int,
        default=60,
        help="Maximum words for long sentences.",
    )
    parser.add_argument(
        "--short-min-words",
        type=int,
        default=6,
        help="Minimum words for short sentences.",
    )
    parser.add_argument(
        "--short-max-words",
        type=int,
        default=14,
        help="Maximum words for short sentences.",
    )
    parser.add_argument(
        "--reference-pairs",
        type=int,
        default=12,
        help="How many pairs to pull from exemplos.txt as style anchors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for final output order.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=140,
        help="Maximum API attempts for each group (long and short).",
    )
    parser.add_argument(
        "--rotate-thread-on-backend-error",
        action="store_true",
        default=False,
        help="Rotate to a fresh thread id after repeated backend processing errors.",
    )
    return parser.parse_args()


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
    if not base:
        base = "thread"
    suffix = uuid.uuid4().hex[:8]
    return f"{base}-{suffix}"


def resolve_api_config(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)

    endpoint = args.endpoint or os.getenv("IAEDU_ENDPOINT")
    api_key = args.api_key or os.getenv("IAEDU_API_KEY")
    channel_id = args.channel_id or os.getenv("IAEDU_CHANNEL_ID")
    thread_id = args.thread_id or os.getenv("IAEDU_THREAD_ID")
    short_thread_id = args.short_thread_id or os.getenv("IAEDU_SHORT_THREAD_ID")

    missing = []
    if not endpoint:
        missing.append("IAEDU_ENDPOINT")
    if not api_key:
        missing.append("IAEDU_API_KEY")
    if not channel_id:
        missing.append("IAEDU_CHANNEL_ID")
    if not thread_id:
        missing.append("IAEDU_THREAD_ID")
    if missing:
        raise SystemExit(
            "Missing IAEDU config. Set these variables in env or --env-file: "
            + ", ".join(missing)
        )
    endpoint = endpoint.replace("/agent-chat//api/", "/agent-chat/api/")

    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "channel_id": channel_id,
        "thread_id": thread_id,
        "short_thread_id": short_thread_id,
        "user_info": parse_json_string("--user-info", args.user_info),
        "user_id": args.user_id,
        "user_context": parse_json_string("--user-context", args.user_context)
        if args.user_context
        else None,
        "request_timeout": args.request_timeout,
    }


def read_reference_pairs(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    pairs: list[dict[str, str]] = []
    for match in PAIR_REGEX.finditer(text):
        pt_pt = match.group("ptpt").strip()
        pt_br = match.group("ptbr").strip()
        if pt_pt and pt_br:
            pairs.append({"pt_PT": pt_pt, "pt_BR": pt_br})
        if len(pairs) >= limit:
            break
    return pairs


def reference_block(pairs: list[dict[str, str]]) -> str:
    if not pairs:
        return "No explicit reference examples provided."
    lines = []
    for i, pair in enumerate(pairs, start=1):
        lines.append(f'{i}. pt_PT: "{pair["pt_PT"]}" | pt_BR: "{pair["pt_BR"]}"')
    return "\n".join(lines)


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def extract_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output did not contain a JSON object.")
    snippet = raw[start : end + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")
    return parsed


def build_prompt(
    n_items: int,
    *,
    mode: str,
    refs: str,
    long_min: int,
    long_max: int,
    short_min: int,
    short_max: int,
    nonce: str,
) -> str:
    if mode == "long":
        length_rule = (
            f"Generate long examples only. Each pt_PT sentence should be at least {long_min} words, "
            "single sentence, natural prose (not subtitle style), and can include a subordinate clause."
        )
        equality_rule = "pt_PT and pt_BR must not be identical."
    elif mode == "short":
        length_rule = (
            f"Generate short examples only. Each pt_PT sentence should be {short_min}-{short_max} words, "
            "single sentence, still natural prose."
        )
        equality_rule = "pt_PT and pt_BR must not be identical."
    elif mode == "equal":
        length_rule = (
            f"Generate equal examples only. Each sentence can be short or long, "
            f"between {short_min} and {long_max} words."
        )
        equality_rule = "pt_PT and pt_BR must be exactly identical strings."
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return f"""
Generate exactly {n_items} NEW bilingual examples with this JSON schema:
{{"exemplos":[{{"pt_PT":"...", "pt_BR":"..."}}]}}

Hard requirements:
1) Keep only one sentence in pt_PT and one sentence in pt_BR for each example.
2) Keep meaning equivalent across variants, but adapt naturally to each variant.
3) Make examples diverse in topic and wording.
4) Avoid repeating prior samples; produce novel wording and scenarios.
5) Keep both variants natural and grammatically correct.
6) Return valid JSON only. No markdown, no comments, no extra keys.
7) {length_rule}
8) {equality_rule}

Batch nonce (for diversity): {nonce}

Reference style examples:
{refs}
""".strip()


def parse_examples(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_examples = payload.get("exemplos")
    if not isinstance(raw_examples, list):
        raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list):
        return []

    out: list[dict[str, str]] = []
    for item in raw_examples:
        if not isinstance(item, dict):
            continue
        pt_pt = str(
            item.get("pt_PT")
            or item.get("pt_pt")
            or item.get("pt-PT")
            or item.get("ptPT")
            or ""
        ).strip()
        pt_br = str(
            item.get("pt_BR")
            or item.get("pt_br")
            or item.get("pt-BR")
            or item.get("ptBR")
            or ""
        ).strip()
        if pt_pt and pt_br:
            out.append({"pt_PT": pt_pt, "pt_BR": pt_br})
    return out


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
            key_lower = key.lower()
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


def find_object_with_examples(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        raw_examples = value.get("exemplos")
        if not isinstance(raw_examples, list):
            raw_examples = value.get("examples")
        if isinstance(raw_examples, list):
            return value
        for item in value.values():
            found = find_object_with_examples(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_object_with_examples(item)
            if found is not None:
                return found
    return None


def extract_examples_from_raw_response(raw: str) -> list[dict[str, str]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        direct = parse_examples(extract_json_object(raw))
        if direct:
            return direct
    except Exception:
        pass

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        found = find_object_with_examples(parsed)
        if found is not None:
            examples = parse_examples(found)
            if examples:
                return examples

        for fragment in collect_text_fragments(parsed):
            try:
                examples = parse_examples(extract_json_object(fragment))
                if examples:
                    return examples
            except Exception:
                continue

    regex_rows: list[dict[str, str]] = []
    for match in FLEX_PAIR_REGEX_A.finditer(raw):
        regex_rows.append({"pt_PT": match.group(1).strip(), "pt_BR": match.group(2).strip()})
    for match in FLEX_PAIR_REGEX_B.finditer(raw):
        regex_rows.append({"pt_PT": match.group(2).strip(), "pt_BR": match.group(1).strip()})
    if regex_rows:
        dedup = {}
        for row in regex_rows:
            key = f"{row['pt_PT']} || {row['pt_BR']}"
            dedup[key] = row
        return list(dedup.values())

    raise ValueError("Could not parse 'exemplos' from agent response.")


def send_agent_message(config: dict[str, Any], message: str) -> str:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'requests'. Install it with: pip install requests"
        ) from exc

    form_data: dict[str, str] = {
        "channel_id": config["channel_id"],
        "thread_id": config["thread_id"],
        "user_info": config["user_info"],
        "message": message,
    }
    if config.get("user_id"):
        form_data["user_id"] = config["user_id"]
    if config.get("user_context"):
        form_data["user_context"] = config["user_context"]

    log(
        "POST IAEDU request "
        f"(endpoint={config['endpoint']}, channel_id={config['channel_id']}, thread_id={config['thread_id']}, "
        f"message_chars={len(message)})"
    )
    response = requests.post(
        config["endpoint"],
        headers={"x-api-key": config["api_key"]},
        data=form_data,
        stream=True,
        timeout=config["request_timeout"],
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    log(f"IAEDU response status={response.status_code}, content_type={content_type or 'unknown'}")
    if "text/event-stream" not in content_type:
        return response.text

    chunks: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event:") or line.startswith("id:") or line.startswith(":"):
            continue

        payload = line[5:].strip() if line.startswith("data:") else line
        if payload == "[DONE]":
            break

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            chunks.append(payload)
            continue

        extracted = collect_text_fragments(parsed)
        if extracted:
            chunks.extend(extracted)
        else:
            chunks.append(json.dumps(parsed, ensure_ascii=False))

    return "".join(chunks)


def request_batch(
    config: dict[str, Any],
    *,
    n_items: int,
    mode: str,
    refs: str,
    long_min: int,
    long_max: int,
    short_min: int,
    short_max: int,
) -> list[dict[str, str]]:
    nonce = uuid.uuid4().hex[:12]
    prompt = build_prompt(
        n_items=n_items,
        mode=mode,
        refs=refs,
        long_min=long_min,
        long_max=long_max,
        short_min=short_min,
        short_max=short_max,
        nonce=nonce,
    )
    raw = send_agent_message(config, prompt)
    if "ProcessingUnexpected processing error" in raw:
        debug_path = Path("data/iaedu_last_response_debug.txt")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(raw, encoding="utf-8")
        raise RuntimeError("IAEDU backend processing error (non-JSON stream payload).")
    try:
        examples = extract_examples_from_raw_response(raw)
        if examples:
            return examples
    except Exception:
        pass

    debug_path = Path("data/iaedu_last_response_debug.txt")
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(raw, encoding="utf-8")
    log(
        "Could not parse batch response. "
        f"Saved raw payload to {debug_path.resolve()} (chars={len(raw)})."
    )
    raise ValueError("Could not parse 'exemplos' from agent response.")


def normalize_pair(pt_pt: str, pt_br: str) -> str:
    return f"{pt_pt.lower().strip()} || {pt_br.lower().strip()}"


def read_existing_rows(path: Path) -> tuple[set[str], int]:
    if not path.exists():
        return set(), 0

    seen_pairs: set[str] = set()
    max_id = 0

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pt_pt = str(row.get("pt_PT", "")).strip()
            pt_br = str(row.get("pt_BR", "")).strip()
            pt_pt_words = count_words(pt_pt)
            pt_br_words = count_words(pt_br)

            if pt_pt and pt_br:
                seen_pairs.add(normalize_pair(pt_pt, pt_br))

            try:
                row_id = int(str(row.get("id", "")).strip())
                max_id = max(max_id, row_id)
            except Exception:
                pass

    return seen_pairs, max_id


class CsvSink:
    def __init__(self, output_csv: Path, *, append: bool, start_id: int) -> None:
        self.output_csv = output_csv
        self.next_id = start_id + 1
        self.written = 0
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        default_fieldnames = ["id", "pt_PT", "pt_BR", "pt_PT_words", "pt_BR_words"]
        mode = "w"
        fieldnames = default_fieldnames
        if append and self.output_csv.exists() and self.output_csv.stat().st_size > 0:
            with self.output_csv.open("r", encoding="utf-8", newline="") as rf:
                reader = csv.reader(rf)
                existing_header = next(reader, None)
            if existing_header:
                fieldnames = existing_header
            mode = "a"

        required = {"id", "pt_PT", "pt_BR", "pt_PT_words", "pt_BR_words"}
        missing_required = [k for k in required if k not in fieldnames]
        if missing_required:
            raise SystemExit(
                f"Existing CSV header is missing required columns: {', '.join(missing_required)}"
            )

        self.fieldnames = fieldnames
        self.has_length_col = "length" in self.fieldnames
        self.fh = self.output_csv.open(mode, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=self.fieldnames)
        if mode == "w":
            self.writer.writeheader()
            self.fh.flush()

        log(
            f"CSV sink ready: path={self.output_csv.resolve()}, mode={mode}, "
            f"start_id={self.next_id}, has_length_col={self.has_length_col}"
        )

    def write(self, row: dict[str, str]) -> None:
        pt_pt = row["pt_PT"]
        pt_br = row["pt_BR"]
        payload = {
            "id": str(self.next_id),
            "pt_PT": pt_pt,
            "pt_BR": pt_br,
            "pt_PT_words": str(count_words(pt_pt)),
            "pt_BR_words": str(count_words(pt_br)),
        }
        if self.has_length_col:
            payload["length"] = ""
        out = {k: payload.get(k, "") for k in self.fieldnames}
        self.writer.writerow(out)
        self.fh.flush()
        self.next_id += 1
        self.written += 1

    def close(self) -> None:
        self.fh.close()


def collect_examples(
    config: dict[str, Any],
    *,
    total: int,
    long_ratio: float,
    equal_ratio: float,
    batch_size: int,
    short_batch_size: int,
    equal_batch_size: int,
    refs: str,
    long_min: int,
    long_max: int,
    short_min: int,
    short_max: int,
    max_attempts: int,
    pause_seconds: float,
    rotate_thread_on_backend_error: bool,
    preexisting_seen: set[str] | None = None,
    accept_callback: Callable[[dict[str, str]], None] | None = None,
) -> list[dict[str, str]]:
    long_target = round(total * long_ratio)
    equal_target = round(total * equal_ratio)
    short_target = total - long_target - equal_target

    log(
        "Generation targets: "
        f"total={total}, long_target={long_target}, short_target={short_target}, equal_target={equal_target}, "
        f"long_batch_size={batch_size}, short_batch_size={short_batch_size}, equal_batch_size={equal_batch_size}"
    )

    seen: set[str] = set(preexisting_seen or set())
    long_rows: list[dict[str, str]] = []
    short_rows: list[dict[str, str]] = []
    equal_rows: list[dict[str, str]] = []
    long_config = dict(config)
    if seen:
        log(f"Loaded {len(seen)} existing pairs; dedup will exclude them.")

    attempts = 0
    long_backend_errors = 0
    long_zero_accept_streak = 0
    while len(long_rows) < long_target and attempts < max_attempts:
        attempts += 1
        needed = min(batch_size, long_target - len(long_rows))
        log(f"[LONG] attempt={attempts}/{max_attempts}, need={needed}, collected={len(long_rows)}/{long_target}")
        try:
            batch = request_batch(
                config=long_config,
                n_items=needed,
                mode="long",
                refs=refs,
                long_min=long_min,
                long_max=long_max,
                short_min=short_min,
                short_max=short_max,
            )
            log(f"[LONG] parsed batch_size={len(batch)}")
        except Exception as exc:
            log(f"[LONG] request failed: {exc}")
            if "backend processing error" in str(exc).lower():
                long_backend_errors += 1
                if rotate_thread_on_backend_error and long_backend_errors >= 2:
                    old_tid = long_config["thread_id"]
                    long_config["thread_id"] = rotated_thread_id(old_tid)
                    long_backend_errors = 0
                    log(f"[LONG] rotated thread_id after backend errors: {old_tid} -> {long_config['thread_id']}")
                backoff = min(12.0, 2.0 + long_backend_errors * 2.0)
                log(f"[LONG] backend error backoff: {backoff:.1f}s")
                time.sleep(backoff)
            else:
                time.sleep(1.0)
            continue
        long_backend_errors = 0

        accepted_before = len(long_rows)
        skipped_dup = 0
        skipped_equal = 0
        skipped_len = 0
        for item in batch:
            pt_pt = item["pt_PT"].strip()
            pt_br = item["pt_BR"].strip()
            pt_pt_wc = count_words(pt_pt)
            key = normalize_pair(pt_pt, pt_br)
            if key in seen:
                skipped_dup += 1
                continue
            if pt_pt == pt_br:
                skipped_equal += 1
                continue
            if pt_pt_wc < long_min:
                skipped_len += 1
                continue
            seen.add(key)
            accepted = {"kind": "long", "pt_PT": pt_pt, "pt_BR": pt_br}
            long_rows.append(accepted)
            if accept_callback is not None:
                accept_callback(accepted)
            if len(long_rows) >= long_target:
                break
        accepted_now = len(long_rows) - accepted_before
        if accepted_now == 0:
            long_zero_accept_streak += 1
        else:
            long_zero_accept_streak = 0
        log(
            f"[LONG] accepted={accepted_now}, total={len(long_rows)}/{long_target}, "
            f"skipped_dup={skipped_dup}, skipped_equal={skipped_equal}, skipped_too_short={skipped_len}"
        )
        if rotate_thread_on_backend_error and long_zero_accept_streak >= 5:
            old_tid = long_config["thread_id"]
            long_config["thread_id"] = rotated_thread_id(old_tid)
            long_zero_accept_streak = 0
            log(
                f"[LONG] rotated thread_id after repeated zero-accept batches: "
                f"{old_tid} -> {long_config['thread_id']}"
            )

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    short_config = config
    if config.get("short_thread_id"):
        short_config = dict(config)
        short_config["thread_id"] = config["short_thread_id"]
        log(f"Using separate short thread_id={short_config['thread_id']}")
    else:
        short_config = dict(config)

    attempts = 0
    short_backend_errors = 0
    while len(short_rows) < short_target and attempts < max_attempts:
        attempts += 1
        needed = min(short_batch_size, short_target - len(short_rows))
        log(
            f"[SHORT] attempt={attempts}/{max_attempts}, need={needed}, collected={len(short_rows)}/{short_target}"
        )
        try:
            batch = request_batch(
                config=short_config,
                n_items=needed,
                mode="short",
                refs=refs,
                long_min=long_min,
                long_max=long_max,
                short_min=short_min,
                short_max=short_max,
            )
            log(f"[SHORT] parsed batch_size={len(batch)}")
        except Exception as exc:
            log(f"[SHORT] request failed: {exc}")
            if "backend processing error" in str(exc).lower():
                short_backend_errors += 1
                if rotate_thread_on_backend_error and short_backend_errors >= 2:
                    old_tid = short_config["thread_id"]
                    short_config["thread_id"] = rotated_thread_id(old_tid)
                    short_backend_errors = 0
                    log(
                        f"[SHORT] rotated thread_id after backend errors: "
                        f"{old_tid} -> {short_config['thread_id']}"
                    )
                if short_backend_errors >= 3 and not config.get("short_thread_id"):
                    log(
                        "[SHORT] repeated backend errors. Set IAEDU_SHORT_THREAD_ID "
                        "or --short-thread-id to use a fresh thread for short examples."
                    )
                backoff = min(12.0, 2.0 + short_backend_errors * 2.0)
                log(f"[SHORT] backend error backoff: {backoff:.1f}s")
                time.sleep(backoff)
            else:
                time.sleep(1.0)
            continue
        short_backend_errors = 0

        accepted_before = len(short_rows)
        skipped_dup = 0
        skipped_equal = 0
        skipped_len = 0
        for item in batch:
            pt_pt = item["pt_PT"].strip()
            pt_br = item["pt_BR"].strip()
            pt_pt_wc = count_words(pt_pt)
            key = normalize_pair(pt_pt, pt_br)
            if key in seen:
                skipped_dup += 1
                continue
            if pt_pt == pt_br:
                skipped_equal += 1
                continue
            if not (short_min <= pt_pt_wc <= short_max):
                skipped_len += 1
                continue
            seen.add(key)
            accepted = {"kind": "short", "pt_PT": pt_pt, "pt_BR": pt_br}
            short_rows.append(accepted)
            if accept_callback is not None:
                accept_callback(accepted)
            if len(short_rows) >= short_target:
                break
        log(
            f"[SHORT] accepted={len(short_rows) - accepted_before}, total={len(short_rows)}/{short_target}, "
            f"skipped_dup={skipped_dup}, skipped_equal={skipped_equal}, skipped_len={skipped_len}"
        )

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    attempts = 0
    equal_backend_errors = 0
    while len(equal_rows) < equal_target and attempts < max_attempts:
        attempts += 1
        needed = min(equal_batch_size, equal_target - len(equal_rows))
        log(
            f"[EQUAL] attempt={attempts}/{max_attempts}, need={needed}, collected={len(equal_rows)}/{equal_target}"
        )
        try:
            batch = request_batch(
                config=short_config,
                n_items=needed,
                mode="equal",
                refs=refs,
                long_min=long_min,
                long_max=long_max,
                short_min=short_min,
                short_max=short_max,
            )
            log(f"[EQUAL] parsed batch_size={len(batch)}")
        except Exception as exc:
            log(f"[EQUAL] request failed: {exc}")
            if "backend processing error" in str(exc).lower():
                equal_backend_errors += 1
                if rotate_thread_on_backend_error and equal_backend_errors >= 2:
                    old_tid = short_config["thread_id"]
                    short_config["thread_id"] = rotated_thread_id(old_tid)
                    equal_backend_errors = 0
                    log(f"[EQUAL] rotated thread_id after backend errors: {old_tid} -> {short_config['thread_id']}")
                backoff = min(12.0, 2.0 + equal_backend_errors * 2.0)
                log(f"[EQUAL] backend error backoff: {backoff:.1f}s")
                time.sleep(backoff)
            else:
                time.sleep(1.0)
            continue
        equal_backend_errors = 0

        accepted_before = len(equal_rows)
        skipped_dup = 0
        skipped_notequal = 0
        skipped_len = 0
        for item in batch:
            pt_pt = item["pt_PT"].strip()
            pt_br = item["pt_BR"].strip()
            pt_pt_wc = count_words(pt_pt)
            key = normalize_pair(pt_pt, pt_br)
            if key in seen:
                skipped_dup += 1
                continue
            if pt_pt != pt_br:
                skipped_notequal += 1
                continue
            if not (short_min <= pt_pt_wc <= long_max):
                skipped_len += 1
                continue
            seen.add(key)
            accepted = {"kind": "equal", "pt_PT": pt_pt, "pt_BR": pt_br}
            equal_rows.append(accepted)
            if accept_callback is not None:
                accept_callback(accepted)
            if len(equal_rows) >= equal_target:
                break
        log(
            f"[EQUAL] accepted={len(equal_rows) - accepted_before}, total={len(equal_rows)}/{equal_target}, "
            f"skipped_dup={skipped_dup}, skipped_not_equal={skipped_notequal}, skipped_len={skipped_len}"
        )

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    rows = long_rows + short_rows + equal_rows
    if len(rows) < total:
        log(
            "WARNING: could not reach target size after max attempts. "
            f"Generated {len(rows)} of {total} "
            f"(long={len(long_rows)}/{long_target}, short={len(short_rows)}/{short_target}, "
            f"equal={len(equal_rows)}/{equal_target})."
        )
    return rows


def write_csv(
    rows: list[dict[str, str]],
    output_csv: Path,
    seed: int,
    *,
    append: bool,
    existing_rows: list[dict[str, str]],
    start_id: int,
) -> None:
    rng = random.Random(seed)
    new_rows = list(rows)
    rng.shuffle(new_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "append" if append and existing_rows else "write"
    log(
        f"Writing CSV to {output_csv.resolve()} "
        f"(mode={mode}, existing_rows={len(existing_rows)}, new_rows={len(new_rows)})"
    )

    final_rows: list[dict[str, str]] = list(existing_rows)
    next_id = start_id + 1
    for row in new_rows:
        pt_pt = row["pt_PT"]
        pt_br = row["pt_BR"]
        final_rows.append(
            {
                "id": str(next_id),
                "pt_PT": pt_pt,
                "pt_BR": pt_br,
                "pt_PT_words": str(count_words(pt_pt)),
                "pt_BR_words": str(count_words(pt_br)),
            }
        )
        next_id += 1

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "pt_PT", "pt_BR", "pt_PT_words", "pt_BR_words"],
        )
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    log(f"Script: {Path(__file__).resolve()}")
    log(f"Current working directory: {Path.cwd().resolve()}")
    log(f"Using env file: {args.env_file.resolve()}")

    if args.total <= 0:
        raise SystemExit("--total must be > 0")
    if not (0.0 <= args.long_ratio <= 1.0):
        raise SystemExit("--long-ratio must be between 0 and 1")
    if not (0.0 <= args.equal_ratio <= 1.0):
        raise SystemExit("--equal-ratio must be between 0 and 1")
    if args.long_ratio + args.equal_ratio > 1.0:
        raise SystemExit("--long-ratio + --equal-ratio must be <= 1")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.short_batch_size <= 0:
        raise SystemExit("--short-batch-size must be > 0")
    if args.equal_batch_size <= 0:
        raise SystemExit("--equal-batch-size must be > 0")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be > 0")
    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be > 0")

    config = resolve_api_config(args)
    if config.get("short_thread_id") and "your_fresh_short_thread_id" in str(config["short_thread_id"]):
        log("Ignoring placeholder short thread id from env: your_fresh_short_thread_id")
        config["short_thread_id"] = None
    log(
        "Resolved IAEDU config: "
        f"endpoint={config['endpoint']}, channel_id={config['channel_id']}, thread_id={config['thread_id']}, "
        f"api_key={mask_secret(config['api_key'])}"
    )
    if config.get("short_thread_id"):
        log(f"Resolved short thread id: {config['short_thread_id']}")
    log(f"Reference examples file: {args.examples_file.resolve()}")
    log(f"Output CSV: {args.output_csv.resolve()}")
    refs = reference_block(read_reference_pairs(args.examples_file, args.reference_pairs))
    existing_seen: set[str] = set()
    start_id = 0
    if args.append:
        existing_seen, start_id = read_existing_rows(args.output_csv)
        log(
            f"Append mode enabled. Loaded existing_pairs={len(existing_seen)}, "
            f"starting_new_ids_from={start_id + 1}"
        )

    sink = CsvSink(args.output_csv, append=args.append, start_id=start_id)
    try:
        rows = collect_examples(
            config=config,
            total=args.total,
            long_ratio=args.long_ratio,
            equal_ratio=args.equal_ratio,
            batch_size=args.batch_size,
            short_batch_size=args.short_batch_size,
            equal_batch_size=args.equal_batch_size,
            refs=refs,
            long_min=args.long_min_words,
            long_max=args.long_max_words,
            short_min=args.short_min_words,
            short_max=args.short_max_words,
            max_attempts=args.max_attempts,
            pause_seconds=args.pause_seconds,
            rotate_thread_on_backend_error=args.rotate_thread_on_backend_error,
            preexisting_seen=existing_seen,
            accept_callback=sink.write,
        )
    finally:
        sink.close()

    long_count = sum(1 for r in rows if r["kind"] == "long")
    short_count = sum(1 for r in rows if r["kind"] == "short")
    equal_count = sum(1 for r in rows if r["kind"] == "equal")

    total_now = start_id + sink.written
    action = "Appended" if args.append else "Wrote"
    print(
        f"{action} {sink.written} rows to {args.output_csv} "
        f"(long={long_count}, short={short_count}, equal={equal_count}, total_now={total_now})."
    )


if __name__ == "__main__":
    main()
