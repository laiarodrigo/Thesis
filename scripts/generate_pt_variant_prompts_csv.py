#!/usr/bin/env python3
"""
Generate pt-PT / pt-BR sentence pairs via IAEDU agent API and save to CSV.

Default behavior:
- 200 generated examples
- 30% long, 30% short, 30% stories, 10% equal (split across long/short/story)
- long sentences: >=16 words
- stories: 3-5 sentences
- style guided by exemplos.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
import uuid
from collections import Counter
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
SIMILARITY_TOKEN_REGEX = re.compile(r"\w+", flags=re.UNICODE)
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?…])\s+")
DEFAULT_TOPICS = [
    "compras",
    "alimentacao",
    "familia",
    "trabalho",
    "escola",
    "saude",
    "transportes",
    "viagens",
    "casa",
    "tecnologia",
    "lazer",
    "servicos",
]


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
            "Defaults to 200 examples with 30% long, 30% short, 30% stories, 10% equal."
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
        default=0.3,
        help="Share of long sentences in total generated rows.",
    )
    parser.add_argument(
        "--short-ratio",
        type=float,
        default=0.3,
        help="Share of short sentences in total generated rows.",
    )
    parser.add_argument(
        "--story-ratio",
        type=float,
        default=0.3,
        help="Share of story examples in total generated rows.",
    )
    parser.add_argument(
        "--equal-ratio",
        type=float,
        default=0.10,
        help="Share of equal sentences where pt_PT == pt_BR.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="How many long examples to request per API call before filtering.",
    )
    parser.add_argument(
        "--short-batch-size",
        type=int,
        default=12,
        help="How many short examples to request per API call before filtering.",
    )
    parser.add_argument(
        "--story-batch-size",
        type=int,
        default=6,
        help="How many story examples to request per API call before filtering.",
    )
    parser.add_argument(
        "--equal-batch-size",
        type=int,
        default=8,
        help="How many equal examples to request per API call before filtering.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=2.0,
        help=(
            "Generate-and-filter factor. Requests this multiple of needed rows "
            "per batch and then filters by quality/diversity."
        ),
    )
    parser.add_argument(
        "--max-candidate-batch",
        type=int,
        default=36,
        help="Maximum rows requested in one API call after multiplier expansion.",
    )
    parser.add_argument(
        "--max-jaccard-similarity",
        type=float,
        default=0.72,
        help="Reject candidate if token Jaccard similarity to accepted data is above this.",
    )
    parser.add_argument(
        "--opening-ngram-size",
        type=int,
        default=4,
        help="Number of starting tokens used to track repetitive openings.",
    )
    parser.add_argument(
        "--max-opening-reuse",
        type=int,
        default=4,
        help="Maximum accepted rows allowed with the same opening n-gram.",
    )
    parser.add_argument(
        "--secondary-opening-ngram-size",
        type=int,
        default=2,
        help="Secondary opening n-gram size used with an additional repetition cap.",
    )
    parser.add_argument(
        "--secondary-max-opening-reuse",
        type=int,
        default=3,
        help="Maximum accepted rows allowed for the secondary opening n-gram.",
    )
    parser.add_argument(
        "--disable-secondary-opening-guard",
        action="store_true",
        default=False,
        help="Disable the secondary short-prefix repetition guard.",
    )
    parser.add_argument(
        "--long-min-words",
        type=int,
        default=16,
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
        "--story-min-sentences",
        type=int,
        default=3,
        help="Minimum number of sentences for story examples.",
    )
    parser.add_argument(
        "--story-max-sentences",
        type=int,
        default=5,
        help="Maximum number of sentences for story examples.",
    )
    parser.add_argument(
        "--story-min-words",
        type=int,
        default=35,
        help="Minimum number of words for story examples.",
    )
    parser.add_argument(
        "--story-max-words",
        type=int,
        default=120,
        help="Maximum number of words for story examples.",
    )
    parser.add_argument(
        "--equal-split",
        choices=("proportional", "even", "story_only"),
        default="proportional",
        help="How to distribute equal examples across long/short/story formats.",
    )
    parser.add_argument(
        "--reference-pairs",
        type=int,
        default=12,
        help="How many pairs to pull from exemplos.txt as style anchors.",
    )
    parser.add_argument(
        "--topics",
        default=",".join(DEFAULT_TOPICS),
        help="Comma-separated list of allowed topic tags used for coverage balancing.",
    )
    parser.add_argument(
        "--disable-topic-tags",
        action="store_true",
        default=False,
        help="Disable topic-tag generation and topic balancing checks.",
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
        help="Maximum API attempts for each generation mode.",
    )
    parser.add_argument(
        "--rotate-thread-on-backend-error",
        action="store_true",
        default=False,
        help="Rotate to a fresh thread id after repeated backend processing errors.",
    )
    parser.add_argument(
        "--max-consecutive-retryable-errors",
        type=int,
        default=30,
        help="Abort a mode after this many consecutive retryable backend errors.",
    )
    parser.add_argument(
        "--diversity-report",
        type=Path,
        default=repo_root / "data" / "pt_variant_diversity_report.json",
        help="Path to JSON report with post-generation diversity statistics.",
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
    # Remove previously appended 8-char rotation suffixes to avoid unbounded growth.
    while True:
        trimmed = re.sub(r"-[0-9a-fA-F]{8}$", "", base)
        if trimmed == base:
            break
        base = trimmed
    if not base:
        base = "thread"
    # Keep thread ids reasonably short for backend compatibility.
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


def parse_topics(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        topic = part.strip().lower()
        if not topic:
            continue
        if topic not in seen:
            out.append(topic)
            seen.add(topic)
    return out


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def count_sentences(text: str) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    parts = [segment.strip() for segment in SENTENCE_SPLIT_REGEX.split(cleaned) if segment.strip()]
    if not parts:
        return 1
    return len(parts)


def allocate_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {key: 0 for key in weights}

    raw = {key: total * max(0.0, weight) for key, weight in weights.items()}
    allocated = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(allocated.values())
    ranked = sorted(
        raw.keys(),
        key=lambda key: (raw[key] - allocated[key], raw[key], key),
        reverse=True,
    )
    for i in range(remaining):
        allocated[ranked[i % len(ranked)]] += 1
    return allocated


def split_equal_targets(
    equal_total: int,
    *,
    strategy: str,
    long_ratio: float,
    short_ratio: float,
    story_ratio: float,
) -> dict[str, int]:
    keys = ("equal_long", "equal_short", "equal_story")
    if equal_total <= 0:
        return {key: 0 for key in keys}
    if strategy == "story_only":
        return {"equal_long": 0, "equal_short": 0, "equal_story": equal_total}
    if strategy == "even":
        return allocate_counts(
            equal_total,
            {"equal_long": 1.0, "equal_short": 1.0, "equal_story": 1.0},
        )

    # proportional
    if long_ratio <= 0 and short_ratio <= 0 and story_ratio <= 0:
        return allocate_counts(
            equal_total,
            {"equal_long": 1.0, "equal_short": 1.0, "equal_story": 1.0},
        )
    return allocate_counts(
        equal_total,
        {
            "equal_long": max(0.0, long_ratio),
            "equal_short": max(0.0, short_ratio),
            "equal_story": max(0.0, story_ratio),
        },
    )


def similarity_tokens(text: str) -> set[str]:
    return set(SIMILARITY_TOKEN_REGEX.findall(text.lower()))


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return intersection / union if union else 0.0


def opening_signature(text: str, n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    toks = SIMILARITY_TOKEN_REGEX.findall(text.lower())
    if not toks:
        return ""
    return " ".join(toks[:n_tokens])


def nearest_neighbor_stats(texts: list[str]) -> dict[str, Any]:
    if len(texts) < 2:
        return {
            "mean_max_jaccard": 0.0,
            "p90_max_jaccard": 0.0,
            "count_ge_0_6": 0,
            "count_ge_0_7": 0,
        }

    token_sets = [similarity_tokens(text) for text in texts]
    max_scores: list[float] = []
    for i, tokens_i in enumerate(token_sets):
        best = 0.0
        for j, tokens_j in enumerate(token_sets):
            if i == j:
                continue
            sim = jaccard_similarity(tokens_i, tokens_j)
            if sim > best:
                best = sim
        max_scores.append(best)

    max_scores_sorted = sorted(max_scores)
    p90_index = max(0, math.ceil(0.9 * len(max_scores_sorted)) - 1)
    return {
        "mean_max_jaccard": round(sum(max_scores) / len(max_scores), 4),
        "p90_max_jaccard": round(max_scores_sorted[p90_index], 4),
        "count_ge_0_6": sum(score >= 0.6 for score in max_scores),
        "count_ge_0_7": sum(score >= 0.7 for score in max_scores),
    }


def write_diversity_report(
    path: Path,
    rows: list[dict[str, str]],
    *,
    opening_ngram_size: int,
    topic_tags_enabled: bool,
) -> None:
    pt_pt_texts = [row["pt_PT"] for row in rows]
    opening_counts: Counter[str] = Counter()
    for text in pt_pt_texts:
        signature = opening_signature(text, opening_ngram_size)
        if signature:
            opening_counts[signature] += 1

    report: dict[str, Any] = {
        "generated_rows": len(rows),
        "nearest_neighbor_jaccard": nearest_neighbor_stats(pt_pt_texts),
        "top_openings": [
            {"opening": opening, "count": count}
            for opening, count in opening_counts.most_common(12)
        ],
    }
    kind_counts = dict(sorted(Counter(row.get("kind", "") for row in rows if row.get("kind")).items()))
    if kind_counts:
        report["kind_counts"] = kind_counts

    if topic_tags_enabled:
        report["topic_distribution"] = dict(
            sorted(Counter(row.get("topic", "") for row in rows if row.get("topic")).items())
        )
        per_kind: dict[str, dict[str, int]] = {}
        kinds = sorted({row.get("kind", "") for row in rows if row.get("kind")})
        for kind in kinds:
            kind_counter = Counter(
                row.get("topic", "")
                for row in rows
                if row.get("kind") == kind and row.get("topic")
            )
            per_kind[kind] = dict(sorted(kind_counter.items()))
        report["topic_distribution_by_kind"] = per_kind

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote diversity report to {path.resolve()}")


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
    story_min_sentences: int,
    story_max_sentences: int,
    story_min_words: int,
    story_max_words: int,
    nonce: str,
    topics: list[str],
    include_topic_tag: bool,
) -> str:
    if mode == "long":
        length_rule = (
            f"Generate long examples only. Each pt_PT sentence should be at least {long_min} words, "
            "single sentence, natural prose (not subtitle style), and can include a subordinate clause. Do not repeat the same beginning across examples."
        )
        equality_rule = "pt_PT and pt_BR must not be identical."
        sentence_rule = "Each variant must be a single sentence."
    elif mode == "short":
        length_rule = (
            f"Generate short examples only. Each pt_PT sentence should be {short_min}-{short_max} words, "
            "single sentence, still natural prose. Do not repeat the same beginning across examples."
        )
        equality_rule = "pt_PT and pt_BR must not be identical."
        sentence_rule = "Each variant must be a single sentence."
    elif mode == "story":
        length_rule = (
            f"Generate story examples only. Each pt_PT text must contain {story_min_sentences}-{story_max_sentences} "
            f"sentences and {story_min_words}-{story_max_words} words. Do not repeat the same beginning across examples."
        )
        equality_rule = "pt_PT and pt_BR must not be identical."
        sentence_rule = "Each variant must be a coherent micro-story with multiple sentences."
    elif mode == "equal_long":
        length_rule = (
            f"Generate equal-long examples only. Each pt_PT sentence should be at least {long_min} words."
        )
        equality_rule = "pt_PT and pt_BR must be exactly identical strings."
        sentence_rule = "Each variant must be a single sentence."
    elif mode == "equal_short":
        length_rule = (
            f"Generate equal-short examples only. Each sentence should be {short_min}-{short_max} words."
        )
        equality_rule = "pt_PT and pt_BR must be exactly identical strings."
        sentence_rule = "Each variant must be a single sentence."
    elif mode == "equal_story":
        length_rule = (
            f"Generate equal-story examples only. Each text must contain {story_min_sentences}-{story_max_sentences} "
            f"sentences and {story_min_words}-{story_max_words} words."
        )
        equality_rule = "pt_PT and pt_BR must be exactly identical strings."
        sentence_rule = "Each variant must be a coherent micro-story with multiple sentences."
    else:
        raise ValueError(f"Unknown mode: {mode}")

    schema = '{"exemplos":[{"pt_PT":"...", "pt_BR":"..."}]}'
    topic_rule = ""
    topic_hint = ""
    if include_topic_tag and topics:
        allowed_topics = ", ".join(topics)
        schema = '{"exemplos":[{"topic":"...", "pt_PT":"...", "pt_BR":"..."}]}'
        topic_rule = (
            f'9) Include a "topic" field and use exactly one of these values: {allowed_topics}.\n'
            "10) Spread examples across different topics and avoid concentrating one topic in a batch."
        )
        topic_hint = f"Allowed topics: {allowed_topics}"

    return f"""
Generate exactly {n_items} NEW bilingual examples with this JSON schema:
{schema}

Hard requirements:
1) {sentence_rule}
2) Keep meaning equivalent across variants, but adapt naturally to each variant.
3) Make examples diverse in topic and wording.
4) Avoid repeating prior samples; produce novel wording and scenarios.
5) Keep both variants natural and grammatically correct.
6) Return valid JSON only. No markdown, no comments, no extra keys.
7) {length_rule}
8) {equality_rule}
{topic_rule}

Batch nonce (for diversity): {nonce}
{topic_hint}

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
        topic = str(
            item.get("topic")
            or item.get("tema")
            or item.get("label")
            or ""
        ).strip().lower()
        if pt_pt and pt_br:
            out.append({"pt_PT": pt_pt, "pt_BR": pt_br, "topic": topic})
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
    story_min_sentences: int,
    story_max_sentences: int,
    story_min_words: int,
    story_max_words: int,
    topics: list[str],
    include_topic_tag: bool,
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
        story_min_sentences=story_min_sentences,
        story_max_sentences=story_max_sentences,
        story_min_words=story_min_words,
        story_max_words=story_max_words,
        nonce=nonce,
        topics=topics,
        include_topic_tag=include_topic_tag,
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


def read_existing_rows(path: Path) -> tuple[set[str], int, list[str]]:
    if not path.exists():
        return set(), 0, []

    seen_pairs: set[str] = set()
    max_id = 0
    pt_pt_texts: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pt_pt = str(row.get("pt_PT", "")).strip()
            pt_br = str(row.get("pt_BR", "")).strip()

            if pt_pt and pt_br:
                seen_pairs.add(normalize_pair(pt_pt, pt_br))
                pt_pt_texts.append(pt_pt)

            try:
                row_id = int(str(row.get("id", "")).strip())
                max_id = max(max_id, row_id)
            except Exception:
                pass

    return seen_pairs, max_id, pt_pt_texts


def validate_mode_candidate(
    mode: str,
    *,
    pt_pt: str,
    pt_br: str,
    long_min: int,
    short_min: int,
    short_max: int,
    story_min_sentences: int,
    story_max_sentences: int,
    story_min_words: int,
    story_max_words: int,
) -> str | None:
    words = count_words(pt_pt)
    sentences = count_sentences(pt_pt)
    is_equal_mode = mode.startswith("equal_")

    if is_equal_mode and pt_pt != pt_br:
        return "skipped_not_equal"
    if not is_equal_mode and pt_pt == pt_br:
        return "skipped_equal"

    if mode in {"long", "equal_long"}:
        if sentences != 1:
            return "skipped_sentence_count"
        if words < long_min:
            return "skipped_too_short"
        return None

    if mode in {"short", "equal_short"}:
        if sentences != 1:
            return "skipped_sentence_count"
        if not (short_min <= words <= short_max):
            return "skipped_len"
        return None

    if mode in {"story", "equal_story"}:
        if not (story_min_sentences <= sentences <= story_max_sentences):
            return "skipped_story_sentence_count"
        if not (story_min_words <= words <= story_max_words):
            return "skipped_story_word_range"
        return None

    return "skipped_mode"


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
    short_ratio: float,
    story_ratio: float,
    equal_ratio: float,
    batch_size: int,
    short_batch_size: int,
    story_batch_size: int,
    equal_batch_size: int,
    refs: str,
    long_min: int,
    long_max: int,
    short_min: int,
    short_max: int,
    story_min_sentences: int,
    story_max_sentences: int,
    story_min_words: int,
    story_max_words: int,
    equal_split: str,
    max_attempts: int,
    pause_seconds: float,
    rotate_thread_on_backend_error: bool,
    candidate_multiplier: float,
    max_candidate_batch: int,
    max_jaccard_similarity: float,
    opening_ngram_size: int,
    max_opening_reuse: int,
    secondary_opening_ngram_size: int,
    secondary_max_opening_reuse: int,
    disable_secondary_opening_guard: bool,
    max_consecutive_retryable_errors: int,
    topics: list[str],
    enable_topic_tags: bool,
    preexisting_seen: set[str] | None = None,
    preexisting_pt_pt: list[str] | None = None,
    accept_callback: Callable[[dict[str, str]], None] | None = None,
) -> list[dict[str, str]]:
    top_level_targets = allocate_counts(
        total,
        {
            "long": long_ratio,
            "short": short_ratio,
            "story": story_ratio,
            "equal_total": equal_ratio,
        },
    )
    equal_targets = split_equal_targets(
        top_level_targets["equal_total"],
        strategy=equal_split,
        long_ratio=long_ratio,
        short_ratio=short_ratio,
        story_ratio=story_ratio,
    )
    mode_targets = {
        "long": top_level_targets["long"],
        "short": top_level_targets["short"],
        "story": top_level_targets["story"],
        "equal_long": equal_targets["equal_long"],
        "equal_short": equal_targets["equal_short"],
        "equal_story": equal_targets["equal_story"],
    }

    topic_tags_enabled = bool(enable_topic_tags and topics)
    topic_lookup = {topic.lower(): topic for topic in topics}
    topic_counts: dict[str, Counter[str]] = {mode: Counter() for mode in mode_targets}
    topic_limits: dict[str, int] = {}
    if topic_tags_enabled and topics and len(topics) > 0:
        topic_limits = {
            mode: max(1, math.ceil(target / len(topics)) + 1)
            for mode, target in mode_targets.items()
            if target > 0
        }

    log(
        "Generation targets: "
        f"total={total}, long={mode_targets['long']}, short={mode_targets['short']}, story={mode_targets['story']}, "
        f"equal_long={mode_targets['equal_long']}, equal_short={mode_targets['equal_short']}, "
        f"equal_story={mode_targets['equal_story']}, "
        f"long_batch_size={batch_size}, short_batch_size={short_batch_size}, "
        f"story_batch_size={story_batch_size}, equal_batch_size={equal_batch_size}, "
        f"candidate_multiplier={candidate_multiplier:.2f}, max_jaccard_similarity={max_jaccard_similarity:.2f}, "
        f"opening_guard={opening_ngram_size}:{max_opening_reuse}, "
        f"secondary_opening_guard={'off' if disable_secondary_opening_guard else f'{secondary_opening_ngram_size}:{secondary_max_opening_reuse}'}, "
        f"topic_tags_enabled={topic_tags_enabled}"
    )

    seen: set[str] = set(preexisting_seen or set())
    similarity_index = [similarity_tokens(text) for text in (preexisting_pt_pt or []) if text.strip()]
    opening_policies: list[tuple[int, int]] = [(opening_ngram_size, max_opening_reuse)]
    if not disable_secondary_opening_guard and secondary_opening_ngram_size != opening_ngram_size:
        opening_policies.append((secondary_opening_ngram_size, secondary_max_opening_reuse))
    opening_counts_by_size: dict[int, Counter[str]] = {size: Counter() for size, _ in opening_policies}
    for text in preexisting_pt_pt or []:
        for size, _ in opening_policies:
            signature = opening_signature(text, size)
            if signature:
                opening_counts_by_size[size][signature] += 1

    rows_by_mode: dict[str, list[dict[str, str]]] = {mode: [] for mode in mode_targets}
    long_config = dict(config)
    if seen:
        log(f"Loaded {len(seen)} existing pairs; dedup will exclude them.")
    if similarity_index:
        log(f"Loaded {len(similarity_index)} existing pt_PT rows for near-duplicate filtering.")

    short_config = config
    if config.get("short_thread_id"):
        short_config = dict(config)
        short_config["thread_id"] = config["short_thread_id"]
        log(f"Using separate short thread_id={short_config['thread_id']}")
    else:
        short_config = dict(config)

    mode_specs: list[dict[str, Any]] = [
        {"mode": "long", "batch_size": batch_size, "config": long_config, "label": "LONG"},
        {"mode": "short", "batch_size": short_batch_size, "config": short_config, "label": "SHORT"},
        {"mode": "story", "batch_size": story_batch_size, "config": short_config, "label": "STORY"},
        {"mode": "equal_long", "batch_size": equal_batch_size, "config": short_config, "label": "EQUAL_LONG"},
        {"mode": "equal_short", "batch_size": equal_batch_size, "config": short_config, "label": "EQUAL_SHORT"},
        {"mode": "equal_story", "batch_size": equal_batch_size, "config": short_config, "label": "EQUAL_STORY"},
    ]

    for spec in mode_specs:
        mode = str(spec["mode"])
        label = str(spec["label"])
        mode_target = int(mode_targets.get(mode, 0))
        if mode_target <= 0:
            continue
        mode_config = spec["config"]
        base_batch_size = int(spec["batch_size"])
        attempts = 0
        backend_errors = 0
        consecutive_retryable_errors = 0
        zero_accept_streak = 0

        while len(rows_by_mode[mode]) < mode_target and attempts < max_attempts:
            attempts += 1
            needed = min(base_batch_size, mode_target - len(rows_by_mode[mode]))
            request_size = min(
                max_candidate_batch,
                max(needed, math.ceil(needed * candidate_multiplier)),
            )
            log(
                f"[{label}] attempt={attempts}/{max_attempts}, need={needed}, request_size={request_size}, "
                f"collected={len(rows_by_mode[mode])}/{mode_target}"
            )
            try:
                batch = request_batch(
                    config=mode_config,
                    n_items=request_size,
                    mode=mode,
                    refs=refs,
                    long_min=long_min,
                    long_max=long_max,
                    short_min=short_min,
                    short_max=short_max,
                    story_min_sentences=story_min_sentences,
                    story_max_sentences=story_max_sentences,
                    story_min_words=story_min_words,
                    story_max_words=story_max_words,
                    topics=topics,
                    include_topic_tag=topic_tags_enabled,
                )
                log(f"[{label}] parsed batch_size={len(batch)}")
                consecutive_retryable_errors = 0
            except Exception as exc:
                log(f"[{label}] request failed: {exc}")
                if is_retryable_backend_error(exc):
                    backend_errors += 1
                    consecutive_retryable_errors += 1
                    if rotate_thread_on_backend_error and backend_errors >= 2:
                        old_tid = mode_config["thread_id"]
                        mode_config["thread_id"] = rotated_thread_id(old_tid)
                        backend_errors = 0
                        log(f"[{label}] rotated thread_id after backend errors: {old_tid} -> {mode_config['thread_id']}")
                    if backend_errors >= 3 and mode != "long" and not config.get("short_thread_id"):
                        log(
                            f"[{label}] repeated backend errors. Set IAEDU_SHORT_THREAD_ID "
                            "or --short-thread-id to use a fresh thread for non-long modes."
                        )
                    if consecutive_retryable_errors >= max_consecutive_retryable_errors:
                        log(
                            f"[{label}] aborting mode after {consecutive_retryable_errors} "
                            "consecutive retryable backend errors."
                        )
                        break
                    backoff = min(30.0, 2.0 + consecutive_retryable_errors * 2.0)
                    log(f"[{label}] backend error backoff: {backoff:.1f}s")
                    time.sleep(backoff)
                else:
                    time.sleep(1.0)
                continue
            backend_errors = 0

            accepted_before = len(rows_by_mode[mode])
            counters: Counter[str] = Counter()
            enforce_topic_cap = (
                topic_tags_enabled
                and attempts <= max(1, math.floor(max_attempts * 0.75))
                and len(rows_by_mode[mode]) < max(1, math.floor(mode_target * 0.95))
            )

            for item in batch:
                pt_pt = item["pt_PT"].strip()
                pt_br = item["pt_BR"].strip()
                topic = topic_lookup.get(str(item.get("topic", "")).strip().lower(), "")
                key = normalize_pair(pt_pt, pt_br)
                if key in seen:
                    counters["skipped_dup"] += 1
                    continue
                if topic_tags_enabled and not topic:
                    counters["skipped_topic"] += 1
                    continue
                if (
                    topic_tags_enabled
                    and enforce_topic_cap
                    and topic_counts[mode][topic] >= topic_limits.get(mode, 0)
                ):
                    counters["skipped_topic_cap"] += 1
                    continue

                mode_reject = validate_mode_candidate(
                    mode,
                    pt_pt=pt_pt,
                    pt_br=pt_br,
                    long_min=long_min,
                    short_min=short_min,
                    short_max=short_max,
                    story_min_sentences=story_min_sentences,
                    story_max_sentences=story_max_sentences,
                    story_min_words=story_min_words,
                    story_max_words=story_max_words,
                )
                if mode_reject is not None:
                    counters[mode_reject] += 1
                    continue

                tokens = similarity_tokens(pt_pt)
                if tokens and any(
                    jaccard_similarity(tokens, previous_tokens) >= max_jaccard_similarity
                    for previous_tokens in similarity_index
                ):
                    counters["skipped_similar"] += 1
                    continue
                opening_signatures: dict[int, str] = {}
                opening_rejected = False
                for size, max_reuse in opening_policies:
                    signature = opening_signature(pt_pt, size)
                    opening_signatures[size] = signature
                    if signature and opening_counts_by_size[size][signature] >= max_reuse:
                        opening_rejected = True
                        break
                if opening_rejected:
                    counters["skipped_opening"] += 1
                    continue

                seen.add(key)
                accepted = {"kind": mode, "pt_PT": pt_pt, "pt_BR": pt_br}
                if topic:
                    accepted["topic"] = topic
                rows_by_mode[mode].append(accepted)
                if tokens:
                    similarity_index.append(tokens)
                for size, _ in opening_policies:
                    signature = opening_signatures.get(size, "")
                    if signature:
                        opening_counts_by_size[size][signature] += 1
                if topic:
                    topic_counts[mode][topic] += 1
                if accept_callback is not None:
                    accept_callback(accepted)
                if len(rows_by_mode[mode]) >= mode_target:
                    break

            accepted_now = len(rows_by_mode[mode]) - accepted_before
            zero_accept_streak = zero_accept_streak + 1 if accepted_now == 0 else 0
            log(
                f"[{label}] accepted={accepted_now}, total={len(rows_by_mode[mode])}/{mode_target}, "
                f"skipped_dup={counters['skipped_dup']}, skipped_equal={counters['skipped_equal']}, "
                f"skipped_not_equal={counters['skipped_not_equal']}, skipped_len={counters['skipped_len']}, "
                f"skipped_too_short={counters['skipped_too_short']}, "
                f"skipped_sentence_count={counters['skipped_sentence_count']}, "
                f"skipped_story_sentence_count={counters['skipped_story_sentence_count']}, "
                f"skipped_story_word_range={counters['skipped_story_word_range']}, "
                f"skipped_similar={counters['skipped_similar']}, skipped_opening={counters['skipped_opening']}, "
                f"skipped_topic={counters['skipped_topic']}, skipped_topic_cap={counters['skipped_topic_cap']}"
            )

            if rotate_thread_on_backend_error and zero_accept_streak >= 5:
                old_tid = mode_config["thread_id"]
                mode_config["thread_id"] = rotated_thread_id(old_tid)
                zero_accept_streak = 0
                log(
                    f"[{label}] rotated thread_id after repeated zero-accept batches: "
                    f"{old_tid} -> {mode_config['thread_id']}"
                )

            if pause_seconds > 0:
                time.sleep(pause_seconds)

    rows = (
        rows_by_mode["long"]
        + rows_by_mode["short"]
        + rows_by_mode["story"]
        + rows_by_mode["equal_long"]
        + rows_by_mode["equal_short"]
        + rows_by_mode["equal_story"]
    )
    if len(rows) < total:
        log(
            "WARNING: could not reach target size after max attempts. "
            f"Generated {len(rows)} of {total} "
            f"(long={len(rows_by_mode['long'])}/{mode_targets['long']}, "
            f"short={len(rows_by_mode['short'])}/{mode_targets['short']}, "
            f"story={len(rows_by_mode['story'])}/{mode_targets['story']}, "
            f"equal_long={len(rows_by_mode['equal_long'])}/{mode_targets['equal_long']}, "
            f"equal_short={len(rows_by_mode['equal_short'])}/{mode_targets['equal_short']}, "
            f"equal_story={len(rows_by_mode['equal_story'])}/{mode_targets['equal_story']})."
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
    if not (0.0 <= args.short_ratio <= 1.0):
        raise SystemExit("--short-ratio must be between 0 and 1")
    if not (0.0 <= args.story_ratio <= 1.0):
        raise SystemExit("--story-ratio must be between 0 and 1")
    if not (0.0 <= args.equal_ratio <= 1.0):
        raise SystemExit("--equal-ratio must be between 0 and 1")
    ratio_total = args.long_ratio + args.short_ratio + args.story_ratio + args.equal_ratio
    if abs(ratio_total - 1.0) > 1e-6:
        raise SystemExit(
            "--long-ratio + --short-ratio + --story-ratio + --equal-ratio must equal 1.0"
        )
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.short_batch_size <= 0:
        raise SystemExit("--short-batch-size must be > 0")
    if args.story_batch_size <= 0:
        raise SystemExit("--story-batch-size must be > 0")
    if args.equal_batch_size <= 0:
        raise SystemExit("--equal-batch-size must be > 0")
    if args.long_min_words <= 0:
        raise SystemExit("--long-min-words must be > 0")
    if args.short_min_words <= 0 or args.short_max_words <= 0:
        raise SystemExit("--short-min-words and --short-max-words must be > 0")
    if args.short_min_words > args.short_max_words:
        raise SystemExit("--short-min-words must be <= --short-max-words")
    if args.story_min_sentences <= 0 or args.story_max_sentences <= 0:
        raise SystemExit("--story-min-sentences and --story-max-sentences must be > 0")
    if args.story_min_sentences > args.story_max_sentences:
        raise SystemExit("--story-min-sentences must be <= --story-max-sentences")
    if args.story_min_words <= 0 or args.story_max_words <= 0:
        raise SystemExit("--story-min-words and --story-max-words must be > 0")
    if args.story_min_words > args.story_max_words:
        raise SystemExit("--story-min-words must be <= --story-max-words")
    if args.candidate_multiplier < 1.0:
        raise SystemExit("--candidate-multiplier must be >= 1.0")
    if args.max_candidate_batch <= 0:
        raise SystemExit("--max-candidate-batch must be > 0")
    if not (0.0 <= args.max_jaccard_similarity <= 1.0):
        raise SystemExit("--max-jaccard-similarity must be between 0 and 1")
    if args.opening_ngram_size <= 0:
        raise SystemExit("--opening-ngram-size must be > 0")
    if args.max_opening_reuse <= 0:
        raise SystemExit("--max-opening-reuse must be > 0")
    if args.secondary_opening_ngram_size <= 0:
        raise SystemExit("--secondary-opening-ngram-size must be > 0")
    if args.secondary_max_opening_reuse <= 0:
        raise SystemExit("--secondary-max-opening-reuse must be > 0")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be > 0")
    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be > 0")
    if args.max_consecutive_retryable_errors <= 0:
        raise SystemExit("--max-consecutive-retryable-errors must be > 0")
    topics = parse_topics(args.topics)
    topic_tags_enabled = not args.disable_topic_tags and bool(topics)
    if not topics and not args.disable_topic_tags:
        log("No topics configured. Topic balancing will be disabled.")

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
    if topic_tags_enabled:
        log(f"Topic balancing enabled with topics={topics}")
    else:
        log("Topic balancing disabled.")
    refs = reference_block(read_reference_pairs(args.examples_file, args.reference_pairs))
    existing_seen: set[str] = set()
    existing_pt_pt: list[str] = []
    start_id = 0
    if args.append:
        existing_seen, start_id, existing_pt_pt = read_existing_rows(args.output_csv)
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
            short_ratio=args.short_ratio,
            story_ratio=args.story_ratio,
            equal_ratio=args.equal_ratio,
            batch_size=args.batch_size,
            short_batch_size=args.short_batch_size,
            story_batch_size=args.story_batch_size,
            equal_batch_size=args.equal_batch_size,
            refs=refs,
            long_min=args.long_min_words,
            long_max=args.long_max_words,
            short_min=args.short_min_words,
            short_max=args.short_max_words,
            story_min_sentences=args.story_min_sentences,
            story_max_sentences=args.story_max_sentences,
            story_min_words=args.story_min_words,
            story_max_words=args.story_max_words,
            equal_split=args.equal_split,
            max_attempts=args.max_attempts,
            pause_seconds=args.pause_seconds,
            rotate_thread_on_backend_error=args.rotate_thread_on_backend_error,
            candidate_multiplier=args.candidate_multiplier,
            max_candidate_batch=args.max_candidate_batch,
            max_jaccard_similarity=args.max_jaccard_similarity,
            opening_ngram_size=args.opening_ngram_size,
            max_opening_reuse=args.max_opening_reuse,
            secondary_opening_ngram_size=args.secondary_opening_ngram_size,
            secondary_max_opening_reuse=args.secondary_max_opening_reuse,
            disable_secondary_opening_guard=args.disable_secondary_opening_guard,
            max_consecutive_retryable_errors=args.max_consecutive_retryable_errors,
            topics=topics,
            enable_topic_tags=topic_tags_enabled,
            preexisting_seen=existing_seen,
            preexisting_pt_pt=existing_pt_pt,
            accept_callback=sink.write,
        )
    finally:
        sink.close()

    long_count = sum(1 for r in rows if r["kind"] == "long")
    short_count = sum(1 for r in rows if r["kind"] == "short")
    story_count = sum(1 for r in rows if r["kind"] == "story")
    equal_long_count = sum(1 for r in rows if r["kind"] == "equal_long")
    equal_short_count = sum(1 for r in rows if r["kind"] == "equal_short")
    equal_story_count = sum(1 for r in rows if r["kind"] == "equal_story")
    equal_count = equal_long_count + equal_short_count + equal_story_count
    write_diversity_report(
        args.diversity_report,
        rows,
        opening_ngram_size=args.opening_ngram_size,
        topic_tags_enabled=topic_tags_enabled,
    )

    total_now = start_id + sink.written
    action = "Appended" if args.append else "Wrote"
    print(
        f"{action} {sink.written} rows to {args.output_csv} "
        f"(long={long_count}, short={short_count}, story={story_count}, "
        f"equal_long={equal_long_count}, equal_short={equal_short_count}, "
        f"equal_story={equal_story_count}, equal_total={equal_count}, total_now={total_now})."
    )


if __name__ == "__main__":
    main()
