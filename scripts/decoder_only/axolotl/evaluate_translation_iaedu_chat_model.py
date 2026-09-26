#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.decoder_only.axolotl.iaedu_agent_client import (
    request_with_retries,
    resolve_api_config,
)
from scripts.decoder_only.axolotl.rebuild_translation_chat_summary import (
    FRMT_BUCKETS,
    build_translation_summary,
    normalize_bucket,
    normalize_label,
    strip_decoder_label_prefix,
)


def debug_log(message: str) -> None:
    if os.getenv("IAEDU_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[iaedu-debug] {message}", flush=True)


SYSTEM_TRANSLATION = (
    "És um assistente especialista em português europeu e português do Brasil. "
    "A tua tarefa é converter frases entre as duas variantes, mantendo o significado, "
    "o registo e um estilo natural. Responde apenas com a tradução final, "
    "sem explicações, sem comentários e sem texto adicional."
)

USER_TRANSL_BR2PT = (
    "Converte o seguinte texto de português do Brasil para português europeu, "
    "mantendo o sentido e soando natural em português europeu. "
    "Responde apenas com a frase convertida.\n\n"
    "Texto: {source}"
)

USER_TRANSL_PT2BR = (
    "Converte o seguinte texto de português europeu para português do Brasil, "
    "mantendo o sentido e soando natural em português do Brasil. "
    "Responde apenas com a frase convertida.\n\n"
    "Texto: {source}"
)

ENCODER_TASK_PREFIX_RE = re.compile(
    r"^\s*(?:<(br-pt|pt-br|pt-pt|id|cls)>|((?:BR|PT|CLS)\b))(?:\s*:\s*|\s+)",
    flags=re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", flags=re.DOTALL)
COMMON_PREFIX_RE = re.compile(
    r"^\s*(?:tradu[cç][aã]o|tradu[cç][aã]o final|texto convertido|resultado|resposta|a tradu[cç][aã]o [ée])\s*[:\-]\s*",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IAEDU chat models on translation JSONL data.")
    parser.add_argument("--model-id", required=True, help="Label used in outputs for the IAEDU-backed model.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=3.0)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / "bla.env")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--channel-id", default=None)
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--short-thread-id", default=None)
    parser.add_argument("--user-info", default="{}")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--user-context", default=None)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=int(os.getenv("IAEDU_PROGRESS_INTERVAL", "100")),
        help="Print progress every N completed rows.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split())


def strip_task_prefix(text: str) -> str:
    raw = text or ""
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_space(raw)


def infer_direction(row: dict[str, Any]) -> str | None:
    value = normalize_space(str(row.get("direction") or row.get("task") or "")).casefold()
    if value in {"translate_br2pt", "br2pt"}:
        return "br2pt"
    if value in {"translate_pt2br", "pt2br"}:
        return "pt2br"

    input_text = normalize_space(str(row.get("input_text") or ""))
    if input_text.startswith("<br-pt>"):
        return "br2pt"
    if input_text.startswith("<pt-br>"):
        return "pt2br"
    return None


def extract_source(row: dict[str, Any]) -> str:
    source = normalize_space(str(row.get("source_text") or ""))
    if source:
        return source
    return strip_task_prefix(str(row.get("input_text") or ""))


def clean_generation_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = THINK_RE.sub("", cleaned).strip()
    cleaned = CODE_FENCE_RE.sub("", cleaned).strip()
    cleaned = COMMON_PREFIX_RE.sub("", cleaned).strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    return normalize_space(cleaned)


def iter_translation_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            direction = infer_direction(row)
            source = extract_source(row)
            target_raw = normalize_space(str(row.get("target_text") or ""))
            target = strip_decoder_label_prefix(target_raw)
            if direction is None or not source or not target:
                continue
            rows.append(
                {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "bucket": row.get("bucket"),
                    "direction": direction,
                    "source": source,
                    "target": target,
                    "target_raw": target_raw,
                }
            )
    return rows


def build_message(source: str, direction: str) -> str:
    if direction == "br2pt":
        user_content = USER_TRANSL_BR2PT.format(source=source)
    elif direction == "pt2br":
        user_content = USER_TRANSL_PT2BR.format(source=source)
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return f"{SYSTEM_TRANSLATION}\n\n{user_content}"


def predict_one(
    config: dict[str, Any],
    row: dict[str, Any],
    *,
    model_id: str,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    debug_log(
        "translation-row-start"
        f" id={row.get('id')}"
        f" direction={row['direction']}"
        f" source_chars={len(row['source'])}"
    )
    raw_response, thread_id = request_with_retries(
        config,
        build_message(row["source"], row["direction"]),
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    pred_raw = clean_generation_text(raw_response)
    pred = strip_decoder_label_prefix(pred_raw)
    gold = normalize_space(row["target"])
    src = normalize_space(row["source"])
    gold_label = normalize_label(row.get("target_raw"))
    pred_label = normalize_label(pred_raw)
    bucket = normalize_bucket(row.get("bucket"))

    prediction_record = {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "bucket": bucket,
        "direction": row["direction"],
        "input_text": row["source"],
        "gold_raw": row.get("target_raw"),
        "gold": gold,
        "pred_raw": pred_raw,
        "pred": pred,
        "api_model": model_id,
        "iaedu_thread_id": thread_id,
    }
    if gold_label is not None:
        prediction_record["gold_source_variant_norm"] = gold_label
    if pred_label is not None:
        prediction_record["pred_source_variant_norm"] = pred_label

    metric_row = {
        "id": row.get("id"),
        "direction": row["direction"],
        "src": src,
        "gold": gold,
        "pred": pred,
        "gold_label": gold_label,
        "pred_label": pred_label,
        "bucket": bucket,
    }
    debug_log(
        "translation-row-done"
        f" id={row.get('id')}"
        f" pred_chars={len(pred)}"
        f" raw_chars={len(pred_raw)}"
        f" thread_id={thread_id}"
    )
    return prediction_record, metric_row


def main() -> None:
    args = parse_args()
    config = resolve_api_config(
        env_file=args.env_file,
        endpoint=args.endpoint,
        api_key=args.api_key,
        channel_id=args.channel_id,
        thread_id=args.thread_id,
        short_thread_id=args.short_thread_id,
        user_info=args.user_info,
        user_id=args.user_id,
        user_context=args.user_context,
        request_timeout=args.request_timeout,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = iter_translation_rows(args.dataset_path)
    if not rows:
        raise RuntimeError(f"No valid translation rows found in {args.dataset_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = args.output_dir / f"{run_id}_translation_predictions.jsonl"
    summary_path = args.output_dir / f"{run_id}_translation_summary.json"
    print(
        "[iaedu] translation setup"
        f" rows={len(rows)}"
        f" concurrency={args.concurrency}"
        f" request_timeout={args.request_timeout}"
        f" predictions={pred_path}"
        f" summary={summary_path}",
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    completed = 0

    progress_interval = max(1, int(args.progress_interval))

    with pred_path.open("w", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        future_map = {
            pool.submit(
                predict_one,
                config,
                row,
                model_id=args.model_id,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            ): row.get("id")
            for row in rows
        }
        for future in as_completed(future_map):
            prediction_record, metric_row = future.result()
            fh.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
            fh.flush()
            metric_rows.append(metric_row)
            completed += 1
            if completed % progress_interval == 0 or completed == len(rows):
                print(f"[progress] {completed}/{len(rows)} translation rows complete")

    summary: dict[str, Any] = {
        "task": "translation",
        "eval_config": {
            "dataset_path": args.dataset_path.as_posix(),
            "model_id": args.model_id,
            "concurrency": args.concurrency,
            "max_retries": args.max_retries,
            "retry_backoff_seconds": args.retry_backoff_seconds,
            "env_file": args.env_file.as_posix(),
            "request_timeout": args.request_timeout,
            "progress_interval": progress_interval,
        },
        **build_translation_summary(metric_rows),
        "predictions_path": pred_path.as_posix(),
        "api_backend": "iaedu_agent",
    }

    per_direction: dict[str, Any] = {}
    for direction in sorted({str(row["direction"]) for row in metric_rows if row.get("direction")}):
        per_direction[direction] = build_translation_summary(
            [row for row in metric_rows if row.get("direction") == direction]
        )
    if per_direction:
        summary["available_directions"] = sorted(per_direction)
        summary["per_direction"] = per_direction

    per_bucket: dict[str, Any] = {}
    for bucket in FRMT_BUCKETS:
        bucket_rows = [row for row in metric_rows if row.get("bucket") == bucket]
        if not bucket_rows:
            continue
        per_bucket[bucket] = build_translation_summary(bucket_rows)
    if per_bucket:
        summary["available_buckets"] = sorted(per_bucket)
        summary["per_bucket"] = per_bucket

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
