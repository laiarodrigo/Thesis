#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "scripts" / "encoder_decoder" / "eval"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from datasets import load_dataset

from scripts.decoder_only.axolotl.iaedu_agent_client import (
    request_with_retries,
    resolve_api_config,
)
from scripts.encoder_decoder.eval.metrics_utils import PT_VARIANT_LABELS


LABELS = ("pt-br", "pt-pt", "equal")
FRMT_BUCKETS = ("random", "entity", "lexical")


def debug_log(message: str) -> None:
    if os.getenv("IAEDU_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[iaedu-debug] {message}", flush=True)


SYSTEM_CLASSIFICATION = (
    "És um linguista especialista em português europeu e português do Brasil. "
    "A tua tarefa é identificar a variante correta do texto."
)

USER_CLASSIFICATION = (
    "Classifica a variante do texto como uma destas etiquetas: {labels}. "
    "Usa 'equal' apenas quando o texto é igual nas duas variantes. "
    "Responde apenas com uma etiqueta.\n\n"
    "Texto: {source}"
)

THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", flags=re.DOTALL)
COMMON_PREFIX_RE = re.compile(
    r"^\s*(?:resposta|r[oó]tulo|etiqueta|classifica[cç][aã]o|resultado)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
TASK_PREFIX_RE = re.compile(
    r"^\s*(?:<(br-pt|pt-br|pt-pt|id|cls)>|((?:BR|PT|CLS)\b))(?:\s*:\s*|\s+)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IAEDU chat models on classification JSONL data.")
    parser.add_argument("--model-id", required=True, help="Label used in outputs for the IAEDU-backed model.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
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
    parser.add_argument(
        "--classification-candidates",
        nargs="+",
        default=None,
        help="Optional explicit labels, e.g. 'pt-br pt-pt equal' or 'BR PT'.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split())


def strip_task_prefix(text: str) -> str:
    return normalize_space(TASK_PREFIX_RE.sub("", text or ""))


def normalize_label(text: str) -> Optional[str]:
    t = normalize_space(text or "").lower()
    if not t:
        return None
    first = t.split(" ", 1)[0].strip(",:;.-_")
    if first == "br":
        return "pt-br"
    if first == "pt":
        return "pt-pt"
    if "equal" in t or "shared" in t or t == "same" or first == "igual":
        return "equal"
    if "pt-br" in t or "ptbr" in t or "brasil" in t:
        return "pt-br"
    if "pt-pt" in t or "ptpt" in t or "europeu" in t or "portugal" in t:
        return "pt-pt"
    return None


def normalize_bucket(raw_bucket: object) -> str:
    text = normalize_space(str(raw_bucket or "")).lower()
    if text in {"rand", "random"}:
        return "random"
    if text in {"entity", "entities"}:
        return "entity"
    if text in {"lexical", "lex"}:
        return "lexical"
    if not text:
        return "n/a"
    return text


def classification_target_values(ds) -> list[str]:
    for key in ("target_text", "label", "gold"):
        if key in ds.column_names:
            return list(ds[key])
    raise KeyError(
        "Classification eval dataset must contain one of: target_text, label, gold. "
        f"Available columns: {sorted(ds.column_names)}"
    )


def extract_classification_targets(batch: dict[str, Any]) -> list[str]:
    for key in ("target_text", "label", "gold"):
        values = batch.get(key)
        if values is None:
            continue
        return [normalize_space(str(value or "")) for value in values]
    raise KeyError(
        "Classification eval batch must contain one of: target_text, label, gold. "
        f"Available keys: {sorted(batch.keys())}"
    )


def extract_source_texts(batch: dict[str, Any]) -> list[str]:
    values = batch.get("text")
    if values is not None:
        return [normalize_space(str(value or "")) for value in values]
    values = batch.get("input_text")
    if values is not None:
        return [strip_task_prefix(str(value or "")) for value in values]
    raise KeyError(
        "Classification eval batch must contain one of: text, input_text. "
        f"Available keys: {sorted(batch.keys())}"
    )


def build_classification_candidates(ds, explicit_candidates: Optional[list[str]]) -> list[str]:
    seen = set()
    candidates: list[str] = []
    raw_values = explicit_candidates if explicit_candidates else classification_target_values(ds)
    for raw in raw_values:
        cand = normalize_space(str(raw or ""))
        if not cand or cand in seen:
            continue
        seen.add(cand)
        candidates.append(cand)
    if not candidates:
        raise ValueError("No valid classification candidates found.")
    return candidates


def filter_classification_dataset_for_candidates(ds, candidates: list[str]):
    allowed = {label for label in (normalize_label(cand) for cand in candidates) if label is not None}
    if not allowed:
        return ds, {}

    raw_targets = classification_target_values(ds)
    keep_indices: list[int] = []
    dropped: Counter[str] = Counter()
    for idx, raw in enumerate(raw_targets):
        norm = normalize_label(str(raw or "")) or "unknown"
        if norm in allowed:
            keep_indices.append(idx)
        else:
            dropped[norm] += 1

    if not dropped:
        return ds, {}
    return ds.select(keep_indices), dict(dropped)


def clean_generation_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = THINK_RE.sub("", cleaned).strip()
    cleaned = CODE_FENCE_RE.sub("", cleaned).strip()
    cleaned = COMMON_PREFIX_RE.sub("", cleaned).strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    return normalize_space(cleaned)


@dataclass
class ClsStats:
    total: int = 0
    correct: int = 0
    tp: Counter | None = None
    fp: Counter | None = None
    fn: Counter | None = None
    cm: Counter | None = None

    def __post_init__(self):
        self.tp = Counter()
        self.fp = Counter()
        self.fn = Counter()
        self.cm = Counter()

    def update(self, gold: str, pred: str):
        self.total += 1
        self.cm[(gold, pred)] += 1
        if pred == gold:
            self.correct += 1
            self.tp[gold] += 1
        else:
            self.fn[gold] += 1
            self.fp[pred] += 1

    def report(self) -> dict[str, Any]:
        acc = self.correct / self.total if self.total else 0.0
        per_class = {}
        f1s = []
        target_f1s = []
        for lab in LABELS:
            tp = self.tp[lab]
            fp = self.fp[lab]
            fn = self.fn[lab]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_class[lab] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": tp + fn,
            }
            f1s.append(f1)
            if lab in PT_VARIANT_LABELS:
                target_f1s.append(f1)
        return {
            "n": self.total,
            "accuracy": acc,
            "macro_f1": sum(target_f1s) / len(target_f1s) if target_f1s else 0.0,
            "macro_f1_all_labels": sum(f1s) / len(f1s) if f1s else 0.0,
            "macro_f1_labels": list(PT_VARIANT_LABELS),
            "per_class": per_class,
            "confusion_matrix": {
                f"{gold}->{pred}": int(self.cm[(gold, pred)])
                for gold in LABELS
                for pred in LABELS
            },
        }


def build_classification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = ClsStats()
    for row in rows:
        stats.update(str(row["gold_norm"]), str(row["pred_norm"]))
    return stats.report()


def build_message(source: str, candidates: list[str]) -> str:
    return (
        f"{SYSTEM_CLASSIFICATION}\n\n"
        + USER_CLASSIFICATION.format(labels=", ".join(candidates), source=source)
    )


def predict_one(
    config: dict[str, Any],
    *,
    row_id: int,
    source_text: str,
    gold_raw: str,
    dataset_name: str | None,
    bucket: str,
    source_id: Any,
    candidates: list[str],
    model_id: str,
    max_retries: int,
    retry_backoff_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    debug_log(
        "classification-row-start"
        f" id={row_id}"
        f" gold={gold_raw}"
        f" source_chars={len(source_text)}"
    )
    raw_response, thread_id = request_with_retries(
        config,
        build_message(source_text, candidates),
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    pred_raw = clean_generation_text(raw_response)
    gold_norm = normalize_label(gold_raw) or "unknown"
    pred_norm = normalize_label(pred_raw) or "unknown"

    record: dict[str, Any] = {
        "id": row_id,
        "input_text": source_text,
        "gold": gold_raw,
        "gold_norm": gold_norm,
        "pred_raw": pred_raw,
        "pred_norm": pred_norm,
        "bucket": bucket,
        "api_model": model_id,
        "iaedu_thread_id": thread_id,
    }
    if dataset_name is not None:
        record["dataset"] = dataset_name
    if source_id is not None:
        record["source_id"] = source_id

    metric_row = {
        "gold_norm": gold_norm,
        "pred_norm": pred_norm,
        "bucket": bucket,
    }
    debug_log(
        "classification-row-done"
        f" id={row_id}"
        f" pred_norm={pred_norm}"
        f" raw_chars={len(pred_raw)}"
        f" thread_id={thread_id}"
    )
    return record, metric_row


def build_candidate_metadata(candidates: list[str]) -> list[dict[str, Any]]:
    return [{"text": cand, "normalized_label": normalize_label(cand)} for cand in candidates]


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
    ds = load_dataset("json", data_files={"eval": args.dataset_path.as_posix()})["eval"]
    classification_candidates = build_classification_candidates(ds, args.classification_candidates)
    ds, filtered_out_counts = filter_classification_dataset_for_candidates(ds, classification_candidates)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = args.output_dir / f"{run_id}_classification_predictions.jsonl"
    summary_path = args.output_dir / f"{run_id}_classification_summary.json"
    print(
        "[iaedu] classification setup"
        f" rows={len(ds)}"
        f" concurrency={args.concurrency}"
        f" request_timeout={args.request_timeout}"
        f" candidates={classification_candidates}"
        f" predictions={pred_path}"
        f" summary={summary_path}",
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    completed = 0

    progress_interval = max(1, int(args.progress_interval))

    with pred_path.open("w", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        future_map = {}
        for start in range(0, len(ds), args.batch_size):
            batch = ds[start : start + args.batch_size]
            source_texts = extract_source_texts(batch)
            gold_targets = extract_classification_targets(batch)
            raw_bucket_values = batch.get("bucket")
            raw_dataset_values = batch.get("dataset")
            raw_source_id_values = batch.get("source_id")

            for row_idx, (source_text, gold_raw) in enumerate(zip(source_texts, gold_targets)):
                bucket = normalize_bucket(raw_bucket_values[row_idx]) if raw_bucket_values is not None else "n/a"
                dataset_name = str(raw_dataset_values[row_idx]) if raw_dataset_values is not None else None
                source_id = raw_source_id_values[row_idx] if raw_source_id_values is not None else None
                future = pool.submit(
                    predict_one,
                    config,
                    row_id=int(start + row_idx),
                    source_text=source_text,
                    gold_raw=gold_raw,
                    dataset_name=dataset_name,
                    bucket=bucket,
                    source_id=source_id,
                    candidates=classification_candidates,
                    model_id=args.model_id,
                    max_retries=args.max_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                )
                future_map[future] = int(start + row_idx)

        for future in as_completed(future_map):
            record, metric_row = future.result()
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            metric_rows.append(metric_row)
            completed += 1
            if completed % progress_interval == 0 or completed == len(future_map):
                print(f"[progress] {completed}/{len(future_map)} classification rows complete")

    summary: dict[str, Any] = {
        "task": "classification",
        "eval_config": {
            "dataset_path": args.dataset_path.as_posix(),
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "concurrency": args.concurrency,
            "max_retries": args.max_retries,
            "retry_backoff_seconds": args.retry_backoff_seconds,
            "env_file": args.env_file.as_posix(),
            "request_timeout": args.request_timeout,
            "progress_interval": progress_interval,
            "classification_candidates": args.classification_candidates,
        },
        "classification_mode": "iaedu-generate",
        "classification_candidates": classification_candidates,
        "classification_candidate_metadata": build_candidate_metadata(classification_candidates),
        "filtered_out_counts": filtered_out_counts or {},
        **build_classification_summary(metric_rows),
    }

    per_bucket: dict[str, Any] = {}
    for bucket in FRMT_BUCKETS:
        bucket_rows = [row for row in metric_rows if row.get("bucket") == bucket]
        if not bucket_rows:
            continue
        per_bucket[bucket] = build_classification_summary(bucket_rows)
    if per_bucket:
        summary["available_buckets"] = sorted(per_bucket)
        summary["per_bucket"] = per_bucket

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Saved predictions: {pred_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
