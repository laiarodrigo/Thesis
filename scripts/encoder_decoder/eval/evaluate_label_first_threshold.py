#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset

from scripts.encoder_decoder.eval.evaluate_encdec import (  # noqa: E402
    ClsStats,
    load_model_and_tokenizer,
    normalize_generation_text,
    normalize_label,
    score_classification_candidates_batch,
    strip_decoder_label_prefix,
    strip_encoder_task_prefix,
)


ENCODER_TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Setup 2 translation models with a BR/PT first-token margin threshold. "
            "If the BR/PT margin is below the threshold, the prediction is mapped to 'equal'."
        )
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--br-token", default="BR")
    parser.add_argument("--pt-token", default="PT")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional single threshold to apply to the normalized BR/PT probability margin.",
    )
    parser.add_argument(
        "--threshold-grid",
        default="0.00,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50",
        help="Comma-separated threshold sweep over normalized BR/PT probability margins.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "encoder_decoder",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=15,
        help="How many example rows to include in the summary previews.",
    )
    return parser.parse_args()


def parse_threshold_grid(raw: str) -> list[float]:
    values: list[float] = []
    for piece in (raw or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = float(piece)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Thresholds must be in [0, 1], got {value!r}")
        values.append(value)
    if not values:
        raise ValueError("Threshold grid is empty.")
    return sorted(set(values))


def infer_source_variant_label_norm(row: dict[str, Any]) -> str | None:
    source_variant_label = normalize_label(str(row.get("source_variant_label") or ""))
    if source_variant_label in {"pt-br", "pt-pt"}:
        return source_variant_label

    for key in ("direction", "task"):
        value = normalize_generation_text(str(row.get(key) or "")).casefold()
        if value in {"translate_br2pt", "br2pt"}:
            return "pt-br"
        if value in {"translate_pt2br", "pt2br"}:
            return "pt-pt"

    raw_input = str(row.get("input_text") or row.get("source_text") or "")
    match = ENCODER_TASK_PREFIX_RE.match(raw_input)
    if match:
        prefix = match.group(1).strip().casefold()
        if prefix == "br-pt":
            return "pt-br"
        if prefix == "pt-br":
            return "pt-pt"

    target_text = str(row.get("target_text") or row.get("gold") or "")
    target_label = normalize_label(target_text)
    if target_label in {"pt-br", "pt-pt"}:
        return target_label
    return None


def row_source_text(row: dict[str, Any]) -> str:
    source = str(row.get("source_text") or row.get("input_text") or "")
    return strip_encoder_task_prefix(source)


def row_target_text(row: dict[str, Any]) -> str:
    target = str(row.get("target_text") or row.get("gold") or "")
    return strip_decoder_label_prefix(target)


def is_equal_row(row: dict[str, Any]) -> bool:
    if row.get("is_equal_pair") is not None:
        return bool(row["is_equal_pair"])
    return row_source_text(row) == row_target_text(row)


def gold_label_norm(row: dict[str, Any]) -> str | None:
    if is_equal_row(row):
        return "equal"
    return infer_source_variant_label_norm(row)


def normalized_candidate_probs(scores: dict[str, float], br_token: str, pt_token: str) -> tuple[float, float]:
    br_score = float(scores[br_token])
    pt_score = float(scores[pt_token])
    max_score = max(br_score, pt_score)
    br_exp = math.exp(br_score - max_score)
    pt_exp = math.exp(pt_score - max_score)
    denom = br_exp + pt_exp
    if denom <= 0:
        return 0.5, 0.5
    return br_exp / denom, pt_exp / denom


def predict_label_norm(
    *,
    br_prob: float,
    pt_prob: float,
    threshold: float,
) -> str:
    if abs(br_prob - pt_prob) <= threshold:
        return "equal"
    return "pt-br" if br_prob > pt_prob else "pt-pt"


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    sorted_vals = sorted(float(v) for v in values)

    def quantile(q: float) -> float:
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        pos = (len(sorted_vals) - 1) * q
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return sorted_vals[lower]
        frac = pos - lower
        return sorted_vals[lower] * (1.0 - frac) + sorted_vals[upper] * frac

    return {
        "count": len(sorted_vals),
        "min": sorted_vals[0],
        "p05": quantile(0.05),
        "p10": quantile(0.10),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "max": sorted_vals[-1],
        "mean": sum(sorted_vals) / len(sorted_vals),
    }


def summarize_margins(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "overall": summarize_numeric([float(record["prob_margin"]) for record in records]),
        "by_gold_label": {},
    }
    for label in ("equal", "pt-br", "pt-pt"):
        out["by_gold_label"][label] = summarize_numeric(
            [float(record["prob_margin"]) for record in records if record["gold_norm"] == label]
        )
    return out


def shorten_text(text: str, max_chars: int = 220) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def preview_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "gold_norm": record.get("gold_norm"),
        "argmax_pred_norm": record.get("argmax_pred_norm"),
        "br_prob": float(record["br_prob"]),
        "pt_prob": float(record["pt_prob"]),
        "prob_margin": float(record["prob_margin"]),
        "input_text": shorten_text(str(record.get("input_text") or "")),
        "target_text": shorten_text(str(record.get("target_text") or "")),
    }


def build_margin_previews(records: list[dict[str, Any]], preview_count: int) -> dict[str, list[dict[str, Any]]]:
    if preview_count <= 0:
        return {}

    smallest_overall = sorted(records, key=lambda item: (item["prob_margin"], str(item.get("id"))))[:preview_count]
    equal_records = [record for record in records if record["gold_norm"] == "equal"]
    smallest_equal = sorted(equal_records, key=lambda item: (item["prob_margin"], str(item.get("id"))))[:preview_count]
    largest_equal = sorted(
        equal_records,
        key=lambda item: (-float(item["prob_margin"]), str(item.get("id"))),
    )[:preview_count]

    return {
        "smallest_margin_overall": [preview_record(record) for record in smallest_overall],
        "smallest_margin_gold_equal": [preview_record(record) for record in smallest_equal],
        "largest_margin_gold_equal": [preview_record(record) for record in largest_equal],
    }


def summarize_threshold(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    stats = ClsStats()
    pred_equal_count = 0
    for record in records:
        pred = predict_label_norm(
            br_prob=record["br_prob"],
            pt_prob=record["pt_prob"],
            threshold=threshold,
        )
        if pred == "equal":
            pred_equal_count += 1
        stats.update(record["gold_norm"], pred)
    report = stats.report()
    report["threshold"] = threshold
    report["pred_equal_count"] = pred_equal_count
    report["pred_equal_rate"] = pred_equal_count / len(records) if records else 0.0
    return report


def main() -> None:
    args = parse_args()
    threshold_grid = parse_threshold_grid(args.threshold_grid)
    if args.threshold is not None and args.threshold not in threshold_grid:
        threshold_grid = sorted(set(threshold_grid + [args.threshold]))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model_and_tokenizer(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        tokenizer_path=args.tokenizer_path,
    )

    ds = load_dataset("json", data_files={"eval": args.dataset_path.as_posix()})["eval"]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = args.output_dir / f"{run_id}_label_first_threshold_predictions.jsonl"
    summary_path = args.output_dir / f"{run_id}_label_first_threshold_summary.json"

    records: list[dict[str, Any]] = []
    skipped = 0
    with pred_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(ds), args.batch_size):
            batch = ds[start : start + args.batch_size]
            rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
            inputs = [row_source_text(row) for row in rows]
            pred_infos = score_classification_candidates_batch(
                model,
                tok,
                inputs=inputs,
                candidates=[args.br_token, args.pt_token],
                max_source_length=args.max_source_length,
                mode="score-first-token",
            )

            for row_idx, (row, source_text, pred_info) in enumerate(zip(rows, inputs, pred_infos)):
                gold_norm = gold_label_norm(row)
                if gold_norm is None:
                    skipped += 1
                    continue
                scores = pred_info["scores"]
                br_prob, pt_prob = normalized_candidate_probs(
                    scores,
                    br_token=args.br_token,
                    pt_token=args.pt_token,
                )
                record = {
                    "id": row.get("id", start + row_idx),
                    "source_id": row.get("source_id"),
                    "input_text": source_text,
                    "target_text": row_target_text(row),
                    "direction": row.get("direction"),
                    "dataset": row.get("dataset"),
                    "bucket": row.get("bucket"),
                    "is_equal_pair": bool(is_equal_row(row)),
                    "gold_norm": gold_norm,
                    "br_log_prob": float(scores[args.br_token]),
                    "pt_log_prob": float(scores[args.pt_token]),
                    "br_prob": br_prob,
                    "pt_prob": pt_prob,
                    "prob_margin": abs(br_prob - pt_prob),
                    "argmax_pred_norm": "pt-br" if br_prob >= pt_prob else "pt-pt",
                }
                if args.threshold is not None:
                    record["threshold_pred_norm"] = predict_label_norm(
                        br_prob=br_prob,
                        pt_prob=pt_prob,
                        threshold=float(args.threshold),
                    )
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)

    argmax_stats = ClsStats()
    for record in records:
        argmax_stats.update(record["gold_norm"], record["argmax_pred_norm"])

    threshold_sweep = [summarize_threshold(records, threshold) for threshold in threshold_grid]
    best_macro = max(threshold_sweep, key=lambda item: item["macro_f1"]) if threshold_sweep else None
    best_equal_f1 = (
        max(threshold_sweep, key=lambda item: item["per_class"]["equal"]["f1"])
        if threshold_sweep
        else None
    )
    best_equal_recall = (
        max(threshold_sweep, key=lambda item: item["per_class"]["equal"]["recall"])
        if threshold_sweep
        else None
    )

    summary: dict[str, Any] = {
        "task": "translation_label_first_threshold",
        "dataset_path": args.dataset_path.as_posix(),
        "model_id": args.model_id,
        "adapter_dir": args.adapter_dir.as_posix() if args.adapter_dir else None,
        "n": len(records),
        "skipped_rows": skipped,
        "br_token": args.br_token,
        "pt_token": args.pt_token,
        "threshold": args.threshold,
        "threshold_grid": threshold_grid,
        "gold_equal_count": sum(1 for record in records if record["gold_norm"] == "equal"),
        "gold_pt_br_count": sum(1 for record in records if record["gold_norm"] == "pt-br"),
        "gold_pt_pt_count": sum(1 for record in records if record["gold_norm"] == "pt-pt"),
        "argmax_report": argmax_stats.report(),
        "margin_stats": summarize_margins(records),
        "margin_previews": build_margin_previews(records, preview_count=int(args.preview_count)),
        "threshold_sweep": threshold_sweep,
        "best_macro_f1_threshold": best_macro["threshold"] if best_macro else None,
        "best_macro_f1": best_macro["macro_f1"] if best_macro else None,
        "best_equal_f1_threshold": best_equal_f1["threshold"] if best_equal_f1 else None,
        "best_equal_f1": (
            best_equal_f1["per_class"]["equal"]["f1"] if best_equal_f1 else None
        ),
        "best_equal_recall_threshold": (
            best_equal_recall["threshold"] if best_equal_recall else None
        ),
        "best_equal_recall": (
            best_equal_recall["per_class"]["equal"]["recall"] if best_equal_recall else None
        ),
    }
    if args.threshold is not None:
        summary["selected_threshold_report"] = summarize_threshold(records, float(args.threshold))

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Saved predictions: {pred_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
