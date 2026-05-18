#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from metrics_utils import PT_VARIANT_LABELS
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.metrics_utils import PT_VARIANT_LABELS


REPO_ROOT = Path(__file__).resolve().parents[3]
BUCKETS = ("random", "entity", "lexical")
FIELDNAMES = [
    "model",
    "eval_set",
    "n",
    "accuracy",
    "f1_macro",
    "bucket_random",
    "bucket_entity",
    "bucket_lexical",
    "accuracy_random",
    "accuracy_entity",
    "accuracy_lexical",
    "f1_macro_random",
    "f1_macro_entity",
    "f1_macro_lexical",
    "predictions_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bucket-ready classification report CSV without TER."
        )
    )
    parser.add_argument(
        "--input-report",
        type=Path,
        default=Path("eval_results/encoder_decoder/classification_scores_report_frmt.csv"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("eval_results/encoder_decoder/classification_bucket_scores_report_frmt.csv"),
    )
    return parser.parse_args()


def normalize_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_predictions_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def normalize_label(text: object) -> str | None:
    raw = str(text or "").strip().lower()
    if not raw:
        return None
    first = raw.split(" ", 1)[0].strip(",:;.-_")
    if first == "br":
        return "pt-br"
    if first == "pt":
        return "pt-pt"
    if "equal" in raw or "shared" in raw or first == "igual" or raw == "same":
        return "equal"
    if "pt-br" in raw or "ptbr" in raw or "brasil" in raw:
        return "pt-br"
    if "pt-pt" in raw or "ptpt" in raw or "europeu" in raw or "portugal" in raw:
        return "pt-pt"
    return None


def normalize_bucket(raw_bucket: object) -> str:
    text = " ".join(str(raw_bucket or "").split()).lower()
    if text in {"rand", "random"}:
        return "random"
    if text in {"entity", "entities"}:
        return "entity"
    if text in {"lexical", "lex"}:
        return "lexical"
    if not text:
        return "n/a"
    return text


def extract_labels(row: dict[str, object]) -> tuple[str | None, str | None]:
    gold = (
        row.get("gold_norm")
        or row.get("gold_source_variant_norm")
        or row.get("gold_label")
        or row.get("label")
        or row.get("gold")
    )
    pred = (
        row.get("pred_norm")
        or row.get("pred_source_variant_norm")
        or row.get("pred_label")
        or row.get("pred")
        or row.get("pred_raw")
    )
    return normalize_label(gold), normalize_label(pred)


def compute_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    gold_labels = [row["gold_norm"] for row in rows]
    pred_labels = [row["pred_norm"] for row in rows]
    n = len(rows)
    if n == 0:
        return {
            "n": 0.0,
            "accuracy": 0.0,
            "f1_macro": 0.0,
        }

    accuracy = sum(gold == pred for gold, pred in zip(gold_labels, pred_labels)) / n
    f1_scores: list[float] = []
    for label in PT_VARIANT_LABELS:
        tp = sum(gold == label and pred == label for gold, pred in zip(gold_labels, pred_labels))
        fp = sum(gold != label and pred == label for gold, pred in zip(gold_labels, pred_labels))
        fn = sum(gold == label and pred != label for gold, pred in zip(gold_labels, pred_labels))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_scores.append(
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )

    return {
        "n": float(n),
        "accuracy": accuracy,
        "f1_macro": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }


def load_prediction_rows(predictions_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with predictions_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            gold_norm, pred_norm = extract_labels(raw)
            if gold_norm is None or pred_norm is None:
                raise ValueError(
                    f"{predictions_path}:{line_no} is missing recognizable gold/pred labels."
                )
            rows.append(
                {
                    "gold_norm": gold_norm,
                    "pred_norm": pred_norm,
                    "bucket": normalize_bucket(raw.get("bucket")),
                }
            )
    return rows


def format_metric(value: object) -> str:
    return f"{float(value):.6f}"


def build_output_row(input_row: dict[str, str]) -> dict[str, str]:
    predictions_path = resolve_predictions_path(input_row["predictions_path"])
    rows = load_prediction_rows(predictions_path)
    overall = compute_metrics(rows)

    out_row = {
        "model": input_row["model"],
        "eval_set": input_row["eval_set"],
        "n": str(int(overall["n"])),
        "accuracy": format_metric(overall["accuracy"]),
        "f1_macro": format_metric(overall["f1_macro"]),
        "predictions_path": normalize_repo_path(predictions_path),
    }

    per_bucket = {
        bucket: compute_metrics([row for row in rows if row.get("bucket") == bucket])
        for bucket in BUCKETS
    }
    for bucket in BUCKETS:
        metrics = per_bucket[bucket]
        out_row[f"bucket_{bucket}"] = str(int(metrics["n"]))
        out_row[f"accuracy_{bucket}"] = (
            format_metric(metrics["accuracy"]) if metrics["n"] else ""
        )
        out_row[f"f1_macro_{bucket}"] = (
            format_metric(metrics["f1_macro"]) if metrics["n"] else ""
        )
    return out_row


def write_rows(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.input_report.open("r", encoding="utf-8", newline="") as fh:
        input_rows = list(csv.DictReader(fh))
    output_rows = [build_output_row(row) for row in input_rows]
    write_rows(args.output_report, output_rows)
    print(f"Wrote {len(output_rows)} rows -> {args.output_report}")


if __name__ == "__main__":
    main()
