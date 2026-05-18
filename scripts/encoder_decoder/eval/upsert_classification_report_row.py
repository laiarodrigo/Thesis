#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = ["model", "eval_set", "accuracy", "f1_macro", "n", "predictions_path"]
REPO_ROOT = Path(__file__).resolve().parents[3]
PT_LABELS = ("pt-br", "pt-pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upsert one classification-eval row into a report CSV from a completed summary JSON."
        )
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--predictions-path", type=Path, default=None)
    return parser.parse_args()


def format_metric(value: object) -> str:
    return f"{float(value):.6f}"


def normalize_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def derive_predictions_path(summary_path: Path) -> Path:
    name = summary_path.name
    if name.endswith("_summary.json"):
        return summary_path.with_name(name.replace("_summary.json", "_predictions.jsonl"))
    raise ValueError(
        "--predictions-path was not provided and summary filename does not end with '_summary.json': "
        f"{summary_path}"
    )


def load_summary(summary_path: Path) -> dict[str, object]:
    with summary_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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


def compute_metrics_from_predictions(predictions_path: Path) -> tuple[float, float, int]:
    gold_labels: list[str] = []
    pred_labels: list[str] = []

    with predictions_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold, pred = extract_labels(row)
            if gold is None or pred is None:
                raise ValueError(
                    f"{predictions_path}:{line_no} is missing recognizable gold/pred labels."
                )
            gold_labels.append(gold)
            pred_labels.append(pred)

    n = len(gold_labels)
    if n == 0:
        return 0.0, 0.0, 0

    accuracy = sum(gold == pred for gold, pred in zip(gold_labels, pred_labels)) / n
    f1_scores: list[float] = []
    for label in PT_LABELS:
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
    return accuracy, sum(f1_scores) / len(f1_scores), n


def build_row(
    *,
    model_name: str,
    eval_set: str,
    summary: dict[str, object],
    predictions_path: Path,
) -> dict[str, str]:
    del summary
    accuracy, f1_macro, n = compute_metrics_from_predictions(predictions_path)

    return {
        "model": model_name,
        "eval_set": eval_set,
        "accuracy": format_metric(accuracy),
        "f1_macro": format_metric(f1_macro),
        "n": str(int(n)),
        "predictions_path": normalize_repo_path(predictions_path),
    }


def read_existing_rows(report_path: Path) -> list[dict[str, str]]:
    if not report_path.exists():
        return []
    with report_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def upsert_rows(
    rows: list[dict[str, str]],
    new_row: dict[str, str],
) -> tuple[list[dict[str, str]], bool]:
    key = (new_row["model"], new_row["eval_set"])
    out_rows: list[dict[str, str]] = []
    replaced = False
    for row in rows:
        row_key = (row.get("model", ""), row.get("eval_set", ""))
        if row_key == key:
            if not replaced:
                out_rows.append(new_row)
                replaced = True
            continue
        out_rows.append({field: row.get(field, "") for field in FIELDNAMES})
    if not replaced:
        out_rows.append(new_row)
    return out_rows, replaced


def write_rows(report_path: Path, rows: list[dict[str, str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary_path = args.summary_path.resolve()
    predictions_path = (
        args.predictions_path.resolve()
        if args.predictions_path is not None
        else derive_predictions_path(summary_path)
    )

    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    summary = load_summary(summary_path)
    new_row = build_row(
        model_name=args.model_name,
        eval_set=args.eval_set,
        summary=summary,
        predictions_path=predictions_path,
    )
    rows = read_existing_rows(args.report_path)
    updated_rows, replaced = upsert_rows(rows, new_row)
    write_rows(args.report_path, updated_rows)

    action = "replaced" if replaced else "appended"
    print(
        f"{action} row for model={args.model_name!r} eval_set={args.eval_set!r} "
        f"in {args.report_path}"
    )
    print(json.dumps(new_row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
