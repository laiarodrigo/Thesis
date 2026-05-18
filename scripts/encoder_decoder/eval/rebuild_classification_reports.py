#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from upsert_classification_report_row import (
        FIELDNAMES,
        compute_metrics_from_predictions,
        normalize_repo_path,
    )
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.upsert_classification_report_row import (
        FIELDNAMES,
        compute_metrics_from_predictions,
        normalize_repo_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild classification report CSVs from the prediction JSONL files, "
            "using binary pt-BR/pt-PT macro F1."
        )
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        action="append",
        required=True,
        help="Existing classification report CSV to rewrite in place. Pass once per CSV.",
    )
    return parser.parse_args()


def rebuild_report(report_path: Path) -> int:
    with report_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    rebuilt_rows: list[dict[str, str]] = []
    for row in rows:
        predictions_path = Path(row["predictions_path"]).resolve()
        accuracy, f1_macro, n = compute_metrics_from_predictions(predictions_path)
        rebuilt_rows.append(
            {
                "model": row["model"],
                "eval_set": row["eval_set"],
                "accuracy": f"{accuracy:.6f}",
                "f1_macro": f"{f1_macro:.6f}",
                "n": str(int(n)),
                "predictions_path": normalize_repo_path(predictions_path),
            }
        )

    with report_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rebuilt_rows)

    return len(rebuilt_rows)


def main() -> None:
    args = parse_args()
    for report_path in args.report_path:
        count = rebuild_report(report_path)
        print(f"Rebuilt {count} rows -> {report_path}")


if __name__ == "__main__":
    main()
