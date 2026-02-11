#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Any

import optuna
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Optuna study results into CSV files.")
    parser.add_argument(
        "--space-files",
        nargs="+",
        default=[
            "hpo/encoder_decoder/translation_search_space.yaml",
            "hpo/encoder_decoder/classification_search_space.yaml",
        ],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--combined-out", default="hpo/encoder_decoder/best_trials.csv")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def trial_sort_value(value: float | None) -> float:
    if value is None:
        return float("inf")
    return float(value)


def main() -> None:
    args = parse_args()
    combined_rows: list[dict[str, Any]] = []

    for path_str in args.space_files:
        space_path = Path(path_str)
        cfg = load_yaml(space_path)
        study_name = cfg["study"]["name"]
        storage = cfg["study"]["storage"]
        metric_key = cfg["study"].get("metric", "eval_loss")

        study = optuna.load_study(study_name=study_name, storage=storage)
        rows: list[dict[str, Any]] = []
        for trial in study.trials:
            row = {
                "study": study_name,
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "metric_key": metric_key,
            }
            for key, val in trial.params.items():
                row[f"param::{key}"] = val
            for key, val in trial.user_attrs.items():
                row[f"attr::{key}"] = val
            rows.append(row)

        rows_sorted = sorted(rows, key=lambda r: trial_sort_value(r.get("value")))
        out_path = space_path.with_name(f"{study_name}_results.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = sorted({k for row in rows_sorted for k in row.keys()})
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)

        print(f"Wrote {len(rows_sorted)} rows to {out_path}")
        combined_rows.extend(rows_sorted[: args.top_k])

    combined_sorted = sorted(combined_rows, key=lambda r: trial_sort_value(r.get("value")))
    combined_path = Path(args.combined_out)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_fields = sorted({k for row in combined_sorted for k in row.keys()})
    with combined_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=combined_fields)
        writer.writeheader()
        writer.writerows(combined_sorted)

    print(f"Wrote combined top-{args.top_k} summary to {combined_path}")


if __name__ == "__main__":
    main()
