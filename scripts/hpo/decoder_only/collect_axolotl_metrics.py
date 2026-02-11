#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Axolotl trial metrics into a summary CSV.")
    parser.add_argument("--manifest-path", default="hpo/decoder_only/qwen3_trials_manifest.csv")
    parser.add_argument("--run-status-path", default="hpo/decoder_only/qwen3_run_status.csv")
    parser.add_argument("--output-csv", default="hpo/decoder_only/qwen3_metrics.csv")
    parser.add_argument("--metric-key", default="eval_loss")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def find_best_metric(output_dir: Path, metric_key: str) -> tuple[float | None, int | None]:
    state_files = sorted(output_dir.glob("**/trainer_state.json"))
    best_val = None
    best_step = None
    for state_path in state_files:
        try:
            with state_path.open("r", encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception:
            continue
        for entry in state.get("log_history", []):
            if metric_key in entry:
                cur = float(entry[metric_key])
                step = entry.get("step")
                if best_val is None or cur < best_val:
                    best_val = cur
                    best_step = int(step) if step is not None else None
    return best_val, best_step


def load_json(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def metric_sort_key(row: dict[str, Any]) -> tuple[int, float]:
    value = row.get("best_eval_loss")
    if value is None:
        return (1, float("inf"))
    return (0, float(value))


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    run_status_path = Path(args.run_status_path)
    output_csv = Path(args.output_csv)
    metric_key = args.metric_key

    manifest_rows = read_csv(manifest_path)
    if not manifest_rows:
        raise RuntimeError(f"No manifest rows found in {manifest_path}")

    status_rows = read_csv(run_status_path)
    status_by_trial = {row.get("trial_id"): row for row in status_rows}

    summary_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        trial_id = row["trial_id"]
        output_dir = Path(row["output_dir"])
        params = load_json(row.get("params_path", ""))
        best_metric, best_step = find_best_metric(output_dir, metric_key=metric_key)

        status_row = status_by_trial.get(trial_id, {})
        summary = {
            "trial_id": trial_id,
            "trial_name": row.get("trial_name"),
            "status": status_row.get("status", "unknown"),
            "best_eval_loss": best_metric,
            "best_eval_step": best_step,
            "output_dir": output_dir.as_posix(),
            "config_path": row.get("config_path"),
        }

        for key, val in params.items():
            summary[f"param::{key}"] = val
        summary_rows.append(summary)

    summary_rows.sort(key=metric_sort_key)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in summary_rows for k in row.keys()})

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} rows to {output_csv}")
    if summary_rows and summary_rows[0].get("best_eval_loss") is not None:
        best = summary_rows[0]
        print(
            f"Best trial: {best.get('trial_name')} "
            f"eval_loss={best.get('best_eval_loss')} "
            f"output_dir={best.get('output_dir')}"
        )


if __name__ == "__main__":
    main()
