#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

try:
    from rebuild_translation_summary import build_translation_summary as build_translation_metrics
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.rebuild_translation_summary import (
        build_translation_summary as build_translation_metrics,
    )


FIELDNAMES = [
    "model",
    "eval_set",
    "direction",
    "n",
    "bleu",
    "model_vs_copy_score_0_100",
    "sentence_model_beats_copy_rate_score_gt_50",
    "ter",
    "copy_baseline_ter",
    "model_vs_copy_ter_score_0_100",
    "sentence_model_beats_copy_rate_ter",
    "sentence_copy_better_or_equal_rate_ter",
    "predictions_path",
]
TWO_DECIMAL_FIELDS = {"bleu", "model_vs_copy_score_0_100"}
THREE_DECIMAL_FIELDS = {"sentence_model_beats_copy_rate_score_gt_50"}
SIX_DECIMAL_FIELDS = {
    "ter",
    "copy_baseline_ter",
    "model_vs_copy_ter_score_0_100",
    "sentence_model_beats_copy_rate_ter",
    "sentence_copy_better_or_equal_rate_ter",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
ENCODER_TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)
DECODER_LABEL_PREFIX_RE = re.compile(r"^\s*(BR|PT|pt-br|pt-pt)\b[:\-\s]*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upsert translation-eval report rows from a completed summary JSON. "
            "One overall row and one row per translation direction are emitted."
        )
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--predictions-path", type=Path, default=None)
    return parser.parse_args()


def format_metric(field: str, value: object) -> str:
    number = float(value)
    if field in TWO_DECIMAL_FIELDS:
        return f"{number:.2f}"
    if field in THREE_DECIMAL_FIELDS:
        return f"{number:.3f}"
    if field in SIX_DECIMAL_FIELDS:
        return f"{number:.6f}"
    raise KeyError(f"Unsupported metric field: {field}")


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


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").replace("\r", " ").split())


def strip_encoder_task_prefix(text: str) -> str:
    raw = text or ""
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_text(raw)


def strip_decoder_label_prefix(text: str) -> str:
    raw = normalize_text(text or "")
    match = DECODER_LABEL_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_text(raw)


def canonicalize_translation_direction(raw_direction: object, input_text: object) -> str | None:
    text = normalize_text(str(raw_direction or "")).lower()
    if text in {"translate_br2pt", "br2pt", "br-pt", "<br-pt>"}:
        return "br2pt"
    if text in {"translate_pt2br", "pt2br", "pt-br", "<pt-br>"}:
        return "pt2br"

    raw = str(input_text or "")
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if not match:
        return None
    prefix = match.group(1).lower()
    if prefix == "br-pt":
        return "br2pt"
    if prefix == "pt-br":
        return "pt2br"
    return None


def load_translation_rows(predictions_path: Path) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    with predictions_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "gold" not in row or "pred_raw" not in row or "input_text" not in row:
                raise ValueError(
                    f"{predictions_path}:{line_no} missing one of required fields: "
                    "'input_text', 'gold', 'pred_raw'"
                )
            rows.append(
                {
                    "direction": canonicalize_translation_direction(
                        row.get("direction"),
                        row.get("input_text"),
                    ),
                    "src": strip_encoder_task_prefix(str(row["input_text"])),
                    "gold": strip_decoder_label_prefix(str(row["gold"])),
                    "pred": strip_decoder_label_prefix(str(row["pred_raw"])),
                }
            )
    return rows


def enrich_summary_with_ter(
    summary: dict[str, object],
    predictions_path: Path,
) -> dict[str, object]:
    translation_rows = load_translation_rows(predictions_path)
    enriched = dict(summary)
    enriched.update(build_translation_metrics(translation_rows))

    per_direction: dict[str, dict[str, object]] = {}
    discovered_directions = sorted(
        {str(row["direction"]) for row in translation_rows if row.get("direction")}
    )
    if discovered_directions:
        enriched["available_directions"] = discovered_directions
    for direction in discovered_directions:
        per_direction[direction] = build_translation_metrics(
            [row for row in translation_rows if row.get("direction") == direction]
        )
    if per_direction:
        enriched["per_direction"] = per_direction
    return enriched


def build_rows(
    *,
    model_name: str,
    eval_set: str,
    summary: dict[str, object],
    predictions_path: Path,
) -> list[dict[str, str]]:
    enriched_summary = enrich_summary_with_ter(summary, predictions_path)

    def build_one(direction: str, metrics: dict[str, object]) -> dict[str, str]:
        required = (
            "n",
            "bleu",
            "model_vs_copy_score_0_100",
            "sentence_model_beats_copy_rate_score_gt_50",
            "ter",
            "copy_baseline_ter",
            "model_vs_copy_ter_score_0_100",
            "sentence_model_beats_copy_rate_ter",
            "sentence_copy_better_or_equal_rate_ter",
        )
        missing = [field for field in required if metrics.get(field) is None]
        if missing:
            raise KeyError(
                "Summary JSON is missing required translation metrics for "
                f"direction={direction!r}: {', '.join(missing)}. "
                f"Available keys: {sorted(metrics)}"
            )

        row = {
            "model": model_name,
            "eval_set": eval_set,
            "direction": direction,
            "n": str(int(metrics["n"])),
            "predictions_path": normalize_repo_path(predictions_path),
        }
        for field in required:
            if field == "n":
                continue
            row[field] = format_metric(field, metrics[field])
        return row

    rows = [build_one("overall", enriched_summary)]
    for direction in sorted((enriched_summary.get("per_direction") or {}).keys()):
        direction_metrics = enriched_summary["per_direction"][direction]
        rows.append(build_one(str(direction), direction_metrics))
    return rows


def read_existing_rows(report_path: Path) -> list[dict[str, str]]:
    if not report_path.exists():
        return []
    with report_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def upsert_rows(
    rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    replacements = {(row["model"], row["eval_set"], row["direction"]): row for row in new_rows}
    out_rows: list[dict[str, str]] = []
    replaced = 0

    for row in rows:
        direction = row.get("direction", "overall")
        row_key = (row.get("model", ""), row.get("eval_set", ""), direction)
        replacement = replacements.pop(row_key, None)
        if replacement is not None:
            out_rows.append(replacement)
            replaced += 1
        else:
            preserved = {field: row.get(field, "") for field in FIELDNAMES}
            if not preserved["direction"]:
                preserved["direction"] = "overall"
            out_rows.append(preserved)

    out_rows.extend(replacements.values())
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
    new_rows = build_rows(
        model_name=args.model_name,
        eval_set=args.eval_set,
        summary=summary,
        predictions_path=predictions_path,
    )
    rows = read_existing_rows(args.report_path)
    updated_rows, replaced = upsert_rows(rows, new_rows)
    write_rows(args.report_path, updated_rows)

    action = "replaced" if replaced else "appended"
    print(
        f"{action} {len(new_rows)} row(s) for model={args.model_name!r} "
        f"eval_set={args.eval_set!r} in {args.report_path}"
    )
    print(json.dumps(new_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
