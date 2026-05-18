#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    from metrics_utils import corpus_ter, sentence_ter
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.metrics_utils import corpus_ter, sentence_ter


REPO_ROOT = Path(__file__).resolve().parents[3]
DIRECTIONS = ("br2pt", "pt2br")
BUCKETS = ("random", "entity", "lexical")
BLEU_METRIC_FIELDS = (
    "bleu",
    "model_vs_copy_score_0_100",
    "sentence_model_beats_copy_rate_score_gt_50",
)
TER_METRIC_FIELDS = (
    "ter",
    "copy_baseline_ter",
    "model_vs_copy_ter_score_0_100",
    "sentence_model_beats_copy_rate_ter",
    "sentence_copy_better_or_equal_rate_ter",
)
METRIC_FIELDS = BLEU_METRIC_FIELDS + TER_METRIC_FIELDS
ENCODER_TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)
DECODER_LABEL_PREFIX_RE = re.compile(r"^\s*(BR|PT|pt-br|pt-pt)\b[:\-\s]*", flags=re.IGNORECASE)
TWO_DECIMAL_FIELDS = {
    "bleu",
    "model_vs_copy_score_0_100",
    "bleu_direction_br2pt",
    "bleu_direction_pt2br",
    "model_vs_copy_score_0_100_direction_br2pt",
    "model_vs_copy_score_0_100_direction_pt2br",
    "bleu_random",
    "bleu_entity",
    "bleu_lexical",
    "model_vs_copy_score_0_100_random",
    "model_vs_copy_score_0_100_entity",
    "model_vs_copy_score_0_100_lexical",
}
THREE_DECIMAL_FIELDS = {
    "sentence_model_beats_copy_rate_score_gt_50",
    "sentence_model_beats_copy_rate_score_gt_50_direction_br2pt",
    "sentence_model_beats_copy_rate_score_gt_50_direction_pt2br",
    "sentence_model_beats_copy_rate_score_gt_50_random",
    "sentence_model_beats_copy_rate_score_gt_50_entity",
    "sentence_model_beats_copy_rate_score_gt_50_lexical",
}
SIX_DECIMAL_FIELDS = set(METRIC_FIELDS) - {"bleu", "model_vs_copy_score_0_100", "sentence_model_beats_copy_rate_score_gt_50"}
for direction in DIRECTIONS:
    for metric in METRIC_FIELDS:
        if metric in {"bleu", "model_vs_copy_score_0_100", "sentence_model_beats_copy_rate_score_gt_50"}:
            continue
        SIX_DECIMAL_FIELDS.add(f"{metric}_direction_{direction}")
for bucket in BUCKETS:
    for metric in METRIC_FIELDS:
        if metric in {"bleu", "model_vs_copy_score_0_100", "sentence_model_beats_copy_rate_score_gt_50"}:
            continue
        SIX_DECIMAL_FIELDS.add(f"{metric}_{bucket}")

FIELDNAMES = [
    "model",
    "eval_set",
    "n",
    "direction_br2pt",
    "direction_pt2br",
    "bucket_random",
    "bucket_entity",
    "bucket_lexical",
    *METRIC_FIELDS,
]
for direction in DIRECTIONS:
    FIELDNAMES.extend(f"{metric}_direction_{direction}" for metric in METRIC_FIELDS)
for bucket in BUCKETS:
    FIELDNAMES.extend(f"{metric}_{bucket}" for metric in METRIC_FIELDS)
FIELDNAMES.append("predictions_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build TER-based translation report CSVs in the new wide schema, with "
            "direction and future bucket columns."
        )
    )
    parser.add_argument(
        "--input-frmt",
        type=Path,
        default=Path("eval_results/encoder_decoder/bleu_copy_wer_scores_report_frmt.csv"),
    )
    parser.add_argument(
        "--output-frmt",
        type=Path,
        default=Path("eval_results/encoder_decoder/bleu_copy_ter_scores_report_frmt.csv"),
    )
    parser.add_argument(
        "--input-golden",
        type=Path,
        default=Path("eval_results/encoder_decoder/bleu_copy_wer_scores_report_golden_collection.csv"),
    )
    parser.add_argument(
        "--output-golden",
        type=Path,
        default=Path("eval_results/encoder_decoder/bleu_copy_ter_scores_report_golden_collection.csv"),
    )
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


def resolve_predictions_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def derive_summary_path(predictions_path: Path) -> Path:
    name = predictions_path.name
    if name.endswith("_predictions.jsonl"):
        return predictions_path.with_name(name.replace("_predictions.jsonl", "_summary.json"))
    raise ValueError(f"Unexpected predictions filename: {predictions_path}")


def load_summary(summary_path: Path) -> dict[str, object]:
    import json

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


def normalize_bucket(raw_bucket: object) -> str:
    text = normalize_text(str(raw_bucket or "")).lower()
    if text in {"rand", "random"}:
        return "random"
    if text in {"entity", "entities"}:
        return "entity"
    if text in {"lexical", "lex"}:
        return "lexical"
    if not text:
        return "n/a"
    return text


def load_translation_rows(predictions_path: Path) -> list[dict[str, str | None]]:
    import json

    rows: list[dict[str, str | None]] = []
    with predictions_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if "gold" not in raw or "pred_raw" not in raw or "input_text" not in raw:
                raise ValueError(
                    f"{predictions_path}:{line_no} missing one of required fields: "
                    "'input_text', 'gold', 'pred_raw'"
                )
            rows.append(
                {
                    "direction": canonicalize_translation_direction(
                        raw.get("direction"),
                        raw.get("input_text"),
                    ),
                    "bucket": normalize_bucket(raw.get("bucket")),
                    "src": strip_encoder_task_prefix(str(raw["input_text"])),
                    "gold": strip_decoder_label_prefix(str(raw["gold"])),
                    "pred": strip_decoder_label_prefix(str(raw["pred_raw"])),
                }
            )
    return rows


def lower_is_better_vs_copy_score_0_100(model_error: float, copy_error: float) -> float:
    denom = model_error + copy_error
    if denom == 0:
        return 50.0
    return 100.0 * copy_error / denom


def build_ter_summary(rows: list[dict[str, str | None]]) -> dict[str, float]:
    refs = [str(row["gold"]) for row in rows]
    hyps = [str(row["pred"]) for row in rows]
    copy_hyps = [str(row["src"]) for row in rows]
    if not refs:
        return {
            "n": 0.0,
            "ter": float("nan"),
            "copy_baseline_ter": float("nan"),
            "model_vs_copy_ter_score_0_100": float("nan"),
            "sentence_model_beats_copy_rate_ter": 0.0,
            "sentence_copy_better_or_equal_rate_ter": 0.0,
        }

    model_better = 0
    copy_better_or_equal = 0
    for row in rows:
        pred_ter = sentence_ter(str(row["pred"]), str(row["gold"]))
        copy_ter = sentence_ter(str(row["src"]), str(row["gold"]))
        if pred_ter <= copy_ter:
            model_better += 1
        if copy_ter <= pred_ter:
            copy_better_or_equal += 1

    model_ter = corpus_ter(hyps, refs)
    copy_baseline_ter = corpus_ter(copy_hyps, refs)
    return {
        "n": float(len(rows)),
        "ter": model_ter,
        "copy_baseline_ter": copy_baseline_ter,
        "model_vs_copy_ter_score_0_100": lower_is_better_vs_copy_score_0_100(
            model_ter,
            copy_baseline_ter,
        ),
        "sentence_model_beats_copy_rate_ter": model_better / len(rows),
        "sentence_copy_better_or_equal_rate_ter": copy_better_or_equal / len(rows),
    }


def set_metric_fields(
    row: dict[str, str],
    *,
    metrics: dict[str, object] | None,
    suffix: str,
    metric_fields: tuple[str, ...],
) -> None:
    if not metrics:
        for metric in metric_fields:
            row[f"{metric}{suffix}"] = ""
        return
    for metric in metric_fields:
        value = metrics.get(metric)
        row[f"{metric}{suffix}"] = "" if value is None else format_metric(f"{metric}{suffix}", value)


def build_output_row(input_row: dict[str, str]) -> dict[str, str]:
    predictions_path = resolve_predictions_path(input_row["predictions_path"])
    summary_path = derive_summary_path(predictions_path)
    summary = load_summary(summary_path)
    translation_rows = load_translation_rows(predictions_path)
    per_direction = summary.get("per_direction") or {}
    per_bucket = summary.get("per_bucket") or {}
    ter_summary = build_ter_summary(translation_rows)
    ter_per_direction = {
        direction: build_ter_summary([row for row in translation_rows if row.get("direction") == direction])
        for direction in DIRECTIONS
    }
    ter_per_bucket = {
        bucket: build_ter_summary([row for row in translation_rows if row.get("bucket") == bucket])
        for bucket in BUCKETS
    }

    row = {
        "model": input_row["model"],
        "eval_set": input_row["eval_set"],
        "n": str(int(ter_summary.get("n", 0) or 0)),
        "direction_br2pt": str(int(ter_per_direction["br2pt"].get("n", 0) or 0)),
        "direction_pt2br": str(int(ter_per_direction["pt2br"].get("n", 0) or 0)),
        "bucket_random": str(int(ter_per_bucket["random"].get("n", 0) or 0)),
        "bucket_entity": str(int(ter_per_bucket["entity"].get("n", 0) or 0)),
        "bucket_lexical": str(int(ter_per_bucket["lexical"].get("n", 0) or 0)),
        "predictions_path": normalize_repo_path(predictions_path),
    }
    set_metric_fields(row, metrics=summary, suffix="", metric_fields=BLEU_METRIC_FIELDS)
    set_metric_fields(row, metrics=ter_summary, suffix="", metric_fields=TER_METRIC_FIELDS)
    for direction in DIRECTIONS:
        set_metric_fields(
            row,
            metrics=per_direction.get(direction),
            suffix=f"_direction_{direction}",
            metric_fields=BLEU_METRIC_FIELDS,
        )
        set_metric_fields(
            row,
            metrics=ter_per_direction.get(direction),
            suffix=f"_direction_{direction}",
            metric_fields=TER_METRIC_FIELDS,
        )
    for bucket in BUCKETS:
        set_metric_fields(
            row,
            metrics=per_bucket.get(bucket),
            suffix=f"_{bucket}",
            metric_fields=BLEU_METRIC_FIELDS,
        )
        set_metric_fields(
            row,
            metrics=ter_per_bucket.get(bucket),
            suffix=f"_{bucket}",
            metric_fields=TER_METRIC_FIELDS,
        )
    return row


def build_output_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as fh:
        input_rows = list(csv.DictReader(fh))
    return [build_output_row(row) for row in input_rows]


def write_rows(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    frmt_rows = build_output_rows(args.input_frmt.resolve())
    golden_rows = build_output_rows(args.input_golden.resolve())
    write_rows(args.output_frmt.resolve(), frmt_rows)
    write_rows(args.output_golden.resolve(), golden_rows)
    print(f"Wrote {len(frmt_rows)} rows -> {args.output_frmt}")
    print(f"Wrote {len(golden_rows)} rows -> {args.output_golden}")


if __name__ == "__main__":
    main()
