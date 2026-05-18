#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sacrebleu import corpus_bleu, sentence_bleu

try:
    from metrics_utils import corpus_ter, sentence_ter
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.metrics_utils import corpus_ter, sentence_ter

ENCODER_TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute copy diagnostics from translation prediction JSONL files."
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=Path,
        required=True,
        help="One or more *_translation_predictions.jsonl files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save computed metrics as JSON.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ").replace("\r", " ")).strip()


def strip_encoder_task_prefix(text: str) -> str:
    raw = text or ""
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_text(raw)


def model_vs_copy_score_0_100(model_bleu: float, copy_bleu: float) -> float | None:
    denom = model_bleu + copy_bleu
    if denom <= 0:
        return None
    return 100.0 * model_bleu / denom


def lower_is_better_vs_copy_score_0_100(model_error: float, copy_error: float) -> float | None:
    denom = model_error + copy_error
    if denom == 0:
        return 50.0
    return 100.0 * copy_error / denom


def compute_metrics(predictions_path: Path) -> dict:
    refs: list[str] = []
    hyps: list[str] = []
    copy_hyps: list[str] = []
    copy_better_or_equal_count = 0
    model_better_or_equal_count = 0
    copy_ter_better_or_equal_count = 0
    model_ter_better_or_equal_count = 0
    exact_input_copy_count = 0

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

            src = strip_encoder_task_prefix(row["input_text"])
            gold = normalize_text(row["gold"])
            pred = normalize_text(row["pred_raw"])

            refs.append(gold)
            hyps.append(pred)
            copy_hyps.append(src)

            pred_bleu = sentence_bleu(pred, [gold]).score
            copy_bleu = sentence_bleu(src, [gold]).score
            pred_ter = sentence_ter(pred, gold)
            copy_ter = sentence_ter(src, gold)
            if copy_bleu >= pred_bleu:
                copy_better_or_equal_count += 1
            if pred_bleu >= copy_bleu:
                model_better_or_equal_count += 1
            if copy_ter <= pred_ter:
                copy_ter_better_or_equal_count += 1
            if pred_ter <= copy_ter:
                model_ter_better_or_equal_count += 1
            if pred == src:
                exact_input_copy_count += 1

    model_bleu = corpus_bleu(hyps, [refs]).score if refs else 0.0
    copy_baseline_bleu = corpus_bleu(copy_hyps, [refs]).score if refs else 0.0
    model_ter = corpus_ter(hyps, refs) if refs else 0.0
    copy_baseline_ter = corpus_ter(copy_hyps, refs) if refs else 0.0
    model_score_0_100 = model_vs_copy_score_0_100(model_bleu, copy_baseline_bleu)
    model_ter_score_0_100 = lower_is_better_vs_copy_score_0_100(model_ter, copy_baseline_ter)
    copy_to_model_bleu_ratio = copy_baseline_bleu / model_bleu if model_bleu > 0 else None
    model_to_copy_bleu_ratio = model_bleu / copy_baseline_bleu if copy_baseline_bleu > 0 else None

    return {
        "predictions_path": predictions_path.as_posix(),
        "n": len(refs),
        "bleu": model_bleu,
        "copy_baseline_bleu": copy_baseline_bleu,
        "ter": model_ter,
        "copy_baseline_ter": copy_baseline_ter,
        "model_vs_copy_score_0_100": model_score_0_100,
        "model_vs_copy_ter_score_0_100": model_ter_score_0_100,
        "model_beats_copy_flag_score_gt_50": (
            bool(model_score_0_100 > 50.0) if model_score_0_100 is not None else None
        ),
        "model_beats_copy_flag_ter_score_gt_50": (
            bool(model_ter_score_0_100 > 50.0)
            if model_ter_score_0_100 is not None
            else None
        ),
        "sentence_model_beats_copy_rate_score_gt_50": (
            model_better_or_equal_count / len(refs) if refs else 0.0
        ),
        "sentence_model_beats_copy_rate_ter": (
            model_ter_better_or_equal_count / len(refs) if refs else 0.0
        ),
        "copy_to_model_bleu_ratio": copy_to_model_bleu_ratio,
        "model_to_copy_bleu_ratio": model_to_copy_bleu_ratio,
        "copying_flag_ratio_gt_1": (
            bool(copy_to_model_bleu_ratio > 1.0) if copy_to_model_bleu_ratio is not None else None
        ),
        "sentence_copy_better_or_equal_rate": (
            copy_better_or_equal_count / len(refs) if refs else 0.0
        ),
        "sentence_copy_better_or_equal_rate_ter": (
            copy_ter_better_or_equal_count / len(refs) if refs else 0.0
        ),
        "exact_input_copy_rate": exact_input_copy_count / len(refs) if refs else 0.0,
    }


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'file':56} {'BLEU':>8} {'COPY_BLEU':>10} {'TER':>8} {'COPY_TER':>10} "
        f"{'BLEU_S50':>8} {'TER_S50':>8} {'EXACT_COPY':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        bleu_score = row["model_vs_copy_score_0_100"]
        bleu_score_txt = f"{bleu_score:.2f}" if bleu_score is not None else "n/a"
        ter_score = row["model_vs_copy_ter_score_0_100"]
        ter_score_txt = f"{ter_score:.2f}" if ter_score is not None else "n/a"
        print(
            f"{Path(row['predictions_path']).name:56} "
            f"{row['bleu']:8.3f} {row['copy_baseline_bleu']:10.3f} "
            f"{row['ter']:8.3f} {row['copy_baseline_ter']:10.3f} "
            f"{bleu_score_txt:>8} {ter_score_txt:>8} "
            f"{row['exact_input_copy_rate']:11.3f}"
        )


def main() -> None:
    args = parse_args()

    rows = []
    for path in args.predictions:
        if not path.exists():
            raise FileNotFoundError(f"Predictions file not found: {path}")
        rows.append(compute_metrics(path))

    print_table(rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report: {args.output_json}")


if __name__ == "__main__":
    main()
