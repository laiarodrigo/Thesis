#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

from sacrebleu import corpus_bleu, sentence_bleu

try:
    from metrics_utils import (
        corpus_ter,
        sentence_ter,
    )
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.metrics_utils import (
        corpus_ter,
        sentence_ter,
    )

WORST_BLEU_K = 10
FRMT_BUCKETS = ("random", "entity", "lexical")
ENCODER_TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)
DECODER_LABEL_PREFIX_RE = re.compile(r"^\s*(BR|PT|pt-br|pt-pt)\b[:\-\s]*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild translation summary JSON from an existing predictions JSONL file."
    )
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to write the rebuilt summary. Defaults to <predictions>_summary.json.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = (text or "").replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


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


def normalize_label(text: str) -> Optional[str]:
    t = normalize_text(text or "").lower()
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


def canonicalize_translation_direction(raw_direction: object, input_text: object) -> Optional[str]:
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


def build_translation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    refs: list[str] = []
    hyps: list[str] = []
    copy_hyps: list[str] = []
    worst_bleu_examples: list[dict[str, Any]] = []
    copy_better_or_equal_count = 0
    model_better_count = 0
    copy_ter_better_or_equal_count = 0
    model_ter_better_count = 0
    exact_input_copy_count = 0
    translation_label_total = 0
    translation_label_parsed = 0
    translation_label_correct = 0

    for row in rows:
        gold_clean = row["gold"]
        pred_clean = row["pred"]
        src_clean = row["src"]
        refs.append(gold_clean)
        hyps.append(pred_clean)
        copy_hyps.append(src_clean)

        ex_bleu = sentence_bleu(pred_clean, [gold_clean]).score
        ex_copy_bleu = sentence_bleu(src_clean, [gold_clean]).score
        ex_ter = sentence_ter(pred_clean, gold_clean)
        ex_copy_ter = sentence_ter(src_clean, gold_clean)

        if ex_copy_bleu >= ex_bleu:
            copy_better_or_equal_count += 1
        if ex_bleu >= ex_copy_bleu:
            model_better_count += 1
        if ex_copy_ter <= ex_ter:
            copy_ter_better_or_equal_count += 1
        if ex_ter <= ex_copy_ter:
            model_ter_better_count += 1
        if pred_clean == src_clean:
            exact_input_copy_count += 1

        gold_label = row.get("gold_label")
        pred_label = row.get("pred_label")
        if gold_label is not None:
            translation_label_total += 1
            if pred_label is not None:
                translation_label_parsed += 1
                if pred_label == gold_label:
                    translation_label_correct += 1

        worst_bleu_examples.append(
            {
                "id": row["id"],
                "sentence_bleu": ex_bleu,
            }
        )

    worst_bleu_examples.sort(key=lambda x: x["sentence_bleu"])
    worst_bleu_examples = worst_bleu_examples[:WORST_BLEU_K]

    model_bleu = corpus_bleu(hyps, [refs]).score if refs else float("nan")
    copy_baseline_bleu = corpus_bleu(copy_hyps, [refs]).score if refs else float("nan")
    copy_to_model_bleu_ratio = copy_baseline_bleu / model_bleu if model_bleu > 0 else None
    model_to_copy_bleu_ratio = model_bleu / copy_baseline_bleu if copy_baseline_bleu > 0 else None
    model_score_0_100 = model_vs_copy_score_0_100(model_bleu, copy_baseline_bleu)
    model_ter = corpus_ter(hyps, refs) if refs else float("nan")
    copy_baseline_ter = corpus_ter(copy_hyps, refs) if refs else float("nan")
    model_ter_score_0_100 = lower_is_better_vs_copy_score_0_100(model_ter, copy_baseline_ter)

    return {
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
            bool(model_ter_score_0_100 > 50.0) if model_ter_score_0_100 is not None else None
        ),
        "sentence_model_beats_copy_rate_score_gt_50": (
            model_better_count / len(refs) if refs else 0.0
        ),
        "sentence_model_beats_copy_rate_ter": (
            model_ter_better_count / len(refs) if refs else 0.0
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
        "translation_source_variant_label_parse_rate": (
            translation_label_parsed / translation_label_total if translation_label_total else None
        ),
        "translation_source_variant_label_accuracy": (
            translation_label_correct / translation_label_total if translation_label_total else None
        ),
        "worst_sentence_bleu_examples": worst_bleu_examples,
    }


def load_translation_rows(predictions_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = json.loads(line)
            src = strip_encoder_task_prefix(raw["input_text"])
            gold_clean = strip_decoder_label_prefix(raw["gold"])
            pred_clean = strip_decoder_label_prefix(raw["pred_raw"])
            gold_label = (
                raw.get("gold_source_variant_norm")
                or normalize_label(raw.get("gold_with_label") or raw["gold"])
            )
            pred_label = (
                raw.get("pred_source_variant_norm")
                or normalize_label(raw.get("pred_with_label") or raw["pred_raw"])
            )
            rows.append(
                {
                    "id": raw.get("id"),
                    "source_id": raw.get("source_id"),
                    "direction": canonicalize_translation_direction(
                        raw.get("direction"),
                        raw.get("input_text"),
                    ),
                    "src": src,
                    "gold": gold_clean,
                    "pred": pred_clean,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "bucket": normalize_bucket(raw.get("bucket")),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    rows = load_translation_rows(args.predictions_path)
    summary = {
        "task": "translation",
        **build_translation_summary(rows),
    }
    per_direction: dict[str, Any] = {}
    for direction in sorted({str(row["direction"]) for row in rows if row.get("direction")}):
        per_direction[direction] = build_translation_summary(
            [row for row in rows if row.get("direction") == direction]
        )
    if per_direction:
        summary["available_directions"] = sorted(per_direction)
        summary["per_direction"] = per_direction
    per_bucket: dict[str, Any] = {}
    for bucket in FRMT_BUCKETS:
        bucket_rows = [row for row in rows if row.get("bucket") == bucket]
        if not bucket_rows:
            continue
        per_bucket[bucket] = build_translation_summary(bucket_rows)
    if per_bucket:
        summary["available_buckets"] = sorted(per_bucket)
        summary["per_bucket"] = per_bucket

    output_path = args.output_path
    if output_path is None:
        name = args.predictions_path.name
        if name.endswith("_predictions.jsonl"):
            stem = name[: -len("_predictions.jsonl")] + "_summary.json"
        elif name.endswith(".jsonl"):
            stem = name[:-6] + "_summary.json"
        else:
            stem = name + "_summary.json"
        output_path = args.predictions_path.with_name(stem)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Saved summary: {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
