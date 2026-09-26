#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

from sacrebleu import corpus_bleu, sentence_bleu

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "scripts" / "encoder_decoder" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from metrics_utils import ter_edit_count, word_edit_distance
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.metrics_utils import (
        ter_edit_count,
        word_edit_distance,
    )


WORST_BLEU_K = 10
FRMT_BUCKETS = ("random", "entity", "lexical")
ENCODER_TASK_PREFIX_RE = re.compile(
    r"^\s*(?:<(br-pt|pt-br|pt-pt|id|cls)>|((?:BR|PT|CLS)\b))(?:\s*:\s*|\s+)",
    flags=re.IGNORECASE,
)
DECODER_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:<(?:pt-br|pt-pt)>\s*:?\s*|(?:BR|PT|pt-br|pt-pt)\b(?:\s*:\s*|\s+))",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild decoder-only translation summary JSON from an existing predictions JSONL file."
    )
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to write the rebuilt summary. Defaults to <predictions>_summary.json.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional original eval dataset path to record in the rebuilt summary.",
    )
    return parser.parse_args()


def normalize_text(text: object) -> str:
    return " ".join(str(text or "").replace("\n", " ").replace("\r", " ").split())


def strip_encoder_task_prefix(text: object) -> str:
    raw = str(text or "")
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_text(raw)


def strip_decoder_label_prefix(text: object) -> str:
    raw = normalize_text(text)
    match = DECODER_LABEL_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_text(raw)


def normalize_label(text: object) -> Optional[str]:
    raw = normalize_text(text).lower()
    if not raw:
        return None
    first = raw.split(" ", 1)[0].strip(",:;.-_")
    if first == "br":
        return "pt-br"
    if first == "pt":
        return "pt-pt"
    if "equal" in raw or "shared" in raw or raw == "same" or first == "igual":
        return "equal"
    if "pt-br" in raw or "ptbr" in raw or "brasil" in raw:
        return "pt-br"
    if "pt-pt" in raw or "ptpt" in raw or "europeu" in raw or "portugal" in raw:
        return "pt-pt"
    return None


def canonicalize_translation_direction(raw_direction: object, input_text: object) -> Optional[str]:
    text = normalize_text(raw_direction).lower()
    if text in {"translate_br2pt", "br2pt", "br-pt", "<br-pt>"}:
        return "br2pt"
    if text in {"translate_pt2br", "pt2br", "pt-br", "<pt-br>"}:
        return "pt2br"

    raw = str(input_text or "")
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if not match:
        return None
    prefix = (match.group(1) or match.group(2)).lower()
    if prefix == "br-pt":
        return "br2pt"
    if prefix == "pt-br":
        return "pt2br"
    if prefix == "br":
        return "br2pt"
    if prefix == "pt":
        return "pt2br"
    return None


def normalize_bucket(raw_bucket: object) -> str:
    text = normalize_text(raw_bucket).lower()
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
    copy_wer_better_or_equal_count = 0
    model_wer_better_count = 0
    exact_input_copy_count = 0
    translation_label_total = 0
    translation_label_parsed = 0
    translation_label_correct = 0
    total_model_ter_edits = 0
    total_copy_ter_edits = 0
    total_ter_ref_tokens = 0
    total_model_wer_edits = 0
    total_copy_wer_edits = 0
    total_wer_ref_tokens = 0

    for row in rows:
        gold_clean = row["gold"]
        pred_clean = row["pred"]
        src_clean = row["src"]
        refs.append(gold_clean)
        hyps.append(pred_clean)
        copy_hyps.append(src_clean)

        ex_bleu = sentence_bleu(pred_clean, [gold_clean]).score
        ex_copy_bleu = sentence_bleu(src_clean, [gold_clean]).score
        ref_tokens = normalize_text(gold_clean).split()
        pred_tokens = normalize_text(pred_clean).split()
        copy_tokens = normalize_text(src_clean).split()

        model_ter_edits, model_ter_ref_len = ter_edit_count(pred_tokens, ref_tokens)
        copy_ter_edits, copy_ter_ref_len = ter_edit_count(copy_tokens, ref_tokens)
        total_model_ter_edits += model_ter_edits
        total_copy_ter_edits += copy_ter_edits
        total_ter_ref_tokens += model_ter_ref_len
        ex_ter = 0.0 if model_ter_ref_len == 0 and model_ter_edits == 0 else (
            1.0 if model_ter_ref_len == 0 else model_ter_edits / model_ter_ref_len
        )
        ex_copy_ter = 0.0 if copy_ter_ref_len == 0 and copy_ter_edits == 0 else (
            1.0 if copy_ter_ref_len == 0 else copy_ter_edits / copy_ter_ref_len
        )

        if ref_tokens:
            model_wer_edits = word_edit_distance(ref_tokens, pred_tokens)
            copy_wer_edits = word_edit_distance(ref_tokens, copy_tokens)
            total_model_wer_edits += model_wer_edits
            total_copy_wer_edits += copy_wer_edits
            total_wer_ref_tokens += len(ref_tokens)
            ex_wer = model_wer_edits / len(ref_tokens)
            ex_copy_wer = copy_wer_edits / len(ref_tokens)
        else:
            model_wer_edits = len(pred_tokens)
            copy_wer_edits = len(copy_tokens)
            total_model_wer_edits += model_wer_edits
            total_copy_wer_edits += copy_wer_edits
            ex_wer = 0.0 if not pred_tokens else 1.0
            ex_copy_wer = 0.0 if not copy_tokens else 1.0

        if ex_copy_bleu >= ex_bleu:
            copy_better_or_equal_count += 1
        if ex_bleu >= ex_copy_bleu:
            model_better_count += 1
        if ex_copy_ter <= ex_ter:
            copy_ter_better_or_equal_count += 1
        if ex_ter <= ex_copy_ter:
            model_ter_better_count += 1
        if ex_copy_wer <= ex_wer:
            copy_wer_better_or_equal_count += 1
        if ex_wer <= ex_copy_wer:
            model_wer_better_count += 1
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

        worst_bleu_examples.append({"id": row["id"], "sentence_bleu": ex_bleu})

    worst_bleu_examples.sort(key=lambda x: x["sentence_bleu"])
    worst_bleu_examples = worst_bleu_examples[:WORST_BLEU_K]

    model_bleu = corpus_bleu(hyps, [refs]).score if refs else float("nan")
    copy_baseline_bleu = corpus_bleu(copy_hyps, [refs]).score if refs else float("nan")
    if total_ter_ref_tokens == 0:
        model_ter = 0.0 if total_model_ter_edits == 0 else 1.0
        copy_baseline_ter = 0.0 if total_copy_ter_edits == 0 else 1.0
    else:
        model_ter = total_model_ter_edits / total_ter_ref_tokens
        copy_baseline_ter = total_copy_ter_edits / total_ter_ref_tokens
    if total_wer_ref_tokens == 0:
        model_wer = 0.0 if total_model_wer_edits == 0 else 1.0
        copy_baseline_wer = 0.0 if total_copy_wer_edits == 0 else 1.0
    else:
        model_wer = total_model_wer_edits / total_wer_ref_tokens
        copy_baseline_wer = total_copy_wer_edits / total_wer_ref_tokens
    copy_to_model_bleu_ratio = copy_baseline_bleu / model_bleu if model_bleu > 0 else None
    model_to_copy_bleu_ratio = model_bleu / copy_baseline_bleu if copy_baseline_bleu > 0 else None
    model_score_0_100 = model_vs_copy_score_0_100(model_bleu, copy_baseline_bleu)
    model_ter_score_0_100 = lower_is_better_vs_copy_score_0_100(model_ter, copy_baseline_ter)
    model_wer_score_0_100 = lower_is_better_vs_copy_score_0_100(model_wer, copy_baseline_wer)

    return {
        "n": len(refs),
        "bleu": model_bleu,
        "copy_baseline_bleu": copy_baseline_bleu,
        "ter": model_ter,
        "copy_baseline_ter": copy_baseline_ter,
        "wer": model_wer,
        "copy_baseline_wer": copy_baseline_wer,
        "model_vs_copy_score_0_100": model_score_0_100,
        "model_vs_copy_ter_score_0_100": model_ter_score_0_100,
        "model_vs_copy_wer_score_0_100": model_wer_score_0_100,
        "model_beats_copy_flag_score_gt_50": (
            bool(model_score_0_100 > 50.0) if model_score_0_100 is not None else None
        ),
        "model_beats_copy_flag_ter_score_gt_50": (
            bool(model_ter_score_0_100 > 50.0) if model_ter_score_0_100 is not None else None
        ),
        "model_beats_copy_flag_wer_score_gt_50": (
            bool(model_wer_score_0_100 > 50.0) if model_wer_score_0_100 is not None else None
        ),
        "sentence_model_beats_copy_rate_score_gt_50": (
            model_better_count / len(refs) if refs else 0.0
        ),
        "sentence_model_beats_copy_rate_ter": (
            model_ter_better_count / len(refs) if refs else 0.0
        ),
        "sentence_model_beats_copy_rate_wer": (
            model_wer_better_count / len(refs) if refs else 0.0
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
        "sentence_copy_better_or_equal_rate_wer": (
            copy_wer_better_or_equal_count / len(refs) if refs else 0.0
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
            input_text = raw.get("input_text") or raw.get("source_text") or raw.get("src") or ""
            src = strip_encoder_task_prefix(input_text)
            gold_with_label = raw.get("gold_raw") or raw.get("gold_with_label") or raw.get("gold") or ""
            pred_with_label = raw.get("pred_with_label") or raw.get("pred_raw") or raw.get("pred") or ""
            gold_clean = strip_decoder_label_prefix(gold_with_label)
            pred_clean = strip_decoder_label_prefix(pred_with_label)
            rows.append(
                {
                    "id": raw.get("id"),
                    "direction": canonicalize_translation_direction(raw.get("direction"), input_text),
                    "src": src,
                    "gold": gold_clean,
                    "pred": pred_clean,
                    "gold_label": raw.get("gold_source_variant_norm") or normalize_label(gold_with_label),
                    "pred_label": raw.get("pred_source_variant_norm") or normalize_label(pred_with_label),
                    "bucket": normalize_bucket(raw.get("bucket")),
                }
            )
    return rows


def default_output_path(predictions_path: Path) -> Path:
    name = predictions_path.name
    if name.endswith("_predictions.jsonl"):
        stem = name[: -len("_predictions.jsonl")] + "_summary.json"
    elif name.endswith(".jsonl"):
        stem = name[:-6] + "_summary.json"
    else:
        stem = name + "_summary.json"
    return predictions_path.with_name(stem)


def main() -> None:
    args = parse_args()
    rows = load_translation_rows(args.predictions_path)
    summary: dict[str, Any] = {
        "task": "translation",
        **build_translation_summary(rows),
        "predictions_path": args.predictions_path.as_posix(),
        "rebuilt_with_label_normalization": True,
    }
    if args.dataset_path is not None:
        summary["dataset_path"] = args.dataset_path.as_posix()

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

    output_path = args.output_path or default_output_path(args.predictions_path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Saved summary: {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
