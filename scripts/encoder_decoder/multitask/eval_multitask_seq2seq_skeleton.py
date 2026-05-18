#!/usr/bin/env python3
"""
Step 4 skeleton: evaluate one multitask seq2seq checkpoint on:
- translation
- classification

This file is intentionally incomplete (TODO markers) so you can finish it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from task_protocol import (
    CLASS_LABELS,
    build_prompted_input,
    build_translation_input_from_encoder_prefix,
    class_label_to_decoder_payload,
    decoder_payload_to_class_label,
    map_task_to_translation_prompt,
    split_translation_prompt,
    strip_encoder_prefix,
)

import sacrebleu

try:
    from scripts.encoder_decoder.eval.metrics_utils import corpus_ter, sentence_ter
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.encoder_decoder.eval.metrics_utils import corpus_ter, sentence_ter

LABELS = tuple(sorted(CLASS_LABELS))
ENCODER_TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skeleton evaluator for multitask seq2seq."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--translation-dataset", type=Path, required=True)
    parser.add_argument("--classification-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens-translation", type=int, default=192)
    parser.add_argument("--max-new-tokens-classification", type=int, default=2)
    parser.add_argument("--adaptive-max-new-tokens", action="store_true")
    parser.add_argument("--adaptive-ratio", type=float, default=1.15)
    parser.add_argument("--adaptive-margin", type=int, default=6)
    parser.add_argument("--adaptive-min-new-tokens", type=int, default=6)
    parser.add_argument("--adaptive-max-new-tokens-ceiling", type=int, default=192)
    parser.add_argument("--translation-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--translation-repetition-penalty", type=float, default=1.1)
    return parser.parse_args()


def load_model_and_tokenizer(model_id: str, adapter_dir: Path | None):
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except (ValueError, ImportError) as exc:
        print(f"Fast tokenizer unavailable, falling back to slow tokenizer: {exc}")
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        dtype=target_dtype,
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base, adapter_dir.as_posix())
    else:
        model = base
    model = model.to(dtype=target_dtype)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tok


def resolve_single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise RuntimeError(
            f"Protocol token {text!r} does not map to a single tokenizer id: {ids}"
        )
    return int(ids[0])


def get_class_label_token_ids(tokenizer) -> list[int]:
    return [
        resolve_single_token_id(tokenizer, class_label_to_decoder_payload(label))
        for label in ("pt-br", "pt-pt", "equal")
    ]


def resolve_eos_pad_ids(model, tokenizer) -> tuple[int | None, int | None]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        cfg_eos = getattr(model.config, "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple)):
            eos_token_id = cfg_eos[0] if cfg_eos else None
        else:
            eos_token_id = cfg_eos

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        cfg_pad = getattr(model.config, "pad_token_id", None)
        if isinstance(cfg_pad, (list, tuple)):
            pad_token_id = cfg_pad[0] if cfg_pad else None
        else:
            pad_token_id = cfg_pad
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return eos_token_id, pad_token_id


def resolve_decoder_start_token_id(model, tokenizer) -> int:
    token_id = getattr(model.config, "decoder_start_token_id", None)
    if token_id is None:
        token_id = tokenizer.pad_token_id
    if token_id is None:
        token_id = tokenizer.bos_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        raise RuntimeError("Could not resolve decoder_start_token_id for classification.")
    return int(token_id)


def generate_from_inputs(
    model,
    tokenizer,
    inputs: list[str],
    *,
    max_source_length: int,
    max_new_tokens: int,
    skip_special_tokens: bool = True,
    adaptive_max_new_tokens: bool = False,
    adaptive_ratio: float = 1.3,
    adaptive_margin: int = 10,
    adaptive_min_new_tokens: int = 8,
    adaptive_max_new_tokens_ceiling: int = 512,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
    allowed_first_step_token_ids: list[int] | None = None,
) -> list[str]:
    device = next(model.parameters()).device
    enc = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)

    eos_token_id, pad_token_id = resolve_eos_pad_ids(model, tokenizer)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": repetition_penalty,
    }
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = int(eos_token_id)
        generate_kwargs["forced_eos_token_id"] = int(eos_token_id)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = int(pad_token_id)
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)

    def per_example_cap(src_len: int) -> int:
        proposed = int(src_len * adaptive_ratio + adaptive_margin)
        proposed = max(int(adaptive_min_new_tokens), proposed)
        proposed = min(int(adaptive_max_new_tokens_ceiling), proposed)
        return max(1, proposed)

    with torch.no_grad():
        if adaptive_max_new_tokens:
            preds = []
            for i in range(enc["input_ids"].shape[0]):
                src_len = int(enc["attention_mask"][i].sum().item())
                sample_kwargs = dict(generate_kwargs)
                sample_kwargs["max_new_tokens"] = per_example_cap(src_len)
                sample_out = model.generate(
                    input_ids=enc["input_ids"][i : i + 1],
                    attention_mask=enc["attention_mask"][i : i + 1],
                    **sample_kwargs,
                )
                preds.append(tokenizer.decode(sample_out[0], skip_special_tokens=skip_special_tokens))
            return preds

        out = model.generate(**enc, **generate_kwargs)
        return tokenizer.batch_decode(out, skip_special_tokens=skip_special_tokens)


def classify_from_first_token(
    model,
    tokenizer,
    inputs: list[str],
    class_label_token_ids: list[int],
    *,
    max_source_length: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    enc = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)

    decoder_start_token_id = resolve_decoder_start_token_id(model, tokenizer)
    decoder_input_ids = torch.full(
        (len(inputs), 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    if logits is None:
        raise RuntimeError("Model did not return logits for classification evaluation.")

    next_token_logits = logits[:, -1, :]
    allowed_ids = torch.tensor(class_label_token_ids, device=next_token_logits.device)
    allowed_logits = next_token_logits.index_select(dim=-1, index=allowed_ids)
    pred_pos = allowed_logits.argmax(dim=-1)
    pred_token_ids = allowed_ids[pred_pos]

    results: list[dict[str, Any]] = []
    for tok_id, label_scores in zip(pred_token_ids.tolist(), allowed_logits.tolist()):
        pred_token = tokenizer.convert_ids_to_tokens(int(tok_id))
        pred_text = tokenizer.decode([int(tok_id)], skip_special_tokens=False)
        scores = {
            tokenizer.convert_ids_to_tokens(int(label_id)): float(score)
            for label_id, score in zip(class_label_token_ids, label_scores)
        }
        results.append(
            {
                "pred_token_id": int(tok_id),
                "pred_token": pred_token,
                "pred_text": pred_text,
                "label_scores": scores,
            }
        )
    return results


def strip_task_prefix_from_target(text: str) -> str:
    t = (text or "").strip()
    for tok in ("<TR_BR2PT>", "<TR_PT2BR>", "<CLS>"):
        if t.startswith(tok):
            return t[len(tok) :].strip()
    return t


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ").replace("\r", " ")).strip()


def strip_encoder_task_prefix(text: str) -> str:
    raw = text or ""
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    _, clean = split_translation_prompt(raw)
    return normalize_text(clean)


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


def resolve_translation_row(row: dict[str, Any]) -> tuple[str, str]:
    """
    Resolve (model_input_text, gold_text) for translation rows.

    Supports:
    - New multitask rows with encoder-side prompt already present.
    - Legacy translation rows with <br-pt> / <pt-br> encoder prefixes.
    """
    task = (row.get("task") or "").strip()
    source_input = (row.get("input_text") or "").strip()
    gold = strip_task_prefix_from_target(
        row["target_text"] if "target_text" in row else row.get("gold", "")
    )

    # Fallback for legacy translation test files.
    prefix, clean_input = strip_encoder_prefix(source_input)
    if prefix == "br-pt":
        return build_translation_input_from_encoder_prefix(prefix, clean_input), gold
    if prefix == "pt-br":
        return build_translation_input_from_encoder_prefix(prefix, clean_input), gold

    prompt = map_task_to_translation_prompt(task)
    if prompt is not None:
        existing_prompt, _ = split_translation_prompt(source_input)
        if existing_prompt is not None:
            return normalize_text(source_input), gold
        return build_prompted_input(prompt, source_input), gold

    existing_prompt, _ = split_translation_prompt(source_input)
    if existing_prompt is not None:
        return normalize_text(source_input), gold

    raise ValueError(
        "Could not resolve translation direction. Expected prompted input, "
        "task hint, or legacy encoder prefix."
    )


def evaluate_translation(
    model,
    tokenizer,
    dataset_path: Path,
    output_dir: Path,
    *,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
    adaptive_max_new_tokens: bool,
    adaptive_ratio: float,
    adaptive_margin: int,
    adaptive_min_new_tokens: int,
    adaptive_max_new_tokens_ceiling: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
) -> None:
    ds = load_dataset("json", data_files={"eval": dataset_path.as_posix()})["eval"]
    pred_path = output_dir / "translation_predictions.jsonl"
    summary_path = output_dir / "translation_summary.json"

    rows = []
    for row in ds:
        model_input, gold = resolve_translation_row(row)
        rows.append({"id": row.get("id"), "input_text": model_input, "gold": gold})

    wrote = 0
    hyps, refs, copy_hyps = [], [], []
    copy_better_or_equal_count = 0
    model_better_count = 0
    copy_ter_better_or_equal_count = 0
    model_ter_better_count = 0
    exact_input_copy_count = 0
    with pred_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [r["input_text"] for r in batch]
            preds = generate_from_inputs(
                model,
                tokenizer,
                texts,
                max_source_length=max_source_length,
                max_new_tokens=max_new_tokens,
                adaptive_max_new_tokens=adaptive_max_new_tokens,
                adaptive_ratio=adaptive_ratio,
                adaptive_margin=adaptive_margin,
                adaptive_min_new_tokens=adaptive_min_new_tokens,
                adaptive_max_new_tokens_ceiling=adaptive_max_new_tokens_ceiling,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
            )
            for r, pred in zip(batch, preds):
                pred_clean = normalize_text(pred)
                src_clean = strip_encoder_task_prefix(r["input_text"])
                hyps.append(pred_clean)
                refs.append(r["gold"])
                copy_hyps.append(src_clean)
                pred_bleu = sacrebleu.sentence_bleu(pred_clean, [r["gold"]]).score
                copy_bleu = sacrebleu.sentence_bleu(src_clean, [r["gold"]]).score
                pred_ter = sentence_ter(pred_clean, r["gold"])
                copy_ter = sentence_ter(src_clean, r["gold"])
                if copy_bleu >= pred_bleu:
                    copy_better_or_equal_count += 1
                if pred_bleu >= copy_bleu:
                    model_better_count += 1
                if copy_ter <= pred_ter:
                    copy_ter_better_or_equal_count += 1
                if pred_ter <= copy_ter:
                    model_ter_better_count += 1
                if pred_clean == src_clean:
                    exact_input_copy_count += 1
                rec = {
                    "id": r["id"],
                    "input_text": r["input_text"],
                    "gold": r["gold"],
                    "pred_raw": re.sub(r"\s+", " ", pred or "").strip(),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1

    model_bleu = sacrebleu.corpus_bleu(hyps, [refs]).score if refs else 0.0
    copy_baseline_bleu = sacrebleu.corpus_bleu(copy_hyps, [refs]).score if refs else 0.0
    model_ter = corpus_ter(hyps, refs) if refs else 0.0
    copy_baseline_ter = corpus_ter(copy_hyps, refs) if refs else 0.0
    model_score_0_100 = model_vs_copy_score_0_100(model_bleu, copy_baseline_bleu)
    model_ter_score_0_100 = lower_is_better_vs_copy_score_0_100(
        model_ter,
        copy_baseline_ter,
    )
    copy_to_model_bleu_ratio = copy_baseline_bleu / model_bleu if model_bleu > 0 else None
    model_to_copy_bleu_ratio = model_bleu / copy_baseline_bleu if copy_baseline_bleu > 0 else None
    summary = {
        "task": "translation",
        "n": wrote,
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
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_label(pred: str) -> str:
    t = normalize_text(pred)
    if not t:
        return "unknown"

    parts = t.split()
    if parts:
        for tok in (t, parts[0], parts[-1]):
            label = decoder_payload_to_class_label(tok)
            if label is not None:
                return label

    low = t.lower()
    if "pt-br" in low:
        return "pt-br"
    if "pt-pt" in low:
        return "pt-pt"
    if "equal" in low:
        return "equal"
    return "unknown"


def prf_for_label(cm: Counter, label: str) -> tuple[float, float, float]:
    tp = cm[(label, label)]
    fp = sum(cm[(g, label)] for g in LABELS if g != label)
    fn = sum(cm[(label, p)] for p in LABELS if p != label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_classification(
    model,
    tokenizer,
    class_label_token_ids: list[int],
    dataset_path: Path,
    output_dir: Path,
    *,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
) -> None:
    ds = load_dataset("json", data_files={"eval": dataset_path.as_posix()})["eval"]
    pred_path = output_dir / "classification_predictions.jsonl"
    summary_path = output_dir / "classification_summary.json"

    rows = []
    for row in ds:
        source_input = (row.get("input_text") or "").strip()
        prefix, clean_input = strip_encoder_prefix(source_input)
        model_input = clean_input if prefix in {"br-pt", "pt-br", "id"} else source_input
        gold = row.get("target_text", row.get("label", row.get("gold", "")))
        gold = strip_task_prefix_from_target(gold)
        rows.append(
            {
                "id": row.get("id"),
                "input_text": model_input,
                "gold": gold,
            }
        )

    cm = Counter()
    total = 0
    correct = 0

    with pred_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [r["input_text"] for r in batch]
            preds = classify_from_first_token(
                model,
                tokenizer,
                texts,
                class_label_token_ids,
                max_source_length=max_source_length,
            )
            for r, pred in zip(batch, preds):
                pred_norm = normalize_label(pred["pred_token"])
                gold_norm = normalize_label(r["gold"])
                cm[(gold_norm, pred_norm)] += 1
                total += 1
                if pred_norm == gold_norm:
                    correct += 1
                fh.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "input_text": r["input_text"],
                            "gold": r["gold"],
                            "pred_raw": pred["pred_text"],
                            "pred_token": pred["pred_token"],
                            "pred_token_id": pred["pred_token_id"],
                            "label_scores": pred["label_scores"],
                            "gold_norm": gold_norm,
                            "pred_norm": pred_norm,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    per_class = {}
    f1_values = []
    for lab in LABELS:
        p, r, f1 = prf_for_label(cm, lab)
        per_class[lab] = {"precision": p, "recall": r, "f1": f1}
        f1_values.append(f1)

    summary = {
        "task": "classification",
        "n": total,
        "accuracy": (correct / total) if total else 0.0,
        "confusion_matrix": {f"{g}->{p}": int(v) for (g, p), v in cm.items()},
        "per_class": per_class,
        "macro_f1": (sum(f1_values) / len(f1_values)) if f1_values else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.adapter_dir)
    class_label_token_ids = get_class_label_token_ids(tokenizer)

    evaluate_translation(
        model,
        tokenizer,
        args.translation_dataset,
        run_dir,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens_translation,
        adaptive_max_new_tokens=args.adaptive_max_new_tokens,
        adaptive_ratio=args.adaptive_ratio,
        adaptive_margin=args.adaptive_margin,
        adaptive_min_new_tokens=args.adaptive_min_new_tokens,
        adaptive_max_new_tokens_ceiling=args.adaptive_max_new_tokens_ceiling,
        no_repeat_ngram_size=args.translation_no_repeat_ngram_size,
        repetition_penalty=args.translation_repetition_penalty,
    )
    evaluate_classification(
        model,
        tokenizer,
        class_label_token_ids,
        args.classification_dataset,
        run_dir,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens_classification,
    )

    print(f"Saved run dir: {run_dir}")


if __name__ == "__main__":
    main()
