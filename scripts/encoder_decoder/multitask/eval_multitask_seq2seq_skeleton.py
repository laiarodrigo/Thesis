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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from task_protocol import CLS, TR_BR2PT, TR_PT2BR, strip_encoder_prefix

import sacrebleu

LABELS = ("pt-br", "pt-pt", "equal")

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
    parser.add_argument("--max-new-tokens-translation", type=int, default=256)
    parser.add_argument("--max-new-tokens-classification", type=int, default=6)
    return parser.parse_args()


def load_model_and_tokenizer(model_id: str, adapter_dir: Path | None):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base, adapter_dir.as_posix())
    else:
        model = base
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tok


def get_task_token_ids(tokenizer) -> dict[str, int]:
    ids = {
        TR_BR2PT: tokenizer.convert_tokens_to_ids(TR_BR2PT),
        TR_PT2BR: tokenizer.convert_tokens_to_ids(TR_PT2BR),
        CLS: tokenizer.convert_tokens_to_ids(CLS),
    }
    for tok, tok_id in ids.items():
        if tok_id is None or tok_id < 0:
            raise RuntimeError(f"Missing token id for {tok}.")
    return ids


def generate_with_forced_task(
    model,
    tokenizer,
    inputs: list[str],
    task_token_id: int,
    *,
    max_source_length: int,
    max_new_tokens: int,
) -> list[str]:
    device = next(model.parameters()).device
    enc = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)

    # Force first decoder token so the model runs in the chosen task mode.
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        forced_decoder_ids=[(0, int(task_token_id))],
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def strip_task_prefix_from_target(text: str) -> str:
    t = (text or "").strip()
    for tok in (TR_BR2PT, TR_PT2BR, CLS):
        if t.startswith(tok):
            return t[len(tok) :].strip()
    return t


def resolve_translation_row(row: dict[str, Any]) -> tuple[str, str, str]:
    """
    Resolve (task_token, model_input_text, gold_text) for translation rows.

    Supports:
    - New multitask rows with explicit task field.
    - Rows where target_text starts with decoder task token.
    - Legacy rows with encoder prefix in input_text.
    """
    task = (row.get("task") or "").strip()
    source_input = (row.get("input_text") or "").strip()
    gold = row["target_text"] if "target_text" in row else row.get("gold", "")

    if task in {"translate_br2pt", TR_BR2PT}:
        return TR_BR2PT, source_input, strip_task_prefix_from_target(gold)
    if task in {"translate_pt2br", TR_PT2BR}:
        return TR_PT2BR, source_input, strip_task_prefix_from_target(gold)

    gold_clean = (gold or "").strip()
    if gold_clean.startswith(TR_BR2PT):
        return TR_BR2PT, source_input, strip_task_prefix_from_target(gold)
    if gold_clean.startswith(TR_PT2BR):
        return TR_PT2BR, source_input, strip_task_prefix_from_target(gold)

    # Fallback for legacy translation test files.
    prefix, clean_input = strip_encoder_prefix(source_input)
    if prefix == "br-pt":
        return TR_BR2PT, clean_input, gold_clean
    if prefix == "pt-br":
        return TR_PT2BR, clean_input, gold_clean

    raise ValueError(
        "Could not resolve translation direction. Expected task field, "
        "decoder target token, or legacy encoder prefix."
    )


def evaluate_translation(
    model,
    tokenizer,
    task_token_ids: dict[str, int],
    dataset_path: Path,
    output_dir: Path,
    *,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
) -> None:
    ds = load_dataset("json", data_files={"eval": dataset_path.as_posix()})["eval"]
    pred_path = output_dir / "translation_predictions.jsonl"
    summary_path = output_dir / "translation_summary.json"

    # TODO(you):
    # 1) group rows by decoder task using row["task"]
    # 2) for each group, call generate_with_forced_task.
    # 3) write records: id/input_text/gold/pred_raw
    # 4) compute BLEU and maybe per-direction BLEU.
    # 5) save summary JSON.

    by_task = {TR_BR2PT: [], TR_PT2BR: []}
    for row in ds:
        task_tok, model_input, gold = resolve_translation_row(row)
        by_task[task_tok].append(
            {
                "id": row.get("id"),
                "input_text": model_input,
                "gold": gold,
            }
        )

    wrote = 0
    hyps, refs = [], []
    with pred_path.open("w", encoding="utf-8") as fh:
        for task_tok, rows in by_task.items():
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                texts = [r["input_text"] for r in batch]
                preds = generate_with_forced_task(
                    model,
                    tokenizer,
                    texts,
                    task_token_ids[task_tok],
                    max_source_length=max_source_length,
                    max_new_tokens=max_new_tokens,
                )
                for r, pred in zip(batch, preds):
                    pred_clean = re.sub(r"\s+", " ", pred or "").strip()
                    hyps.append(pred_clean)
                    refs.append(r["gold"])
                    rec = {
                        "id": r["id"],
                        "input_text": r["input_text"],
                        "gold": r["gold"],
                        "pred_raw": re.sub(r"\s+", " ", pred or "").strip(),
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    wrote += 1

    summary = {
        "task": "translation",
        "n": wrote,
        "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score if refs else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_label(pred: str) -> str:
    t = (pred or "").strip().lower()
    if "pt-br" in t:
        return "pt-br"
    if "pt-pt" in t:
        return "pt-pt"
    if "equal" in t:
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
    task_token_ids: dict[str, int],
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
        gold = row["target_text"] if "target_text" in row else row.get("gold", "")
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
            preds = generate_with_forced_task(
                model,
                tokenizer,
                texts,
                task_token_ids[CLS],
                max_source_length=max_source_length,
                max_new_tokens=max_new_tokens,
            )
            for r, pred in zip(batch, preds):
                pred_norm = normalize_label(pred)
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
                            "pred_raw": pred,
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
    task_token_ids = get_task_token_ids(tokenizer)

    evaluate_translation(
        model,
        tokenizer,
        task_token_ids,
        args.translation_dataset,
        run_dir,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens_translation,
    )
    evaluate_classification(
        model,
        tokenizer,
        task_token_ids,
        args.classification_dataset,
        run_dir,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens_classification,
    )

    print(f"Saved run dir: {run_dir}")


if __name__ == "__main__":
    main()
