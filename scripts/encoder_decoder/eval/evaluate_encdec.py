#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sacrebleu import corpus_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


LABELS = ("pt-br", "pt-pt", "equal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate encoder-decoder translation/classification runs.")
    parser.add_argument("--task", choices=["translation", "classification"], required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results") / "encoder_decoder")
    return parser.parse_args()


def normalize_label(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if not t:
        return None
    if "equal" in t or "shared" in t or t == "same":
        return "equal"
    if "pt-br" in t or "ptbr" in t or "brasil" in t:
        return "pt-br"
    if "pt-pt" in t or "ptpt" in t or "europeu" in t or "portugal" in t:
        return "pt-pt"
    return None


def load_model_and_tokenizer(model_id: str, adapter_dir: Optional[Path], tokenizer_path: Optional[Path]):
    tokenizer_candidates = []
    if tokenizer_path:
        tokenizer_candidates.append(tokenizer_path.as_posix())
    if adapter_dir:
        tokenizer_candidates.append(adapter_dir.as_posix())
    tokenizer_candidates.append(model_id)

    tok = None
    for cand in tokenizer_candidates:
        try:
            tok = AutoTokenizer.from_pretrained(cand, use_fast=True)
            print(f"Tokenizer loaded from: {cand}")
            break
        except Exception:
            continue
    if tok is None:
        raise RuntimeError("Unable to load tokenizer from tokenizer-path/adapter/model.")

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base_model, adapter_dir.as_posix())
    else:
        model = base_model

    model.eval()
    if not torch.cuda.is_available():
        model.to("cpu")
    return model, tok


def generate_batch(model, tok, inputs: list[str], max_source_length: int, max_new_tokens: int) -> list[str]:
    device = next(model.parameters()).device
    enc = tok(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    return tok.batch_decode(out, skip_special_tokens=True)


@dataclass
class ClsStats:
    total: int = 0
    correct: int = 0
    tp: Counter = None
    fp: Counter = None
    fn: Counter = None
    cm: Counter = None

    def __post_init__(self):
        self.tp = Counter()
        self.fp = Counter()
        self.fn = Counter()
        self.cm = Counter()

    def update(self, gold: str, pred: str):
        self.total += 1
        self.cm[(gold, pred)] += 1
        if pred == gold:
            self.correct += 1
            self.tp[gold] += 1
        else:
            self.fn[gold] += 1
            self.fp[pred] += 1

    def report(self) -> dict:
        acc = self.correct / self.total if self.total else 0.0
        per_class = {}
        f1s = []
        for lab in LABELS:
            tp = self.tp[lab]
            fp = self.fp[lab]
            fn = self.fn[lab]
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
            per_class[lab] = {
                "precision": p,
                "recall": r,
                "f1": f1,
                "support": tp + fn,
            }
            f1s.append(f1)
        return {
            "n": self.total,
            "accuracy": acc,
            "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "per_class": per_class,
            "confusion_matrix": {
                f"{g}->{p}": int(self.cm[(g, p)])
                for g in LABELS
                for p in LABELS
            },
        }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model_and_tokenizer(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        tokenizer_path=args.tokenizer_path,
    )

    ds = load_dataset("json", data_files={"eval": args.dataset_path.as_posix()})["eval"]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = args.output_dir / f"{run_id}_{args.task}_predictions.jsonl"
    summary_path = args.output_dir / f"{run_id}_{args.task}_summary.json"

    cls_stats = ClsStats()
    refs = []
    hyps = []

    with pred_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(ds), args.batch_size):
            batch = ds[start : start + args.batch_size]
            inputs = batch["input_text"]
            targets = batch["target_text"]
            preds = generate_batch(
                model,
                tok,
                inputs=inputs,
                max_source_length=args.max_source_length,
                max_new_tokens=args.max_new_tokens,
            )

            for src, gold, pred in zip(inputs, targets, preds):
                pred_clean = (pred or "").strip()
                if args.task == "translation":
                    refs.append(gold)
                    hyps.append(pred_clean)
                else:
                    gold_norm = normalize_label(gold) or "unknown"
                    pred_norm = normalize_label(pred_clean) or "unknown"
                    cls_stats.update(gold_norm, pred_norm)

                rec = {
                    "input_text": src,
                    "gold": gold,
                    "pred_raw": pred_clean,
                }
                if args.task == "classification":
                    rec["gold_norm"] = normalize_label(gold) or "unknown"
                    rec["pred_norm"] = normalize_label(pred_clean) or "unknown"
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.task == "translation":
        summary = {
            "task": "translation",
            "n": len(refs),
            "bleu": corpus_bleu(hyps, [refs]).score if refs else float("nan"),
        }
    else:
        summary = {"task": "classification", **cls_stats.report()}

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Saved predictions: {pred_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
