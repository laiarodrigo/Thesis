#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sacrebleu import corpus_bleu, sentence_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


LABELS = ("pt-br", "pt-pt", "equal")
WORST_BLEU_K = 10


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
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Anti-repetition ngram size for generation; set 0 to disable.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty > 1.0 discourages loops in generated text.",
    )
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


def normalize_generation_text(text: str) -> str:
    # Keep output single-line and compact for CSV/JSONL inspection.
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def generate_batch(
    model,
    tok,
    inputs: list[str],
    max_source_length: int,
    max_new_tokens: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
) -> list[str]:
    device = next(model.parameters()).device
    enc = tok(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)

    eos_token_id = tok.eos_token_id
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else eos_token_id

    with torch.no_grad():
        generate_kwargs = dict(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty=repetition_penalty,
        )
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            generate_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
        out = model.generate(**generate_kwargs)
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
    worst_bleu_examples = []
    has_id_column = "id" in ds.column_names

    with pred_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(ds), args.batch_size):
            batch = ds[start : start + args.batch_size]
            inputs = batch["input_text"]
            targets = batch["target_text"]
            if has_id_column:
                batch_ids = batch["id"]
            else:
                batch_ids = list(range(start, start + len(inputs)))
            preds = generate_batch(
                model,
                tok,
                inputs=inputs,
                max_source_length=args.max_source_length,
                max_new_tokens=args.max_new_tokens,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
            )

            for ex_id, src, gold, pred in zip(batch_ids, inputs, targets, preds):
                pred_clean = normalize_generation_text(pred or "")
                if args.task == "translation":
                    refs.append(gold)
                    hyps.append(pred_clean)
                    ex_bleu = sentence_bleu(pred_clean, [gold]).score
                    worst_bleu_examples.append(
                        {
                            "id": ex_id,
                            "sentence_bleu": ex_bleu,
                            "gold": gold,
                            "pred_raw": pred_clean,
                        }
                    )
                else:
                    gold_norm = normalize_label(gold) or "unknown"
                    pred_norm = normalize_label(pred_clean) or "unknown"
                    cls_stats.update(gold_norm, pred_norm)

                rec = {
                    "id": ex_id,
                    "input_text": src,
                    "gold": gold,
                    "pred_raw": pred_clean,
                }
                if args.task == "classification":
                    rec["gold_norm"] = normalize_label(gold) or "unknown"
                    rec["pred_norm"] = normalize_label(pred_clean) or "unknown"
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.task == "translation":
        worst_bleu_examples.sort(key=lambda x: x["sentence_bleu"])
        worst_bleu_examples = worst_bleu_examples[:WORST_BLEU_K]
        summary = {
            "task": "translation",
            "n": len(refs),
            "bleu": corpus_bleu(hyps, [refs]).score if refs else float("nan"),
        }
        summary["worst_sentence_bleu_examples"] = [
            {"id": x["id"], "sentence_bleu": x["sentence_bleu"]} for x in worst_bleu_examples
        ]
    else:
        summary = {"task": "classification", **cls_stats.report()}

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Saved predictions: {pred_path}")
    print(f"Saved summary: {summary_path}")
    if args.task == "translation":
        print(f"Worst {len(worst_bleu_examples)} sentence-BLEU examples:")
        for ex in worst_bleu_examples:
            print(f"  id={ex['id']} sentence_bleu={ex['sentence_bleu']:.4f}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
