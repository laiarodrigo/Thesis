#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sacrebleu import corpus_bleu, sentence_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from metrics_utils import (
        PT_VARIANT_LABELS,
        corpus_ter,
        sentence_ter,
    )
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.metrics_utils import (
        PT_VARIANT_LABELS,
        corpus_ter,
        sentence_ter,
    )


LABELS = ("pt-br", "pt-pt", "equal")
WORST_BLEU_K = 10
FRMT_BUCKETS = ("random", "entity", "lexical")
ENCODER_TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)
DECODER_LABEL_PREFIX_RE = re.compile(r"^\s*(BR|PT|pt-br|pt-pt)\b[:\-\s]*", flags=re.IGNORECASE)


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
        "--num-beams",
        type=int,
        default=1,
        help="Beam size for generation. Use 1 for greedy decoding.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty used with beam search.",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable beam-search early stopping when all beams finish.",
    )
    parser.add_argument(
        "--adaptive-max-new-tokens",
        action="store_true",
        help="Compute max_new_tokens per example from source length.",
    )
    parser.add_argument(
        "--adaptive-ratio",
        type=float,
        default=1.3,
        help="Per-example cap formula: src_len * adaptive_ratio + adaptive_margin.",
    )
    parser.add_argument(
        "--adaptive-margin",
        type=int,
        default=10,
        help="Per-example cap formula: src_len * adaptive_ratio + adaptive_margin.",
    )
    parser.add_argument(
        "--adaptive-min-new-tokens",
        type=int,
        default=8,
        help="Lower bound for adaptive max_new_tokens.",
    )
    parser.add_argument(
        "--adaptive-max-new-tokens-ceiling",
        type=int,
        default=512,
        help="Upper bound for adaptive max_new_tokens.",
    )
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
    parser.add_argument(
        "--classification-mode",
        choices=["score-sequences", "score-first-token", "generate"],
        default="score-sequences",
        help=(
            "Classification inference mode. "
            "'score-sequences' scores candidate labels directly under the decoder and is the safest default. "
            "'score-first-token' compares only the first decoder token and requires one-token labels. "
            "'generate' uses standard free generation."
        ),
    )
    parser.add_argument(
        "--classification-candidates",
        nargs="+",
        default=None,
        help=(
            "Optional explicit list of classification target strings to score, e.g. 'BR PT' or "
            "'pt-br pt-pt equal'. Defaults to the unique target_text values found in the eval dataset."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results") / "encoder_decoder")
    return parser.parse_args()


def normalize_label(text: str) -> Optional[str]:
    t = normalize_generation_text(text or "").lower()
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
    for value in (raw_direction,):
        text = normalize_generation_text(str(value or "")).lower()
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
    text = normalize_generation_text(str(raw_bucket or "")).lower()
    if text in {"rand", "random"}:
        return "random"
    if text in {"entity", "entities"}:
        return "entity"
    if text in {"lexical", "lex"}:
        return "lexical"
    if not text:
        return "n/a"
    return text


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

    target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        dtype=target_dtype,
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base_model, adapter_dir.as_posix())
    else:
        model = base_model

    # Some PEFT checkpoints restore LoRA weights in fp32 even when the base model
    # is loaded in bf16. Align everything to one dtype before generation.
    model = model.to(dtype=target_dtype)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")
    return model, tok


def normalize_generation_text(text: str) -> str:
    # Keep output single-line and compact for CSV/JSONL inspection.
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def strip_encoder_task_prefix(text: str) -> str:
    raw = text or ""
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_generation_text(raw)


def strip_decoder_label_prefix(text: str) -> str:
    raw = normalize_generation_text(text or "")
    match = DECODER_LABEL_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_generation_text(raw)


def build_classification_candidates(ds, explicit_candidates: Optional[list[str]]) -> list[str]:
    seen = set()
    candidates: list[str] = []
    raw_values = explicit_candidates if explicit_candidates else classification_target_values(ds)
    for raw in raw_values:
        cand = normalize_generation_text(str(raw or ""))
        if not cand or cand in seen:
            continue
        seen.add(cand)
        candidates.append(cand)
    if not candidates:
        raise ValueError("No valid classification candidates found.")
    return candidates


def classification_target_values(ds) -> list[str]:
    for key in ("target_text", "label", "gold"):
        if key in ds.column_names:
            return list(ds[key])
    raise KeyError(
        "Classification eval dataset must contain one of: target_text, label, gold. "
        f"Available columns: {sorted(ds.column_names)}"
    )


def extract_classification_targets(batch: dict[str, Any]) -> list[str]:
    for key in ("target_text", "label", "gold"):
        values = batch.get(key)
        if values is None:
            continue
        return [normalize_generation_text(str(value or "")) for value in values]
    raise KeyError(
        "Classification eval batch must contain one of: target_text, label, gold. "
        f"Available keys: {sorted(batch.keys())}"
    )


def extract_translation_targets(batch: dict[str, Any]) -> list[str]:
    for key in ("target_text", "gold"):
        values = batch.get(key)
        if values is None:
            continue
        return [str(value or "") for value in values]
    raise KeyError(
        "Translation eval batch must contain one of: target_text, gold. "
        f"Available keys: {sorted(batch.keys())}"
    )


def classification_label_alias_map(candidates: list[str]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for cand in candidates:
        norm = normalize_label(cand)
        if norm and norm not in alias_map:
            alias_map[norm] = cand
    return alias_map


def filter_classification_dataset_for_candidates(ds, candidates: list[str]):
    allowed = {label for label in (normalize_label(cand) for cand in candidates) if label is not None}
    if not allowed:
        return ds, {}

    raw_targets = classification_target_values(ds)
    keep_indices: list[int] = []
    dropped: Counter[str] = Counter()
    for idx, raw in enumerate(raw_targets):
        norm = normalize_label(str(raw or "")) or "unknown"
        if norm in allowed:
            keep_indices.append(idx)
        else:
            dropped[norm] += 1

    if not dropped:
        return ds, {}
    return ds.select(keep_indices), dict(dropped)


def classification_candidate_metadata(tok, candidates: list[str]) -> list[dict[str, Any]]:
    meta: list[dict[str, Any]] = []
    for cand in candidates:
        core_ids = tok.encode(cand, add_special_tokens=False)
        core_tokens = tok.convert_ids_to_tokens(core_ids) if core_ids else []
        meta.append(
            {
                "text": cand,
                "core_token_ids": [int(x) for x in core_ids],
                "core_tokens": list(core_tokens),
                "core_token_count": len(core_ids),
                "normalized_label": normalize_label(cand),
            }
        )
    return meta


def _decoder_start_token_id(model, tok) -> int:
    start_id = getattr(model.config, "decoder_start_token_id", None)
    if start_id is None:
        start_id = tok.pad_token_id
    if start_id is None:
        start_id = tok.eos_token_id
    if start_id is None:
        raise ValueError("Could not determine decoder_start_token_id / pad_token_id / eos_token_id.")
    return int(start_id)


def _prepare_decoder_input_ids_from_labels(model, labels_for_shift: torch.Tensor) -> torch.Tensor:
    prep_fn = getattr(model, "prepare_decoder_input_ids_from_labels", None)
    if prep_fn is None:
        raise AttributeError("Model does not expose prepare_decoder_input_ids_from_labels.")
    try:
        return prep_fn(labels=labels_for_shift)
    except TypeError as exc:
        if "unexpected keyword argument 'labels'" not in str(exc):
            raise
        return prep_fn(labels_for_shift)


def score_classification_candidates_batch(
    model,
    tok,
    *,
    inputs: list[str],
    candidates: list[str],
    max_source_length: int,
    mode: str,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    enc = tok(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)

    if mode == "score-first-token":
        candidate_token_ids: list[int] = []
        for cand in candidates:
            ids = tok.encode(cand, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"--classification-mode=score-first-token requires one-token labels, "
                    f"but candidate {cand!r} tokenizes to {ids!r}"
                )
            candidate_token_ids.append(int(ids[0]))

        decoder_input_ids = torch.full(
            (len(inputs), 1),
            _decoder_start_token_id(model, tok),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            ).logits[:, 0, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            candidate_log_probs = log_probs[:, candidate_token_ids]

        results: list[dict[str, Any]] = []
        for row_idx in range(len(inputs)):
            scores = {
                cand: float(candidate_log_probs[row_idx, cand_idx].item())
                for cand_idx, cand in enumerate(candidates)
            }
            best_idx = int(torch.argmax(candidate_log_probs[row_idx]).item())
            results.append(
                {
                    "pred_text": candidates[best_idx],
                    "scores": scores,
                }
            )
        return results

    if mode != "score-sequences":
        raise ValueError(f"Unsupported classification scoring mode: {mode}")

    batch_scores: list[dict[str, float]] = [dict() for _ in inputs]
    for cand in candidates:
        dec = tok(
            [cand] * len(inputs),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        target_ids = dec["input_ids"]
        target_mask = dec["attention_mask"]
        labels_for_shift = target_ids.masked_fill(target_mask == 0, -100)
        decoder_input_ids = _prepare_decoder_input_ids_from_labels(model, labels_for_shift)
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            ).logits
            token_log_probs = torch.log_softmax(logits, dim=-1).gather(
                -1,
                target_ids.unsqueeze(-1),
            ).squeeze(-1)
            masked = token_log_probs * target_mask
            seq_scores = masked.sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)

        for row_idx in range(len(inputs)):
            batch_scores[row_idx][cand] = float(seq_scores[row_idx].item())

    results = []
    for scores in batch_scores:
        pred_text = max(scores.items(), key=lambda kv: kv[1])[0]
        results.append({"pred_text": pred_text, "scores": scores})
    return results


def model_vs_copy_score_0_100(model_bleu: float, copy_bleu: float) -> Optional[float]:
    if not (np.isfinite(model_bleu) and np.isfinite(copy_bleu)):
        return None
    denom = model_bleu + copy_bleu
    if denom <= 0:
        return None
    return 100.0 * model_bleu / denom


def lower_is_better_vs_copy_score_0_100(model_error: float, copy_error: float) -> Optional[float]:
    if not (np.isfinite(model_error) and np.isfinite(copy_error)):
        return None
    denom = model_error + copy_error
    if denom == 0:
        return 50.0
    return 100.0 * copy_error / denom


def build_translation_summary(
    rows: list[dict[str, Any]],
    *,
    worst_k: int = WORST_BLEU_K,
) -> dict[str, Any]:
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
    worst_bleu_examples = worst_bleu_examples[:worst_k]

    model_bleu = corpus_bleu(hyps, [refs]).score if refs else float("nan")
    copy_baseline_bleu = corpus_bleu(copy_hyps, [refs]).score if refs else float("nan")
    copy_to_model_bleu_ratio = (
        copy_baseline_bleu / model_bleu if np.isfinite(model_bleu) and model_bleu > 0 else None
    )
    model_to_copy_bleu_ratio = (
        model_bleu / copy_baseline_bleu
        if np.isfinite(copy_baseline_bleu) and copy_baseline_bleu > 0
        else None
    )
    model_score_0_100 = model_vs_copy_score_0_100(model_bleu, copy_baseline_bleu)
    model_ter = corpus_ter(hyps, refs) if refs else float("nan")
    copy_baseline_ter = corpus_ter(copy_hyps, refs) if refs else float("nan")
    model_ter_score_0_100 = lower_is_better_vs_copy_score_0_100(
        model_ter,
        copy_baseline_ter,
    )

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
        "translation_source_variant_label_parse_rate": (
            translation_label_parsed / translation_label_total if translation_label_total else None
        ),
        "translation_source_variant_label_accuracy": (
            translation_label_correct / translation_label_total if translation_label_total else None
        ),
        "worst_sentence_bleu_examples": worst_bleu_examples,
    }


def build_classification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = ClsStats()
    for row in rows:
        gold_norm = str(row["gold_norm"])
        pred_norm = str(row["pred_norm"])
        stats.update(gold_norm, pred_norm)
    return stats.report()


def generate_batch(
    model,
    tok,
    inputs: list[str],
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    early_stopping: bool,
    adaptive_max_new_tokens: bool,
    adaptive_ratio: float,
    adaptive_margin: int,
    adaptive_min_new_tokens: int,
    adaptive_max_new_tokens_ceiling: int,
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
    if eos_token_id is None:
        cfg_eos = getattr(model.config, "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple)):
            eos_token_id = cfg_eos[0] if cfg_eos else None
        else:
            eos_token_id = cfg_eos

    pad_token_id = tok.pad_token_id
    if pad_token_id is None:
        cfg_pad = getattr(model.config, "pad_token_id", None)
        if isinstance(cfg_pad, (list, tuple)):
            pad_token_id = cfg_pad[0] if cfg_pad else None
        else:
            pad_token_id = cfg_pad
    if pad_token_id is None:
        pad_token_id = eos_token_id

    generate_base_kwargs = dict(
        do_sample=False,
        repetition_penalty=repetition_penalty,
    )
    if eos_token_id is not None:
        generate_base_kwargs["eos_token_id"] = int(eos_token_id)
        # Ensure termination token is injected if generation reaches the cap.
        generate_base_kwargs["forced_eos_token_id"] = int(eos_token_id)
    if pad_token_id is not None:
        generate_base_kwargs["pad_token_id"] = int(pad_token_id)
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        generate_base_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    if num_beams and num_beams > 1:
        generate_base_kwargs["num_beams"] = int(num_beams)
        generate_base_kwargs["length_penalty"] = float(length_penalty)
        generate_base_kwargs["early_stopping"] = bool(early_stopping)

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
                sample_kwargs = dict(
                    input_ids=enc["input_ids"][i : i + 1],
                    attention_mask=enc["attention_mask"][i : i + 1],
                    max_new_tokens=per_example_cap(src_len),
                    **generate_base_kwargs,
                )
                sample_out = model.generate(**sample_kwargs)
                preds.append(tok.decode(sample_out[0], skip_special_tokens=True))
            return preds

        generate_kwargs = dict(
            **enc,
            max_new_tokens=max_new_tokens,
            **generate_base_kwargs,
        )
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
        target_f1s = []
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
            if lab in PT_VARIANT_LABELS:
                target_f1s.append(f1)
        return {
            "n": self.total,
            "accuracy": acc,
            "macro_f1": sum(target_f1s) / len(target_f1s) if target_f1s else 0.0,
            "macro_f1_all_labels": sum(f1s) / len(f1s) if f1s else 0.0,
            "macro_f1_labels": list(PT_VARIANT_LABELS),
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
    classification_candidates: list[str] | None = None
    classification_candidate_meta: list[dict[str, Any]] | None = None
    classification_aliases: dict[str, str] | None = None
    filtered_out_counts: dict[str, int] | None = None
    if args.task == "classification":
        classification_candidates = build_classification_candidates(ds, args.classification_candidates)
        ds, filtered_out_counts = filter_classification_dataset_for_candidates(ds, classification_candidates)
        classification_aliases = classification_label_alias_map(classification_candidates)
        classification_candidate_meta = classification_candidate_metadata(tok, classification_candidates)
        print("Classification candidates:")
        for meta in classification_candidate_meta:
            print(
                f"  {meta['text']!r}: core_token_count={meta['core_token_count']} "
                f"core_token_ids={meta['core_token_ids']}"
            )
        if filtered_out_counts:
            print(
                "Filtered classification rows not covered by scoring candidates:"
                f" {filtered_out_counts}"
            )

    cls_stats = ClsStats()
    translation_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    has_id_column = "id" in ds.column_names
    has_source_id_column = "source_id" in ds.column_names
    has_direction_column = "direction" in ds.column_names
    has_bucket_column = "bucket" in ds.column_names
    has_dataset_column = "dataset" in ds.column_names

    with pred_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(ds), args.batch_size):
            batch = ds[start : start + args.batch_size]
            inputs = batch["input_text"]
            if args.task == "classification":
                inputs = [strip_encoder_task_prefix(text) for text in inputs]
            if args.task == "classification":
                targets = extract_classification_targets(batch)
            else:
                targets = extract_translation_targets(batch)
            if has_id_column:
                batch_ids = batch["id"]
            else:
                batch_ids = list(range(start, start + len(inputs)))
            batch_source_ids = batch["source_id"] if has_source_id_column else [None] * len(inputs)
            batch_directions = batch["direction"] if has_direction_column else [None] * len(inputs)
            batch_buckets = batch["bucket"] if has_bucket_column else [None] * len(inputs)
            batch_datasets = batch["dataset"] if has_dataset_column else [None] * len(inputs)
            pred_infos: list[dict[str, Any]]
            if args.task == "classification" and args.classification_mode != "generate":
                pred_infos = score_classification_candidates_batch(
                    model,
                    tok,
                    inputs=inputs,
                    candidates=classification_candidates or [],
                    max_source_length=args.max_source_length,
                    mode=args.classification_mode,
                )
            else:
                preds = generate_batch(
                    model,
                    tok,
                    inputs=inputs,
                    max_source_length=args.max_source_length,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=args.early_stopping,
                    adaptive_max_new_tokens=args.adaptive_max_new_tokens,
                    adaptive_ratio=args.adaptive_ratio,
                    adaptive_margin=args.adaptive_margin,
                    adaptive_min_new_tokens=args.adaptive_min_new_tokens,
                    adaptive_max_new_tokens_ceiling=args.adaptive_max_new_tokens_ceiling,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                )
                pred_infos = [{"pred_text": pred, "scores": None} for pred in preds]

            for ex_id, source_id, raw_direction, raw_bucket, raw_dataset, src, gold, pred_info in zip(
                batch_ids,
                batch_source_ids,
                batch_directions,
                batch_buckets,
                batch_datasets,
                inputs,
                targets,
                pred_infos,
            ):
                pred_full = normalize_generation_text(pred_info.get("pred_text") or "")
                bucket = normalize_bucket(raw_bucket)
                dataset_name = normalize_generation_text(str(raw_dataset or ""))
                if args.task == "translation":
                    direction = canonicalize_translation_direction(raw_direction, src)
                    gold_label = normalize_label(gold)
                    pred_label = normalize_label(pred_full)
                    gold_clean = strip_decoder_label_prefix(gold)
                    pred_clean = strip_decoder_label_prefix(pred_full)
                    src_clean = strip_encoder_task_prefix(src)
                    translation_rows.append(
                        {
                            "id": ex_id,
                            "source_id": source_id,
                            "direction": direction,
                            "src": src_clean,
                            "gold": gold_clean,
                            "pred": pred_clean,
                            "gold_label": gold_label,
                            "pred_label": pred_label,
                            "bucket": bucket,
                            "dataset": dataset_name,
                        }
                    )
                else:
                    pred_clean = pred_full
                    gold_norm = normalize_label(gold) or "unknown"
                    pred_norm = normalize_label(pred_clean) or "unknown"
                    cls_stats.update(gold_norm, pred_norm)
                    classification_rows.append(
                        {
                            "id": ex_id,
                            "gold_norm": gold_norm,
                            "pred_norm": pred_norm,
                            "bucket": bucket,
                            "dataset": dataset_name,
                        }
                    )
                    gold_out = (
                        classification_aliases.get(gold_norm, gold) if classification_aliases else gold
                    )

                rec = {
                    "id": ex_id,
                    "input_text": src,
                    "gold": gold_clean if args.task == "translation" else gold_out,
                    "pred_raw": pred_clean,
                }
                if args.task == "translation":
                    if source_id is not None:
                        rec["source_id"] = source_id
                    if direction is not None:
                        rec["direction"] = direction
                    if bucket:
                        rec["bucket"] = bucket
                    if dataset_name:
                        rec["dataset"] = dataset_name
                    gold_with_label = normalize_generation_text(gold)
                    if gold_with_label != rec["gold"]:
                        rec["gold_with_label"] = gold_with_label
                    if pred_full != pred_clean:
                        rec["pred_with_label"] = pred_full
                    if gold_label is not None:
                        rec["gold_source_variant_norm"] = gold_label
                    if pred_label is not None:
                        rec["pred_source_variant_norm"] = pred_label
                if args.task == "classification":
                    rec["gold_norm"] = normalize_label(gold) or "unknown"
                    rec["pred_norm"] = normalize_label(pred_clean) or "unknown"
                    if bucket:
                        rec["bucket"] = bucket
                    if dataset_name:
                        rec["dataset"] = dataset_name
                    if pred_info.get("scores") is not None:
                        rec["candidate_scores"] = pred_info["scores"]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.task == "translation":
        summary = {
            "task": "translation",
            **build_translation_summary(translation_rows),
        }
        per_direction: dict[str, Any] = {}
        for direction in sorted(
            {str(row["direction"]) for row in translation_rows if row.get("direction")}
        ):
            per_direction[direction] = build_translation_summary(
                [row for row in translation_rows if row.get("direction") == direction]
            )
        if per_direction:
            summary["available_directions"] = sorted(per_direction)
            summary["per_direction"] = per_direction
        per_bucket: dict[str, Any] = {}
        for bucket in FRMT_BUCKETS:
            bucket_rows = [row for row in translation_rows if row.get("bucket") == bucket]
            if not bucket_rows:
                continue
            per_bucket[bucket] = build_translation_summary(bucket_rows)
        if per_bucket:
            summary["available_buckets"] = sorted(per_bucket)
            summary["per_bucket"] = per_bucket
    else:
        summary = {
            "task": "classification",
            "classification_mode": args.classification_mode,
            "classification_candidates": classification_candidates,
            "classification_candidate_metadata": classification_candidate_meta,
            "filtered_out_counts": filtered_out_counts or {},
            **cls_stats.report(),
        }
        per_bucket: dict[str, Any] = {}
        for bucket in FRMT_BUCKETS:
            bucket_rows = [row for row in classification_rows if row.get("bucket") == bucket]
            if not bucket_rows:
                continue
            per_bucket[bucket] = build_classification_summary(bucket_rows)
        if per_bucket:
            summary["available_buckets"] = sorted(per_bucket)
            summary["per_bucket"] = per_bucket

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Saved predictions: {pred_path}")
    print(f"Saved summary: {summary_path}")
    if args.task == "translation":
        overall_worst = summary.get("worst_sentence_bleu_examples", [])
        print(f"Worst {len(overall_worst)} sentence-BLEU examples:")
        for ex in overall_worst:
            print(f"  id={ex['id']} sentence_bleu={ex['sentence_bleu']:.4f}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
