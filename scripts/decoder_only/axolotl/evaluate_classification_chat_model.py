#!/usr/bin/env python3
from __future__ import annotations

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
EVAL_DIR = REPO_ROOT / "scripts" / "encoder_decoder" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from metrics_utils import PT_VARIANT_LABELS
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.metrics_utils import PT_VARIANT_LABELS


LABELS = ("pt-br", "pt-pt", "equal")
FRMT_BUCKETS = ("random", "entity", "lexical")

SYSTEM_CLASSIFICATION = (
    "És um linguista especialista em português europeu e português do Brasil. "
    "A tua tarefa é identificar a variante correta do texto."
)

USER_CLASSIFICATION = (
    "Classifica a variante do texto como uma destas etiquetas: {labels}. "
    "Usa 'equal' apenas quando o texto é igual nas duas variantes. "
    "Responde apenas com uma etiqueta.\n\n"
    "Texto: {source}"
)

THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", flags=re.DOTALL)
COMMON_PREFIX_RE = re.compile(
    r"^\s*(?:resposta|r[oó]tulo|etiqueta|classifica[cç][aã]o|resultado)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
TASK_PREFIX_RE = re.compile(
    r"^\s*(?:<(br-pt|pt-br|pt-pt|id|cls)>|((?:BR|PT|CLS)\b))(?:\s*:\s*|\s+)",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decoder-only chat models on classification JSONL data.")
    parser.add_argument("--model-id", required=True, help="Base model id or merged local model dir.")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="Optional LoRA/QLoRA adapter dir.")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Optional tokenizer override.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--classification-mode",
        choices=("score-sequences", "generate"),
        default="score-sequences",
        help="Candidate scoring is the safest default for chat models.",
    )
    parser.add_argument(
        "--classification-candidates",
        nargs="+",
        default=None,
        help="Optional explicit labels, e.g. 'pt-br pt-pt equal'.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split())


def strip_task_prefix(text: str) -> str:
    return normalize_space(TASK_PREFIX_RE.sub("", text or ""))


def normalize_label(text: str) -> Optional[str]:
    t = normalize_space(text or "").lower()
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


def normalize_bucket(raw_bucket: object) -> str:
    text = normalize_space(str(raw_bucket or "")).lower()
    if text in {"rand", "random"}:
        return "random"
    if text in {"entity", "entities"}:
        return "entity"
    if text in {"lexical", "lex"}:
        return "lexical"
    if not text:
        return "n/a"
    return text


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
        return [normalize_space(str(value or "")) for value in values]
    raise KeyError(
        "Classification eval batch must contain one of: target_text, label, gold. "
        f"Available keys: {sorted(batch.keys())}"
    )


def extract_source_texts(batch: dict[str, Any]) -> list[str]:
    values = batch.get("text")
    if values is not None:
        return [normalize_space(str(value or "")) for value in values]
    values = batch.get("input_text")
    if values is not None:
        return [strip_task_prefix(str(value or "")) for value in values]
    raise KeyError(
        "Classification eval batch must contain one of: text, input_text. "
        f"Available keys: {sorted(batch.keys())}"
    )


def build_classification_candidates(ds, explicit_candidates: Optional[list[str]]) -> list[str]:
    seen = set()
    candidates: list[str] = []
    raw_values = explicit_candidates if explicit_candidates else classification_target_values(ds)
    for raw in raw_values:
        cand = normalize_space(str(raw or ""))
        if not cand or cand in seen:
            continue
        seen.add(cand)
        candidates.append(cand)
    if not candidates:
        raise ValueError("No valid classification candidates found.")
    return candidates


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


def load_model_and_tokenizer(
    model_id: str,
    adapter_dir: Optional[Path],
    tokenizer_path: Optional[Path],
    *,
    trust_remote_code: bool,
):
    tokenizer_candidates: list[str] = []
    if tokenizer_path is not None:
        tokenizer_candidates.append(tokenizer_path.as_posix())
    if adapter_dir is not None:
        tokenizer_candidates.append(adapter_dir.as_posix())
    tokenizer_candidates.append(model_id)

    tok = None
    for cand in tokenizer_candidates:
        try:
            tok = AutoTokenizer.from_pretrained(
                cand,
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )
            print(f"Tokenizer loaded from: {cand}")
            break
        except Exception:
            continue
    if tok is None:
        raise RuntimeError("Unable to load tokenizer from tokenizer-path/adapter/model.")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(base_model, adapter_dir.as_posix())
    else:
        model = base_model
    model.eval()
    if not torch.cuda.is_available():
        model.to("cpu")
    return model, tok


def build_prompt(tok, source: str, candidates: list[str], *, disable_thinking: bool) -> str:
    user_content = USER_CLASSIFICATION.format(
        labels=", ".join(candidates),
        source=source,
    )
    messages = [
        {"role": "system", "content": SYSTEM_CLASSIFICATION},
        {"role": "user", "content": user_content},
    ]

    apply_template = getattr(tok, "apply_chat_template", None)
    if callable(apply_template):
        if disable_thinking:
            try:
                return apply_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                pass
        try:
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return (
        f"System: {SYSTEM_CLASSIFICATION}\n\n"
        f"User: {user_content}\n\n"
        "Assistant:"
    )


def clean_generation_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = THINK_RE.sub("", cleaned).strip()
    cleaned = CODE_FENCE_RE.sub("", cleaned).strip()
    cleaned = COMMON_PREFIX_RE.sub("", cleaned).strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    return normalize_space(cleaned)


def pad_rows(rows: list[list[int]], pad_token_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(row) for row in rows)
    input_ids = []
    attention_masks = []
    for row in rows:
        pad_len = max_len - len(row)
        input_ids.append([pad_token_id] * pad_len + row)
        attention_masks.append([0] * pad_len + [1] * len(row))
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_masks, dtype=torch.long, device=device),
    )


def score_classification_candidates_batch(
    model,
    tok,
    *,
    prompts: list[str],
    candidates: list[str],
    max_input_length: int,
    reserve_candidate_tokens: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    prompt_max_length = max(1, int(max_input_length) - int(reserve_candidate_tokens))
    prompt_rows = tok(
        prompts,
        add_special_tokens=False,
        truncation=True,
        max_length=prompt_max_length,
    )["input_ids"]

    batch_scores: list[dict[str, float]] = [dict() for _ in prompts]
    for cand in candidates:
        cand_ids = tok.encode(cand, add_special_tokens=False)
        if not cand_ids:
            raise ValueError(f"Candidate {cand!r} tokenized to an empty sequence.")

        model_rows: list[list[int]] = []
        loss_masks: list[list[int]] = []
        for prompt_ids in prompt_rows:
            row_ids = list(prompt_ids) + list(cand_ids)
            model_rows.append(row_ids)
            loss_masks.append([0] * len(prompt_ids) + [1] * len(cand_ids))

        input_ids, attention_mask = pad_rows(model_rows, int(tok.pad_token_id), device)
        loss_mask, _ = pad_rows(loss_masks, 0, device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            shift_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            shift_target_ids = input_ids[:, 1:]
            token_log_probs = shift_log_probs.gather(
                -1,
                shift_target_ids.unsqueeze(-1),
            ).squeeze(-1)
            shift_loss_mask = loss_mask[:, 1:].to(dtype=token_log_probs.dtype)
            seq_scores = (token_log_probs * shift_loss_mask).sum(dim=-1) / shift_loss_mask.sum(dim=-1).clamp_min(1.0)

        for row_idx in range(len(prompts)):
            batch_scores[row_idx][cand] = float(seq_scores[row_idx].item())

    results: list[dict[str, Any]] = []
    for scores in batch_scores:
        pred_text = max(scores.items(), key=lambda kv: kv[1])[0]
        results.append(
            {
                "pred_text": pred_text,
                "scores": scores,
            }
        )
    return results


def generate_batch(
    model,
    tok,
    prompts: list[str],
    *,
    max_input_length: int,
    max_new_tokens: int,
) -> list[str]:
    device = next(model.parameters()).device
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=False,
    ).to(device)

    eos_token_id = tok.eos_token_id
    if eos_token_id is None:
        cfg_eos = getattr(model.config, "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple)):
            eos_token_id = cfg_eos[0] if cfg_eos else None
        else:
            eos_token_id = cfg_eos

    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos_token_id,
        )

    prompt_width = enc["input_ids"].shape[1]
    preds: list[str] = []
    for i in range(out.size(0)):
        gen_ids = out[i, prompt_width:]
        preds.append(tok.decode(gen_ids, skip_special_tokens=True).strip())
    return preds


@dataclass
class ClsStats:
    total: int = 0
    correct: int = 0
    tp: Counter | None = None
    fp: Counter | None = None
    fn: Counter | None = None
    cm: Counter | None = None

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

    def report(self) -> dict[str, Any]:
        acc = self.correct / self.total if self.total else 0.0
        per_class = {}
        f1s = []
        target_f1s = []
        for lab in LABELS:
            tp = self.tp[lab]
            fp = self.fp[lab]
            fn = self.fn[lab]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_class[lab] = {
                "precision": precision,
                "recall": recall,
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
                f"{gold}->{pred}": int(self.cm[(gold, pred)])
                for gold in LABELS
                for pred in LABELS
            },
        }


def build_classification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = ClsStats()
    for row in rows:
        stats.update(str(row["gold_norm"]), str(row["pred_norm"]))
    return stats.report()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model_and_tokenizer(
        args.model_id,
        args.adapter_dir,
        args.tokenizer_path,
        trust_remote_code=bool(args.trust_remote_code),
    )

    ds = load_dataset("json", data_files={"eval": args.dataset_path.as_posix()})["eval"]
    classification_candidates = build_classification_candidates(ds, args.classification_candidates)
    ds, filtered_out_counts = filter_classification_dataset_for_candidates(ds, classification_candidates)
    classification_candidate_meta = classification_candidate_metadata(tok, classification_candidates)
    reserve_candidate_tokens = max((meta["core_token_count"] for meta in classification_candidate_meta), default=1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = args.output_dir / f"{run_id}_classification_predictions.jsonl"
    summary_path = args.output_dir / f"{run_id}_classification_summary.json"

    cls_stats = ClsStats()
    classification_rows: list[dict[str, Any]] = []

    with pred_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(ds), args.batch_size):
            batch = ds[start : start + args.batch_size]
            source_texts = extract_source_texts(batch)
            gold_targets = extract_classification_targets(batch)
            prompts = [
                build_prompt(
                    tok,
                    source,
                    classification_candidates,
                    disable_thinking=bool(args.disable_thinking),
                )
                for source in source_texts
            ]

            if args.classification_mode == "score-sequences":
                pred_infos = score_classification_candidates_batch(
                    model,
                    tok,
                    prompts=prompts,
                    candidates=classification_candidates,
                    max_input_length=args.max_input_length,
                    reserve_candidate_tokens=reserve_candidate_tokens,
                )
            else:
                raw_preds = generate_batch(
                    model,
                    tok,
                    prompts,
                    max_input_length=args.max_input_length,
                    max_new_tokens=args.max_new_tokens,
                )
                pred_infos = [
                    {
                        "pred_text": clean_generation_text(raw_pred),
                        "scores": None,
                    }
                    for raw_pred in raw_preds
                ]

            for row_idx, (source_text, gold_raw, pred_info) in enumerate(zip(source_texts, gold_targets, pred_infos)):
                gold_norm = normalize_label(gold_raw) or "unknown"
                pred_raw = clean_generation_text(str(pred_info["pred_text"] or ""))
                pred_norm = normalize_label(pred_raw) or "unknown"
                cls_stats.update(gold_norm, pred_norm)

                raw_bucket_values = batch.get("bucket")
                raw_dataset_values = batch.get("dataset")
                raw_source_id_values = batch.get("source_id")

                rec: dict[str, Any] = {
                    "id": int(start + row_idx),
                    "input_text": source_text,
                    "gold": gold_raw,
                    "gold_norm": gold_norm,
                    "pred_raw": pred_raw,
                    "pred_norm": pred_norm,
                }
                if raw_source_id_values is not None:
                    rec["source_id"] = raw_source_id_values[row_idx]
                if raw_dataset_values is not None:
                    rec["dataset"] = str(raw_dataset_values[row_idx])
                if raw_bucket_values is not None:
                    rec["bucket"] = normalize_bucket(raw_bucket_values[row_idx])
                if pred_info.get("scores") is not None:
                    rec["candidate_scores"] = pred_info["scores"]

                classification_rows.append(
                    {
                        "gold_norm": gold_norm,
                        "pred_norm": pred_norm,
                        "bucket": rec.get("bucket", "n/a"),
                    }
                )
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "task": "classification",
        "eval_config": {
            "dataset_path": args.dataset_path.as_posix(),
            "model_id": args.model_id,
            "adapter_dir": args.adapter_dir.as_posix() if args.adapter_dir else None,
            "tokenizer_path": args.tokenizer_path.as_posix() if args.tokenizer_path else None,
            "batch_size": args.batch_size,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "classification_mode": args.classification_mode,
            "classification_candidates": args.classification_candidates,
            "trust_remote_code": bool(args.trust_remote_code),
            "disable_thinking": bool(args.disable_thinking),
        },
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
