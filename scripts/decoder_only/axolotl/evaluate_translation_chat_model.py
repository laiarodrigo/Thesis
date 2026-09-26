#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from peft import PeftModel
from sacrebleu import corpus_bleu, sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer

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


SYSTEM_TRANSLATION = (
    "És um assistente especialista em português europeu e português do Brasil. "
    "A tua tarefa é converter frases entre as duas variantes, mantendo o significado, "
    "o registo e um estilo natural. Responde apenas com a tradução final, "
    "sem explicações, sem comentários e sem texto adicional."
)

USER_TRANSL_BR2PT = (
    "Converte o seguinte texto de português do Brasil para português europeu, "
    "mantendo o sentido e soando natural em português europeu. "
    "Responde apenas com a frase convertida.\n\n"
    "Texto: {source}"
)

USER_TRANSL_PT2BR = (
    "Converte o seguinte texto de português europeu para português do Brasil, "
    "mantendo o sentido e soando natural em português do Brasil. "
    "Responde apenas com a frase convertida.\n\n"
    "Texto: {source}"
)

ENCODER_TASK_PREFIX_RE = re.compile(
    r"^\s*(?:<(br-pt|pt-br|pt-pt|id|cls)>|((?:BR|PT|CLS)\b))(?:\s*:\s*|\s+)",
    flags=re.IGNORECASE,
)
DECODER_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:<(?:pt-br|pt-pt)>\s*:?\s*|(?:BR|PT|pt-br|pt-pt)\b(?:\s*:\s*|\s+))",
    flags=re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", flags=re.DOTALL)
COMMON_PREFIX_RE = re.compile(
    r"^\s*(?:tradu[cç][aã]o|tradu[cç][aã]o final|texto convertido|resultado|resposta|a tradu[cç][aã]o [ée])\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
FRMT_BUCKETS = ("random", "entity", "lexical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decoder-only translation chat models on JSONL data.")
    parser.add_argument("--model-id", required=True, help="Base model id or merged local model dir.")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="Optional LoRA/QLoRA adapter dir.")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Optional tokenizer override.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split())


def strip_task_prefix(text: str) -> str:
    raw = text or ""
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_space(raw)


def strip_decoder_label_prefix(text: str) -> str:
    raw = normalize_space(text or "")
    match = DECODER_LABEL_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_space(raw)


def normalize_label(text: str) -> str | None:
    raw = normalize_space(text or "").lower()
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


def infer_direction(row: dict[str, Any]) -> str | None:
    value = normalize_space(str(row.get("direction") or row.get("task") or "")).casefold()
    if value in {"translate_br2pt", "br2pt"}:
        return "br2pt"
    if value in {"translate_pt2br", "pt2br"}:
        return "pt2br"

    input_text = normalize_space(str(row.get("input_text") or ""))
    if input_text.startswith("<br-pt>"):
        return "br2pt"
    if input_text.startswith("<pt-br>"):
        return "pt2br"
    return None


def extract_source(row: dict[str, Any]) -> str:
    source = normalize_space(str(row.get("source_text") or ""))
    if source:
        return source
    return strip_task_prefix(str(row.get("input_text") or ""))


def build_prompt(tok, source: str, direction: str, *, disable_thinking: bool) -> str:
    if direction == "br2pt":
        user_content = USER_TRANSL_BR2PT.format(source=source)
    elif direction == "pt2br":
        user_content = USER_TRANSL_PT2BR.format(source=source)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    messages = [
        {"role": "system", "content": SYSTEM_TRANSLATION},
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
        f"System: {SYSTEM_TRANSLATION}\n\n"
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


def iter_translation_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            direction = infer_direction(row)
            source = extract_source(row)
            target_raw = normalize_space(str(row.get("target_text") or ""))
            target = strip_decoder_label_prefix(target_raw)
            if direction is None or not source or not target:
                continue
            rows.append(
                {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "bucket": row.get("bucket"),
                    "direction": direction,
                    "source": source,
                    "target": target,
                    "target_raw": target_raw,
                }
            )
    return rows


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
        ref_tokens = normalize_space(gold_clean).split()
        pred_tokens = normalize_space(pred_clean).split()
        copy_tokens = normalize_space(src_clean).split()

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
    worst_bleu_examples = worst_bleu_examples[:10]

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
    if total_ter_ref_tokens == 0:
        model_ter = 0.0 if total_model_ter_edits == 0 else 1.0
        copy_baseline_ter = 0.0 if total_copy_ter_edits == 0 else 1.0
    else:
        model_ter = total_model_ter_edits / total_ter_ref_tokens
        copy_baseline_ter = total_copy_ter_edits / total_ter_ref_tokens
    model_ter_score_0_100 = lower_is_better_vs_copy_score_0_100(model_ter, copy_baseline_ter)
    if total_wer_ref_tokens == 0:
        model_wer = 0.0 if total_model_wer_edits == 0 else 1.0
        copy_baseline_wer = 0.0 if total_copy_wer_edits == 0 else 1.0
    else:
        model_wer = total_model_wer_edits / total_wer_ref_tokens
        copy_baseline_wer = total_copy_wer_edits / total_wer_ref_tokens
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = iter_translation_rows(args.dataset_path)
    if not rows:
        raise RuntimeError(f"No valid translation rows found in {args.dataset_path}")

    model, tok = load_model_and_tokenizer(
        args.model_id,
        args.adapter_dir,
        args.tokenizer_path,
        trust_remote_code=bool(args.trust_remote_code),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = args.output_dir / f"{timestamp}_translation_predictions.jsonl"
    summary_path = args.output_dir / f"{timestamp}_translation_summary.json"

    translation_rows: list[dict[str, Any]] = []

    with pred_path.open("w", encoding="utf-8") as out_fh:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            prompts = [
                build_prompt(tok, row["source"], row["direction"], disable_thinking=bool(args.disable_thinking))
                for row in chunk
            ]
            raw_preds = generate_batch(
                model,
                tok,
                prompts,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
            )
            for row, pred_raw in zip(chunk, raw_preds):
                pred_cleaned = clean_generation_text(pred_raw)
                pred = strip_decoder_label_prefix(pred_cleaned)
                ref = normalize_space(row["target"])
                copy_hyp = normalize_space(row["source"])
                gold_label = normalize_label(row.get("target_raw"))
                pred_label = normalize_label(pred_cleaned)
                bucket = normalize_bucket(row.get("bucket"))

                translation_rows.append(
                    {
                        "id": row.get("id"),
                        "direction": row["direction"],
                        "src": copy_hyp,
                        "gold": ref,
                        "pred": pred,
                        "gold_label": gold_label,
                        "pred_label": pred_label,
                        "bucket": bucket,
                    }
                )

                rec = {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "bucket": bucket,
                    "direction": row["direction"],
                    "input_text": row["source"],
                    "gold_raw": row.get("target_raw"),
                    "gold": ref,
                    "pred_raw": pred_cleaned,
                    "pred": pred,
                }
                if gold_label is not None:
                    rec["gold_source_variant_norm"] = gold_label
                if pred_label is not None:
                    rec["pred_source_variant_norm"] = pred_label
                out_fh.write(
                    json.dumps(rec, ensure_ascii=False)
                    + "\n"
                )

    summary: dict[str, Any] = {
        "task": "translation",
        "eval_config": {
            "dataset_path": args.dataset_path.as_posix(),
            "model_id": args.model_id,
            "adapter_dir": args.adapter_dir.as_posix() if args.adapter_dir else None,
            "tokenizer_path": args.tokenizer_path.as_posix() if args.tokenizer_path else None,
            "batch_size": args.batch_size,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "trust_remote_code": bool(args.trust_remote_code),
            "disable_thinking": bool(args.disable_thinking),
        },
        **build_translation_summary(translation_rows),
        "predictions_path": pred_path.as_posix(),
    }
    per_direction: dict[str, Any] = {}
    for direction in sorted({str(row["direction"]) for row in translation_rows if row.get("direction")}):
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

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
