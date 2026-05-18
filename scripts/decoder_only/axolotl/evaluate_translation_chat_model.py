#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from peft import PeftModel
from sacrebleu import corpus_bleu, sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_TRANSLATION = (
    "Es um assistente especialista em portugues europeu e portugues do Brasil. "
    "A tua tarefa e converter frases entre as duas variantes, mantendo o significado, "
    "o registo e um estilo natural. Responde apenas com a traducao final, "
    "sem explicacoes, sem comentarios e sem texto adicional."
)

USER_TRANSL_BR2PT = (
    "Converte o seguinte texto de portugues do Brasil para portugues europeu, "
    "mantendo o sentido e soando natural em portugues europeu. "
    "Responde apenas com a frase convertida.\n\n"
    "Texto: {source}"
)

USER_TRANSL_PT2BR = (
    "Converte o seguinte texto de portugues europeu para portugues do Brasil, "
    "mantendo o sentido e soando natural em portugues do Brasil. "
    "Responde apenas com a frase convertida.\n\n"
    "Texto: {source}"
)

TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", flags=re.DOTALL)
COMMON_PREFIX_RE = re.compile(
    r"^\s*(?:traducao|traducao final|texto convertido|resultado|resposta|a traducao e)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)


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
    return normalize_space(TASK_PREFIX_RE.sub("", text or ""))


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

    if disable_thinking:
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            pass

    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
            target = normalize_space(str(row.get("target_text") or ""))
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


def word_edit_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> int:
    if not ref_tokens:
        return len(hyp_tokens)
    if not hyp_tokens:
        return len(ref_tokens)

    prev = list(range(len(hyp_tokens) + 1))
    for i, ref_tok in enumerate(ref_tokens, start=1):
        curr = [i]
        for j, hyp_tok in enumerate(hyp_tokens, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def word_error_rate(hyp_text: str, ref_text: str) -> float:
    ref_tokens = normalize_space(ref_text).split()
    hyp_tokens = normalize_space(hyp_text).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return word_edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def corpus_word_error_rate(hyps: list[str], refs: list[str]) -> float:
    total_edits = 0
    total_ref_tokens = 0
    for hyp_text, ref_text in zip(hyps, refs):
        ref_tokens = normalize_space(ref_text).split()
        hyp_tokens = normalize_space(hyp_text).split()
        if not ref_tokens:
            if hyp_tokens:
                total_edits += len(hyp_tokens)
            continue
        total_edits += word_edit_distance(ref_tokens, hyp_tokens)
        total_ref_tokens += len(ref_tokens)
    if total_ref_tokens == 0:
        return 0.0
    return total_edits / total_ref_tokens


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

    refs: list[str] = []
    hyps: list[str] = []
    copy_hyps: list[str] = []
    model_better_count = 0
    model_wer_better_count = 0
    copy_better_or_equal_count = 0
    copy_wer_better_or_equal_count = 0
    exact_input_copy_count = 0

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
                pred = clean_generation_text(pred_raw)
                ref = normalize_space(row["target"])
                copy_hyp = normalize_space(row["source"])
                refs.append(ref)
                hyps.append(pred)
                copy_hyps.append(copy_hyp)
                exact_input_copy_count += int(pred == copy_hyp)

                model_sent_bleu = sentence_bleu(pred, [ref]).score
                copy_sent_bleu = sentence_bleu(copy_hyp, [ref]).score
                model_sent_wer = word_error_rate(pred, ref)
                copy_sent_wer = word_error_rate(copy_hyp, ref)

                model_score = model_vs_copy_score_0_100(model_sent_bleu, copy_sent_bleu)
                if model_score is not None and model_score >= 50.0:
                    model_better_count += 1
                if model_score is not None and model_score <= 50.0:
                    copy_better_or_equal_count += 1

                model_wer_score = lower_is_better_vs_copy_score_0_100(model_sent_wer, copy_sent_wer)
                if model_wer_score is not None and model_wer_score >= 50.0:
                    model_wer_better_count += 1
                if model_wer_score is not None and model_wer_score <= 50.0:
                    copy_wer_better_or_equal_count += 1

                out_fh.write(
                    json.dumps(
                        {
                            "id": row.get("id"),
                            "dataset": row.get("dataset"),
                            "bucket": row.get("bucket"),
                            "direction": row["direction"],
                            "input_text": row["source"],
                            "gold": ref,
                            "pred_raw": pred,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    model_bleu = corpus_bleu(hyps, [refs]).score if refs else float("nan")
    copy_baseline_bleu = corpus_bleu(copy_hyps, [refs]).score if refs else float("nan")
    model_wer = corpus_word_error_rate(hyps, refs) if refs else float("nan")
    copy_baseline_wer = corpus_word_error_rate(copy_hyps, refs) if refs else float("nan")
    model_score_0_100 = model_vs_copy_score_0_100(model_bleu, copy_baseline_bleu)
    model_wer_score_0_100 = lower_is_better_vs_copy_score_0_100(model_wer, copy_baseline_wer)

    summary = {
        "task": "translation",
        "dataset_path": args.dataset_path.as_posix(),
        "n": len(refs),
        "bleu": model_bleu,
        "copy_baseline_bleu": copy_baseline_bleu,
        "wer": model_wer,
        "copy_baseline_wer": copy_baseline_wer,
        "model_vs_copy_score_0_100": model_score_0_100,
        "model_vs_copy_wer_score_0_100": model_wer_score_0_100,
        "model_beats_copy_flag_score_gt_50": (
            bool(model_score_0_100 > 50.0) if model_score_0_100 is not None else None
        ),
        "model_beats_copy_flag_wer_score_gt_50": (
            bool(model_wer_score_0_100 > 50.0) if model_wer_score_0_100 is not None else None
        ),
        "sentence_model_beats_copy_rate_score_gt_50": model_better_count / len(refs) if refs else 0.0,
        "sentence_model_beats_copy_rate_wer": model_wer_better_count / len(refs) if refs else 0.0,
        "sentence_copy_better_or_equal_rate": copy_better_or_equal_count / len(refs) if refs else 0.0,
        "sentence_copy_better_or_equal_rate_wer": (
            copy_wer_better_or_equal_count / len(refs) if refs else 0.0
        ),
        "exact_input_copy_rate": exact_input_copy_count / len(refs) if refs else 0.0,
        "predictions_path": pred_path.as_posix(),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
