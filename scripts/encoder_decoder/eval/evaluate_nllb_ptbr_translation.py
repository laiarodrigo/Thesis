#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from sacrebleu import corpus_bleu, sentence_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.encoder_decoder.eval.metrics_utils import corpus_ter, sentence_ter
except ModuleNotFoundError:
    from metrics_utils import corpus_ter, sentence_ter


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
        description=(
            "Evaluate NLLB as a zero-shot Portuguese-to-Portuguese translation baseline. "
            "NLLB has only generic Portuguese (por_Latn), so this is not variant-controlled."
        )
    )
    parser.add_argument("--model-id", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/encoder_decoder/t5gemma2/control_string_ptbr_eval"),
    )
    parser.add_argument("--view", default="encoder_unified")
    parser.add_argument("--datasets", nargs="+", default=["golden", "frmt"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("eval_results/encoder_decoder/external_seq2seq_prompt_models"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--src-lang", default="por_Latn")
    parser.add_argument("--tgt-lang", default="por_Latn")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    return parser.parse_args()


def normalize_space(text: object) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def strip_encoder_prefix(text: object) -> str:
    raw = str(text or "")
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_space(raw)


def strip_decoder_label_prefix(text: object) -> str:
    raw = normalize_space(text)
    match = DECODER_LABEL_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_space(raw)


def infer_direction(row: dict[str, Any]) -> str | None:
    raw = normalize_space(row.get("direction") or row.get("task")).lower()
    if raw in {"translate_br2pt", "br2pt", "br-pt", "<br-pt>"}:
        return "br2pt"
    if raw in {"translate_pt2br", "pt2br", "pt-br", "<pt-br>"}:
        return "pt2br"
    match = ENCODER_TASK_PREFIX_RE.match(str(row.get("input_text") or ""))
    if match:
        prefix = (match.group(1) or match.group(2) or "").lower()
        if prefix in {"br-pt", "br"}:
            return "br2pt"
        if prefix in {"pt-br", "pt"}:
            return "pt2br"
    return None


def extract_source(row: dict[str, Any]) -> str:
    for key in ("source_text", "text"):
        value = normalize_space(row.get(key))
        if value:
            return value
    return strip_encoder_prefix(row.get("input_text"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def forced_bos_token_id(tokenizer, tgt_lang: str) -> int:
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_to_id, dict) and tgt_lang in lang_code_to_id:
        return int(lang_code_to_id[tgt_lang])
    token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise RuntimeError(f"Could not resolve target language token id for {tgt_lang!r}.")
    return int(token_id)


def load_model(model_id: str, src_lang: str, trust_remote_code: bool):
    tok = AutoTokenizer.from_pretrained(model_id, src_lang=src_lang, trust_remote_code=trust_remote_code)
    tok.src_lang = src_lang
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tok


def generate_batch(
    model,
    tokenizer,
    sources: list[str],
    *,
    max_source_length: int,
    max_new_tokens: int,
    tgt_lang: str,
) -> list[str]:
    device = next(model.parameters()).device
    enc = tokenizer(
        sources,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            forced_bos_token_id=forced_bos_token_id(tokenizer, tgt_lang),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return [normalize_space(text) for text in tokenizer.batch_decode(out, skip_special_tokens=True)]


def compute_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    preds = [row["pred"] for row in rows]
    refs = [row["gold"] for row in rows]
    return {
        "n": len(rows),
        "bleu": corpus_bleu(preds, [refs]).score if rows else 0.0,
        "ter": corpus_ter(preds, refs) if rows else 0.0,
    }


def eval_dataset(
    model,
    tokenizer,
    dataset_path: Path,
    output_dir: Path,
    *,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
    tgt_lang: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw in iter_jsonl(dataset_path):
        direction = infer_direction(raw)
        source = extract_source(raw)
        gold = strip_decoder_label_prefix(raw.get("target_text") or raw.get("gold"))
        if direction and source and gold:
            rows.append(
                {
                    "id": raw.get("id"),
                    "dataset": raw.get("dataset"),
                    "bucket": raw.get("bucket"),
                    "direction": direction,
                    "source": source,
                    "gold": gold,
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = output_dir / f"{timestamp}_translation_predictions.jsonl"
    eval_rows: list[dict[str, Any]] = []
    with pred_path.open("w", encoding="utf-8") as out:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            preds = generate_batch(
                model,
                tokenizer,
                [row["source"] for row in chunk],
                max_source_length=max_source_length,
                max_new_tokens=max_new_tokens,
                tgt_lang=tgt_lang,
            )
            for row, pred in zip(chunk, preds):
                rec = {
                    **row,
                    "input_text": row["source"],
                    "pred": pred,
                    "sentence_bleu": sentence_bleu(pred, [row["gold"]]).score,
                    "sentence_ter": sentence_ter(pred, row["gold"]),
                }
                eval_rows.append(rec)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "task": "translation",
        "dataset_path": dataset_path.as_posix(),
        "predictions_path": pred_path.as_posix(),
        "note": "NLLB uses generic Portuguese por_Latn for both source and target; this is not variant-controlled.",
        **compute_report(eval_rows),
        "per_direction": {
            direction: compute_report([row for row in eval_rows if row["direction"] == direction])
            for direction in sorted({row["direction"] for row in eval_rows})
        },
    }
    summary_path = output_dir / f"{timestamp}_translation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args.model_id, args.src_lang, bool(args.trust_remote_code))
    root_out = args.output_root / args.run_name / "control_strings_ptbr" / args.view
    for dataset in args.datasets:
        dataset_dir = args.data_root / dataset / args.view
        eval_dataset(
            model,
            tokenizer,
            dataset_dir / "translation_test.jsonl",
            root_out / dataset / "translation",
            batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens,
            tgt_lang=args.tgt_lang,
        )


if __name__ == "__main__":
    main()
