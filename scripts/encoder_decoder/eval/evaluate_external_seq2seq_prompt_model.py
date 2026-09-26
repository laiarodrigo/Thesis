#!/usr/bin/env python3
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
from sacrebleu import corpus_bleu, sentence_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.encoder_decoder.eval.metrics_utils import corpus_ter, sentence_ter
except ModuleNotFoundError:
    from metrics_utils import corpus_ter, sentence_ter


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

SYSTEM_CLASSIFICATION = (
    "És um linguista especialista em português europeu e português do Brasil. "
    "A tua tarefa é identificar a variante correta do texto."
)

USER_CLASSIFICATION = (
    "Classifica a variante do texto como uma destas opções: {labels}. "
    "Responde apenas com uma opção.\n\n"
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
COMMON_PREFIX_RE = re.compile(
    r"^\s*(?:tradu[cç][aã]o|tradu[cç][aã]o final|texto convertido|resultado|resposta|"
    r"r[oó]tulo|etiqueta|classifica[cç][aã]o|a tradu[cç][aã]o [ée])\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", flags=re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot prompt evaluation for external encoder-decoder seq2seq models."
    )
    parser.add_argument("--model-id", required=True)
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--classification-batch-size", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-new-tokens-translation", type=int, default=160)
    parser.add_argument("--max-new-tokens-classification", type=int, default=16)
    parser.add_argument(
        "--classification-mode",
        choices=("score-sequences", "generate"),
        default="score-sequences",
    )
    parser.add_argument(
        "--classification-candidates",
        nargs="+",
        default=["português do Brasil", "português europeu"],
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    return parser.parse_args()


def normalize_space(text: object) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def clean_output(text: object) -> str:
    raw = normalize_space(text)
    raw = CODE_FENCE_RE.sub("", raw).strip()
    raw = COMMON_PREFIX_RE.sub("", raw).strip()
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        raw = raw[1:-1].strip()
    return normalize_space(raw)


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


def normalize_label(text: object) -> str | None:
    raw = normalize_space(text).lower()
    if not raw:
        return None
    first = raw.split(" ", 1)[0].strip(",:;.-_")
    if first == "br":
        return "BR"
    if first == "pt":
        return "PT"
    if "pt-br" in raw or "ptbr" in raw or "brasil" in raw or "brasileir" in raw:
        return "BR"
    if "pt-pt" in raw or "ptpt" in raw or "europeu" in raw or "europeia" in raw or "portugal" in raw:
        return "PT"
    if "equal" in raw or first == "igual":
        return "equal"
    return None


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


def build_translation_prompt(source: str, direction: str) -> str:
    if direction == "br2pt":
        user = USER_TRANSL_BR2PT.format(source=source)
    elif direction == "pt2br":
        user = USER_TRANSL_PT2BR.format(source=source)
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return f"System: {SYSTEM_TRANSLATION}\n\nUser: {user}\n\nAssistant:"


def build_classification_prompt(source: str, candidates: list[str]) -> str:
    labels = " / ".join(candidates)
    user = USER_CLASSIFICATION.format(source=source, labels=labels)
    return f"System: {SYSTEM_CLASSIFICATION}\n\nUser: {user}\n\nAssistant:"


def load_model(model_id: str, trust_remote_code: bool):
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tok


def generate_batch(model, tok, inputs: list[str], *, max_source_length: int, max_new_tokens: int) -> list[str]:
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
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return [clean_output(x) for x in tok.batch_decode(out, skip_special_tokens=True)]


def pad_label_rows(rows: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(row) for row in rows)
    padded = []
    mask = []
    for row in rows:
        pad_len = max_len - len(row)
        padded.append(row + [pad_id] * pad_len)
        mask.append([1] * len(row) + [0] * pad_len)
    labels = torch.tensor(padded, dtype=torch.long, device=device)
    loss_mask = torch.tensor(mask, dtype=torch.float32, device=device)
    labels_for_model = labels.masked_fill(loss_mask.eq(0), -100)
    return labels_for_model, loss_mask


def score_classification_candidates(
    model,
    tok,
    prompts: list[str],
    candidates: list[str],
    *,
    max_source_length: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)

    all_scores: list[dict[str, float]] = [dict() for _ in prompts]
    for candidate in candidates:
        label_ids = tok(candidate, add_special_tokens=True).input_ids
        label_rows = [list(label_ids) for _ in prompts]
        labels, loss_mask = pad_label_rows(label_rows, int(tok.pad_token_id or 0), device)
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=labels,
            ).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            gather_labels = labels.masked_fill(labels.eq(-100), 0)
            token_scores = log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
            seq_scores = (token_scores * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp_min(1.0)
        for idx, score in enumerate(seq_scores.tolist()):
            all_scores[idx][candidate] = float(score)

    out = []
    for scores in all_scores:
        pred_text = max(scores.items(), key=lambda kv: kv[1])[0]
        out.append({"pred_text": pred_text, "scores": scores})
    return out


def compute_classification_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("BR", "PT")
    cm = Counter((row["gold_norm"], row["pred_norm"]) for row in rows)
    observed_preds = sorted({str(row["pred_norm"]) for row in rows} | set(labels))
    total = len(rows)
    correct = sum(1 for row in rows if row["gold_norm"] == row["pred_norm"])
    per_class = {}
    f1s = []
    for label in labels:
        tp = cm[(label, label)]
        fp = sum(cm[(gold, label)] for gold in labels if gold != label)
        fn = sum(cm[(label, pred)] for pred in observed_preds if pred != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1s.append(f1)
    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": {f"{gold}->{pred}": int(cm[(gold, pred)]) for gold in labels for pred in labels},
        "confusion_matrix_with_unknown": {
            f"{gold}->{pred}": int(cm[(gold, pred)])
            for gold in labels
            for pred in observed_preds
        },
    }


def compute_translation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    preds = [row["pred"] for row in rows]
    refs = [row["gold"] for row in rows]
    return {
        "n": len(rows),
        "bleu": corpus_bleu(preds, [refs]).score if rows else 0.0,
        "ter": corpus_ter(preds, refs) if rows else 0.0,
    }


def eval_translation_dataset(
    model,
    tok,
    dataset_path: Path,
    output_dir: Path,
    *,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
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
            prompts = [build_translation_prompt(row["source"], row["direction"]) for row in chunk]
            preds = generate_batch(
                model,
                tok,
                prompts,
                max_source_length=max_source_length,
                max_new_tokens=max_new_tokens,
            )
            for row, pred_raw in zip(chunk, preds):
                pred = strip_decoder_label_prefix(pred_raw)
                rec = {
                    **row,
                    "input_text": row["source"],
                    "pred_raw": pred_raw,
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
        **compute_translation_report(eval_rows),
        "per_direction": {
            direction: compute_translation_report([row for row in eval_rows if row["direction"] == direction])
            for direction in sorted({row["direction"] for row in eval_rows})
        },
    }
    summary_path = output_dir / f"{timestamp}_translation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def eval_classification_dataset(
    model,
    tok,
    dataset_path: Path,
    output_dir: Path,
    *,
    candidates: list[str],
    mode: str,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw in iter_jsonl(dataset_path):
        source = extract_source(raw)
        gold = normalize_label(raw.get("target_text") or raw.get("label") or raw.get("gold"))
        if source and gold in {"BR", "PT"}:
            rows.append({"id": raw.get("id"), "source": source, "gold_norm": gold})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = output_dir / f"{timestamp}_classification_predictions.jsonl"
    eval_rows: list[dict[str, Any]] = []
    with pred_path.open("w", encoding="utf-8") as out:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            prompts = [build_classification_prompt(row["source"], candidates) for row in chunk]
            if mode == "score-sequences":
                pred_infos = score_classification_candidates(
                    model,
                    tok,
                    prompts,
                    candidates,
                    max_source_length=max_source_length,
                )
            else:
                preds = generate_batch(
                    model,
                    tok,
                    prompts,
                    max_source_length=max_source_length,
                    max_new_tokens=max_new_tokens,
                )
                pred_infos = [{"pred_text": pred, "scores": None} for pred in preds]
            for row, pred_info in zip(chunk, pred_infos):
                pred_raw = clean_output(pred_info["pred_text"])
                pred_norm = normalize_label(pred_raw) or "unknown"
                rec = {
                    **row,
                    "input_text": row["source"],
                    "pred_raw": pred_raw,
                    "pred_norm": pred_norm,
                }
                if pred_info.get("scores") is not None:
                    rec["candidate_scores"] = pred_info["scores"]
                eval_rows.append(rec)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    filtered_rows = [row for row in eval_rows if row["pred_norm"] in {"BR", "PT"}]
    # Unknown predictions are counted as incorrect through an extra confusion bucket.
    report_rows = [
        row if row["pred_norm"] in {"BR", "PT"} else {**row, "pred_norm": "UNKNOWN"}
        for row in eval_rows
    ]
    summary = {
        "task": "classification",
        "dataset_path": dataset_path.as_posix(),
        "classification_mode": mode,
        "classification_candidates": candidates,
        "predictions_path": pred_path.as_posix(),
        **compute_classification_report(report_rows),
        "pred_label_counts": dict(Counter(row["pred_norm"] for row in eval_rows)),
        "valid_parsed_rate": len(filtered_rows) / len(eval_rows) if eval_rows else 0.0,
    }
    summary_path = output_dir / f"{timestamp}_classification_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    model, tok = load_model(args.model_id, bool(args.trust_remote_code))

    root_out = args.output_root / args.run_name / "control_strings_ptbr" / args.view
    for dataset in args.datasets:
        dataset_dir = args.data_root / dataset / args.view
        eval_translation_dataset(
            model,
            tok,
            dataset_dir / "translation_test.jsonl",
            root_out / dataset / "translation",
            batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens_translation,
        )
        eval_classification_dataset(
            model,
            tok,
            dataset_dir / "classification_test.jsonl",
            root_out / dataset / "classification",
            candidates=args.classification_candidates,
            mode=args.classification_mode,
            batch_size=args.classification_batch_size,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens_classification,
        )


if __name__ == "__main__":
    main()
