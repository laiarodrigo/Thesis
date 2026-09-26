#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from sacrebleu import corpus_bleu, sentence_bleu

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "scripts" / "encoder_decoder" / "eval"
for path in (REPO_ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.encoder_decoder.eval.evaluate_encdec import (  # noqa: E402
    canonicalize_translation_direction,
    load_model_and_tokenizer,
    normalize_generation_text,
    score_classification_candidates_batch,
    strip_decoder_label_prefix,
    strip_encoder_task_prefix,
)
from scripts.encoder_decoder.eval.metrics_utils import corpus_ter, sentence_ter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quality-aware translation decoding: generate beam-search plus sampled "
            "candidates, then select the candidate with highest target-variant "
            "classification probability."
        )
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model-id", default="google/t5gemma-2-4b-4b")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--family", choices=["E", "D"], required=True)
    parser.add_argument("--classification-prefix", default="CLS")
    parser.add_argument("--classification-candidates", nargs=2, default=["BR", "PT"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classification-batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.8)
    parser.add_argument("--beam-early-stopping", action="store_true")
    parser.add_argument("--adaptive-max-new-tokens", action="store_true")
    parser.add_argument("--adaptive-ratio", type=float, default=1.3)
    parser.add_argument("--adaptive-margin", type=int, default=10)
    parser.add_argument("--adaptive-min-new-tokens", type=int, default=16)
    parser.add_argument("--adaptive-max-new-tokens-ceiling", type=int, default=256)
    parser.add_argument("--num-sampled-candidates", type=int, default=9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--classification-mode", choices=["score-sequences", "score-first-token"], default="score-sequences")
    parser.add_argument(
        "--selection-strategy",
        choices=["target-prob", "conservative"],
        default="target-prob",
        help=(
            "How to choose among beam and sampled candidates. target-prob keeps the "
            "original classifier-only selector. conservative only replaces the beam "
            "when the classifier gain is large enough and length remains close to "
            "the beam/source lengths."
        ),
    )
    parser.add_argument("--min-target-prob-gain", type=float, default=0.15)
    parser.add_argument("--min-length-ratio-to-beam", type=float, default=0.70)
    parser.add_argument("--max-length-ratio-to-beam", type=float, default=1.30)
    parser.add_argument("--max-length-ratio-to-source", type=float, default=1.40)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def softmax_scores(scores: dict[str, float]) -> dict[str, float]:
    values = list(scores.values())
    max_score = max(values)
    exp_values = {key: math.exp(value - max_score) for key, value in scores.items()}
    denom = sum(exp_values.values())
    return {key: value / denom for key, value in exp_values.items()}


def target_candidate_for_direction(direction: str | None, candidates: list[str]) -> str:
    br_candidate, pt_candidate = candidates
    if direction == "br2pt":
        return pt_candidate
    if direction == "pt2br":
        return br_candidate
    raise ValueError(f"Cannot infer target variant from direction={direction!r}")


def classifier_input_for_candidate(candidate_text: str, *, family: str, classification_prefix: str) -> str:
    clean = strip_decoder_label_prefix(normalize_generation_text(candidate_text))
    if family == "E":
        return f"{classification_prefix} {clean}".strip()
    return clean


def normalize_translation_candidate(text: str) -> str:
    return strip_decoder_label_prefix(normalize_generation_text(text))


def token_count(text: str) -> int:
    return len(str(text).split())


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("inf") if numerator > 0 else 1.0
    return numerator / denominator


def select_quality_aware_candidate(
    scored_candidates: list[dict[str, Any]],
    *,
    beam: dict[str, Any],
    src_clean: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.selection_strategy == "target-prob":
        selected = max(scored_candidates, key=lambda item: item["target_variant_prob"])
        selected["selection_reason"] = "highest_target_prob"
        return selected

    beam_prob = float(beam["target_variant_prob"])
    beam_len = token_count(beam["clean_text"])
    src_len = token_count(src_clean)
    eligible: list[dict[str, Any]] = []

    for cand in scored_candidates:
        cand_len = token_count(cand["clean_text"])
        prob_gain = float(cand["target_variant_prob"]) - beam_prob
        ratio_to_beam = safe_ratio(cand_len, beam_len)
        ratio_to_source = safe_ratio(cand_len, src_len)
        cand["target_prob_gain_vs_beam"] = prob_gain
        cand["length_tokens"] = cand_len
        cand["length_ratio_to_beam"] = ratio_to_beam
        cand["length_ratio_to_source"] = ratio_to_source

        if cand is beam:
            cand["selection_reason"] = "beam_baseline"
            eligible.append(cand)
            continue
        if prob_gain < float(args.min_target_prob_gain):
            cand["selection_reason"] = "rejected_low_prob_gain"
            continue
        if ratio_to_beam < float(args.min_length_ratio_to_beam):
            cand["selection_reason"] = "rejected_too_short_vs_beam"
            continue
        if ratio_to_beam > float(args.max_length_ratio_to_beam):
            cand["selection_reason"] = "rejected_too_long_vs_beam"
            continue
        if ratio_to_source > float(args.max_length_ratio_to_source):
            cand["selection_reason"] = "rejected_too_long_vs_source"
            continue
        cand["selection_reason"] = "eligible_classifier_gain_length_safe"
        eligible.append(cand)

    selected = max(
        eligible,
        key=lambda item: (
            float(item["target_variant_prob"]),
            -abs(token_count(item["clean_text"]) - beam_len),
        ),
    )
    if selected is beam:
        selected["selection_reason"] = "kept_beam_conservative"
    return selected


def generation_ids(model, tok) -> tuple[int | None, int | None]:
    eos_token_id = tok.eos_token_id
    if eos_token_id is None:
        cfg_eos = getattr(model.config, "eos_token_id", None)
        eos_token_id = cfg_eos[0] if isinstance(cfg_eos, (list, tuple)) and cfg_eos else cfg_eos
    pad_token_id = tok.pad_token_id
    if pad_token_id is None:
        cfg_pad = getattr(model.config, "pad_token_id", None)
        pad_token_id = cfg_pad[0] if isinstance(cfg_pad, (list, tuple)) and cfg_pad else cfg_pad
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return (
        int(eos_token_id) if eos_token_id is not None else None,
        int(pad_token_id) if pad_token_id is not None else None,
    )


def per_example_cap(
    src_len: int,
    *,
    max_new_tokens: int,
    adaptive: bool,
    ratio: float,
    margin: int,
    min_tokens: int,
    ceiling: int,
) -> int:
    if not adaptive:
        return int(max_new_tokens)
    proposed = int(src_len * ratio + margin)
    proposed = max(int(min_tokens), proposed)
    proposed = min(int(ceiling), proposed)
    return max(1, proposed)


def generate_candidates_for_input(
    model,
    tok,
    input_text: str,
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    enc = tok(
        [input_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_source_length,
    ).to(device)
    src_len = int(enc["attention_mask"][0].sum().item())
    max_new_tokens = per_example_cap(
        src_len,
        max_new_tokens=args.max_new_tokens,
        adaptive=bool(args.adaptive_max_new_tokens),
        ratio=args.adaptive_ratio,
        margin=args.adaptive_margin,
        min_tokens=args.adaptive_min_new_tokens,
        ceiling=args.adaptive_max_new_tokens_ceiling,
    )
    eos_token_id, pad_token_id = generation_ids(model, tok)
    base_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": float(args.repetition_penalty),
    }
    if eos_token_id is not None:
        base_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        base_kwargs["pad_token_id"] = pad_token_id
    if args.no_repeat_ngram_size > 0:
        base_kwargs["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)

    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        beam = model.generate(
            **enc,
            do_sample=False,
            num_beams=int(args.num_beams),
            length_penalty=float(args.length_penalty),
            early_stopping=bool(args.beam_early_stopping),
            **base_kwargs,
        )
        beam_text = normalize_generation_text(tok.decode(beam[0], skip_special_tokens=True))
        outputs.append({"kind": f"beam{args.num_beams}", "text": beam_text})

        if args.num_sampled_candidates > 0:
            sampled = model.generate(
                **enc,
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                num_return_sequences=int(args.num_sampled_candidates),
                **base_kwargs,
            )
            for text in tok.batch_decode(sampled, skip_special_tokens=True):
                outputs.append({"kind": "sample", "text": normalize_generation_text(text)})

    seen = set()
    deduped: list[dict[str, Any]] = []
    for cand in outputs:
        key = normalize_translation_candidate(cand["text"])
        if key in seen:
            continue
        seen.add(key)
        cand["clean_text"] = key
        deduped.append(cand)
    return deduped


def build_translation_summary(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    refs = [row["gold"] for row in rows]
    hyps = [row[pred_key] for row in rows]
    srcs = [row["src"] for row in rows]
    return {
        "n": len(rows),
        "bleu": corpus_bleu(hyps, [refs]).score if rows else float("nan"),
        "ter": corpus_ter(hyps, refs) if rows else float("nan"),
        "copy_baseline_bleu": corpus_bleu(srcs, [refs]).score if rows else float("nan"),
        "copy_baseline_ter": corpus_ter(srcs, refs) if rows else float("nan"),
        "sentence_bleu_mean": sum(sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)) / max(len(rows), 1),
        "sentence_ter_mean": sum(sentence_ter(h, r) for h, r in zip(hyps, refs)) / max(len(rows), 1),
        "exact_input_copy_rate": sum(int(h == s) for h, s in zip(hyps, srcs)) / max(len(rows), 1),
    }


def main() -> None:
    args = parse_args()
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("json", data_files={"eval": args.dataset_path.as_posix()})["eval"]
    if args.limit is not None:
        ds = ds.select(range(min(int(args.limit), len(ds))))
    model, tok = load_model_and_tokenizer(args.model_id, args.adapter_dir, args.tokenizer_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = args.output_dir / f"{run_id}_quality_aware_predictions.jsonl"
    csv_path = args.output_dir / f"{run_id}_quality_aware_rows.csv"
    summary_path = args.output_dir / f"{run_id}_quality_aware_summary.json"

    rows: list[dict[str, Any]] = []
    candidate_count_hist: Counter[int] = Counter()
    selection_kind_counts: Counter[str] = Counter()

    with pred_path.open("w", encoding="utf-8") as pred_fh:
        for idx, row in enumerate(ds):
            input_text = str(row["input_text"])
            gold = normalize_translation_candidate(str(row.get("target_text") or row.get("gold") or ""))
            src_clean = strip_encoder_task_prefix(input_text)
            direction = canonicalize_translation_direction(row.get("direction"), input_text)
            target_candidate = target_candidate_for_direction(direction, args.classification_candidates)

            candidates = generate_candidates_for_input(model, tok, input_text, args=args)
            candidate_count_hist[len(candidates)] += 1
            classifier_inputs = [
                classifier_input_for_candidate(cand["text"], family=args.family, classification_prefix=args.classification_prefix)
                for cand in candidates
            ]
            score_infos: list[dict[str, Any]] = []
            for start in range(0, len(classifier_inputs), args.classification_batch_size):
                score_infos.extend(
                    score_classification_candidates_batch(
                        model,
                        tok,
                        inputs=classifier_inputs[start : start + args.classification_batch_size],
                        candidates=args.classification_candidates,
                        max_source_length=args.max_source_length,
                        mode=args.classification_mode,
                    )
                )

            scored_candidates: list[dict[str, Any]] = []
            for cand, score_info in zip(candidates, score_infos):
                probs = softmax_scores(score_info["scores"])
                scored_candidates.append(
                    {
                        **cand,
                        "classifier_input": classifier_input_for_candidate(
                            cand["text"],
                            family=args.family,
                            classification_prefix=args.classification_prefix,
                        ),
                        "classifier_scores": score_info["scores"],
                        "classifier_probs": probs,
                        "target_variant_prob": probs[target_candidate],
                        "target_variant_score": score_info["scores"][target_candidate],
                    }
                )
            beam_kind = f"beam{args.num_beams}"
            beam = next(item for item in scored_candidates if item["kind"] == beam_kind)
            selected = select_quality_aware_candidate(
                scored_candidates,
                beam=beam,
                src_clean=src_clean,
                args=args,
            )
            selection_kind_counts[str(selected["kind"])] += 1

            out_row = {
                "id": row.get("id", idx),
                "source_id": row.get("source_id"),
                "direction": direction,
                "target_candidate": target_candidate,
                "src": src_clean,
                "gold": gold,
                "beam_pred": beam["clean_text"],
                "selected_pred": selected["clean_text"],
                "selected_kind": selected["kind"],
                "selected_target_prob": selected["target_variant_prob"],
                "beam_target_prob": beam["target_variant_prob"],
                "selected_target_prob_gain_vs_beam": float(selected["target_variant_prob"]) - float(beam["target_variant_prob"]),
                "selected_length_tokens": token_count(selected["clean_text"]),
                "beam_length_tokens": token_count(beam["clean_text"]),
                "source_length_tokens": token_count(src_clean),
                "selected_length_ratio_to_beam": safe_ratio(token_count(selected["clean_text"]), token_count(beam["clean_text"])),
                "selected_length_ratio_to_source": safe_ratio(token_count(selected["clean_text"]), token_count(src_clean)),
                "selected_reason": selected.get("selection_reason"),
                "candidate_count": len(scored_candidates),
                "beam_sentence_bleu": sentence_bleu(beam["clean_text"], [gold]).score,
                "selected_sentence_bleu": sentence_bleu(selected["clean_text"], [gold]).score,
                "beam_sentence_ter": sentence_ter(beam["clean_text"], gold),
                "selected_sentence_ter": sentence_ter(selected["clean_text"], gold),
            }
            rows.append(out_row)
            pred_fh.write(
                json.dumps(
                    {
                        **out_row,
                        "input_text": input_text,
                        "candidates": scored_candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (idx + 1) % 50 == 0:
                print(f"processed={idx + 1}/{len(ds)}")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "task": "quality_aware_decoding",
        "run_id": run_id,
        "eval_config": {
            "dataset_path": args.dataset_path.as_posix(),
            "model_id": args.model_id,
            "adapter_dir": args.adapter_dir.as_posix(),
            "tokenizer_path": args.tokenizer_path.as_posix() if args.tokenizer_path else None,
            "family": args.family,
            "classification_candidates": args.classification_candidates,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "beam_early_stopping": bool(args.beam_early_stopping),
            "num_sampled_candidates": args.num_sampled_candidates,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "selection_strategy": args.selection_strategy,
            "min_target_prob_gain": args.min_target_prob_gain,
            "min_length_ratio_to_beam": args.min_length_ratio_to_beam,
            "max_length_ratio_to_beam": args.max_length_ratio_to_beam,
            "max_length_ratio_to_source": args.max_length_ratio_to_source,
            "limit": args.limit,
        },
        "beam": build_translation_summary(rows, "beam_pred"),
        "quality_aware_selected": build_translation_summary(rows, "selected_pred"),
        "selection": {
            "selected_sample_rate": selection_kind_counts["sample"] / max(len(rows), 1),
            "selected_beam_rate": selection_kind_counts[f"beam{args.num_beams}"] / max(len(rows), 1),
            "candidate_count_histogram": {str(k): int(v) for k, v in sorted(candidate_count_hist.items())},
            "target_prob_mean_selected": sum(float(row["selected_target_prob"]) for row in rows) / max(len(rows), 1),
            "target_prob_mean_beam": sum(float(row["beam_target_prob"]) for row in rows) / max(len(rows), 1),
        },
        "outputs": {
            "predictions_jsonl": pred_path.as_posix(),
            "rows_csv": csv_path.as_posix(),
            "summary_json": summary_path.as_posix(),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved rows: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
