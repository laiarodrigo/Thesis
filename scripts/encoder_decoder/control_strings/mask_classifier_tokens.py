#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "scripts" / "encoder_decoder" / "eval"
for path in (REPO_ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.encoder_decoder.eval.evaluate_encdec import (  # noqa: E402
    classification_candidate_metadata,
    classification_label_alias_map,
    load_model_and_tokenizer,
    normalize_generation_text,
    normalize_label,
    score_classification_candidates_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mask one classifier input token at a time and measure the drop in "
            "target-label probability."
        )
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model-id", default="google/t5gemma-2-4b-4b")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classification-candidates", nargs=2, default=["BR", "PT"])
    parser.add_argument("--classification-mode", choices=["score-sequences", "score-first-token"], default="score-sequences")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--max-tokens-per-example", type=int, default=96)
    parser.add_argument("--mask-token", default=None)
    parser.add_argument("--preserve-leading-control", default="CLS")
    parser.add_argument("--top-k-examples", type=int, default=100)
    return parser.parse_args()


def softmax_scores(scores: dict[str, float]) -> dict[str, float]:
    max_score = max(scores.values())
    exp_values = {key: math.exp(value - max_score) for key, value in scores.items()}
    denom = sum(exp_values.values())
    return {key: value / denom for key, value in exp_values.items()}


def choose_mask_token(tok, requested: str | None) -> str:
    candidates = []
    if requested:
        candidates.append(requested)
    if getattr(tok, "mask_token", None):
        candidates.append(str(tok.mask_token))
    candidates.extend(["<mask>", "<extra_id_0>"])
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        ids = tok.encode(candidate, add_special_tokens=False)
        unk_id = getattr(tok, "unk_token_id", None)
        if ids and (unk_id is None or not all(int(x) == int(unk_id) for x in ids)):
            return candidate
    raise RuntimeError("Could not find a valid mask/sentinel token for this tokenizer.")


def target_candidate_from_gold(raw_gold: object, alias_map: dict[str, str]) -> str | None:
    norm = normalize_label(str(raw_gold or ""))
    if norm is None:
        return None
    return alias_map.get(norm)


def token_rows_for_input(
    tok,
    text: str,
    *,
    mask_token: str,
    preserve_leading_control: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    token_ids = tok.encode(text, add_special_tokens=False)
    tokens = tok.convert_ids_to_tokens(token_ids)
    skip_indices = set()
    if preserve_leading_control:
        control_ids = tok.encode(preserve_leading_control, add_special_tokens=False)
        if control_ids and token_ids[: len(control_ids)] == control_ids:
            skip_indices.update(range(len(control_ids)))

    rows = []
    for idx, token in enumerate(tokens[:max_tokens]):
        if idx in skip_indices:
            continue
        masked_ids = list(token_ids)
        mask_ids = tok.encode(mask_token, add_special_tokens=False)
        masked_ids = masked_ids[:idx] + mask_ids + masked_ids[idx + 1 :]
        masked_text = normalize_generation_text(tok.decode(masked_ids, skip_special_tokens=False))
        rows.append(
            {
                "token_index": idx,
                "token": str(token),
                "token_id": int(token_ids[idx]),
                "masked_text": masked_text,
            }
        )
    return rows


def score_inputs(
    model,
    tok,
    inputs: list[str],
    *,
    candidates: list[str],
    max_source_length: int,
    mode: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    all_scores = []
    for start in range(0, len(inputs), batch_size):
        all_scores.extend(
            score_classification_candidates_batch(
                model,
                tok,
                inputs=inputs[start : start + batch_size],
                candidates=candidates,
                max_source_length=max_source_length,
                mode=mode,
            )
        )
    return all_scores


def main() -> None:
    args = parse_args()
    if not args.dataset_path.is_file():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("json", data_files={"eval": args.dataset_path.as_posix()})["eval"]
    if args.max_examples is not None:
        ds = ds.select(range(min(int(args.max_examples), len(ds))))
    model, tok = load_model_and_tokenizer(args.model_id, args.adapter_dir, args.tokenizer_path)
    mask_token = choose_mask_token(tok, args.mask_token)

    alias_map = classification_label_alias_map(args.classification_candidates)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows_path = args.output_dir / f"{run_id}_mask_token_rows.csv"
    examples_path = args.output_dir / f"{run_id}_mask_examples.jsonl"
    summary_path = args.output_dir / f"{run_id}_mask_summary.json"

    print("Classification candidates:")
    for meta in classification_candidate_metadata(tok, args.classification_candidates):
        print(json.dumps(meta, ensure_ascii=False))
    print(f"Mask token: {mask_token!r} ids={tok.encode(mask_token, add_special_tokens=False)}")

    token_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    skipped = Counter()

    for ex_idx, row in enumerate(ds):
        input_text = normalize_generation_text(str(row["input_text"]))
        target_candidate = target_candidate_from_gold(row.get("target_text", row.get("label", row.get("gold"))), alias_map)
        if target_candidate is None:
            skipped["unknown_target"] += 1
            continue

        base_score = score_inputs(
            model,
            tok,
            [input_text],
            candidates=args.classification_candidates,
            max_source_length=args.max_source_length,
            mode=args.classification_mode,
            batch_size=1,
        )[0]
        base_probs = softmax_scores(base_score["scores"])
        base_prob = float(base_probs[target_candidate])

        masked_specs = token_rows_for_input(
            tok,
            input_text,
            mask_token=mask_token,
            preserve_leading_control=args.preserve_leading_control,
            max_tokens=args.max_tokens_per_example,
        )
        if not masked_specs:
            skipped["no_maskable_tokens"] += 1
            continue

        masked_scores = score_inputs(
            model,
            tok,
            [spec["masked_text"] for spec in masked_specs],
            candidates=args.classification_candidates,
            max_source_length=args.max_source_length,
            mode=args.classification_mode,
            batch_size=args.batch_size,
        )

        per_example = []
        for spec, score_info in zip(masked_specs, masked_scores):
            probs = softmax_scores(score_info["scores"])
            masked_prob = float(probs[target_candidate])
            drop = base_prob - masked_prob
            rec = {
                "id": row.get("id", ex_idx),
                "target_candidate": target_candidate,
                "base_target_prob": base_prob,
                "masked_target_prob": masked_prob,
                "target_prob_drop": drop,
                "token_index": spec["token_index"],
                "token": spec["token"],
                "token_id": spec["token_id"],
                "masked_pred": score_info["pred_text"],
                "input_text": input_text,
            }
            token_rows.append(rec)
            per_example.append(rec)

        per_example.sort(key=lambda item: item["target_prob_drop"], reverse=True)
        example_rows.append(
            {
                "id": row.get("id", ex_idx),
                "target_candidate": target_candidate,
                "base_pred": base_score["pred_text"],
                "base_target_prob": base_prob,
                "input_text": input_text,
                "top_tokens": [
                    {
                        "token_index": rec["token_index"],
                        "token": rec["token"],
                        "token_id": rec["token_id"],
                        "target_prob_drop": rec["target_prob_drop"],
                        "masked_target_prob": rec["masked_target_prob"],
                    }
                    for rec in per_example[:10]
                ],
            }
        )
        if (ex_idx + 1) % 25 == 0:
            print(f"processed={ex_idx + 1}/{len(ds)} token_rows={len(token_rows)}")

    with rows_path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "id",
            "target_candidate",
            "base_target_prob",
            "masked_target_prob",
            "target_prob_drop",
            "token_index",
            "token",
            "token_id",
            "masked_pred",
            "input_text",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(token_rows)

    example_rows.sort(
        key=lambda item: item["top_tokens"][0]["target_prob_drop"] if item["top_tokens"] else 0.0,
        reverse=True,
    )
    with examples_path.open("w", encoding="utf-8") as fh:
        for row in example_rows[: args.top_k_examples]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_token: dict[str, list[float]] = defaultdict(list)
    for row in token_rows:
        by_token[str(row["token"])].append(float(row["target_prob_drop"]))
    top_token_summary = sorted(
        [
            {
                "token": token,
                "count": len(values),
                "mean_target_prob_drop": sum(values) / len(values),
                "max_target_prob_drop": max(values),
            }
            for token, values in by_token.items()
        ],
        key=lambda item: (item["mean_target_prob_drop"], item["max_target_prob_drop"]),
        reverse=True,
    )[:100]

    summary = {
        "task": "mask_classifier_tokens",
        "run_id": run_id,
        "eval_config": {
            "dataset_path": args.dataset_path.as_posix(),
            "model_id": args.model_id,
            "adapter_dir": args.adapter_dir.as_posix(),
            "tokenizer_path": args.tokenizer_path.as_posix() if args.tokenizer_path else None,
            "classification_candidates": args.classification_candidates,
            "classification_mode": args.classification_mode,
            "max_examples": args.max_examples,
            "max_tokens_per_example": args.max_tokens_per_example,
            "mask_token": mask_token,
            "preserve_leading_control": args.preserve_leading_control,
        },
        "n_examples": len(example_rows),
        "n_token_masks": len(token_rows),
        "skipped": dict(skipped),
        "mean_target_prob_drop": (
            sum(float(row["target_prob_drop"]) for row in token_rows) / len(token_rows)
            if token_rows
            else None
        ),
        "top_tokens_by_mean_drop": top_token_summary,
        "outputs": {
            "token_rows_csv": rows_path.as_posix(),
            "top_examples_jsonl": examples_path.as_posix(),
            "summary_json": summary_path.as_posix(),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved token rows: {rows_path}")
    print(f"Saved top examples: {examples_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
