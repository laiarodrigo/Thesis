#!/usr/bin/env python3
"""Generate one-sentence-at-a-time PT/BR thesis candidates after adapter restore.

This wrapper intentionally has no usable default adapter. It reuses the thesis
loader, beam-4 generator, and decoder-label cleaner from evaluate_encdec.py.
The final dissertation's decoder-controlled protocol receives an unprefixed
source sentence and emits the source-variety label before the rewrite.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cmu_evaluator_project.scripts.campaign_common import canonicalize_source

SYSTEM_SPECS = {
    "B4-D-G-PTBR": {
        "adapter_basename": "t5gemma2_4b_stageB_gpt_wiki_decoder_ptbr",
        "config_basename": "stageB_gpt_wiki_decoder_ptbr.yaml",
    },
    "B4-D-GF-PTBR": {
        "adapter_basename": "t5gemma2_4b_stageB_gpt_wiki_frmt_decoder_ptbr",
        "config_basename": "stageB_gpt_wiki_frmt_decoder_ptbr.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-id", choices=sorted(SYSTEM_SPECS), required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--model-id", default="google/t5gemma-2-4b-4b")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.8)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    adapter = args.adapter_dir.resolve()
    if not adapter.is_dir():
        raise SystemExit(f"Adapter directory does not exist: {adapter}")
    training_config = args.training_config.resolve()
    if not training_config.is_file():
        raise SystemExit(f"Training config does not exist: {training_config}")
    spec = SYSTEM_SPECS[args.system_id]
    if adapter.name != spec["adapter_basename"]:
        raise SystemExit(
            f"Wrong adapter for {args.system_id}: expected basename "
            f"{spec['adapter_basename']!r}, got {adapter.name!r}"
        )
    if training_config.name != spec["config_basename"]:
        raise SystemExit(
            f"Wrong config for {args.system_id}: expected basename "
            f"{spec['config_basename']!r}, got {training_config.name!r}"
        )
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")

    source = load_json(args.source)
    source_rows = canonicalize_source(source)

    # Importing here ensures --help and static validation never load a model.
    from scripts.encoder_decoder.eval.evaluate_encdec import (
        generate_batch,
        load_model_and_tokenizer,
        normalize_generation_text,
        strip_decoder_label_prefix,
    )
    import peft
    import torch
    import transformers

    model, tokenizer = load_model_and_tokenizer(
        args.model_id, adapter, args.tokenizer_path
    )
    generation_config = {
        "deterministic": True,
        "do_sample": False,
        "one_sentence_at_a_time": True,
        "max_source_length": args.max_source_length,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "early_stopping": True,
        "adaptive_max_new_tokens": True,
        "adaptive_ratio": 1.15,
        "adaptive_margin": 6,
        "adaptive_min_new_tokens": 8,
        "adaptive_max_new_tokens_ceiling": 384,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "forced_eos": True,
        "input_prefix": None,
        "decoder_controlled": True,
        "control_label_semantics": "source_variety",
        "expected_decoder_label": "BR",
    }

    rows = []
    batch_parameters = inspect.signature(generate_batch).parameters
    supports_forced_eos_flag = "forced_eos" in batch_parameters
    for sentence in source_rows:
        model_input = sentence["source_bp"]
        batch_kwargs = {
            "inputs": [model_input],
            "max_source_length": args.max_source_length,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "early_stopping": True,
            "adaptive_max_new_tokens": True,
            "adaptive_ratio": 1.15,
            "adaptive_margin": 6,
            "adaptive_min_new_tokens": 8,
            "adaptive_max_new_tokens_ceiling": 384,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "repetition_penalty": args.repetition_penalty,
        }
        # The newer evaluator exposes this switch. The Slurm checkout's older
        # helper has no switch because it already forces EOS unconditionally.
        if supports_forced_eos_flag:
            batch_kwargs["forced_eos"] = True
        decoded = generate_batch(model, tokenizer, **batch_kwargs)[0]
        raw_model_output = decoded
        normalized = normalize_generation_text(decoded)
        label_match = re.match(r"^\s*(BR|PT)\b", normalized, flags=re.IGNORECASE)
        generated_label = label_match.group(1).upper() if label_match else None
        cleaned = strip_decoder_label_prefix(normalized)
        rows.append(
            {
                "sentence_id": sentence["sentence_id"],
                "position": sentence["position"],
                "document_id": sentence["document_id"],
                "passage_id": sentence["passage_id"],
                "system_id": args.system_id,
                "source_bp": sentence["source_bp"],
                "model_input": model_input,
                "raw_model_output": raw_model_output,
                "generated_control_label": generated_label,
                "control_label_matches_expected": generated_label == "BR",
                "cleaned_candidate": cleaned,
            }
        )

    payload = {
        "schema_version": "1.0",
        "source_dataset_id": source.get("dataset_id"),
        "source_item_count": len(source_rows),
        "system_id": args.system_id,
        "model": {
            "base_model": args.model_id,
            "base_model_resolved_revision": getattr(
                model.get_base_model().config, "_commit_hash", None
            ),
            "adapter_path": str(adapter),
            "training_config": str(training_config),
            "tokenizer_path": str(args.tokenizer_path.resolve()) if args.tokenizer_path else str(adapter),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(next(model.parameters()).device),
            "model_dtype": str(next(model.parameters()).dtype),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "generation_config": generation_config,
        "sentences": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
