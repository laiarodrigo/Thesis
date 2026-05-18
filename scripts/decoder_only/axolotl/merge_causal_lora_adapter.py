#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a decoder-only LoRA adapter into a standalone model dir.")
    parser.add_argument("--base-model", required=True, help="Base model id or path used for LoRA training.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    for cand in (args.adapter_dir.as_posix(), args.base_model):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                cand,
                use_fast=True,
                trust_remote_code=bool(args.trust_remote_code),
            )
            break
        except Exception:
            continue
    if tokenizer is None:
        raise RuntimeError("Failed to load tokenizer from adapter dir or base model.")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir.as_posix())
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out_dir.as_posix())
    tokenizer.save_pretrained(args.out_dir.as_posix())
    print(f"Merged adapter saved to {args.out_dir}")


if __name__ == "__main__":
    main()
