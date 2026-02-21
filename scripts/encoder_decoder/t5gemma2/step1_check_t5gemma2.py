#!/usr/bin/env python3
"""
Step 1: Check whether a checkpoint is usable as encoder-decoder for T5Gemma2 workflow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect model/tokenizer config to verify encoder-decoder readiness."
    )
    parser.add_argument(
        "--model-id",
        default="google/t5gemma2-base",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True when loading config/tokenizer.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional output path to write the inspection report JSON.",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help=(
            "Environment variable containing Hugging Face token for gated repos "
            "(falls back to HUGGINGFACE_HUB_TOKEN if present)."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=False,
        help="Do not hit network; only load from local HF cache/files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'transformers' in this Python environment. "
            "Activate your training venv and install transformers."
        ) from exc

    hf_token = os.getenv(args.hf_token_env) or os.getenv("HUGGINGFACE_HUB_TOKEN")

    common_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if hf_token:
        common_kwargs["token"] = hf_token

    try:
        cfg = AutoConfig.from_pretrained(
            args.model_id,
            **common_kwargs,
        )
        tok = AutoTokenizer.from_pretrained(
            args.model_id,
            use_fast=True,
            **common_kwargs,
        )
    except Exception as exc:
        msg = str(exc)
        if "gated repo" in msg.lower() or "401" in msg.lower():
            print("ERROR: model repo is gated and authentication is required.")
            print("Fix:")
            print("1) Request access on the model page and accept its license.")
            print("2) Login: `hf auth login` (or equivalent).")
            print(
                f"3) If env `{args.hf_token_env}` is set, it overrides CLI login. "
                f"Unset it (`unset {args.hf_token_env}`) or set a valid token."
            )
            print(
                f"4) Retry command: python3 {Path(__file__).as_posix()} --model-id {args.model_id}"
            )
            raise SystemExit(2) from exc
        raise

    report = {
        "model_id": args.model_id,
        "config_class": cfg.__class__.__name__,
        "model_type": getattr(cfg, "model_type", None),
        "is_encoder_decoder": bool(getattr(cfg, "is_encoder_decoder", False)),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "decoder_start_token_id": getattr(cfg, "decoder_start_token_id", None),
        "pad_token_id": getattr(cfg, "pad_token_id", None),
        "eos_token_id": getattr(cfg, "eos_token_id", None),
        "tokenizer_class": tok.__class__.__name__,
        "tokenizer_vocab_size": tok.vocab_size,
        "tokenizer_model_max_length": tok.model_max_length,
        "special_tokens": tok.special_tokens_map,
    }

    print("Model inspection report:")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not report["is_encoder_decoder"]:
        print(
            "\nWARNING: config.is_encoder_decoder is False. "
            "This checkpoint is not suitable for encoder-decoder fine-tuning."
        )
    else:
        print("\nOK: checkpoint reports encoder-decoder architecture.")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report to {args.save_json}")


if __name__ == "__main__":
    main()
