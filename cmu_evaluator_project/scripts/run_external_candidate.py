#!/usr/bin/env python3
"""Generate independent conservative BP→EP candidates with Gemma 4."""

from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path

from campaign_common import canonicalize_source


MODEL_ID = "google/gemma-4-E4B-it"
SYSTEM_ID = "gemma-4-E4B-it"
TASK_INSTRUCTION = """You are given one sentence in Brazilian Portuguese.

Rewrite it in standard European Portuguese while preserving the original
meaning, wording, tone, and structure as much as possible.

Make only changes that are necessary for an acceptable European Portuguese
version.

Do not paraphrase, summarize, expand, explain, or stylistically improve the
sentence.

If the original sentence is already acceptable in European Portuguese,
return it unchanged.

Return only the rewritten sentence and no explanation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize_response(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def technical_flags(raw: str, cleaned: str, hit_token_limit: bool) -> list[str]:
    flags: list[str] = []
    if not cleaned:
        flags.append("empty_output")
    if hit_token_limit:
        flags.append("possible_truncation_max_new_tokens")
    nonempty_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(nonempty_lines) > 1:
        flags.append("multiple_nonempty_lines_possible_explanation")
    if re.match(
        r"^(?:sure[,.! ]|here(?:'s| is)|rewritten sentence\s*:|"
        r"european portuguese\s*:|portugu[eê]s europeu\s*:|vers[aã]o\s*:)",
        cleaned,
        flags=re.IGNORECASE,
    ):
        flags.append("possible_explanation_or_label")
    return flags


def json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def main() -> None:
    args = parse_args()
    if args.model_id != MODEL_ID:
        raise SystemExit(f"This frozen pilot runner only permits {MODEL_ID}")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")

    source = read_json(args.source)
    sentences = canonicalize_source(source)

    # Imports occur only after argument and input validation. --help never loads a model.
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing to run Gemma 4 on CPU")

    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_token_id)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = int(eos_token_id)

    rows = []
    for sentence in sentences:
        user_message = f"{TASK_INSTRUCTION}\n\nSource sentence:\n{sentence['source_bp']}"
        messages = [{"role": "user", "content": user_message}]
        rendered_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(text=rendered_prompt, return_tensors="pt")
        input_device = next(model.parameters()).device
        inputs = inputs.to(input_device)
        input_tokens = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        continuation = generated[0, input_tokens:]
        output_tokens = int(continuation.shape[-1])
        raw_response = tokenizer.decode(continuation, skip_special_tokens=True)
        cleaned = normalize_response(raw_response)
        hit_token_limit = output_tokens >= args.max_new_tokens

        rows.append(
            {
                "document_id": sentence["document_id"],
                "passage_id": sentence["passage_id"],
                "sentence_id": sentence["sentence_id"],
                "position": sentence["position"],
                "source_bp": sentence["source_bp"],
                "system_id": SYSTEM_ID,
                "messages": messages,
                "rendered_prompt": rendered_prompt,
                "raw_model_output": raw_response,
                "cleaned_candidate": cleaned,
                "input_token_count": input_tokens,
                "output_token_count": output_tokens,
                "technical_flags": technical_flags(raw_response, cleaned, hit_token_limit),
            }
        )

    default_generation = {
        key: json_safe(value)
        for key, value in model.generation_config.to_dict().items()
    }
    device_map = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    payload = {
        "schema_version": "1.0",
        "source_dataset_id": source.get("dataset_id"),
        "source_item_count": len(sentences),
        "system_id": SYSTEM_ID,
        "model": {
            "model_id": args.model_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "architecture": model.__class__.__name__,
            "processor": processor.__class__.__name__,
        },
        "prompt": {
            "task_instruction": TASK_INSTRUCTION,
            "source_separator": "\n\nSource sentence:\n",
            "chat_template": "official processor.apply_chat_template",
            "add_generation_prompt": True,
            "enable_thinking": False,
            "one_sentence_at_a_time": True,
            "passage_context_provided": False,
        },
        "generation_config": {
            "deterministic": True,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "model_generation_defaults": default_generation,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(),
            "model_device": str(next(model.parameters()).device),
            "model_dtype": str(next(model.parameters()).dtype),
            "hf_device_map": device_map,
        },
        "sentences": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
