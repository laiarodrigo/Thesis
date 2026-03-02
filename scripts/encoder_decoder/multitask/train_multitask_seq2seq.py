#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from more_itertools import one
import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from task_protocol import CLS, TR_BR2PT, TR_PT2BR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skeleton trainer for multitask seq2seq (decoder task control)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Run training. Without this flag, dry-run only.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_special_tokens(tokenizer) -> dict[str, int]:
    """
    Register decoder task tokens and return their ids.
    """
    tokens = [TR_BR2PT, TR_PT2BR, CLS]
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if added:
        print(f"Added {added} special tokens: {tokens}")
    else:
        print("Special tokens already present.")

    ids = {}
    for tok in tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None or tok_id < 0:
            raise RuntimeError(f"Token id not found for {tok}")
        ids[tok] = int(tok_id)
    return ids


def build_training_args(train_cfg: dict[str, Any]) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg.get("max_steps", -1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        logging_steps=train_cfg["logging_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        evaluation_strategy=train_cfg.get("eval_strategy", "steps"),
        predict_with_generate=True,
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=True,
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        use_fast=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and train_cfg.get("bf16", False) else None,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    task_token_ids = ensure_special_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Task token ids: {task_token_ids}")

    data_files = {
        "train": data_cfg["train_path"],
        "validation": data_cfg["valid_path"],
    }
    raw_ds = load_dataset("json", data_files=data_files)
    print(raw_ds)

    max_source_length = int(model_cfg.get("max_source_length", 512))
    max_target_length = int(model_cfg.get("max_target_length", 128))
    enforce_first_label_task_token = bool(
        data_cfg.get("enforce_first_label_task_token", model_cfg.get("enforce_first_label_task_token", False))
    )
    valid_task_token_ids = frozenset(task_token_ids.values())
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        cfg_eos = getattr(model.config, "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple)):
            eos_token_id = cfg_eos[0] if cfg_eos else None
        else:
            eos_token_id = cfg_eos
    if eos_token_id is None:
        print("WARNING: eos_token_id is undefined; labels will not be forced to end with EOS.")

    target_texts = raw_ds["train"]["target_text"]
    valid_prefixes = {TR_BR2PT, TR_PT2BR, CLS}
    valid_cls_labels = {'pt-br', 'pt-pt', 'equal'}

    counts = {"translation": 0, "classification": 0}
    bad = 0
    
    for i, t_text in enumerate(target_texts):
        parts = t_text.strip().split()
        if not parts or parts[0] not in valid_prefixes:
            print(f"WARNING: invalid target prefix at train index {i}: {t_text!r}")
            bad += 1
            continue
        if parts[0] == CLS:
            if len(parts) != 2 or parts[1] not in valid_cls_labels:
                print(f"WARNING: invalid CLS format at train index {i}: {t_text!r}")
                bad += 1
                continue
            counts["classification"] += 1
        else:
            counts["translation"] += 1
    print(f"Task distribution: {counts} | bad_rows={bad}")


    def preprocess_batch(batch: dict[str, list[str]]) -> dict[str, Any]:

        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
        label_ids = labels["input_ids"]
        if eos_token_id is not None:
            fixed_label_ids = []
            for seq in label_ids:
                if not seq:
                    fixed_label_ids.append([int(eos_token_id)])
                    continue
                if seq[-1] == eos_token_id:
                    fixed_label_ids.append(seq)
                    continue
                if len(seq) >= max_target_length:
                    seq = seq[: max_target_length - 1]
                fixed_label_ids.append(seq + [int(eos_token_id)])
            label_ids = fixed_label_ids
        if enforce_first_label_task_token:
            for i, seq in enumerate(label_ids):
                if not seq or seq[0] not in valid_task_token_ids:
                    raise ValueError(
                        f"Invalid target prefix at batch index {i}: first label token id={seq[0] if seq else None}, "
                        f"expected one of {sorted(valid_task_token_ids)}"
                    )
        model_inputs["labels"] = label_ids
        return model_inputs

    tokenized = raw_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
    )

    training_args = build_training_args(train_cfg)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    if not args.execute:
        print("Dry-run only. Use --execute to train.")
        print(f"Config: {args.config}")
        print(f"Train rows: {len(raw_ds['train'])} | Valid rows: {len(raw_ds['validation'])}")
        return

    print("Starting training...")
    trainer.train()
    trainer.evaluate()

    out_dir = Path(train_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir.as_posix())
    tokenizer.save_pretrained(out_dir.as_posix())
    print(f"Saved model/tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
