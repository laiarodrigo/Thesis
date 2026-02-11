#!/usr/bin/env python3
import argparse
import inspect
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an encoder-decoder model with LoRA.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def build_training_args(training_cfg: dict[str, Any]) -> Seq2SeqTrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": training_cfg["output_dir"],
        "per_device_train_batch_size": training_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
        "learning_rate": training_cfg["learning_rate"],
        "weight_decay": training_cfg["weight_decay"],
        "warmup_steps": training_cfg["warmup_steps"],
        "max_steps": training_cfg.get("max_steps", -1),
        "num_train_epochs": training_cfg.get("num_train_epochs", 1),
        "logging_steps": training_cfg["logging_steps"],
        "save_steps": training_cfg["save_steps"],
        "save_total_limit": training_cfg["save_total_limit"],
        "predict_with_generate": True,
        "bf16": training_cfg.get("bf16", False),
        "fp16": training_cfg.get("fp16", False),
        "gradient_checkpointing": training_cfg.get("gradient_checkpointing", False),
        "dataloader_num_workers": training_cfg.get("dataloader_num_workers", 4),
        "remove_unused_columns": True,
        "load_best_model_at_end": training_cfg.get("load_best_model_at_end", True),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": training_cfg.get("greater_is_better", False),
    }

    eval_enabled = bool(training_cfg.get("do_eval", True))
    eval_strategy = training_cfg.get("eval_strategy", "steps")
    if eval_enabled:
        kwargs["eval_steps"] = training_cfg["eval_steps"]

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = eval_strategy if eval_enabled else "no"
    else:
        kwargs["eval_strategy"] = eval_strategy if eval_enabled else "no"

    return Seq2SeqTrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]
    seed = cfg.get("seed", 123)

    set_seed(seed)

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and train_cfg.get("bf16", False) else None,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules"),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    print("Loading datasets...")
    data_files = {
        "train": data_cfg["train_path"],
        "validation": data_cfg["valid_path"],
    }
    raw_ds = load_dataset("json", data_files=data_files)

    max_source_length = model_cfg.get("max_source_length", 512)
    max_target_length = model_cfg.get("max_target_length", 128)

    def preprocess_batch(batch):
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
        model_inputs["labels"] = labels["input_ids"]
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

    print("Starting training...")
    trainer.train()
    trainer.evaluate()

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir.as_posix())
    tokenizer.save_pretrained(output_dir.as_posix())
    print(f"Saved LoRA adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
