#!/usr/bin/env python3
import argparse
import inspect
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def maybe_override_output_dir(training_cfg: dict[str, Any]) -> dict[str, Any]:
    override = os.environ.get("THESIS_OUTPUT_DIR_OVERRIDE", "").strip()
    if not override:
        return training_cfg
    updated = dict(training_cfg)
    original = updated["output_dir"]
    updated["output_dir"] = override
    print(
        "Output dir override:"
        f" configured={original}"
        f" effective={override}"
    )
    return updated


def format_gpu_stats() -> str:
    if not torch.cuda.is_available():
        return "cuda_available=False"
    parts = ["cuda_available=True"]
    for idx in range(torch.cuda.device_count()):
        try:
            allocated_mb = torch.cuda.memory_allocated(idx) / (1024**2)
            reserved_mb = torch.cuda.memory_reserved(idx) / (1024**2)
            parts.append(
                f"cuda:{idx}_alloc_mb={allocated_mb:.1f}"
                f" cuda:{idx}_reserved_mb={reserved_mb:.1f}"
            )
        except Exception as exc:
            parts.append(f"cuda:{idx}_stats_error={exc!r}")
    return " ".join(parts)


def cast_trainable_params_to_fp32(model) -> None:
    converted = 0
    for _, param in model.named_parameters():
        if not param.requires_grad or not torch.is_floating_point(param):
            continue
        if param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            converted += 1
    if converted:
        print(f"Promoted trainable params to float32: converted_tensors={converted}")


def cast_norm_modules_to_fp32(model) -> None:
    converted_modules = 0
    converted_tensors = 0
    sample_names: list[str] = []
    for name, module in model.named_modules():
        normalized_name = name.lower()
        class_name = module.__class__.__name__.lower()
        if "norm" not in normalized_name and "norm" not in class_name:
            continue
        touched = False
        for param_name, param in module.named_parameters(recurse=False):
            if not torch.is_floating_point(param) or param.dtype == torch.float32:
                continue
            param.data = param.data.to(torch.float32)
            converted_tensors += 1
            touched = True
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if not torch.is_floating_point(buffer) or buffer.dtype == torch.float32:
                continue
            module._buffers[buffer_name] = buffer.to(torch.float32)
            converted_tensors += 1
            touched = True
        if touched:
            converted_modules += 1
            if len(sample_names) < 8:
                sample_names.append(name or "<root>")
    if converted_tensors:
        sample_suffix = f" sample={sample_names}" if sample_names else ""
        print(
            "Promoted norm modules to float32:"
            f" converted_modules={converted_modules}"
            f" converted_tensors={converted_tensors}"
            f"{sample_suffix}"
        )


def finite_tensor_report(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.detach()
    finite_mask = torch.isfinite(flat)
    finite = int(finite_mask.sum().item())
    total = flat.numel()
    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    return (
        f"{name}: dtype={flat.dtype}"
        f" shape={tuple(flat.shape)}"
        f" finite={finite}/{total}"
        f" nan={nan_count}"
        f" inf={inf_count}"
    )


def label_batch_report(inputs: dict[str, Any]) -> str:
    labels = inputs.get("labels")
    if labels is None or not isinstance(labels, torch.Tensor):
        return "labels=<missing>"
    valid_mask = labels.ne(-100)
    per_row = valid_mask.sum(dim=1)
    return (
        f"labels_shape={tuple(labels.shape)}"
        f" valid_tokens={int(valid_mask.sum().item())}"
        f" rows_without_supervision={int(per_row.eq(0).sum().item())}"
        f" min_valid_tokens={int(per_row.min().item()) if per_row.numel() else 0}"
        f" max_valid_tokens={int(per_row.max().item()) if per_row.numel() else 0}"
    )


class InstrumentedSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            loss = outputs["loss"]
            logits = outputs.get("logits")
        else:
            loss = outputs.loss
            logits = getattr(outputs, "logits", None)
        if not torch.isfinite(loss.detach()).all():
            raise RuntimeError(
                "Non-finite trainer loss detected:"
                f" {finite_tensor_report('loss', loss)}"
                f" {label_batch_report(inputs)}"
            )
        if logits is not None and not torch.isfinite(logits.detach()).all():
            raise RuntimeError(
                "Non-finite trainer logits detected:"
                f" {finite_tensor_report('logits', logits)}"
                f" {label_batch_report(inputs)}"
            )
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, *args, **kwargs):
        step = int(getattr(self.state, "global_step", 0))
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
            f" EVENT evaluate_start step={step} {format_gpu_stats()}",
            flush=True,
        )
        started_at = time.time()
        metrics = super().evaluate(*args, **kwargs)
        duration = time.time() - started_at
        metric_bits = []
        for key in ("eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"):
            value = metrics.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                metric_bits.append(f"{key}={value:.4f}")
            else:
                metric_bits.append(f"{key}={value}")
        metric_suffix = (" " + " ".join(metric_bits)) if metric_bits else ""
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
            f" EVENT evaluate_end step={step} duration_s={duration:.2f}"
            f"{metric_suffix} {format_gpu_stats()}",
            flush=True,
        )
        return metrics

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        step = int(getattr(self.state, "global_step", 0))
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{step}"
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
            f" EVENT save_start step={step} checkpoint_dir={checkpoint_dir}"
            f" {format_gpu_stats()}",
            flush=True,
        )
        started_at = time.time()
        result = super()._save_checkpoint(model, trial, *args, **kwargs)
        duration = time.time() - started_at
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
            f" EVENT save_end step={step} checkpoint_dir={checkpoint_dir}"
            f" duration_s={duration:.2f} {format_gpu_stats()}",
            flush=True,
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an encoder-decoder model with LoRA or full fine-tuning."
    )
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


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    return text in {"1", "true", "yes", "y"}


def coerce_nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def apply_prefix_loss_mask(seq: list[int], prefix_tokens_to_mask: int) -> list[int]:
    mask_n = min(coerce_nonnegative_int(prefix_tokens_to_mask), len(seq))
    if mask_n <= 0:
        return list(seq)
    masked = list(seq)
    for idx in range(mask_n):
        masked[idx] = -100
    return masked


def print_trainable_stats(model: torch.nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(
        f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}"
    )


def print_param_dtype_summary(model: torch.nn.Module, *, max_examples: int = 12) -> None:
    all_dtypes: dict[str, int] = {}
    trainable_dtypes: dict[str, int] = {}
    trainable_examples: list[tuple[str, str, tuple[int, ...]]] = []

    for name, param in model.named_parameters():
        dtype_name = str(param.dtype)
        all_dtypes[dtype_name] = all_dtypes.get(dtype_name, 0) + param.numel()
        if not param.requires_grad:
            continue
        trainable_dtypes[dtype_name] = trainable_dtypes.get(dtype_name, 0) + param.numel()
        if len(trainable_examples) < max_examples:
            trainable_examples.append((name, dtype_name, tuple(param.shape)))

    print(f"Parameter dtype summary (all): {all_dtypes}")
    print(f"Parameter dtype summary (trainable): {trainable_dtypes}")
    if trainable_examples:
        print("Sample trainable params:")
        for name, dtype_name, shape in trainable_examples:
            print(f"  {name}: dtype={dtype_name} shape={shape}")


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
        "predict_with_generate": training_cfg.get("predict_with_generate", False),
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
    if "save_safetensors" in sig.parameters:
        kwargs["save_safetensors"] = training_cfg.get("save_safetensors", True)
    if "save_only_model" in sig.parameters:
        kwargs["save_only_model"] = training_cfg.get("save_only_model", False)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = eval_strategy if eval_enabled else "no"
    else:
        kwargs["eval_strategy"] = eval_strategy if eval_enabled else "no"

    optional_training_args = (
        "adafactor",
        "dataloader_pin_memory",
        "eval_accumulation_steps",
        "gradient_checkpointing_kwargs",
        "group_by_length",
        "length_column_name",
        "optim",
        "torch_empty_cache_steps",
    )
    for arg_name in optional_training_args:
        if arg_name in training_cfg and arg_name in sig.parameters:
            kwargs[arg_name] = training_cfg[arg_name]

    return Seq2SeqTrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["dataset"]
    train_cfg = maybe_override_output_dir(cfg["training"])
    lora_cfg = cfg.get("lora", {})
    use_lora = bool(lora_cfg.get("enabled", True))
    init_adapter_path = lora_cfg.get("init_adapter_path")
    seed = cfg.get("seed", 123)

    set_seed(seed)

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], use_fast=True)
    target_dtype = None
    load_dtype = None
    if torch.cuda.is_available():
        if train_cfg.get("bf16", False):
            target_dtype = torch.bfloat16
            load_dtype = torch.bfloat16
        elif train_cfg.get("fp16", False):
            # Load fp16 runs in half precision so 4B LoRA training fits on
            # 11GB-class GPUs such as the RTX 2080 Ti. The previous float32
            # load path doubled resident weight memory and OOMed before the
            # Trainer step on those nodes.
            target_dtype = torch.float16
            load_dtype = torch.float16
    print(
        "Runtime:"
        f" cuda_available={torch.cuda.is_available()}"
        f" device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        f" bf16={bool(train_cfg.get('bf16', False))}"
        f" fp16={bool(train_cfg.get('fp16', False))}"
        f" target_dtype={target_dtype}"
        f" load_dtype={load_dtype}"
    )
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(idx)
            except Exception as exc:
                name = f"<error: {exc!r}>"
            print(f"  cuda:{idx} name={name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=load_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    if train_cfg.get("gradient_checkpointing", False):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if use_lora:
        if init_adapter_path:
            print(f"Training mode: LoRA (continue from adapter: {init_adapter_path})")
            try:
                model = PeftModel.from_pretrained(
                    model,
                    str(init_adapter_path),
                    is_trainable=True,
                )
            except TypeError:
                model = PeftModel.from_pretrained(model, str(init_adapter_path))
                # Backward-compatible fallback for older PEFT versions.
                for name, param in model.named_parameters():
                    if "lora_" in name or "modules_to_save" in name:
                        param.requires_grad = True
            if load_dtype is not None:
                model = model.to(dtype=load_dtype)
        else:
            missing = [k for k in ("r", "alpha", "dropout") if k not in lora_cfg]
            if missing:
                raise SystemExit(
                    "LoRA mode is enabled but lora config is missing keys: "
                    + ", ".join(missing)
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
            if load_dtype is not None:
                model = model.to(dtype=load_dtype)
            print("Training mode: LoRA")
        if load_dtype is not None:
            # Keep the frozen backbone in reduced precision while restoring
            # trainable LoRA weights to fp32 so AMP/GradScaler can unscale
            # gradients correctly during Trainer-driven mixed-precision runs.
            cast_trainable_params_to_fp32(model)
            # fp16-only 4B runs on g06 are more stable when normalization
            # modules stay in fp32. This mirrors common mixed-precision
            # fine-tuning practice without materially affecting memory use.
            cast_norm_modules_to_fp32(model)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        else:
            print_trainable_stats(model)
    else:
        print("Training mode: full fine-tuning (LoRA disabled)")
        if load_dtype is not None:
            model = model.to(dtype=load_dtype)
        print_trainable_stats(model)
    print_param_dtype_summary(model)

    print("Loading datasets...")
    data_files = {
        "train": data_cfg["train_path"],
        "validation": data_cfg["valid_path"],
    }
    raw_ds = load_dataset("json", data_files=data_files)
    print(raw_ds)
    print(
        f"Dataset sizes: train={len(raw_ds['train'])} validation={len(raw_ds['validation'])}"
    )
    if len(raw_ds["train"]) < 1000:
        print(
            "WARNING: train split is unexpectedly small for a 4B run. "
            "Double-check the exported JSONL path and line count."
        )

    max_source_length = model_cfg.get("max_source_length", 512)
    max_target_length = model_cfg.get("max_target_length", 128)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        cfg_eos = getattr(model.config, "eos_token_id", None)
        if isinstance(cfg_eos, (list, tuple)):
            eos_token_id = cfg_eos[0] if cfg_eos else None
        else:
            eos_token_id = cfg_eos
    if eos_token_id is None:
        print("WARNING: eos_token_id is undefined; labels will not be forced to end with EOS.")

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
        label_ids = labels["input_ids"]
        first_token_only_flags = batch.get("loss_on_first_token_only")
        if first_token_only_flags is None:
            first_token_only_flags = [False] * len(label_ids)
        loss_mask_prefix_tokens = batch.get("loss_mask_prefix_tokens")
        if loss_mask_prefix_tokens is None:
            loss_mask_prefix_tokens = [0] * len(label_ids)
        if eos_token_id is not None:
            fixed_label_ids = []
            for seq, first_token_only, prefix_tokens_to_mask in zip(
                label_ids,
                first_token_only_flags,
                loss_mask_prefix_tokens,
            ):
                if coerce_bool(first_token_only):
                    fixed_label_ids.append(seq[:1] if seq else [])
                    continue
                if not seq:
                    fixed_label_ids.append([int(eos_token_id)])
                    continue
                if seq[-1] == eos_token_id:
                    fixed_label_ids.append(
                        apply_prefix_loss_mask(seq, prefix_tokens_to_mask)
                    )
                    continue
                if len(seq) >= max_target_length:
                    seq = seq[: max_target_length - 1]
                fixed_label_ids.append(
                    apply_prefix_loss_mask(seq + [int(eos_token_id)], prefix_tokens_to_mask)
                )
            label_ids = fixed_label_ids
        else:
            fixed_label_ids = []
            for seq, first_token_only, prefix_tokens_to_mask in zip(
                label_ids,
                first_token_only_flags,
                loss_mask_prefix_tokens,
            ):
                if coerce_bool(first_token_only):
                    fixed_label_ids.append(seq[:1] if seq else [])
                else:
                    fixed_label_ids.append(
                        apply_prefix_loss_mask(seq, prefix_tokens_to_mask)
                    )
            label_ids = fixed_label_ids
        model_inputs["labels"] = label_ids
        return model_inputs

    tokenized = raw_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
    )

    training_args = build_training_args(train_cfg)
    print(
        f"Trainer device: device={training_args.device} n_gpu={training_args.n_gpu} "
        f"fp16={training_args.fp16} bf16={training_args.bf16}"
    )
    print(
        "Trainer config:"
        f" output_dir={train_cfg['output_dir']}"
        f" do_eval={bool(train_cfg.get('do_eval', True))}"
        f" predict_with_generate={bool(training_args.predict_with_generate)}"
        f" save_steps={train_cfg['save_steps']}"
        f" eval_steps={train_cfg.get('eval_steps', '<disabled>')}"
        f" save_total_limit={train_cfg['save_total_limit']}"
    )

    # Some seq2seq checkpoints expose prepare_decoder_input_ids_from_labels with
    # non-standard signatures (e.g., no `labels=` kwarg). In that case, avoid
    # passing model into the collator so training can proceed.
    collator_model = model
    prep_fn = getattr(model, "prepare_decoder_input_ids_from_labels", None)
    if callable(prep_fn):
        try:
            prep_sig = inspect.signature(prep_fn)
            params = prep_sig.parameters
            has_labels_kw = "labels" in params
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not (has_labels_kw or has_var_kw):
                print(
                    "WARNING: prepare_decoder_input_ids_from_labels has no `labels` kwarg; "
                    "using DataCollatorForSeq2Seq without model."
                )
                collator_model = None
        except (TypeError, ValueError):
            # If signature introspection fails, keep default behavior.
            pass

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=collator_model)
    callbacks = []
    early_stopping_patience = train_cfg.get("early_stopping_patience")
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping_patience),
            )
        )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": collator,
        "callbacks": callbacks,
    }
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = InstrumentedSeq2SeqTrainer(**trainer_kwargs)

    resume_from_checkpoint = (
        train_cfg.get("resume_from_checkpoint")
        or os.environ.get("THESIS_RESUME_FROM_CHECKPOINT")
        or os.environ.get("RESUME_FROM_CHECKPOINT")
        or None
    )
    if resume_from_checkpoint:
        resume_path = Path(str(resume_from_checkpoint))
        if not resume_path.exists():
            raise SystemExit(
                f"Requested resume checkpoint does not exist: {resume_path}"
            )
        resume_from_checkpoint = resume_path.as_posix()
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if bool(train_cfg.get("final_evaluate", False)):
        trainer.evaluate()

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir.as_posix())
    tokenizer.save_pretrained(output_dir.as_posix())
    if use_lora:
        print(f"Saved LoRA adapter and tokenizer to {output_dir}")
    else:
        print(f"Saved fully fine-tuned model and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
