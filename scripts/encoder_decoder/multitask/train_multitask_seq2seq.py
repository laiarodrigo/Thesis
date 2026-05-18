#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:
    np = None
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from task_protocol import (
    VALID_CLASS_PAYLOADS,
    build_translation_input_from_encoder_prefix,
    class_label_to_decoder_payload,
    decoder_payload_to_class_label,
    strip_encoder_prefix,
)


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
    if np is not None:
        np.random.seed(seed)
    else:
        print("WARNING: numpy is unavailable; skipping numpy random seed.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def harmonize_vocab_sizes(model, vocab_size: int) -> None:
    """
    Keep encoder/decoder vocab config fields aligned for T5Gemma2-style configs.
    """
    target = int(vocab_size)
    cfg = model.config

    for attr in ("vocab_size", "encoder_vocab_size", "decoder_vocab_size"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, target)

    for sub_name in ("text_config", "encoder", "decoder"):
        sub_cfg = getattr(cfg, sub_name, None)
        if sub_cfg is None:
            continue
        for attr in ("vocab_size", "encoder_vocab_size", "decoder_vocab_size"):
            if hasattr(sub_cfg, attr):
                setattr(sub_cfg, attr, target)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def print_trainable_stats(model: torch.nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(
        f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}"
    )


def normalize_dataset_name(value: object) -> str:
    text = (str(value).strip().lower() if value is not None else "")
    return text or "unknown"


def parse_ratio(value: Any) -> float:
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    return ratio


def parse_ratio_overrides(spec: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    raw = str(spec or "").strip()
    if not raw:
        return out
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid ratio override {part!r}. Expected dataset=ratio."
            )
        key, value = part.split("=", 1)
        out[normalize_dataset_name(key)] = parse_ratio(value.strip())
    return out


def max_equal_count(non_equal_count: int, ratio: float) -> int:
    if ratio <= 0.0:
        return 0
    if ratio >= 1.0:
        return 10**18
    return int((ratio * non_equal_count) / (1.0 - ratio))


class MixedObjectiveSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        cls_task_token_id: int | None = None,
        cls_ptpt_token_id: int | None = None,
        cls_ptbr_token_id: int | None = None,
        cls_equal_token_id: int | None = None,
        translation_loss_weight: float = 1.0,
        classification_loss_weight: float = 1.0,
        classification_decoder_ce_weight: float = 0.25,
        equal_soft_target: float = 0.5,
        loss_mode: str = "hybrid",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cls_task_token_id = (
            None if cls_task_token_id is None else int(cls_task_token_id)
        )
        self.cls_ptpt_token_id = (
            None if cls_ptpt_token_id is None else int(cls_ptpt_token_id)
        )
        self.cls_ptbr_token_id = (
            None if cls_ptbr_token_id is None else int(cls_ptbr_token_id)
        )
        self.cls_equal_token_id = (
            None if cls_equal_token_id is None else int(cls_equal_token_id)
        )
        self.translation_loss_weight = float(translation_loss_weight)
        self.classification_loss_weight = float(classification_loss_weight)
        self.classification_decoder_ce_weight = float(classification_decoder_ce_weight)
        self.equal_soft_target = float(equal_soft_target)
        if not 0.0 <= self.equal_soft_target <= 1.0:
            raise ValueError("equal_soft_target must be in [0, 1]")
        self.loss_mode = str(loss_mode or "hybrid").strip().lower()
        if self.loss_mode not in {"hybrid", "cross_entropy_only"}:
            raise ValueError(
                "loss_mode must be one of: hybrid, cross_entropy_only"
            )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        labels = inputs.get("labels")
        if labels is None:
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            return (loss, outputs) if return_outputs else loss

        model_inputs = dict(inputs)
        model_inputs.pop("num_items_in_batch", None)
        outputs = model(**model_inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        if logits is None:
            raise RuntimeError("Model did not return logits; cannot compute mixed objective.")

        labels = labels.to(logits.device)
        if logits.size(1) > 1:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels)

            token_mask = shift_labels.ne(-100)
            token_counts = token_mask.sum(dim=1)
            valid_examples = token_counts.gt(0)
            seq_loss = (token_loss * token_mask).sum(dim=1) / token_counts.clamp_min(1)

            if self.loss_mode == "cross_entropy_only":
                if valid_examples.any():
                    loss = seq_loss[valid_examples].mean()
                else:
                    fallback = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                    loss = logits.new_zeros(()) if fallback is None else fallback
                return (loss, outputs) if return_outputs else loss

            if None in (
                self.cls_task_token_id,
                self.cls_ptpt_token_id,
                self.cls_ptbr_token_id,
                self.cls_equal_token_id,
            ):
                raise ValueError(
                    "Hybrid multitask loss requires explicit classification token ids."
                )

            cls_mask = labels[:, 0].eq(self.cls_task_token_id)
            trans_mask = ~cls_mask

            losses: list[torch.Tensor] = []
            if valid_examples.any() and trans_mask.any():
                trans_valid = valid_examples & trans_mask
                if trans_valid.any():
                    losses.append(self.translation_loss_weight * seq_loss[trans_valid].mean())

            if valid_examples.any() and cls_mask.any() and self.classification_decoder_ce_weight > 0.0:
                cls_valid = valid_examples & cls_mask
                if cls_valid.any():
                    losses.append(self.classification_decoder_ce_weight * seq_loss[cls_valid].mean())

            if cls_mask.any() and logits.size(1) > 1:
                cls_logits = logits[cls_mask, 1, :]
                cls_payload_token = labels[cls_mask, 1]

                targets = torch.full_like(cls_payload_token, -1.0, dtype=torch.float32)
                targets = torch.where(
                    cls_payload_token.eq(self.cls_ptpt_token_id),
                    torch.ones_like(targets),
                    targets,
                )
                targets = torch.where(
                    cls_payload_token.eq(self.cls_ptbr_token_id),
                    torch.zeros_like(targets),
                    targets,
                )
                targets = torch.where(
                    cls_payload_token.eq(self.cls_equal_token_id),
                    torch.full_like(targets, self.equal_soft_target),
                    targets,
                )

                valid = targets.ge(0.0)
                if valid.any():
                    z_ptpt = cls_logits[:, self.cls_ptpt_token_id]
                    z_ptbr = cls_logits[:, self.cls_ptbr_token_id]
                    delta = z_ptpt - z_ptbr
                    loss_classification = F.binary_cross_entropy_with_logits(
                        delta[valid],
                        targets[valid],
                    )
                    losses.append(self.classification_loss_weight * loss_classification)

            if losses:
                loss = losses[0]
                for extra in losses[1:]:
                    loss = loss + extra
            elif valid_examples.any():
                loss = seq_loss[valid_examples].mean()
            else:
                fallback = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                loss = logits.new_zeros(()) if fallback is None else fallback
        else:
            # Fallback to model CE loss (should rarely happen).
            fallback = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            if fallback is None:
                loss = logits.new_zeros(())
            else:
                loss = fallback

        return (loss, outputs) if return_outputs else loss
def build_training_args(train_cfg: dict[str, Any]) -> Seq2SeqTrainingArguments:
    kwargs = {
        "output_dir": train_cfg["output_dir"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": train_cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "weight_decay": train_cfg["weight_decay"],
        "warmup_steps": train_cfg["warmup_steps"],
        "max_steps": train_cfg.get("max_steps", -1),
        "num_train_epochs": train_cfg.get("num_train_epochs", 1),
        "logging_steps": train_cfg["logging_steps"],
        "eval_steps": train_cfg["eval_steps"],
        "save_steps": train_cfg["save_steps"],
        "save_total_limit": train_cfg["save_total_limit"],
        "predict_with_generate": True,
        "bf16": train_cfg.get("bf16", False),
        "fp16": train_cfg.get("fp16", False),
        "gradient_checkpointing": train_cfg.get("gradient_checkpointing", False),
        "dataloader_num_workers": train_cfg.get("dataloader_num_workers", 4),
        "remove_unused_columns": True,
        "load_best_model_at_end": train_cfg.get("load_best_model_at_end", True),
        "metric_for_best_model": train_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": train_cfg.get("greater_is_better", False),
    }
    arg_names = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters.keys())
    if "evaluation_strategy" in arg_names:
        kwargs["evaluation_strategy"] = train_cfg.get("eval_strategy", "steps")
    else:
        kwargs["eval_strategy"] = train_cfg.get("eval_strategy", "steps")
    return Seq2SeqTrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    lora_cfg = cfg.get("lora", {})
    use_lora = bool(lora_cfg.get("enabled", False))
    init_adapter_path = lora_cfg.get("init_adapter_path")
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    print("Loading tokenizer/model...")
    tokenizer_source = str(init_adapter_path) if init_adapter_path else model_cfg["base_model"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=True,
        )
    except (ValueError, ImportError) as exc:
        print(f"Fast tokenizer unavailable, falling back to slow tokenizer: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=False,
        )
    target_dtype = torch.bfloat16 if torch.cuda.is_available() and train_cfg.get("bf16", False) else None
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=target_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    harmonize_vocab_sizes(model, len(tokenizer))
    print("Protocol: encoder prompts BR/PT for translation; decoder labels BR/PT/igual for classification.")

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
                for name, param in model.named_parameters():
                    if "lora_" in name or "modules_to_save" in name:
                        param.requires_grad = True
            if target_dtype is not None:
                model = model.to(dtype=target_dtype)
        else:
            missing = [k for k in ("r", "alpha", "dropout") if k not in lora_cfg]
            if missing:
                raise SystemExit(
                    "LoRA mode is enabled but lora config is missing keys: "
                    + ", ".join(missing)
                )
            peft_cfg = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=int(lora_cfg["r"]),
                lora_alpha=int(lora_cfg["alpha"]),
                lora_dropout=float(lora_cfg["dropout"]),
                bias=str(lora_cfg.get("bias", "none")),
                target_modules=lora_cfg.get("target_modules"),
                modules_to_save=lora_cfg.get("modules_to_save"),
            )
            model = get_peft_model(model, peft_cfg)
            if target_dtype is not None:
                model = model.to(dtype=target_dtype)
            print("Training mode: LoRA")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        else:
            print_trainable_stats(model)
    else:
        print("Training mode: full fine-tuning (LoRA disabled)")
        print_trainable_stats(model)

    data_files = {
        "train": data_cfg["train_path"],
        "validation": data_cfg["valid_path"],
    }
    raw_ds = load_dataset("json", data_files=data_files)

    equal_max_ratio = data_cfg.get("equal_max_ratio")
    equal_ratio_overrides = parse_ratio_overrides(data_cfg.get("equal_max_ratio_by_dataset", ""))
    equal_sampling_seed = int(data_cfg.get("equal_sampling_seed", seed))

    def apply_equal_ratio_filter(split_name: str) -> None:
        if equal_max_ratio is None and not equal_ratio_overrides:
            return
        ds = raw_ds[split_name]
        if "target_text" not in ds.column_names:
            return
        default_ratio = parse_ratio(equal_max_ratio if equal_max_ratio is not None else 1.0)
        rng = random.Random(equal_sampling_seed + (0 if split_name == "train" else 10_000))

        by_dataset_equal: dict[str, list[int]] = defaultdict(list)
        by_dataset_non_equal: dict[str, int] = defaultdict(int)
        dataset_values = (
            ds["dataset"]
            if "dataset" in ds.column_names
            else [None] * len(ds)
        )
        target_values = ds["target_text"]

        task_values = (
            ds["task"]
            if "task" in ds.column_names
            else [None] * len(ds)
        )

        for idx, (task_value, target_text, dataset_value) in enumerate(
            zip(task_values, target_values, dataset_values)
        ):
            if str(task_value or "").strip().lower() != "classification":
                continue
            cls_label = decoder_payload_to_class_label(str(target_text or ""))
            if cls_label is None:
                continue
            ds_key = normalize_dataset_name(dataset_value)
            if cls_label == "equal":
                by_dataset_equal[ds_key].append(idx)
            else:
                by_dataset_non_equal[ds_key] += 1

        if not by_dataset_equal:
            print(f"No classification `equal` rows found for split={split_name}; skipping equal filter.")
            return

        drop_idx: set[int] = set()
        per_dataset_logs: list[str] = []
        for ds_key in sorted(set(by_dataset_equal) | set(by_dataset_non_equal)):
            ratio = equal_ratio_overrides.get(ds_key, default_ratio)
            equal_indices = by_dataset_equal.get(ds_key, [])
            non_equal_count = by_dataset_non_equal.get(ds_key, 0)
            equal_before = len(equal_indices)
            keep_equal = min(equal_before, max_equal_count(non_equal_count, ratio))
            if keep_equal < equal_before:
                keep_set = set(rng.sample(equal_indices, k=keep_equal))
                for i in equal_indices:
                    if i not in keep_set:
                        drop_idx.add(i)
            per_dataset_logs.append(
                f"{ds_key}: ratio={ratio:.3f} non_equal={non_equal_count} equal_before={equal_before} equal_keep={keep_equal}"
            )

        if not drop_idx:
            print(f"Equal filter split={split_name}: no rows dropped.")
            return

        keep_indices = [i for i in range(len(ds)) if i not in drop_idx]
        raw_ds[split_name] = ds.select(keep_indices)
        print(
            f"Equal filter split={split_name}: dropped={len(drop_idx)} kept={len(keep_indices)} "
            f"(default_ratio={default_ratio:.3f})"
        )
        for line in per_dataset_logs:
            print(f"  {line}")

    apply_equal_ratio_filter("train")
    apply_equal_ratio_filter("validation")

    max_train_rows = data_cfg.get("max_train_rows")
    max_valid_rows = data_cfg.get("max_valid_rows")
    sample_shuffle = bool(data_cfg.get("sample_shuffle", False))
    sample_seed = int(data_cfg.get("sample_seed", seed))

    def apply_optional_limit(split_name: str, max_rows: Any) -> None:
        if max_rows in (None, 0, "0", ""):
            return
        limit = int(max_rows)
        ds = raw_ds[split_name]
        if len(ds) <= limit:
            return
        if sample_shuffle:
            ds = ds.shuffle(seed=sample_seed)
        ds = ds.select(range(limit))
        raw_ds[split_name] = ds
        print(
            f"Applied dataset limit: {split_name}={len(ds)} rows "
            f"(limit={limit}, shuffle={sample_shuffle})"
        )

    apply_optional_limit("train", max_train_rows)
    apply_optional_limit("validation", max_valid_rows)
    print(raw_ds)

    max_source_length = int(model_cfg.get("max_source_length", 512))
    max_target_length = int(model_cfg.get("max_target_length", 128))
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
    cls_counts = {"pt-br": 0, "pt-pt": 0, "equal": 0}

    counts = {"translation": 0, "classification": 0}
    bad = 0
    task_texts = (
        raw_ds["train"]["task"]
        if "task" in raw_ds["train"].column_names
        else [None] * len(target_texts)
    )

    for i, (task_text, target_text) in enumerate(zip(task_texts, target_texts)):
        task_name = str(task_text or "").strip().lower()
        if task_name == "classification":
            if str(target_text or "").strip() not in VALID_CLASS_PAYLOADS:
                print(f"WARNING: invalid classification payload at train index {i}: {target_text!r}")
                bad += 1
                continue
            cls_label = decoder_payload_to_class_label(str(target_text or ""))
            if cls_label is None:
                print(f"WARNING: invalid classification label at train index {i}: {target_text!r}")
                bad += 1
                continue
            counts["classification"] += 1
            cls_counts[cls_label] += 1
        elif task_name == "translation":
            counts["translation"] += 1
        else:
            print(f"WARNING: invalid task at train index {i}: task={task_text!r} target={target_text!r}")
            bad += 1
    print(f"Task distribution: {counts} | cls_labels={cls_counts} | bad_rows={bad}")


    def preprocess_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        tasks = batch.get("task") or [""] * len(batch["input_text"])
        normalized_inputs: list[str] = []
        normalized_targets: list[str] = []
        for task_name, input_text, target_text in zip(
            tasks,
            batch["input_text"],
            batch["target_text"],
        ):
            task_low = str(task_name or "").strip().lower()
            raw_input = str(input_text or "").strip()
            prefix, clean_input = strip_encoder_prefix(raw_input)

            if task_low == "translation" and prefix in {"br-pt", "pt-br"}:
                normalized_inputs.append(
                    build_translation_input_from_encoder_prefix(prefix, clean_input)
                )
            elif prefix in {"br-pt", "pt-br", "id"}:
                normalized_inputs.append(clean_input)
            else:
                normalized_inputs.append(raw_input)

            if task_low == "classification":
                cls_label = decoder_payload_to_class_label(str(target_text or ""))
                if cls_label is None:
                    raise ValueError(
                        f"Invalid classification payload in target_text: {target_text!r}"
                    )
                normalized_targets.append(class_label_to_decoder_payload(cls_label))
            else:
                normalized_targets.append(str(target_text or "").strip())

        model_inputs = tokenizer(
            normalized_inputs,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=normalized_targets,
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
        model_inputs["labels"] = label_ids
        return model_inputs

    tokenized = raw_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
    )

    training_args = build_training_args(train_cfg)
    # Keep collator model-agnostic: some trust_remote_code seq2seq models expose
    # prepare_decoder_input_ids_from_labels with a non-HF signature.
    # Our trainer computes losses directly from logits+labels, so decoder inputs
    # do not need to be precomputed in the collator.
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": collator,
    }
    trainer_arg_names = set(inspect.signature(Seq2SeqTrainer.__init__).parameters.keys())
    if "tokenizer" in trainer_arg_names:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        trainer_kwargs["processing_class"] = tokenizer

    loss_mode = str(train_cfg.get("loss_mode", "hybrid")).strip().lower()
    translation_loss_weight = float(train_cfg.get("translation_loss_weight", 1.0))
    classification_loss_weight = float(train_cfg.get("classification_loss_weight", 1.0))
    classification_decoder_ce_weight = float(train_cfg.get("classification_decoder_ce_weight", 0.25))
    equal_soft_target = float(train_cfg.get("classification_equal_soft_target", 0.5))
    print(f"Loss mode: {loss_mode}")
    if loss_mode != "cross_entropy_only":
        raise ValueError(
            "The existing-vocabulary multitask protocol only supports loss_mode=cross_entropy_only."
        )
    trainer = MixedObjectiveSeq2SeqTrainer(
        **trainer_kwargs,
        translation_loss_weight=translation_loss_weight,
        classification_loss_weight=classification_loss_weight,
        classification_decoder_ce_weight=classification_decoder_ce_weight,
        equal_soft_target=equal_soft_target,
        loss_mode=loss_mode,
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
    harmonize_vocab_sizes(model, len(tokenizer))
    trainer.save_model(out_dir.as_posix())
    tokenizer.save_pretrained(out_dir.as_posix())
    print(f"Saved model/tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
