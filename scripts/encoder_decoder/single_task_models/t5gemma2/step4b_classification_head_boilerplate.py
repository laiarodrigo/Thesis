#!/usr/bin/env python3
"""
Step 4B (runnable boilerplate): classification-head training for T5Gemma2.

- Uses encoder representations + linear classification head.
- Keeps dry-run by default. Add --execute to train.
- Intended for pipeline validation before HPO.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput


def print_trainable_stats(model: torch.nn.Module, *, prefix: str = "") -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    msg_prefix = f"{prefix} " if prefix else ""
    print(
        f"{msg_prefix}trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}"
    )


def maybe_limit_split(dataset, *, max_rows: int | None, seed: int, split_name: str):
    if max_rows is None:
        return dataset
    limit = int(max_rows)
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    print(f"[dataset] limiting {split_name} rows: {len(dataset)} -> {limit}")
    return dataset.shuffle(seed=seed).select(range(limit))


class EncoderClassifier(torch.nn.Module):
    def __init__(
        self,
        base_model: str,
        *,
        num_labels: int,
        trust_remote_code: bool,
        local_files_only: bool = False,
        torch_dtype: torch.dtype | None = None,
        freeze_decoder: bool = True,
        dropout: float = 0.1,
        lora_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        load_errors = []
        trust_options = [trust_remote_code]
        if trust_remote_code:
            trust_options.append(False)
        for trust_opt in trust_options:
            common_kwargs = {
                "trust_remote_code": trust_opt,
                "local_files_only": local_files_only,
            }
            try:
                if torch_dtype is None:
                    self.base = AutoModelForSeq2SeqLM.from_pretrained(
                        base_model,
                        **common_kwargs,
                    )
                else:
                    try:
                        self.base = AutoModelForSeq2SeqLM.from_pretrained(
                            base_model,
                            dtype=torch_dtype,
                            **common_kwargs,
                        )
                    except TypeError:
                        # Backward compatibility with transformers versions that still expect torch_dtype.
                        self.base = AutoModelForSeq2SeqLM.from_pretrained(
                            base_model,
                            torch_dtype=torch_dtype,
                            **common_kwargs,
                        )
                break
            except Exception as e:  # pragma: no cover - fallback path
                load_errors.append(e)
        else:
            if load_errors:
                raise load_errors[-1]
            raise RuntimeError("Failed to load base model.")

        hidden_size = self._infer_hidden_size()
        if hidden_size is None:
            raise ValueError(
                "Could not infer hidden size from model config. "
                "Tried base config, text_config, and encoder config."
            )

        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(int(hidden_size), int(num_labels))
        print(f"[classifier] inferred hidden_size={int(hidden_size)}")

        self.using_lora = False
        if lora_cfg and bool(lora_cfg.get("enabled", False)):
            from peft import LoraConfig, TaskType, get_peft_model

            missing = [k for k in ("r", "alpha", "dropout") if k not in lora_cfg]
            if missing:
                raise ValueError(
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
            )
            self.base = get_peft_model(self.base, peft_cfg)
            self.using_lora = True
            print("[classifier] LoRA enabled on base model")

        if freeze_decoder:
            for name, param in self.base.named_parameters():
                if (
                    name.startswith("decoder")
                    or ".decoder." in name
                    or name.startswith("lm_head")
                    or ".lm_head." in name
                ):
                    param.requires_grad = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        """Forward Trainer gradient-checkpointing hook to wrapped HF model."""
        fn = getattr(self.base, "gradient_checkpointing_enable", None)
        if fn is None:
            return
        if gradient_checkpointing_kwargs:
            try:
                fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                return
            except TypeError:
                pass
        fn()

    def gradient_checkpointing_disable(self) -> None:
        fn = getattr(self.base, "gradient_checkpointing_disable", None)
        if fn is not None:
            fn()

    def _read_hidden_size(self, obj: Any) -> int | None:
        if obj is None:
            return None
        for key in ("d_model", "hidden_size", "model_dim", "dim"):
            val = getattr(obj, key, None)
            if isinstance(val, int) and val > 0:
                return val
        return None

    def _infer_hidden_size(self) -> int | None:
        # 1) Common top-level config keys.
        encoder = self.base.get_encoder()
        for cfg in (self.base.config, getattr(self.base.config, "text_config", None), encoder.config):
            hs = self._read_hidden_size(cfg)
            if hs is not None:
                return hs

        # 2) Common embedding entry points.
        emb_getters = [
            lambda: self.base.get_input_embeddings(),
            lambda: encoder.get_input_embeddings() if hasattr(encoder, "get_input_embeddings") else None,
            lambda: getattr(self.base, "embed_tokens", None),
            lambda: getattr(encoder, "embed_tokens", None),
        ]
        for get_emb in emb_getters:
            try:
                emb = get_emb()
                if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
                    shape = tuple(emb.weight.shape)
                    if len(shape) == 2 and shape[1] > 0:
                        return int(shape[1])
            except Exception:
                pass

        # 3) State dict key fallback.
        try:
            state = self.base.state_dict()
            for key in (
                "model.embed_tokens.weight",
                "encoder.embed_tokens.weight",
                "shared.weight",
                "embed_tokens.weight",
            ):
                tensor = state.get(key)
                if tensor is not None and tensor.ndim == 2 and tensor.shape[1] > 0:
                    return int(tensor.shape[1])
        except Exception:
            pass

        # 4) Fallback through dict-like configs (covers nested model-specific structures).
        cfg_dict = {}
        if hasattr(self.base.config, "to_dict"):
            try:
                cfg_dict = self.base.config.to_dict()
            except Exception:
                cfg_dict = {}

        for key in ("hidden_size", "d_model", "model_dim", "dim"):
            val = cfg_dict.get(key)
            if isinstance(val, int) and val > 0:
                return val

        text_cfg = cfg_dict.get("text_config")
        if isinstance(text_cfg, dict):
            for key in ("hidden_size", "d_model", "model_dim", "dim"):
                val = text_cfg.get(key)
                if isinstance(val, int) and val > 0:
                    return val

        # 5) Last-resort: tiny forward through encoder and read output width.
        try:
            vocab = None
            emb = self.base.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight") and emb.weight is not None and emb.weight.ndim == 2:
                vocab = int(emb.weight.shape[0])
            token_id = 1
            if vocab is not None and vocab > 1:
                token_id = min(1, vocab - 1)
            input_ids = torch.tensor([[token_id]], dtype=torch.long)
            with torch.no_grad():
                out = encoder(input_ids=input_ids, return_dict=True)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is not None and hidden.ndim == 3 and hidden.shape[-1] > 0:
                return int(hidden.shape[-1])
        except Exception:
            pass

        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: Any,
    ) -> SequenceClassifierOutput:
        enc_out = self.base.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = enc_out.last_hidden_state  # [B, T, H]

        if attention_mask is None:
            pooled = hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom

        # Keep classifier matmul dtype-consistent across bf16/fp32 environments.
        pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Boilerplate trainer for classification head.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "encoder_decoder" / "t5gemma2" / "classification_head_boilerplate.yaml",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Execute training. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--smoke-run",
        action="store_true",
        default=False,
        help="Override with very small limits for a quick pipeline test.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def normalize_label(raw_label: Any) -> str:
    text = " ".join(str(raw_label).strip().split()).lower().replace("_", "-")
    candidates = [text]
    if ":" in text:
        candidates.append(text.split(":")[-1].strip())
    if " " in text:
        candidates.append(text.split(" ")[-1].strip())

    for cand in candidates:
        if cand in {"pt-br", "ptbr", "br"}:
            return "pt-br"
        if cand in {"pt-pt", "ptpt", "pt"}:
            return "pt-pt"
        if cand in {"equal", "same", "shared"}:
            return "equal"

    raise ValueError(f"Unsupported classification label value: {raw_label!r}")


def resolve_model_load_dtype(training_cfg: dict[str, Any]) -> torch.dtype | None:
    if not torch.cuda.is_available():
        return None
    if bool(training_cfg.get("bf16", False)):
        return torch.bfloat16
    if bool(training_cfg.get("fp16", False)):
        return torch.float16
    return None


def build_training_args(training_cfg: dict[str, Any], *, smoke_run: bool) -> TrainingArguments:
    max_steps = 30 if smoke_run else training_cfg.get("max_steps", -1)
    eval_steps = 10 if smoke_run else training_cfg.get("eval_steps", 200)
    save_steps = 30 if smoke_run else training_cfg.get("save_steps", 400)
    save_checkpoints = bool(training_cfg.get("save_checkpoints", False)) and not smoke_run
    load_best_model = bool(training_cfg.get("load_best_model_at_end", True)) and save_checkpoints

    kwargs: dict[str, Any] = {
        "output_dir": training_cfg["output_dir"],
        "per_device_train_batch_size": training_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
        "learning_rate": training_cfg["learning_rate"],
        "weight_decay": training_cfg["weight_decay"],
        "warmup_steps": training_cfg["warmup_steps"],
        "max_steps": max_steps,
        "num_train_epochs": training_cfg.get("num_train_epochs", 1),
        "logging_steps": training_cfg["logging_steps"],
        "save_total_limit": training_cfg["save_total_limit"],
        "bf16": training_cfg.get("bf16", False),
        "fp16": training_cfg.get("fp16", False),
        "gradient_checkpointing": training_cfg.get("gradient_checkpointing", False),
        "dataloader_num_workers": training_cfg.get("dataloader_num_workers", 4),
        "remove_unused_columns": True,
        "load_best_model_at_end": load_best_model,
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": training_cfg.get("greater_is_better", False),
        "label_names": ["labels"],
    }

    # The wrapped seq2seq backbone has tied/shared weights, which can trigger
    # safetensors shared-memory save errors in Trainer checkpoints.
    sig = inspect.signature(TrainingArguments.__init__)
    if "save_safetensors" in sig.parameters:
        kwargs["save_safetensors"] = False
    if "save_strategy" in sig.parameters:
        kwargs["save_strategy"] = "steps" if save_checkpoints else "no"
    if save_checkpoints:
        kwargs["save_steps"] = save_steps

    eval_enabled = True
    eval_strategy = training_cfg.get("eval_strategy", "steps")
    if eval_enabled:
        kwargs["eval_steps"] = eval_steps

    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = eval_strategy if eval_enabled else "no"
    else:
        kwargs["eval_strategy"] = eval_strategy if eval_enabled else "no"

    return TrainingArguments(**kwargs)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)

    acc = float((preds == labels).mean()) if len(labels) else 0.0

    f1s = []
    for lab in sorted(set(labels.tolist()) | set(preds.tolist())):
        tp = int(((preds == lab) & (labels == lab)).sum())
        fp = int(((preds == lab) & (labels != lab)).sum())
        fn = int(((preds != lab) & (labels == lab)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        f1s.append(f1)

    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0
    return {"accuracy": acc, "f1_macro": macro_f1}


def _remap_state_dict_keys_for_layout(
    state: dict[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())

    model_uses_text_model = any(".encoder.text_model." in k for k in model_keys)
    state_uses_text_model = any(".encoder.text_model." in k for k in state_keys)

    if model_uses_text_model == state_uses_text_model:
        return state

    remapped: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        new_key = key
        if model_uses_text_model and not state_uses_text_model:
            if key.startswith("base.model.encoder.layers."):
                new_key = key.replace("base.model.encoder.", "base.model.encoder.text_model.", 1)
            elif key.startswith("base.model.encoder.embed_tokens."):
                new_key = key.replace("base.model.encoder.", "base.model.encoder.text_model.", 1)
            elif key.startswith("base.model.encoder.norm."):
                new_key = key.replace("base.model.encoder.", "base.model.encoder.text_model.", 1)
            elif key.startswith("encoder.layers."):
                new_key = key.replace("encoder.", "base.model.encoder.text_model.", 1)
            elif key.startswith("encoder.embed_tokens."):
                new_key = key.replace("encoder.", "base.model.encoder.text_model.", 1)
            elif key.startswith("encoder.norm."):
                new_key = key.replace("encoder.", "base.model.encoder.text_model.", 1)
        elif (not model_uses_text_model) and state_uses_text_model:
            if key.startswith("base.model.encoder.text_model."):
                new_key = key.replace("base.model.encoder.text_model.", "base.model.encoder.", 1)
            elif key.startswith("encoder.text_model."):
                new_key = key.replace("encoder.text_model.", "encoder.", 1)
        remapped[new_key] = val
    return remapped


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora", {})
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    seed = cfg.get("seed", 42)

    print("Step 4B boilerplate")
    print(f"  config: {args.config}")
    print(f"  model: {model_cfg['base_model']}")
    print(f"  train data: {data_cfg['train_path']}")
    print(f"  valid data: {data_cfg['valid_path']}")
    print(f"  output dir: {train_cfg['output_dir']}")
    print(f"  lora_enabled={bool(lora_cfg.get('enabled', False))}")
    print(f"  smoke_run={args.smoke_run}")

    if not args.execute:
        print("\nDry-run mode. Add --execute when ready.")
        print("Checks you should do before execute:")
        print("1) Step 1 model access passes.")
        print("2) classification_{train,valid}.jsonl exist.")
        print("3) Start with --smoke-run first.")
        return

    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        use_fast=True,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    load_dtype = resolve_model_load_dtype(train_cfg)
    print(f"  model_load_dtype={str(load_dtype) if load_dtype is not None else 'default'}")

    model = EncoderClassifier(
        model_cfg["base_model"],
        num_labels=int(model_cfg["num_labels"]),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        torch_dtype=load_dtype,
        freeze_decoder=bool(model_cfg.get("freeze_decoder", True)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        lora_cfg=lora_cfg,
    )

    init_from_dir = train_cfg.get("init_from_dir")
    if init_from_dir:
        init_dir = Path(str(init_from_dir))
        init_state_path = init_dir / "classifier_state_dict.pt"
        if not init_state_path.exists():
            raise SystemExit(f"init_from_dir provided but missing state dict: {init_state_path}")
        print(f"Loading init weights from: {init_state_path}")
        init_state = torch.load(init_state_path, map_location="cpu")
        try:
            model.load_state_dict(init_state, strict=True)
        except RuntimeError as first_error:
            remapped_state = _remap_state_dict_keys_for_layout(init_state, model)
            load_result = model.load_state_dict(remapped_state, strict=False)
            missing = list(load_result.missing_keys)
            critical_missing = [k for k in missing if k.startswith("classifier.")]
            if critical_missing:
                raise RuntimeError(
                    "State dict init load failed; classifier head weights are missing after remap.\n"
                    f"Original error: {first_error}\n"
                    f"Critical missing keys: {critical_missing[:10]}"
                ) from first_error
            print(
                "Warning: non-strict init state_dict load after key remap. "
                f"missing={len(missing)}, unexpected={len(load_result.unexpected_keys)}"
            )

    print_trainable_stats(model, prefix="[classifier]")

    print("Loading datasets...")
    raw_ds = load_dataset(
        "json",
        data_files={"train": data_cfg["train_path"], "validation": data_cfg["valid_path"]},
    )
    raw_ds["train"] = maybe_limit_split(
        raw_ds["train"],
        max_rows=data_cfg.get("max_train_rows"),
        seed=seed,
        split_name="train",
    )
    raw_ds["validation"] = maybe_limit_split(
        raw_ds["validation"],
        max_rows=data_cfg.get("max_valid_rows"),
        seed=seed,
        split_name="validation",
    )

    max_len = int(model_cfg.get("max_source_length", 512))

    def preprocess(batch):
        out = tok(
            batch["input_text"],
            truncation=True,
            max_length=max_len,
        )
        label2id = model_cfg["label2id"]
        if "label_id" in batch:
            out["labels"] = [int(x) for x in batch["label_id"]]
        elif "label" in batch:
            out["labels"] = [int(label2id[normalize_label(lbl)]) for lbl in batch["label"]]
        elif "target_text" in batch:
            out["labels"] = [int(label2id[normalize_label(lbl)]) for lbl in batch["target_text"]]
        else:
            raise KeyError("Classification dataset must include one of: label_id, label, target_text")
        return out

    tokenized = raw_ds.map(
        preprocess,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
    )

    def collate(batch):
        return tok.pad(batch, padding=True, return_tensors="pt")

    train_args = build_training_args(train_cfg, smoke_run=args.smoke_run)

    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": collate,
        "compute_metrics": compute_metrics,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok

    trainer = Trainer(**trainer_kwargs)

    print("Starting classification-head training...")
    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model state + tokenizer + label maps for reproducibility.
    torch.save(model.state_dict(), output_dir / "classifier_state_dict.pt")
    tok.save_pretrained(output_dir.as_posix())

    meta = {
        "base_model": model_cfg["base_model"],
        "num_labels": int(model_cfg["num_labels"]),
        "label2id": model_cfg["label2id"],
        "id2label": model_cfg["id2label"],
        "max_source_length": max_len,
        "lora": lora_cfg if bool(lora_cfg.get("enabled", False)) else None,
    }
    (output_dir / "classifier_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved classification artifacts to {output_dir}")


if __name__ == "__main__":
    main()
