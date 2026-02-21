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


class EncoderClassifier(torch.nn.Module):
    def __init__(
        self,
        base_model: str,
        *,
        num_labels: int,
        trust_remote_code: bool,
        freeze_decoder: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.base = AutoModelForSeq2SeqLM.from_pretrained(
                base_model,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        except TypeError:
            # Backward compatibility with transformers versions that still expect torch_dtype.
            self.base = AutoModelForSeq2SeqLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

        hidden_size = self._infer_hidden_size()
        if hidden_size is None:
            raise ValueError(
                "Could not infer hidden size from model config. "
                "Tried base config, text_config, and encoder config."
            )

        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(int(hidden_size), int(num_labels))
        print(f"[classifier] inferred hidden_size={int(hidden_size)}")

        if freeze_decoder:
            for name, param in self.base.named_parameters():
                if (
                    name.startswith("decoder")
                    or ".decoder." in name
                    or name.startswith("lm_head")
                    or ".lm_head." in name
                ):
                    param.requires_grad = False

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


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    seed = cfg.get("seed", 42)

    print("Step 4B boilerplate")
    print(f"  config: {args.config}")
    print(f"  model: {model_cfg['base_model']}")
    print(f"  train data: {data_cfg['train_path']}")
    print(f"  valid data: {data_cfg['valid_path']}")
    print(f"  output dir: {train_cfg['output_dir']}")
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

    model = EncoderClassifier(
        model_cfg["base_model"],
        num_labels=int(model_cfg["num_labels"]),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        freeze_decoder=bool(model_cfg.get("freeze_decoder", True)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    print("Loading datasets...")
    raw_ds = load_dataset(
        "json",
        data_files={"train": data_cfg["train_path"], "validation": data_cfg["valid_path"]},
    )

    max_len = int(model_cfg.get("max_source_length", 512))

    def preprocess(batch):
        out = tok(
            batch["input_text"],
            truncation=True,
            max_length=max_len,
        )
        if "label_id" in batch:
            out["labels"] = [int(x) for x in batch["label_id"]]
        else:
            label2id = model_cfg["label2id"]
            out["labels"] = [int(label2id[str(lbl)]) for lbl in batch["label"]]
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
    }
    (output_dir / "classifier_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved classification artifacts to {output_dir}")


if __name__ == "__main__":
    main()
