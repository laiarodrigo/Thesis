#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import torch
import torch.nn.functional as F
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
    TrainerCallback,
)

from task_balanced_sampler import FixedTaskMixSampler, normalize_task_kind

VOCAB_CONFIG_KEYS = {"vocab_size", "encoder_vocab_size", "decoder_vocab_size"}
TASK_KIND_TO_ID = {"translation": 0, "classification": 1}
TASK_ID_TO_KIND = {value: key for key, value in TASK_KIND_TO_ID.items()}


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


def maybe_override_precision(training_cfg: dict[str, Any]) -> dict[str, Any]:
    override = os.environ.get("THESIS_PRECISION_OVERRIDE", "").strip().casefold()
    if not override:
        return training_cfg
    if override not in {"bf16", "fp16"}:
        raise ValueError("THESIS_PRECISION_OVERRIDE must be 'bf16' or 'fp16'")
    if training_cfg.get("require_bf16", False) and override != "bf16":
        raise ValueError(
            "This config requires bf16 and cannot be overridden to fp16"
        )
    updated = dict(training_cfg)
    updated["bf16"] = override == "bf16"
    updated["fp16"] = override == "fp16"
    updated["require_bf16"] = bool(training_cfg.get("require_bf16", False)) or override == "bf16"
    print(
        "Precision override:"
        f" configured_bf16={training_cfg.get('bf16', False)}"
        f" configured_fp16={training_cfg.get('fp16', False)}"
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


class TrainingExampleCounter:
    """Count task-labelled training occurrences actually passed to the model."""

    def __init__(self, dataset_counts: dict[str, int]) -> None:
        self.dataset_counts = {
            str(key): int(value) for key, value in dataset_counts.items()
        }
        self.seen: Counter[str] = Counter()
        self.micro_batches = 0

    def update(self, task_kind_ids: torch.Tensor) -> None:
        values = task_kind_ids.detach().reshape(-1).cpu().tolist()
        for raw_value in values:
            task_kind = TASK_ID_TO_KIND.get(int(raw_value))
            if task_kind is None:
                raise RuntimeError(f"Unknown task-kind ID in training batch: {raw_value}")
            self.seen[task_kind] += 1
            self.seen["total"] += 1
        self.micro_batches += 1

    def report(self, global_step: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "count_semantics": "training row occurrences consumed; evaluation excluded",
            "global_step": int(global_step),
            "micro_batches": int(self.micro_batches),
            "dataset_rows": dict(self.dataset_counts),
            "consumed_rows": {
                "total": int(self.seen["total"]),
                "translation": int(self.seen["translation"]),
                "classification": int(self.seen["classification"]),
            },
        }

    def write(self, path: Path, global_step: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(self.report(global_step), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        consumed = payload.get("consumed_rows", {})
        self.seen = Counter(
            {
                "total": int(consumed.get("total", 0)),
                "translation": int(consumed.get("translation", 0)),
                "classification": int(consumed.get("classification", 0)),
            }
        )
        self.micro_batches = int(payload.get("micro_batches", 0))
        print(f"Restored training example counts from {path}: {self.report(payload.get('global_step', 0))}")


class TrainingExampleCountCallback(TrainerCallback):
    def __init__(self, counter: TrainingExampleCounter) -> None:
        self.counter = counter

    def _write(self, args, state, *, checkpoint: bool = False) -> None:
        report = self.counter.report(state.global_step)
        print(
            "TRAINING_EXAMPLE_COUNTS"
            f" step={report['global_step']}"
            f" total={report['consumed_rows']['total']}"
            f" translation={report['consumed_rows']['translation']}"
            f" classification={report['consumed_rows']['classification']}"
        )
        output_dir = Path(args.output_dir)
        self.counter.write(output_dir / "training_example_counts.json", state.global_step)
        if checkpoint:
            self.counter.write(
                output_dir
                / f"checkpoint-{state.global_step}"
                / "training_example_counts.json",
                state.global_step,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            report = self.counter.report(state.global_step)["consumed_rows"]
            logs["examples_seen_total"] = report["total"]
            logs["examples_seen_translation"] = report["translation"]
            logs["examples_seen_classification"] = report["classification"]
        self._write(args, state)

    def on_save(self, args, state, control, **kwargs):
        self._write(args, state, checkpoint=True)

    def on_train_end(self, args, state, control, **kwargs):
        self._write(args, state)


class InstrumentedSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        task_balanced_sampler=None,
        restricted_classification_token_ids=None,
        training_example_counter=None,
        **kwargs,
    ):
        self.task_balanced_sampler = task_balanced_sampler
        self.restricted_classification_token_ids = (
            [int(token_id) for token_id in restricted_classification_token_ids]
            if restricted_classification_token_ids
            else []
        )
        self.training_example_counter = training_example_counter
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, *args, **kwargs):
        if self.task_balanced_sampler is not None:
            return self.task_balanced_sampler
        return super()._get_train_sampler(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        restricted_label_ids = inputs.pop("restricted_classification_label_id", None)
        task_kind_ids = inputs.pop("task_kind_id", None)
        if model.training:
            if self.training_example_counter is None or task_kind_ids is None:
                raise RuntimeError("Training example counting is required but task IDs are missing")
            self.training_example_counter.update(task_kind_ids)
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            loss = outputs["loss"]
            logits = outputs.get("logits")
        else:
            loss = outputs.loss
            logits = getattr(outputs, "logits", None)
        if (
            self.restricted_classification_token_ids
            and restricted_label_ids is not None
            and logits is not None
        ):
            loss = self._mixed_translation_and_restricted_classification_loss(
                logits=logits,
                labels=inputs["labels"],
                restricted_label_ids=restricted_label_ids,
            )
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

    def _mixed_translation_and_restricted_classification_loss(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        restricted_label_ids: torch.Tensor,
    ) -> torch.Tensor:
        candidate_ids = torch.tensor(
            self.restricted_classification_token_ids,
            device=logits.device,
            dtype=torch.long,
        )
        if candidate_ids.numel() != 2:
            raise RuntimeError(
                "restricted_classification_token_ids must contain exactly two token IDs"
            )

        restricted_label_ids = restricted_label_ids.to(device=logits.device, dtype=torch.long)
        classification_mask = restricted_label_ids.ne(-100)

        token_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        translation_token_mask = labels.ne(-100)
        if classification_mask.any():
            translation_token_mask = translation_token_mask & ~classification_mask.unsqueeze(1)

        loss_sum = token_loss[translation_token_mask].sum()
        denom = translation_token_mask.sum().to(dtype=logits.dtype)

        if classification_mask.any():
            class_label_ids = restricted_label_ids[classification_mask]
            invalid = ~torch.isin(class_label_ids, candidate_ids)
            if invalid.any():
                bad_ids = sorted(set(int(x) for x in class_label_ids[invalid].detach().cpu()))
                raise RuntimeError(
                    "Restricted classification labels contain token IDs outside "
                    f"the configured class vocabulary: {bad_ids}"
                )
            class_targets = (class_label_ids == candidate_ids[1]).long()
            class_logits = logits[classification_mask, 0, :].index_select(
                dim=1,
                index=candidate_ids,
            )
            class_loss = F.cross_entropy(
                class_logits.float(),
                class_targets,
                reduction="sum",
            )
            loss_sum = loss_sum + class_loss
            denom = denom + classification_mask.sum().to(dtype=logits.dtype)

        if denom.item() <= 0:
            raise RuntimeError(f"Batch has no supervised tokens: {label_batch_report({'labels': labels})}")
        return loss_sum / denom

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


def strip_leading_target_special_tokens(seq: list[int], tokenizer) -> list[int]:
    """Remove tokenizer-inserted BOS/decoder-start tokens from labels.

    T5Gemma's target tokenizer can prepend a special token before the actual
    text target. The training objective should supervise the real target text:
    for final classification rows this means the first supervised token must be
    <pt-br> or <pt-pt>, and for decoder-label translation rows it means the
    first supervised token is the source-variety label.
    """

    special_ids = {
        int(token_id)
        for token_id in getattr(tokenizer, "all_special_ids", []) or []
        if token_id is not None
    }
    if not special_ids:
        return list(seq)
    cleaned = list(seq)
    while len(cleaned) > 1 and int(cleaned[0]) in special_ids:
        cleaned.pop(0)
    return cleaned


def harmonize_vocab_sizes(model, vocab_size: int) -> None:
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


def collect_config_vocab_values(obj: Any, out: list[int]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in VOCAB_CONFIG_KEYS:
                try:
                    out.append(int(value))
                except (TypeError, ValueError):
                    pass
            collect_config_vocab_values(value, out)
    elif isinstance(obj, list):
        for item in obj:
            collect_config_vocab_values(item, out)


def patch_config_vocab_values(obj: Any, target: int) -> int:
    updates = 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in VOCAB_CONFIG_KEYS:
                if int(value) != target:
                    obj[key] = int(target)
                    updates += 1
            else:
                updates += patch_config_vocab_values(value, target)
    elif isinstance(obj, list):
        for item in obj:
            updates += patch_config_vocab_values(item, target)
    return updates


def maybe_patch_local_config_vocab_before_load(model_name_or_path: str, target_vocab: int) -> None:
    """Repair local T5Gemma checkpoint configs saved after tokenizer expansion.

    T5Gemma2 refuses to instantiate if any saved encoder/decoder vocab-size
    fields disagree. Some Transformers save paths preserve the resized decoder
    vocabulary while leaving encoder-side fields at the original value, so a
    later Stage B load can fail before resize_model_vocab() gets a chance to
    normalize the model config. Remote pretrained checkpoints are not touched.
    """

    model_dir = Path(str(model_name_or_path))
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    before_values: list[int] = []
    collect_config_vocab_values(cfg, before_values)
    if not before_values:
        return

    target = max(int(target_vocab), max(before_values))
    if sorted(set(before_values)) == [target]:
        return

    updates = patch_config_vocab_values(cfg, target)
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "Patched local config vocab before load:"
        f" path={cfg_path}"
        f" target_vocab={target}"
        f" updates={updates}"
        f" before_unique={sorted(set(before_values))}"
    )


def add_control_tokens(tokenizer, model_cfg: dict[str, Any]) -> tuple[list[str], list[int], int]:
    tokens = [str(token).strip() for token in model_cfg.get("control_tokens", [])]
    tokens = list(dict.fromkeys(token for token in tokens if token))
    if not tokens:
        return [], [], 0

    # Keep labels visible during decoding: these are atomic control tokens, not
    # tokenizer special tokens that disappear under skip_special_tokens=True.
    added = int(tokenizer.add_tokens(tokens, special_tokens=False))
    token_ids: list[int] = []
    for token in tokens:
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise RuntimeError(
                f"Control token is not atomic after tokenizer update: {token!r} -> {encoded}"
            )
        token_ids.append(token_id)
    print(f"Control tokens: tokens={tokens} ids={token_ids} newly_added={added}")
    return tokens, token_ids, added


def inspect_control_strings(tokenizer, model_cfg: dict[str, Any]) -> None:
    strings = [str(value).strip() for value in model_cfg.get("control_strings", [])]
    strings = list(dict.fromkeys(value for value in strings if value))
    if not strings:
        return

    added_tokens = {
        str(value).strip()
        for value in model_cfg.get("control_tokens", [])
        if str(value).strip()
    }
    overlap = sorted(set(strings).intersection(added_tokens))
    if overlap:
        raise ValueError(
            "Control strings must not also be configured as added control tokens: "
            f"{overlap}"
        )

    require_non_atomic = bool(model_cfg.get("require_control_strings_non_atomic", False))
    require_atomic = bool(model_cfg.get("require_control_strings_atomic", False))
    if require_atomic and require_non_atomic:
        raise ValueError(
            "Only one of require_control_strings_atomic and "
            "require_control_strings_non_atomic can be true."
        )
    print("Control strings (base-tokenizer text; no vocabulary expansion):")
    for value in strings:
        token_ids = [int(token_id) for token_id in tokenizer.encode(value, add_special_tokens=False)]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        if not token_ids:
            raise RuntimeError(f"Control string tokenizes to an empty sequence: {value!r}")
        if require_non_atomic and len(token_ids) == 1:
            raise RuntimeError(
                "Control string unexpectedly tokenizes as one atomic token; this run "
                f"would not test the ordinary-string protocol: {value!r} -> {token_ids}"
            )
        if require_atomic and len(token_ids) != 1:
            raise RuntimeError(
                "Control string must tokenize as one existing base-tokenizer token "
                f"for this protocol: {value!r} -> {token_ids}"
            )
        print(f"  {value!r}: token_ids={token_ids} tokens={tokens}")


def resize_model_vocab(model, tokenizer) -> None:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise RuntimeError("Model does not expose input embeddings for tokenizer resizing.")
    old_size = int(input_embeddings.num_embeddings)
    new_size = len(tokenizer)
    if old_size != new_size:
        print(f"Resizing token embeddings: old_size={old_size} new_size={new_size}")
        model.resize_token_embeddings(new_size)
    harmonize_vocab_sizes(model, new_size)


def verify_control_token_setup(
    tokenizer,
    model,
    *,
    tokens: list[str],
    token_ids: list[int],
) -> None:
    if not tokens:
        return
    if len(tokens) != len(token_ids):
        raise RuntimeError("Control-token verification received mismatched tokens and IDs.")

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError(
            "Model must expose both input embeddings and output projections "
            "for control-token verification."
        )
    input_vocab_size = int(input_embeddings.weight.shape[0])
    output_vocab_size = int(output_embeddings.weight.shape[0])

    for token, expected_id in zip(tokens, token_ids):
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if token_id != expected_id or encoded != [expected_id]:
            raise RuntimeError(
                f"Control token is not recognized atomically: {token!r} "
                f"expected_id={expected_id} converted_id={token_id} encoded={encoded}"
            )
        if decoded != token:
            raise RuntimeError(
                f"Control token disappears or changes during decoding: "
                f"{token!r} -> {decoded!r}"
            )
        if expected_id >= input_vocab_size or expected_id >= output_vocab_size:
            raise RuntimeError(
                f"Control token ID is outside model vocabulary: {token!r} "
                f"id={expected_id} input_vocab={input_vocab_size} "
                f"output_vocab={output_vocab_size}"
            )

    print(
        "Control-token startup check passed:"
        f" tokens={dict(zip(tokens, token_ids))}"
        f" input_vocab={input_vocab_size}"
        f" output_vocab={output_vocab_size}"
    )


def resolve_restricted_classification_token_ids(tokenizer, model_cfg: dict[str, Any]) -> list[int]:
    tokens = [str(token).strip() for token in model_cfg.get("restricted_classification_tokens", [])]
    tokens = list(dict.fromkeys(token for token in tokens if token))
    if not tokens:
        return []
    if len(tokens) != 2:
        raise ValueError("model.restricted_classification_tokens must contain exactly two tokens")

    token_ids: list[int] = []
    for token in tokens:
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise RuntimeError(
                f"Restricted classification token is not atomic: {token!r} -> {encoded}"
            )
        token_ids.append(token_id)
    print(f"Restricted classification tokens: {dict(zip(tokens, token_ids))}")
    return token_ids


def adapter_trains_control_tokens(model) -> bool:
    peft_configs = getattr(model, "peft_config", {})
    if not isinstance(peft_configs, dict):
        return False
    for peft_cfg in peft_configs.values():
        if isinstance(peft_cfg, dict):
            indices = peft_cfg.get("trainable_token_indices")
        else:
            indices = getattr(peft_cfg, "trainable_token_indices", None)
        if indices:
            return True
    return False


def build_task_balanced_sampler(
    raw_train_dataset,
    *,
    task_batching_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    seed: int,
) -> FixedTaskMixSampler | None:
    if not bool(task_batching_cfg.get("enabled", False)):
        return None
    task_column = str(task_batching_cfg.get("task_column", "task"))
    if task_column not in raw_train_dataset.column_names:
        raise ValueError(
            f"Balanced task batching requires dataset column {task_column!r}."
        )

    micro_batch_size = int(train_cfg["per_device_train_batch_size"])
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    optimizer_window_size = micro_batch_size * grad_accum
    window_size = int(
        task_batching_cfg.get("window_size", optimizer_window_size)
    )
    if window_size != optimizer_window_size:
        raise ValueError(
            "Balanced task batching must cover one local optimizer-update window: "
            f"expected {optimizer_window_size}, got {window_size}"
        )
    translation_rows = task_batching_cfg.get("translation_rows_per_window")
    classification_rows = task_batching_cfg.get("classification_rows_per_window")
    if translation_rows is None and classification_rows is None:
        fraction = float(task_batching_cfg.get("translation_fraction", 0.5))
        if not 0.0 < fraction < 1.0:
            raise ValueError("task_batching.translation_fraction must be in (0, 1)")
        translation_rows = round(window_size * fraction)
        classification_rows = window_size - int(translation_rows)
    elif translation_rows is None:
        translation_rows = window_size - int(classification_rows)
    elif classification_rows is None:
        classification_rows = window_size - int(translation_rows)

    sampler = FixedTaskMixSampler(
        raw_train_dataset[task_column],
        window_size=window_size,
        translation_rows_per_window=int(translation_rows),
        classification_rows_per_window=int(classification_rows),
        seed=int(task_batching_cfg.get("seed", seed)),
    )
    translation_count = len(sampler.indices_by_task["translation"])
    classification_count = len(sampler.indices_by_task["classification"])
    dataset_translation_fraction = translation_count / (
        translation_count + classification_count
    )
    window_translation_fraction = (
        sampler.translation_rows_per_window / sampler.window_size
    )
    fraction_error = abs(dataset_translation_fraction - window_translation_fraction)
    max_fraction_error = task_batching_cfg.get("max_dataset_fraction_error")
    if max_fraction_error is not None and fraction_error > float(max_fraction_error):
        raise ValueError(
            "Balanced task window does not preserve the dataset task ratio closely enough: "
            f"dataset_translation_fraction={dataset_translation_fraction:.6f} "
            f"window_translation_fraction={window_translation_fraction:.6f} "
            f"absolute_error={fraction_error:.6f} "
            f"max_allowed={float(max_fraction_error):.6f}"
        )
    print(
        "Balanced task batching:"
        f" window_size={sampler.window_size}"
        f" translation_rows={sampler.translation_rows_per_window}"
        f" classification_rows={sampler.classification_rows_per_window}"
        f" dataset_translation_count={translation_count}"
        f" dataset_classification_count={classification_count}"
        f" dataset_translation_fraction={dataset_translation_fraction:.6f}"
        f" window_translation_fraction={window_translation_fraction:.6f}"
        f" fraction_error={fraction_error:.6f}"
        f" windows_per_epoch={sampler.windows_per_epoch}"
        f" sampled_rows_per_epoch={len(sampler)}"
    )
    return sampler


def validate_early_stopping_config(train_cfg: dict[str, Any]) -> None:
    patience = train_cfg.get("early_stopping_patience")
    if patience is None:
        if train_cfg.get("require_early_stopping", False):
            raise ValueError(
                "training.require_early_stopping=true requires "
                "training.early_stopping_patience"
            )
        return

    required_settings = {
        "do_eval": True,
        "load_best_model_at_end": True,
    }
    for key, expected in required_settings.items():
        if train_cfg.get(key, True) != expected:
            raise ValueError(
                f"early_stopping_patience requires {key}={expected!r}, "
                f"got {train_cfg.get(key, True)!r}"
            )

    eval_strategy = train_cfg.get("eval_strategy", "no")
    save_strategy = train_cfg.get("save_strategy", "steps")
    if eval_strategy == "no":
        raise ValueError("early_stopping_patience requires an evaluation strategy")
    if save_strategy != eval_strategy:
        raise ValueError(
            "load_best_model_at_end requires matching evaluation and save strategies"
        )

    if eval_strategy == "steps":
        eval_steps = int(train_cfg.get("eval_steps") or 0)
        save_steps = int(train_cfg.get("save_steps") or 0)
        if eval_steps <= 0 or save_steps <= 0:
            raise ValueError(
                "step-based early stopping requires positive eval_steps and save_steps"
            )
        if save_steps % eval_steps:
            raise ValueError(
                "save_steps must be a multiple of eval_steps when loading the best model"
            )


def validate_precision_config(train_cfg: dict[str, Any]) -> None:
    bf16 = bool(train_cfg.get("bf16", False))
    fp16 = bool(train_cfg.get("fp16", False))
    if bf16 and fp16:
        raise ValueError("training.bf16 and training.fp16 cannot both be true")
    if train_cfg.get("require_bf16", False) and not (bf16 and not fp16):
        raise ValueError(
            "training.require_bf16=true requires training.bf16=true and "
            "training.fp16=false"
        )


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


def count_dataset_tasks(dataset, task_column: str) -> dict[str, int]:
    if task_column not in dataset.column_names:
        raise ValueError(
            f"Training example counting requires dataset column {task_column!r}"
        )
    raw_counts = pc.value_counts(dataset.data.column(task_column)).to_pylist()
    counts: Counter[str] = Counter()
    for item in raw_counts:
        task_kind = normalize_task_kind(item["values"])
        counts[task_kind] += int(item["counts"])
    counts["total"] = counts["translation"] + counts["classification"]
    if counts["total"] != len(dataset):
        raise RuntimeError(
            f"Task-count mismatch: counted={counts['total']} dataset={len(dataset)}"
        )
    return {
        "total": int(counts["total"]),
        "translation": int(counts["translation"]),
        "classification": int(counts["classification"]),
    }


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
        "remove_unused_columns": training_cfg.get("remove_unused_columns", False),
        "do_eval": training_cfg.get("do_eval", True),
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
    if "save_strategy" in sig.parameters:
        kwargs["save_strategy"] = training_cfg.get("save_strategy", "steps")
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
    train_cfg = maybe_override_precision(train_cfg)
    lora_cfg = cfg.get("lora", {})
    use_lora = bool(lora_cfg.get("enabled", True))
    init_adapter_path = lora_cfg.get("init_adapter_path")
    seed = cfg.get("seed", 123)

    set_seed(seed)
    validate_precision_config(train_cfg)
    validate_early_stopping_config(train_cfg)

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], use_fast=True)
    control_tokens, control_token_ids, _ = add_control_tokens(tokenizer, model_cfg)
    inspect_control_strings(tokenizer, model_cfg)
    restricted_classification_token_ids = resolve_restricted_classification_token_ids(
        tokenizer,
        model_cfg,
    )
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
    maybe_patch_local_config_vocab_before_load(model_cfg["base_model"], len(tokenizer))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=load_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    resize_model_vocab(model, tokenizer)

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
            if (
                bool(lora_cfg.get("train_control_token_embeddings", False))
                and not adapter_trains_control_tokens(model)
            ):
                raise RuntimeError(
                    "The continued LoRA adapter does not train the configured control-token "
                    "embeddings. Rerun its preceding stage with the same canonical-token settings."
                )
            if load_dtype is not None:
                model = model.to(dtype=load_dtype)
        else:
            missing = [k for k in ("r", "alpha", "dropout") if k not in lora_cfg]
            if missing:
                raise SystemExit(
                    "LoRA mode is enabled but lora config is missing keys: "
                    + ", ".join(missing)
                )
            lora_kwargs: dict[str, Any] = {
                "task_type": TaskType.SEQ_2_SEQ_LM,
                "r": lora_cfg["r"],
                "lora_alpha": lora_cfg["alpha"],
                "lora_dropout": lora_cfg["dropout"],
                "bias": lora_cfg.get("bias", "none"),
                "target_modules": lora_cfg.get("target_modules"),
            }
            modules_to_save = lora_cfg.get("modules_to_save")
            if modules_to_save:
                lora_kwargs["modules_to_save"] = modules_to_save
            if bool(lora_cfg.get("train_control_token_embeddings", False)):
                if not control_token_ids:
                    raise ValueError(
                        "train_control_token_embeddings requires model.control_tokens"
                    )
                if "trainable_token_indices" not in inspect.signature(LoraConfig).parameters:
                    raise RuntimeError(
                        "This PEFT version cannot selectively train added token embeddings. "
                        "Upgrade PEFT to a version whose LoraConfig supports "
                        "trainable_token_indices."
                    )
                lora_kwargs["trainable_token_indices"] = control_token_ids
            peft_cfg = LoraConfig(**lora_kwargs)
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
    verify_control_token_setup(
        tokenizer,
        model,
        tokens=control_tokens,
        token_ids=control_token_ids,
    )
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
    task_column = str(data_cfg.get("task_column", "task"))
    training_dataset_task_counts = count_dataset_tasks(raw_ds["train"], task_column)
    validation_dataset_task_counts = count_dataset_tasks(raw_ds["validation"], task_column)
    print(f"Training dataset task counts: {training_dataset_task_counts}")
    print(f"Validation dataset task counts: {validation_dataset_task_counts}")
    training_example_counter = TrainingExampleCounter(training_dataset_task_counts)
    if len(raw_ds["train"]) < 1000:
        print(
            "WARNING: train split is unexpectedly small for a 4B run. "
            "Double-check the exported JSONL path and line count."
        )
    task_balanced_sampler = build_task_balanced_sampler(
        raw_ds["train"],
        task_batching_cfg=cfg.get("task_batching", {}),
        train_cfg=train_cfg,
        seed=seed,
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
        label_ids = [
            strip_leading_target_special_tokens(list(seq), tokenizer)
            for seq in labels["input_ids"]
        ]
        first_token_only_flags = batch.get("loss_on_first_token_only")
        if first_token_only_flags is None:
            first_token_only_flags = [False] * len(label_ids)
        loss_mask_prefix_tokens = batch.get("loss_mask_prefix_tokens")
        if loss_mask_prefix_tokens is None:
            loss_mask_prefix_tokens = [0] * len(label_ids)
        task_kind_ids = [
            TASK_KIND_TO_ID[normalize_task_kind(value)]
            for value in batch[task_column]
        ]
        restricted_label_ids: list[int] = []
        if restricted_classification_token_ids:
            allowed_restricted_ids = set(restricted_classification_token_ids)
            for seq, task_kind_id in zip(label_ids, task_kind_ids):
                if task_kind_id != TASK_KIND_TO_ID["classification"]:
                    restricted_label_ids.append(-100)
                    continue
                if not seq:
                    raise ValueError(
                        "Restricted classification row has an empty target sequence"
                    )
                first_label_id = int(seq[0])
                if first_label_id not in allowed_restricted_ids:
                    raise ValueError(
                        "Restricted classification target is outside the configured "
                        f"class vocabulary: token_id={first_label_id} "
                        f"seq_head={seq[:8]} allowed={sorted(allowed_restricted_ids)}"
                    )
                restricted_label_ids.append(first_label_id)
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
        model_inputs["task_kind_id"] = task_kind_ids
        if restricted_classification_token_ids:
            model_inputs["restricted_classification_label_id"] = restricted_label_ids
        return model_inputs

    tokenized = raw_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
    )

    training_args = build_training_args(train_cfg)
    if int(training_args.world_size) != 1:
        raise RuntimeError(
            "Exact training-example accounting currently requires one process; "
            f"got world_size={training_args.world_size}"
        )
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
    callbacks.append(TrainingExampleCountCallback(training_example_counter))
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
        "task_balanced_sampler": task_balanced_sampler,
        "restricted_classification_token_ids": restricted_classification_token_ids,
        "training_example_counter": training_example_counter,
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
        count_path = resume_path / "training_example_counts.json"
        if not count_path.exists():
            raise SystemExit(
                "Resume checkpoint is missing training example counts: "
                f"{count_path}"
            )
        training_example_counter.load(count_path)

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if bool(train_cfg.get("final_evaluate", False)):
        trainer.evaluate()

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir.as_posix())
    tokenizer.save_pretrained(output_dir.as_posix())
    training_example_counter.write(
        output_dir / "training_example_counts.json",
        trainer.state.global_step,
    )
    maybe_patch_local_config_vocab_before_load(output_dir.as_posix(), len(tokenizer))
    if use_lora:
        print(f"Saved LoRA adapter and tokenizer to {output_dir}")
    else:
        print(f"Saved fully fine-tuned model and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
