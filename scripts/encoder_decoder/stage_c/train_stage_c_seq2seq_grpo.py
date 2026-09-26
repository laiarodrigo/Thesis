#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sacrebleu import sentence_bleu
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_scheduler

try:
    from scripts.encoder_decoder.eval.metrics_utils import sentence_ter
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.encoder_decoder.eval.metrics_utils import sentence_ter


TR_BR_SOURCE = "<pt-br>"
TR_PT_SOURCE = "<pt-pt>"
WORD_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass
class StageCExample:
    record_id: str
    source_text: str
    target_text: str
    direction: str
    dataset: str
    bucket: str
    stage_bucket: str


@dataclass(frozen=True)
class DecoderTargetConfig:
    encoder_input_format: str
    target_format: str
    br_token: str
    pt_token: str
    encoder_br_token: str
    encoder_pt_token: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage C seq2seq RL (Dr. GRPO-style).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", default=False)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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


def load_tokenizer(model_cfg: dict[str, Any], init_adapter_path: str | None):
    candidates = [
        model_cfg.get("tokenizer_path"),
        init_adapter_path,
        model_cfg["base_model"],
    ]
    tokenizer = None
    tokenizer_source = None
    for candidate in dict.fromkeys(str(item) for item in candidates if item):
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True)
            tokenizer_source = candidate
            break
        except Exception:
            continue
    if tokenizer is None:
        raise RuntimeError("Unable to load tokenizer from tokenizer_path/adapter/base model.")

    tokens = [str(token).strip() for token in model_cfg.get("control_tokens", [])]
    tokens = list(dict.fromkeys(token for token in tokens if token))
    added = int(tokenizer.add_tokens(tokens, special_tokens=False)) if tokens else 0
    token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]
    print(
        f"Tokenizer loaded from: {tokenizer_source}"
        f" control_tokens={dict(zip(tokens, token_ids))}"
        f" newly_added={added}"
    )
    return tokenizer, tokens, token_ids


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


def verify_control_token_setup(tokenizer, model, tokens: list[str], token_ids: list[int]) -> None:
    if not tokens:
        return
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("Model does not expose both input and output token embeddings.")
    input_vocab_size = int(input_embeddings.weight.shape[0])
    output_vocab_size = int(output_embeddings.weight.shape[0])
    for token, expected_id in zip(tokens, token_ids):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(
            [expected_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if encoded != [expected_id] or decoded != token:
            raise RuntimeError(
                f"Control token is not recognized atomically: {token!r} "
                f"id={expected_id} encoded={encoded} decoded={decoded!r}"
            )
        if expected_id >= input_vocab_size or expected_id >= output_vocab_size:
            raise RuntimeError(
                f"Control token ID is outside model vocabulary: {token!r} id={expected_id}"
            )
    print(
        "Control-token startup check passed:"
        f" tokens={dict(zip(tokens, token_ids))}"
        f" input_vocab={input_vocab_size}"
        f" output_vocab={output_vocab_size}"
    )


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
        for _, param in module.named_parameters(recurse=False):
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


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").replace("\r", " ").split())


def build_decoder_target_config(model_cfg: dict[str, Any]) -> DecoderTargetConfig:
    encoder_input_format = normalize_text(str(model_cfg.get("encoder_input_format", "task_prefix"))).lower()
    if encoder_input_format not in {"task_prefix", "plain_source"}:
        raise ValueError(
            "Unsupported model.encoder_input_format="
            f"{encoder_input_format!r}. Use 'task_prefix' or 'plain_source'."
        )
    target_format = normalize_text(str(model_cfg.get("decoder_target_format", "plain"))).lower()
    if target_format not in {"plain", "label_first"}:
        raise ValueError(
            f"Unsupported model.decoder_target_format={target_format!r}. Use 'plain' or 'label_first'."
        )
    br_token = normalize_text(str(model_cfg.get("br_token", "BR"))) or "BR"
    pt_token = normalize_text(str(model_cfg.get("pt_token", "PT"))) or "PT"
    encoder_br_token = normalize_text(str(model_cfg.get("encoder_br_token", TR_BR_SOURCE))) or TR_BR_SOURCE
    encoder_pt_token = normalize_text(str(model_cfg.get("encoder_pt_token", TR_PT_SOURCE))) or TR_PT_SOURCE
    return DecoderTargetConfig(
        encoder_input_format=encoder_input_format,
        target_format=target_format,
        br_token=br_token,
        pt_token=pt_token,
        encoder_br_token=encoder_br_token,
        encoder_pt_token=encoder_pt_token,
    )


def label_alias_map(decoder_cfg: DecoderTargetConfig) -> dict[str, str]:
    aliases = {
        "br": "pt-br",
        "pt": "pt-pt",
        "pt-br": "pt-br",
        "pt-pt": "pt-pt",
    }
    custom_aliases = {
        normalize_text(decoder_cfg.br_token).lower(): "pt-br",
        normalize_text(decoder_cfg.pt_token).lower(): "pt-pt",
    }
    for raw, normalized in custom_aliases.items():
        if raw:
            aliases[raw] = normalized
    return aliases


def normalize_label(text: str, decoder_cfg: DecoderTargetConfig) -> str | None:
    normalized = normalize_text(text or "").lower()
    if not normalized:
        return None

    aliases = label_alias_map(decoder_cfg)
    first = normalized.split(" ", 1)[0].strip(",:;.-_")
    if first in aliases:
        return aliases[first]
    if "equal" in normalized or "shared" in normalized or normalized == "same":
        return "equal"
    if "pt-br" in normalized or "ptbr" in normalized or "brasil" in normalized:
        return "pt-br"
    if "pt-pt" in normalized or "ptpt" in normalized or "europeu" in normalized or "portugal" in normalized:
        return "pt-pt"
    return None


def strip_decoder_label_prefix(text: str, decoder_cfg: DecoderTargetConfig) -> str:
    raw = normalize_text(text or "")
    if not raw:
        return raw
    aliases = label_alias_map(decoder_cfg)
    parts = raw.split(" ", 1)
    first = parts[0].strip(",:;.-_").lower()
    if first not in aliases:
        return raw
    if len(parts) == 1:
        return ""
    return normalize_text(parts[1])


def word_edit_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> int:
    if not ref_tokens:
        return len(hyp_tokens)
    if not hyp_tokens:
        return len(ref_tokens)

    prev = list(range(len(hyp_tokens) + 1))
    for i, ref_tok in enumerate(ref_tokens, start=1):
        curr = [i]
        for j, hyp_tok in enumerate(hyp_tokens, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def word_error_rate(hyp_text: str, ref_text: str) -> float:
    ref_tokens = normalize_text(ref_text).split()
    hyp_tokens = normalize_text(hyp_text).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return word_edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def canonical_direction(value: str) -> str:
    text = normalize_text(value).casefold()
    if text in {"translate_br2pt", "br2pt"}:
        return "translate_br2pt"
    if text in {"translate_pt2br", "pt2br"}:
        return "translate_pt2br"
    raise ValueError(f"Unsupported direction: {value!r}")


def encoder_input_text(direction: str, source_text: str, decoder_cfg: DecoderTargetConfig) -> str:
    if decoder_cfg.encoder_input_format == "plain_source":
        return normalize_text(source_text)
    if direction == "translate_br2pt":
        return f"{decoder_cfg.encoder_br_token} {source_text}".strip()
    if direction == "translate_pt2br":
        return f"{decoder_cfg.encoder_pt_token} {source_text}".strip()
    raise ValueError(
        f"Unsupported direction={direction!r} for encoder_input_format={decoder_cfg.encoder_input_format!r}"
    )


def expected_label_norm_for_direction(direction: str) -> str:
    if direction == "translate_br2pt":
        return "pt-br"
    if direction == "translate_pt2br":
        return "pt-pt"
    raise ValueError(f"Unsupported direction: {direction!r}")


def target_label_norm_for_direction(direction: str) -> str:
    if direction == "translate_br2pt":
        return "pt-pt"
    if direction == "translate_pt2br":
        return "pt-br"
    raise ValueError(f"Unsupported direction: {direction!r}")


def label_candidate_for_norm(label_norm: str, decoder_cfg: DecoderTargetConfig) -> str:
    if label_norm == "pt-br":
        return decoder_cfg.br_token
    if label_norm == "pt-pt":
        return decoder_cfg.pt_token
    raise ValueError(f"Unsupported label norm: {label_norm!r}")


def expected_label_token_for_direction(direction: str, decoder_cfg: DecoderTargetConfig) -> str:
    normalized = expected_label_norm_for_direction(direction)
    if normalized == "pt-br":
        return decoder_cfg.br_token
    return decoder_cfg.pt_token


def format_model_target_text(ex: StageCExample, decoder_cfg: DecoderTargetConfig) -> str:
    target = normalize_text(ex.target_text)
    if decoder_cfg.target_format == "label_first":
        label = expected_label_token_for_direction(ex.direction, decoder_cfg)
        return f"{label} {target}".strip()
    return target


def normalize_candidate_text_for_metrics(text: str, decoder_cfg: DecoderTargetConfig) -> str:
    normalized = normalize_text(text or "")
    if decoder_cfg.target_format == "label_first":
        return strip_decoder_label_prefix(normalized, decoder_cfg)
    return normalized


def classifier_input_text_for_candidate(
    candidate_clean: str,
    *,
    decoder_cfg: DecoderTargetConfig,
    reward_cfg: dict[str, Any],
) -> str:
    input_format = normalize_text(str(reward_cfg.get("classifier_input_format", "auto"))).lower()
    classification_control = normalize_text(str(reward_cfg.get("classification_control", "CLS"))) or "CLS"
    if input_format == "auto":
        input_format = "classification_prefix" if decoder_cfg.target_format == "plain" else "plain_source"
    if input_format == "classification_prefix":
        return f"{classification_control} {candidate_clean}".strip()
    if input_format == "plain_source":
        return candidate_clean
    raise ValueError(
        f"Unsupported reward.classifier_input_format={input_format!r}. "
        "Use 'auto', 'classification_prefix', or 'plain_source'."
    )


def reward_classification_candidates(
    reward_cfg: dict[str, Any],
    decoder_cfg: DecoderTargetConfig,
) -> list[str]:
    raw_candidates = reward_cfg.get("classification_candidates")
    if raw_candidates is None:
        raw_candidates = [decoder_cfg.br_token, decoder_cfg.pt_token]
    candidates = [normalize_text(str(candidate)) for candidate in raw_candidates]
    candidates = list(dict.fromkeys(candidate for candidate in candidates if candidate))
    if not candidates:
        raise ValueError("reward.classification_candidates must contain at least one label.")
    return candidates


def resolve_classification_candidate_for_norm(
    label_norm: str,
    candidates: list[str],
    decoder_cfg: DecoderTargetConfig,
) -> str:
    aliases = label_alias_map(decoder_cfg)
    for candidate in candidates:
        normalized = normalize_text(candidate).lower()
        first = normalized.split(" ", 1)[0].strip(",:;.-_")
        if aliases.get(first) == label_norm:
            return candidate
    fallback = label_candidate_for_norm(label_norm, decoder_cfg)
    if fallback in candidates:
        return fallback
    raise ValueError(
        f"Cannot map label {label_norm!r} to reward.classification_candidates={candidates!r}."
    )


def softmax_prob_for_candidate(scores: dict[str, float], target_candidate: str) -> float:
    max_score = max(scores.values())
    exp_values = {key: math.exp(value - max_score) for key, value in scores.items()}
    denom = sum(exp_values.values())
    if denom <= 0.0:
        return 0.0
    return float(exp_values[target_candidate] / denom)


def decoder_start_token_id(model, tokenizer) -> int:
    start_id = getattr(model.config, "decoder_start_token_id", None)
    if start_id is None:
        start_id = tokenizer.pad_token_id
    if start_id is None:
        start_id = tokenizer.eos_token_id
    if start_id is None:
        raise ValueError("Could not determine decoder_start_token_id / pad_token_id / eos_token_id.")
    return int(start_id)


def prepare_decoder_input_ids_from_labels(model, labels_for_shift: torch.Tensor) -> torch.Tensor:
    prep_fn = getattr(model, "prepare_decoder_input_ids_from_labels", None)
    if prep_fn is None:
        raise AttributeError("Model does not expose prepare_decoder_input_ids_from_labels.")
    try:
        return prep_fn(labels=labels_for_shift)
    except TypeError as exc:
        if "unexpected keyword argument 'labels'" not in str(exc):
            raise
        return prep_fn(labels_for_shift)


def score_classification_candidates_for_reward(
    model,
    tokenizer,
    *,
    inputs: list[str],
    candidates: list[str],
    max_source_length: int,
    mode: str,
    device: torch.device,
) -> list[dict[str, float]]:
    enc = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)

    if mode == "score-first-token":
        candidate_token_ids: list[int] = []
        for candidate in candidates:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    "reward.classifier_mode=score-first-token requires one-token labels, "
                    f"but candidate {candidate!r} tokenizes to {ids!r}"
                )
            candidate_token_ids.append(int(ids[0]))
        decoder_input_ids = torch.full(
            (len(inputs), 1),
            decoder_start_token_id(model, tokenizer),
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            ).logits[:, 0, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            candidate_log_probs = log_probs[:, candidate_token_ids]
        return [
            {
                candidate: float(candidate_log_probs[row_idx, candidate_idx].item())
                for candidate_idx, candidate in enumerate(candidates)
            }
            for row_idx in range(len(inputs))
        ]

    if mode != "score-sequences":
        raise ValueError(
            f"Unsupported reward.classifier_mode={mode!r}. Use 'score-sequences' or 'score-first-token'."
        )

    batch_scores: list[dict[str, float]] = [dict() for _ in inputs]
    for candidate in candidates:
        dec = tokenizer(
            [candidate] * len(inputs),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        target_ids = dec["input_ids"]
        target_mask = dec["attention_mask"]
        labels_for_shift = target_ids.masked_fill(target_mask == 0, -100)
        decoder_input_ids = prepare_decoder_input_ids_from_labels(model, labels_for_shift)
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            ).logits
            token_log_probs = torch.log_softmax(logits, dim=-1).gather(
                -1,
                target_ids.unsqueeze(-1),
            ).squeeze(-1)
            masked = token_log_probs * target_mask
            seq_scores = masked.sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)
        for row_idx in range(len(inputs)):
            batch_scores[row_idx][candidate] = float(seq_scores[row_idx].item())
    return batch_scores


def compute_classifier_target_probs(
    model,
    tokenizer,
    batch: list[StageCExample],
    grouped_candidates: list[list[str]],
    *,
    reward_cfg: dict[str, Any],
    decoder_cfg: DecoderTargetConfig,
    model_cfg: dict[str, Any],
    device: torch.device,
) -> list[float]:
    candidates = reward_classification_candidates(reward_cfg, decoder_cfg)
    mode = normalize_text(str(reward_cfg.get("classifier_mode", "score-sequences"))).lower()
    max_source_length = int(reward_cfg.get("classifier_max_source_length", model_cfg.get("max_source_length", 512)))
    classifier_batch_size = int(reward_cfg.get("classifier_batch_size", 16))

    inputs: list[str] = []
    target_candidates: list[str] = []
    for ex, candidate_group in zip(batch, grouped_candidates):
        target_norm = target_label_norm_for_direction(ex.direction)
        target_candidate = resolve_classification_candidate_for_norm(target_norm, candidates, decoder_cfg)
        for candidate in candidate_group:
            candidate_clean = normalize_candidate_text_for_metrics(candidate, decoder_cfg)
            inputs.append(
                classifier_input_text_for_candidate(
                    candidate_clean,
                    decoder_cfg=decoder_cfg,
                    reward_cfg=reward_cfg,
                )
            )
            target_candidates.append(target_candidate)

    was_training = bool(model.training)
    model.eval()
    probs: list[float] = []
    try:
        for start in range(0, len(inputs), classifier_batch_size):
            batch_inputs = inputs[start : start + classifier_batch_size]
            batch_targets = target_candidates[start : start + classifier_batch_size]
            score_rows = score_classification_candidates_for_reward(
                model,
                tokenizer,
                inputs=batch_inputs,
                candidates=candidates,
                max_source_length=max_source_length,
                mode=mode,
                device=device,
            )
            probs.extend(
                softmax_prob_for_candidate(scores, target_candidate)
                for scores, target_candidate in zip(score_rows, batch_targets)
            )
    finally:
        if was_training:
            model.train()
    return probs


def word_tokens_for_reward(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_TOKEN_RE.finditer(normalize_text(text))]


def token_conservatism_reward(candidate_clean: str, source: str, target: str) -> float:
    hyp_tokens = word_tokens_for_reward(candidate_clean)
    if not hyp_tokens:
        return 0.0
    source_tokens = set(word_tokens_for_reward(source))
    target_tokens = set(word_tokens_for_reward(target))
    score = 0.0
    for token in hyp_tokens:
        if token not in target_tokens:
            continue
        score += 0.5 if token in source_tokens else 1.0
    return float(score / len(hyp_tokens))


def load_stage_c_dataset(path: Path) -> list[StageCExample]:
    rows: list[StageCExample] = []
    skipped_direction_rows: list[dict[str, str]] = []
    skipped_direction_count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                direction = canonical_direction(str(row.get("direction") or ""))
            except ValueError:
                skipped_direction_count += 1
                if len(skipped_direction_rows) < 5:
                    skipped_direction_rows.append(
                        {
                            "record_id": str(row.get("record_id") or f"{path.name}:{line_no}"),
                            "direction": str(row.get("direction") or ""),
                            "source_path": str(row.get("source_path") or ""),
                        }
                    )
                continue
            rows.append(
                StageCExample(
                    record_id=str(row.get("record_id") or f"{path.name}:{line_no}"),
                    source_text=normalize_text(str(row["source_text"])),
                    target_text=normalize_text(str(row["target_text"])),
                    direction=direction,
                    dataset=str(row.get("dataset") or "unknown"),
                    bucket=str(row.get("bucket") or "n/a"),
                    stage_bucket=str(row.get("stage_bucket") or "n/a"),
                )
            )
    if skipped_direction_count:
        print(
            "Skipped Stage C rows with unsupported direction:"
            f" count={skipped_direction_count}"
            f" samples={json.dumps(skipped_direction_rows, ensure_ascii=False)}"
        )
    return rows


class CyclingBatchIterator:
    def __init__(self, rows: list[StageCExample], batch_size: int, seed: int) -> None:
        self.rows = rows
        self.batch_size = int(batch_size)
        self.rng = random.Random(seed)
        self.order = list(range(len(rows)))
        self.pos = 0
        self._shuffle()

    def _shuffle(self) -> None:
        self.rng.shuffle(self.order)
        self.pos = 0

    def next_batch(self) -> list[StageCExample]:
        if not self.rows:
            raise RuntimeError("Stage C dataset is empty.")
        batch: list[StageCExample] = []
        while len(batch) < self.batch_size:
            if self.pos >= len(self.order):
                self._shuffle()
            idx = self.order[self.pos]
            self.pos += 1
            batch.append(self.rows[idx])
        return batch


def prefixed_counts(counter: Counter[str], prefix: str) -> dict[str, int]:
    return {
        key[len(prefix) :]: int(value)
        for key, value in sorted(counter.items())
        if key.startswith(prefix)
    }


def stage_c_example_counts_payload(step: int, counter: Counter[str]) -> dict[str, Any]:
    return {
        "step": int(step),
        "total": int(counter["total"]),
        "by_dataset": prefixed_counts(counter, "dataset:"),
        "by_direction": prefixed_counts(counter, "direction:"),
        "by_bucket": prefixed_counts(counter, "bucket:"),
        "by_stage_bucket": prefixed_counts(counter, "stage_bucket:"),
    }


def write_stage_c_example_counts(path: Path, step: int, counter: Counter[str]) -> None:
    payload = stage_c_example_counts_payload(step, counter)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_model_and_tokenizer(cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    lora_cfg = cfg.get("lora", {})
    use_lora = bool(lora_cfg.get("enabled", True))
    init_adapter_path = lora_cfg.get("init_adapter_path")

    tokenizer, control_tokens, control_token_ids = load_tokenizer(
        model_cfg,
        str(init_adapter_path) if init_adapter_path else None,
    )
    target_dtype = None
    load_dtype = None
    if torch.cuda.is_available():
        if train_cfg.get("bf16", False):
            target_dtype = torch.bfloat16
            load_dtype = torch.bfloat16
        elif train_cfg.get("fp16", False):
            # Keep fp16 Stage C runs in half precision so adapter-based 4B
            # training fits on 11GB-class GPUs such as the RTX 2080 Ti. The
            # previous float32 load path OOMed before the first optimizer step.
            target_dtype = torch.float16
            load_dtype = torch.float16
    print(
        "Model load:"
        f" cuda_available={torch.cuda.is_available()}"
        f" bf16={bool(train_cfg.get('bf16', False))}"
        f" fp16={bool(train_cfg.get('fp16', False))}"
        f" target_dtype={target_dtype}"
        f" load_dtype={load_dtype}"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_cfg["base_model"],
        torch_dtype=load_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    resize_model_vocab(model, tokenizer)

    if use_lora:
        if init_adapter_path:
            try:
                model = PeftModel.from_pretrained(model, str(init_adapter_path), is_trainable=True)
            except TypeError:
                model = PeftModel.from_pretrained(model, str(init_adapter_path))
                for name, param in model.named_parameters():
                    if "lora_" in name or "modules_to_save" in name:
                        param.requires_grad = True
            if load_dtype is not None:
                model = model.to(dtype=load_dtype)
        else:
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
        if load_dtype is not None:
            # Keep the frozen backbone in reduced precision while restoring
            # trainable LoRA weights to fp32 so optimizer math remains stable
            # and mixed-precision gradient handling works as expected.
            cast_trainable_params_to_fp32(model)
            # fp16-only 4B runs are more numerically stable when normalization
            # modules remain in fp32, which is a common mixed-precision
            # fine-tuning setup and only changes a tiny fraction of weights.
            cast_norm_modules_to_fp32(model)
    elif load_dtype is not None:
        model = model.to(dtype=load_dtype)

    if train_cfg.get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    verify_control_token_setup(tokenizer, model, control_tokens, control_token_ids)
    return model, tokenizer, device


def autocast_context(train_cfg: dict[str, Any]):
    if not torch.cuda.is_available():
        return nullcontext()
    if train_cfg.get("bf16", False):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if train_cfg.get("fp16", False):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


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


def finite_tensor_report(tensor: torch.Tensor) -> str:
    data = tensor.detach()
    if not torch.is_floating_point(data):
        return f"dtype={data.dtype} shape={tuple(data.shape)}"
    finite = torch.isfinite(data)
    nan_count = int(torch.isnan(data).sum().item())
    inf_count = int(torch.isinf(data).sum().item())
    finite_count = int(finite.sum().item())
    total = int(data.numel())
    return (
        f"dtype={data.dtype} shape={tuple(data.shape)}"
        f" finite={finite_count}/{total} nan={nan_count} inf={inf_count}"
    )


def ensure_finite_tensor(name: str, tensor: torch.Tensor, *, step: int, batch: list[StageCExample]) -> None:
    if torch.isfinite(tensor).all():
        return
    record_ids = [ex.record_id for ex in batch[:5]]
    raise RuntimeError(
        f"Non-finite tensor detected: {name} step={step} "
        f"records={record_ids} {finite_tensor_report(tensor)}"
    )


def ensure_finite_gradients(model, *, step: int, batch: list[StageCExample]) -> None:
    bad: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad.append(name)
            if len(bad) >= 5:
                break
    if not bad:
        return
    record_ids = [ex.record_id for ex in batch[:5]]
    raise RuntimeError(
        f"Non-finite gradients detected step={step} records={record_ids} params={bad}"
    )


def append_nonfinite_debug_log(
    path: Path,
    *,
    error: str,
    step: int,
    micro_batch: list[StageCExample],
    grouped_candidates: list[list[str]] | None,
    rewards: list[float] | None,
    advantages: list[float] | None,
    include_reference_candidate: bool,
    decoder_cfg: DecoderTargetConfig,
    include_first_token_reward: bool,
) -> None:
    payload: dict[str, Any] = {
        "nonfinite_step": step,
        "error": error,
        "records": [
            {
                "record_id": ex.record_id,
                "direction": ex.direction,
                "dataset": ex.dataset,
                "bucket": ex.bucket,
                "stage_bucket": ex.stage_bucket,
                "source_text": ex.source_text,
                "target_text": ex.target_text,
            }
            for ex in micro_batch
        ],
    }
    if grouped_candidates is not None and rewards is not None and advantages is not None:
        payload["samples"] = build_candidate_debug_samples(
            micro_batch,
            grouped_candidates,
            rewards,
            advantages,
            text_char_limit=500,
            include_reference_candidate=include_reference_candidate,
            decoder_cfg=decoder_cfg,
            include_first_token_reward=include_first_token_reward,
            max_examples=max(len(micro_batch), 1),
        )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def generation_safety_kwargs(model, rl_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    generation_cfg = getattr(model, "generation_config", None)
    if generation_cfg is None:
        return kwargs
    if hasattr(generation_cfg, "remove_invalid_values"):
        kwargs["remove_invalid_values"] = bool(rl_cfg.get("remove_invalid_values", True))
    if hasattr(generation_cfg, "renormalize_logits"):
        kwargs["renormalize_logits"] = bool(rl_cfg.get("renormalize_logits", True))
    return kwargs


def clip_text(text: str, char_limit: int) -> str:
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    if char_limit <= 3:
        return text[:char_limit]
    return text[: char_limit - 3] + "..."


def generate_candidates(
    model,
    tokenizer,
    batch: list[StageCExample],
    *,
    model_cfg: dict[str, Any],
    rl_cfg: dict[str, Any],
    decoder_cfg: DecoderTargetConfig,
    device: torch.device,
) -> list[list[str]]:
    prompts = [encoder_input_text(ex.direction, ex.source_text, decoder_cfg) for ex in batch]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(model_cfg.get("max_source_length", 512)),
    ).to(device)

    model.eval()
    requested_generations = int(rl_cfg.get("num_sampled_candidates", rl_cfg.get("num_generations", 4)))
    generate_kwargs = {
        "do_sample": bool(rl_cfg.get("do_sample", True)),
        "temperature": float(rl_cfg.get("temperature", 0.8)),
        "top_p": float(rl_cfg.get("top_p", 0.95)),
        "max_new_tokens": int(rl_cfg.get("max_completion_length", 128)),
        "num_return_sequences": requested_generations,
        "no_repeat_ngram_size": int(rl_cfg.get("no_repeat_ngram_size", 0)),
        "early_stopping": bool(rl_cfg.get("early_stopping", True)),
    }
    generate_kwargs.update(generation_safety_kwargs(model, rl_cfg))
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            **generate_kwargs,
        )
    decoded = [normalize_text(x) for x in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
    grouped: list[list[str]] = []
    k = requested_generations
    for i in range(0, len(decoded), k):
        grouped.append(decoded[i : i + k])

    model.train()
    return grouped


def nearest_unused_index(sorted_rows: list[dict[str, Any]], target_idx: int, used: set[int]) -> int | None:
    if target_idx not in used:
        return target_idx
    for distance in range(1, len(sorted_rows)):
        left = target_idx - distance
        right = target_idx + distance
        if left >= 0 and left not in used:
            return left
        if right < len(sorted_rows) and right not in used:
            return right
    return None


def select_quantile_ter_diverse_rows(rows: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    if len(rows) <= keep:
        return rows
    sorted_rows = sorted(rows, key=lambda row: (float(row["ter"]), int(row["original_idx"])))
    if keep <= 1:
        return [sorted_rows[0]]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for slot in range(keep):
        quantile = slot / max(keep - 1, 1)
        target_idx = int(round(quantile * (len(sorted_rows) - 1)))
        chosen_idx = nearest_unused_index(sorted_rows, target_idx, used)
        if chosen_idx is None:
            break
        used.add(chosen_idx)
        selected.append(sorted_rows[chosen_idx])
    return selected


def select_ter_diverse_candidates(
    batch: list[StageCExample],
    grouped_candidates: list[list[str]],
    *,
    rl_cfg: dict[str, Any],
    decoder_cfg: DecoderTargetConfig,
) -> list[list[str]]:
    selection = normalize_text(str(rl_cfg.get("candidate_selection", "all"))).lower()
    if selection in {"", "all", "none"}:
        return grouped_candidates
    if selection not in {"ter_diverse", "ter-diverse"}:
        raise ValueError(
            f"Unsupported rl.candidate_selection={selection!r}. Use 'all' or 'ter_diverse'."
        )

    keep = int(rl_cfg.get("ter_diverse_keep", rl_cfg.get("num_generations", 4)))
    keep = max(1, keep)
    drop_exact_target = bool(rl_cfg.get("ter_diverse_drop_exact_target", True))
    filter_invalid_first_token = bool(rl_cfg.get("ter_diverse_filter_invalid_first_token", False))
    selected_groups: list[list[str]] = []

    for ex, candidates in zip(batch, grouped_candidates):
        target = normalize_text(ex.target_text)
        expected_label_norm = expected_label_norm_for_direction(ex.direction)
        base_rows: list[dict[str, Any]] = []
        seen_clean: set[str] = set()
        for idx, candidate in enumerate(candidates):
            candidate_clean = normalize_candidate_text_for_metrics(candidate, decoder_cfg)
            if candidate_clean in seen_clean:
                continue
            seen_clean.add(candidate_clean)
            base_rows.append(
                {
                    "candidate": candidate,
                    "clean": candidate_clean,
                    "ter": sentence_ter(candidate_clean, target),
                    "original_idx": idx,
                }
            )

        rows = list(base_rows)
        if drop_exact_target:
            rows = [row for row in rows if row["clean"] != target]
        if filter_invalid_first_token and decoder_cfg.target_format == "label_first":
            rows = [
                row for row in rows
                if normalize_label(str(row["candidate"]), decoder_cfg) == expected_label_norm
            ]
        if not rows:
            rows = base_rows
        selected_rows = select_quantile_ter_diverse_rows(rows, keep)
        selected_groups.append([str(row["candidate"]) for row in selected_rows])

    return selected_groups


def add_reference_candidates(
    batch: list[StageCExample],
    grouped_candidates: list[list[str]],
    *,
    include_reference_candidate: bool,
    decoder_cfg: DecoderTargetConfig,
) -> list[list[str]]:
    if not include_reference_candidate:
        return grouped_candidates
    return [
        [*candidates, format_model_target_text(ex, decoder_cfg)]
        for ex, candidates in zip(batch, grouped_candidates)
    ]


def compute_rewards(
    batch: list[StageCExample],
    grouped_candidates: list[list[str]],
    *,
    reward_cfg: dict[str, Any],
    include_reference_candidate: bool,
    decoder_cfg: DecoderTargetConfig,
    classifier_target_probs: list[float] | None = None,
) -> tuple[list[float], dict[str, float | None]]:
    reward_metric = str(reward_cfg.get("metric", "bleu")).strip().lower()
    if reward_metric == "wer":
        reward_metric = "ter"
    elif reward_metric in {"bleu_wer", "bleu_ter"}:
        reward_metric = "bleu_copy_ter"
    if reward_metric not in {"bleu", "ter", "bleu_copy_ter"}:
        raise ValueError(
            "Unsupported reward.metric="
            f"{reward_metric!r}. Use 'bleu', 'ter', or 'bleu_copy_ter'."
        )
    include_first_token_reward = bool(reward_cfg.get("include_first_token_reward", False))
    hard_first_token_gate = bool(reward_cfg.get("hard_first_token_gate", False))
    include_classifier_reward = bool(reward_cfg.get("include_classifier_reward", False))
    include_token_conservatism_reward = bool(reward_cfg.get("include_token_conservatism_reward", False))
    track_first_token = include_first_token_reward or hard_first_token_gate
    if track_first_token and decoder_cfg.target_format != "label_first":
        raise ValueError(
            "reward.include_first_token_reward=true or reward.hard_first_token_gate=true "
            "requires model.decoder_target_format='label_first'."
        )

    eps = float(reward_cfg.get("epsilon", 1e-8))
    raw_copy_penalty = float(reward_cfg.get("copy_penalty", 0.0))
    use_bleu = reward_metric in {"bleu", "bleu_copy_ter"}
    use_ter = reward_metric in {"ter", "bleu_copy_ter"}
    copy_penalty = raw_copy_penalty if use_bleu else 0.0
    component_weights: dict[str, float] = {}
    if use_bleu:
        component_weights["bleu"] = float(reward_cfg.get("bleu_weight", 1.0))
    if use_ter:
        component_weights["ter"] = float(
            reward_cfg.get("ter_weight", reward_cfg.get("wer_weight", 1.0))
        )
    if include_first_token_reward:
        component_weights["first_token"] = float(reward_cfg.get("first_token_weight", 1.0))
    if include_classifier_reward:
        component_weights["classifier"] = float(reward_cfg.get("classifier_weight", 1.0))
    if include_token_conservatism_reward:
        component_weights["token_conservatism"] = float(reward_cfg.get("token_conservatism_weight", 1.0))
    total_weight = sum(component_weights.values())
    if total_weight <= 0.0:
        raise ValueError("Active reward component weights must sum to a value > 0.")
    component_weights = {
        name: weight / total_weight
        for name, weight in component_weights.items()
    }

    rewards: list[float] = []
    model_bleus: list[float] = []
    copy_bleus: list[float] = []
    model_ters: list[float] = []
    copy_ters: list[float] = []
    exact_copy_flags: list[float] = []
    distinct_rates: list[float] = []
    first_token_parse_flags: list[float] = []
    first_token_match_flags: list[float] = []
    first_token_gate_pass_flags: list[float] = []
    bleu_reward_components: list[float] = []
    ter_reward_components: list[float] = []
    first_token_reward_components: list[float] = []
    classifier_reward_components: list[float] = []
    token_conservatism_reward_components: list[float] = []
    classifier_prob_pos = 0
    if include_classifier_reward and classifier_target_probs is None:
        raise ValueError("classifier_target_probs is required when reward.include_classifier_reward=true.")

    for ex, candidates in zip(batch, grouped_candidates):
        source = normalize_text(ex.source_text)
        target = normalize_text(ex.target_text)
        copy_bleu = sentence_bleu(source, [target]).score
        copy_ter = sentence_ter(source, target)
        copy_ok = source == target
        metric_candidates = candidates[:-1] if include_reference_candidate and candidates else candidates
        metric_candidates_clean = [
            normalize_candidate_text_for_metrics(candidate, decoder_cfg)
            for candidate in metric_candidates
        ]
        distinct_rates.append(len(set(metric_candidates_clean)) / max(len(metric_candidates_clean), 1))
        expected_label_norm = expected_label_norm_for_direction(ex.direction)
        for idx, candidate in enumerate(candidates):
            candidate_clean = normalize_candidate_text_for_metrics(candidate, decoder_cfg)
            model_bleu = sentence_bleu(candidate_clean, [target]).score
            model_ter = sentence_ter(candidate_clean, target)
            reward_parts: list[float] = []
            bleu_reward = None
            ter_reward = None
            first_token_reward = None
            first_token_gate_pass = None
            classifier_reward = None
            token_reward = None
            if use_bleu:
                bleu_reward = model_bleu / (model_bleu + copy_bleu + eps)
                reward_parts.append(component_weights["bleu"] * bleu_reward)
            if use_ter:
                ter_reward = (copy_ter + eps) / (model_ter + copy_ter + (2.0 * eps))
                reward_parts.append(component_weights["ter"] * ter_reward)
            parsed_label = normalize_label(candidate, decoder_cfg) if track_first_token else None
            if include_first_token_reward:
                first_token_reward = 1.0 if parsed_label == expected_label_norm else 0.0
                reward_parts.append(component_weights["first_token"] * first_token_reward)
            if hard_first_token_gate:
                first_token_gate_pass = parsed_label == expected_label_norm
            if include_classifier_reward:
                assert classifier_target_probs is not None
                classifier_reward = float(classifier_target_probs[classifier_prob_pos])
                classifier_prob_pos += 1
                reward_parts.append(component_weights["classifier"] * classifier_reward)
            if include_token_conservatism_reward:
                token_reward = token_conservatism_reward(candidate_clean, source, target)
                reward_parts.append(component_weights["token_conservatism"] * token_reward)

            reward = sum(reward_parts)
            if hard_first_token_gate and not first_token_gate_pass:
                reward = 0.0
            elif copy_penalty > 0.0 and candidate_clean == source and not copy_ok:
                reward -= copy_penalty
            rewards.append(float(reward))
            is_reference_candidate = include_reference_candidate and idx == len(candidates) - 1
            if not is_reference_candidate:
                model_bleus.append(float(model_bleu))
                copy_bleus.append(float(copy_bleu))
                model_ters.append(float(model_ter))
                copy_ters.append(float(copy_ter))
                exact_copy_flags.append(1.0 if candidate_clean == source else 0.0)
                if bleu_reward is not None:
                    bleu_reward_components.append(float(bleu_reward))
                if ter_reward is not None:
                    ter_reward_components.append(float(ter_reward))
                if include_first_token_reward:
                    first_token_reward_components.append(float(first_token_reward or 0.0))
                if track_first_token:
                    first_token_parse_flags.append(1.0 if parsed_label is not None else 0.0)
                    first_token_match_flags.append(1.0 if parsed_label == expected_label_norm else 0.0)
                if hard_first_token_gate:
                    first_token_gate_pass_flags.append(1.0 if first_token_gate_pass else 0.0)
                if include_classifier_reward:
                    classifier_reward_components.append(float(classifier_reward or 0.0))
                if include_token_conservatism_reward:
                    token_conservatism_reward_components.append(float(token_reward or 0.0))

    if include_classifier_reward and classifier_target_probs is not None:
        if classifier_prob_pos != len(classifier_target_probs):
            raise ValueError(
                "Classifier reward probability count mismatch:"
                f" consumed={classifier_prob_pos} available={len(classifier_target_probs)}"
            )

    metrics: dict[str, float | None] = {
        "reward_mean": sum(rewards) / max(len(rewards), 1),
        "model_bleu_mean": sum(model_bleus) / max(len(model_bleus), 1),
        "copy_bleu_mean": sum(copy_bleus) / max(len(copy_bleus), 1),
        "model_ter_mean": sum(model_ters) / max(len(model_ters), 1),
        "copy_ter_mean": sum(copy_ters) / max(len(copy_ters), 1),
        "exact_copy_rate": sum(exact_copy_flags) / max(len(exact_copy_flags), 1),
        "distinct_candidate_rate": sum(distinct_rates) / max(len(distinct_rates), 1),
        "bleu_reward_component_mean": (
            sum(bleu_reward_components) / max(len(bleu_reward_components), 1)
            if use_bleu
            else None
        ),
        "ter_reward_component_mean": (
            sum(ter_reward_components) / max(len(ter_reward_components), 1)
            if use_ter
            else None
        ),
    }
    if include_first_token_reward:
        metrics["first_token_reward_component_mean"] = (
            sum(first_token_reward_components) / max(len(first_token_reward_components), 1)
        )
    else:
        metrics["first_token_reward_component_mean"] = None
    if track_first_token:
        metrics["first_token_parse_rate"] = sum(first_token_parse_flags) / max(len(first_token_parse_flags), 1)
        metrics["first_token_accuracy"] = sum(first_token_match_flags) / max(len(first_token_match_flags), 1)
    else:
        metrics["first_token_parse_rate"] = None
        metrics["first_token_accuracy"] = None
    if hard_first_token_gate:
        metrics["first_token_gate_pass_rate"] = (
            sum(first_token_gate_pass_flags) / max(len(first_token_gate_pass_flags), 1)
        )
    else:
        metrics["first_token_gate_pass_rate"] = None
    if include_classifier_reward:
        metrics["classifier_reward_component_mean"] = (
            sum(classifier_reward_components) / max(len(classifier_reward_components), 1)
        )
    else:
        metrics["classifier_reward_component_mean"] = None
    if include_token_conservatism_reward:
        metrics["token_conservatism_reward_component_mean"] = (
            sum(token_conservatism_reward_components) / max(len(token_conservatism_reward_components), 1)
        )
    else:
        metrics["token_conservatism_reward_component_mean"] = None
    return rewards, metrics


def compute_group_advantages(rewards: list[float], group_size: int | list[int], eps: float) -> list[float]:
    if isinstance(group_size, int):
        group_sizes = [group_size for _ in range(0, len(rewards), group_size)]
    else:
        group_sizes = list(group_size)
    advantages: list[float] = []
    start = 0
    for size in group_sizes:
        group = rewards[start : start + size]
        start += size
        if not group:
            continue
        mean = sum(group) / len(group)
        var = sum((x - mean) ** 2 for x in group) / len(group)
        std = math.sqrt(max(var, 0.0))
        if std <= eps:
            advantages.extend([0.0 for _ in group])
        else:
            advantages.extend([(x - mean) / (std + eps) for x in group])
    if start != len(rewards):
        raise ValueError(
            f"Group sizes do not match reward count: consumed={start} rewards={len(rewards)}"
        )
    return advantages


def tokenize_generated_targets(
    tokenizer,
    prompts: list[str],
    generated_texts: list[str],
    *,
    max_source_length: int,
    max_completion_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)
    label_batch = tokenizer(
        text_target=generated_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_completion_length,
    )
    labels = label_batch["input_ids"].to(device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    labels = labels.masked_fill(labels.eq(pad_token_id), -100)
    return enc, labels


def compute_sft_loss(
    model,
    tokenizer,
    batch: list[StageCExample],
    *,
    max_source_length: int,
    max_target_length: int,
    device: torch.device,
    decoder_cfg: DecoderTargetConfig,
) -> torch.Tensor:
    prompts = [encoder_input_text(ex.direction, ex.source_text, decoder_cfg) for ex in batch]
    targets = [format_model_target_text(ex, decoder_cfg) for ex in batch]
    enc_inputs, gold_labels = tokenize_generated_targets(
        tokenizer,
        prompts,
        targets,
        max_source_length=max_source_length,
        max_completion_length=max_target_length,
        device=device,
    )
    outputs = model(**enc_inputs, labels=gold_labels)
    return masked_ce_loss_from_logits(outputs.logits, gold_labels)


def resolve_loss_weights(
    rl_cfg: dict[str, Any],
    *,
    include_reference_candidate: bool,
) -> tuple[str, float, float]:
    loss_mix_mode = normalize_text(str(rl_cfg.get("loss_mix_mode", "manual"))).lower()
    rl_weight = float(rl_cfg.get("rl_weight", rl_cfg.get("lambda_rl", 1.0)))
    sft_weight = float(rl_cfg.get("sft_weight", rl_cfg.get("lambda_sft", 0.05)))
    if loss_mix_mode == "manual":
        return loss_mix_mode, rl_weight, sft_weight
    if loss_mix_mode in {"reference_candidate_auto", "auto_by_reference_candidate"}:
        effective_sft_weight = 0.0 if include_reference_candidate else sft_weight
        return loss_mix_mode, rl_weight, effective_sft_weight
    raise ValueError(
        f"Unsupported rl.loss_mix_mode={loss_mix_mode!r}. Use 'manual' or 'reference_candidate_auto'."
    )


def build_candidate_debug_samples(
    batch: list[StageCExample],
    grouped_candidates: list[list[str]],
    rewards: list[float],
    advantages: list[float],
    *,
    text_char_limit: int,
    include_reference_candidate: bool,
    decoder_cfg: DecoderTargetConfig,
    include_first_token_reward: bool,
    max_examples: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    reward_pos = 0
    for ex, candidates in zip(batch, grouped_candidates):
        group_rewards = rewards[reward_pos : reward_pos + len(candidates)]
        group_advantages = advantages[reward_pos : reward_pos + len(candidates)]
        reward_pos += len(candidates)
        target = normalize_text(ex.target_text)
        model_target = format_model_target_text(ex, decoder_cfg)
        source = normalize_text(ex.source_text)
        expected_label_norm = (
            expected_label_norm_for_direction(ex.direction) if include_first_token_reward else None
        )
        rows: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            is_reference_candidate = include_reference_candidate and idx == len(candidates) - 1
            row = {
                "idx": idx,
                "is_reference_candidate": is_reference_candidate,
                "matches_gold": candidate == model_target,
                "matches_source": normalize_candidate_text_for_metrics(candidate, decoder_cfg) == source,
                "reward": round(float(group_rewards[idx]), 6),
                "advantage": round(float(group_advantages[idx]), 6),
                "text": clip_text(candidate, text_char_limit),
                "text_len": len(candidate),
            }
            if include_first_token_reward:
                parsed_label = normalize_label(candidate, decoder_cfg)
                row["pred_source_variant_norm"] = parsed_label
                row["expected_source_variant_norm"] = expected_label_norm
                row["first_token_correct"] = parsed_label == expected_label_norm
            rows.append(row)
        sample = {
            "record_id": ex.record_id,
            "direction": ex.direction,
            "dataset": ex.dataset,
            "bucket": ex.bucket,
            "stage_bucket": ex.stage_bucket,
            "source_text": clip_text(source, text_char_limit),
            "target_text": clip_text(target, text_char_limit),
            "source_text_len": len(source),
            "target_text_len": len(target),
            "group_size": len(candidates),
            "unique_candidates": len(set(candidates)),
            "reference_candidate_added": bool(candidates)
            and include_reference_candidate
            and candidates[-1] == model_target,
            "candidates": rows,
        }
        if model_target != target:
            sample["model_target_text"] = clip_text(model_target, text_char_limit)
        samples.append(sample)
        if len(samples) >= max_examples:
            break
    return samples


def masked_log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    gather_labels = labels.masked_fill(labels.eq(-100), 0)
    token_log_probs = log_probs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
    mask = labels.ne(-100)
    token_log_probs = token_log_probs * mask.to(token_log_probs.dtype)
    return token_log_probs, mask


def masked_ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    token_log_probs, mask = masked_log_probs_from_logits(logits, labels)
    token_nll = -token_log_probs
    denom = mask.sum().clamp_min(1).to(token_nll.dtype)
    return token_nll.sum() / denom


def sequence_logprobs(model, enc_inputs: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    outputs = model(**enc_inputs, labels=labels)
    token_log_probs, _ = masked_log_probs_from_logits(outputs.logits, labels)
    return token_log_probs.sum(dim=-1)


def save_checkpoint(
    model,
    tokenizer,
    out_dir: Path,
    step: int,
    metrics_log_path: Path,
    *,
    save_total_limit: int,
) -> None:
    ckpt_dir = out_dir / f"checkpoint-{step}"
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
        f" EVENT save_start step={step} checkpoint_dir={ckpt_dir}"
        f" {format_gpu_stats()}",
        flush=True,
    )
    started_at = time.time()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir.as_posix())
    tokenizer.save_pretrained(ckpt_dir.as_posix())
    state = {
        "global_step": step,
        "metrics_log_path": metrics_log_path.as_posix(),
    }
    (ckpt_dir / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    checkpoints = sorted(
        [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    while len(checkpoints) > save_total_limit:
        old = checkpoints.pop(0)
        shutil.rmtree(old, ignore_errors=True)
    duration = time.time() - started_at
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"
        f" EVENT save_end step={step} checkpoint_dir={ckpt_dir}"
        f" duration_s={duration:.2f} {format_gpu_stats()}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    dataset_cfg = cfg["dataset"]
    train_cfg = maybe_override_output_dir(cfg["training"])
    validate_precision_config(train_cfg)
    rl_cfg = cfg["rl"]
    reward_cfg = cfg.get("reward", {})
    model_cfg = cfg["model"]
    decoder_cfg = build_decoder_target_config(model_cfg)

    rows = load_stage_c_dataset(Path(dataset_cfg["train_path"]))
    if not rows:
        raise SystemExit("Stage C dataset is empty.")

    out_dir = Path(train_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = out_dir / "stage_c_metrics.jsonl"
    (out_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"Loaded Stage C dataset: rows={len(rows)} path={dataset_cfg['train_path']}")
    print(
        "Stage bucket counts:",
        json.dumps(
            {
                key: sum(1 for row in rows if row.stage_bucket == key)
                for key in sorted({row.stage_bucket for row in rows})
            },
            ensure_ascii=False,
        ),
    )

    model, tokenizer, device = load_model_and_tokenizer(cfg)

    batch_size = int(train_cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    max_steps = int(train_cfg["max_steps"])
    logging_steps = int(train_cfg.get("logging_steps", 10))
    save_steps = int(train_cfg.get("save_steps", 50))
    save_total_limit = int(train_cfg.get("save_total_limit", 2))
    debug_candidate_every_steps = int(train_cfg.get("debug_candidate_every_steps", 5))
    debug_candidate_examples = int(train_cfg.get("debug_candidate_examples", 1))
    debug_text_char_limit = int(train_cfg.get("debug_text_char_limit", 200))
    learning_rate = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    max_source_length = int(model_cfg.get("max_source_length", 512))
    max_completion_length = int(rl_cfg.get("max_completion_length", 128))
    max_target_length = int(model_cfg.get("max_target_length", max_completion_length))
    num_generations = int(rl_cfg.get("num_generations", 4))
    num_sampled_candidates = int(rl_cfg.get("num_sampled_candidates", num_generations))
    candidate_selection = normalize_text(str(rl_cfg.get("candidate_selection", "all"))).lower() or "all"
    include_reference_candidate = bool(rl_cfg.get("include_reference_candidate", True))
    reward_has_first_token = bool(reward_cfg.get("include_first_token_reward", False))
    reward_hard_first_token_gate = bool(reward_cfg.get("hard_first_token_gate", False))
    reward_has_classifier = bool(reward_cfg.get("include_classifier_reward", False))
    reward_has_token_conservatism = bool(reward_cfg.get("include_token_conservatism_reward", False))
    reward_tracks_first_token = reward_has_first_token or reward_hard_first_token_gate
    generation_safety = generation_safety_kwargs(model, rl_cfg)
    if reward_tracks_first_token and decoder_cfg.target_format != "label_first":
        raise SystemExit(
            "reward.include_first_token_reward=true or reward.hard_first_token_gate=true "
            "requires model.decoder_target_format='label_first'."
        )
    try:
        loss_mix_mode, rl_weight, sft_weight = resolve_loss_weights(
            rl_cfg,
            include_reference_candidate=include_reference_candidate,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if rl_cfg.get("beta", 0.0) not in (0, 0.0):
        raise SystemExit("beta > 0 is not implemented in this custom Stage C trainer yet.")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = get_scheduler(
        name=str(train_cfg.get("lr_scheduler_type", "linear")),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(torch.cuda.is_available() and train_cfg.get("fp16", False) and not train_cfg.get("bf16", False))
    )
    iterator = CyclingBatchIterator(rows, batch_size=batch_size, seed=seed)

    if not args.execute:
        print("Dry-run only. Use --execute to train.")
        print(f"Config: {args.config}")
        print(f"Output dir: {out_dir}")
        return

    candidate_debug_log_path = out_dir / "stage_c_candidate_debug.jsonl"
    nonfinite_debug_log_path = out_dir / "stage_c_nonfinite_debug.jsonl"
    example_counts_path = out_dir / "stage_c_training_example_counts.json"
    max_nonfinite_skips = int(train_cfg.get("max_nonfinite_skips", 100))
    print(
        "Runtime:"
        f" output_dir={out_dir}"
        f" batch_size={batch_size}"
        f" grad_accum={grad_accum}"
        f" max_steps={max_steps}"
        f" save_steps={save_steps}"
        f" logging_steps={logging_steps}"
        f" debug_candidate_every_steps={debug_candidate_every_steps}"
        f" debug_text_char_limit={debug_text_char_limit}"
        f" encoder_input_format={decoder_cfg.encoder_input_format}"
        f" decoder_target_format={decoder_cfg.target_format}"
        f" encoder_br_token={decoder_cfg.encoder_br_token}"
        f" encoder_pt_token={decoder_cfg.encoder_pt_token}"
        f" br_token={decoder_cfg.br_token}"
        f" pt_token={decoder_cfg.pt_token}"
        f" num_generations={num_generations}"
        f" num_sampled_candidates={num_sampled_candidates}"
        f" candidate_selection={candidate_selection}"
        f" ter_diverse_keep={rl_cfg.get('ter_diverse_keep', '<unset>')}"
        f" include_reference_candidate={include_reference_candidate}"
        f" loss_mix_mode={loss_mix_mode}"
        f" rl_weight={rl_weight}"
        f" sft_weight={sft_weight}"
        f" reward_metric={reward_cfg.get('metric', 'bleu')}"
        f" include_first_token_reward={reward_has_first_token}"
        f" hard_first_token_gate={reward_hard_first_token_gate}"
        f" include_classifier_reward={reward_has_classifier}"
        f" include_token_conservatism_reward={reward_has_token_conservatism}"
        f" remove_invalid_values={generation_safety.get('remove_invalid_values', '<unsupported>')}"
        f" renormalize_logits={generation_safety.get('renormalize_logits', '<unsupported>')}"
        f" grad_scaler_enabled={scaler.is_enabled()}"
        f" max_nonfinite_skips={max_nonfinite_skips}"
        f" device={device}"
        f" {format_gpu_stats()}",
        flush=True,
    )

    global_step = 0
    running_loss = 0.0
    running_rl_loss = 0.0
    running_sft_loss = 0.0
    running_reward = 0.0
    running_model_bleu = 0.0
    running_copy_bleu = 0.0
    running_model_ter = 0.0
    running_copy_ter = 0.0
    running_exact_copy = 0.0
    running_distinct = 0.0
    running_bleu_reward_component = 0.0
    running_ter_reward_component = 0.0
    running_first_token_reward_component = 0.0
    running_first_token_parse = 0.0
    running_first_token_accuracy = 0.0
    running_first_token_gate_pass = 0.0
    running_classifier_reward_component = 0.0
    running_token_conservatism_reward_component = 0.0
    running_micro_batches = 0
    nonfinite_skips_total = 0
    examples_seen: Counter[str] = Counter()

    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        next_step = global_step + 1
        applied_micro_batches = 0
        last_micro_batch: list[StageCExample] | None = None
        should_debug_candidates = bool(
            debug_candidate_every_steps > 0
            and (next_step == 1 or next_step % debug_candidate_every_steps == 0)
        )
        debug_snapshot: list[dict[str, Any]] | None = None

        for _ in range(grad_accum):
            micro_batch = iterator.next_batch()
            last_micro_batch = micro_batch
            grouped_candidates = generate_candidates(
                model,
                tokenizer,
                micro_batch,
                model_cfg=model_cfg,
                rl_cfg=rl_cfg,
                decoder_cfg=decoder_cfg,
                device=device,
            )
            grouped_candidates = select_ter_diverse_candidates(
                micro_batch,
                grouped_candidates,
                rl_cfg=rl_cfg,
                decoder_cfg=decoder_cfg,
            )
            grouped_candidates = add_reference_candidates(
                micro_batch,
                grouped_candidates,
                include_reference_candidate=include_reference_candidate,
                decoder_cfg=decoder_cfg,
            )
            classifier_target_probs = None
            if reward_has_classifier:
                classifier_target_probs = compute_classifier_target_probs(
                    model,
                    tokenizer,
                    micro_batch,
                    grouped_candidates,
                    reward_cfg=reward_cfg,
                    decoder_cfg=decoder_cfg,
                    model_cfg=model_cfg,
                    device=device,
                )
            rewards, reward_metrics = compute_rewards(
                micro_batch,
                grouped_candidates,
                reward_cfg=reward_cfg,
                include_reference_candidate=include_reference_candidate,
                decoder_cfg=decoder_cfg,
                classifier_target_probs=classifier_target_probs,
            )
            group_sizes = [len(candidates) for candidates in grouped_candidates]
            advantages = compute_group_advantages(
                rewards,
                group_size=group_sizes,
                eps=float(rl_cfg.get("advantage_eps", 1e-6)),
            )
            if should_debug_candidates and debug_snapshot is None:
                debug_snapshot = build_candidate_debug_samples(
                    micro_batch,
                    grouped_candidates,
                    rewards,
                    advantages,
                    text_char_limit=debug_text_char_limit,
                    include_reference_candidate=include_reference_candidate,
                    decoder_cfg=decoder_cfg,
                    include_first_token_reward=reward_tracks_first_token,
                    max_examples=max(debug_candidate_examples, 1),
                )

            prompts_expanded: list[str] = []
            generated_flat: list[str] = []
            for ex, candidates in zip(micro_batch, grouped_candidates):
                prompt = encoder_input_text(ex.direction, ex.source_text, decoder_cfg)
                prompts_expanded.extend([prompt] * len(candidates))
                generated_flat.extend(candidates)

            enc_inputs, labels = tokenize_generated_targets(
                tokenizer,
                prompts_expanded,
                generated_flat,
                max_source_length=max_source_length,
                max_completion_length=max_completion_length,
                device=device,
            )

            with autocast_context(train_cfg):
                seq_log_probs = sequence_logprobs(model, enc_inputs, labels)
                adv_tensor = torch.tensor(advantages, dtype=seq_log_probs.dtype, device=device)
                denom = float(max_completion_length)
                rl_loss = -((adv_tensor * seq_log_probs) / denom).mean()
                if sft_weight != 0.0:
                    sft_loss = compute_sft_loss(
                        model,
                        tokenizer,
                        micro_batch,
                        max_source_length=max_source_length,
                        max_target_length=max_target_length,
                        device=device,
                        decoder_cfg=decoder_cfg,
                    )
                else:
                    sft_loss = torch.zeros((), dtype=seq_log_probs.dtype, device=device)
                loss = ((rl_weight * rl_loss) + (sft_weight * sft_loss)) / grad_accum

            try:
                ensure_finite_tensor("seq_log_probs", seq_log_probs, step=next_step, batch=micro_batch)
                ensure_finite_tensor("rl_loss", rl_loss, step=next_step, batch=micro_batch)
                ensure_finite_tensor("sft_loss", sft_loss, step=next_step, batch=micro_batch)
                ensure_finite_tensor("loss", loss, step=next_step, batch=micro_batch)
            except RuntimeError as exc:
                nonfinite_skips_total += 1
                append_nonfinite_debug_log(
                    nonfinite_debug_log_path,
                    error=str(exc),
                    step=next_step,
                    micro_batch=micro_batch,
                    grouped_candidates=grouped_candidates,
                    rewards=rewards,
                    advantages=advantages,
                    include_reference_candidate=include_reference_candidate,
                    decoder_cfg=decoder_cfg,
                    include_first_token_reward=reward_tracks_first_token,
                )
                print(
                    json.dumps(
                        {
                            "nonfinite_skip_step": next_step,
                            "nonfinite_skips_total": nonfinite_skips_total,
                            "record_ids": [ex.record_id for ex in micro_batch],
                            "error": str(exc),
                            "nonfinite_debug_path": nonfinite_debug_log_path.as_posix(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if nonfinite_skips_total > max_nonfinite_skips:
                    raise
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            applied_micro_batches += 1

            running_loss += float(loss.item() * grad_accum)
            running_rl_loss += float(rl_loss.item())
            running_sft_loss += float(sft_loss.item())
            running_reward += reward_metrics["reward_mean"]
            running_model_bleu += reward_metrics["model_bleu_mean"]
            running_copy_bleu += reward_metrics["copy_bleu_mean"]
            running_model_ter += reward_metrics["model_ter_mean"]
            running_copy_ter += reward_metrics["copy_ter_mean"]
            running_exact_copy += reward_metrics["exact_copy_rate"]
            running_distinct += reward_metrics["distinct_candidate_rate"]
            running_bleu_reward_component += float(reward_metrics["bleu_reward_component_mean"] or 0.0)
            running_ter_reward_component += float(reward_metrics["ter_reward_component_mean"] or 0.0)
            if reward_has_first_token:
                running_first_token_reward_component += float(
                    reward_metrics["first_token_reward_component_mean"] or 0.0
                )
            if reward_tracks_first_token:
                running_first_token_parse += float(reward_metrics["first_token_parse_rate"] or 0.0)
                running_first_token_accuracy += float(reward_metrics["first_token_accuracy"] or 0.0)
            if reward_hard_first_token_gate:
                running_first_token_gate_pass += float(reward_metrics["first_token_gate_pass_rate"] or 0.0)
            if reward_has_classifier:
                running_classifier_reward_component += float(
                    reward_metrics["classifier_reward_component_mean"] or 0.0
                )
            if reward_has_token_conservatism:
                running_token_conservatism_reward_component += float(
                    reward_metrics["token_conservatism_reward_component_mean"] or 0.0
                )
            running_micro_batches += 1
            for ex in micro_batch:
                examples_seen["total"] += 1
                examples_seen[f"dataset:{ex.dataset}"] += 1
                examples_seen[f"direction:{ex.direction}"] += 1
                examples_seen[f"bucket:{ex.bucket}"] += 1
                examples_seen[f"stage_bucket:{ex.stage_bucket}"] += 1

        if applied_micro_batches == 0:
            optimizer.zero_grad(set_to_none=True)
            print(
                json.dumps(
                    {
                        "skipped_step": next_step,
                        "reason": "all_micro_batches_nonfinite",
                        "nonfinite_skips_total": nonfinite_skips_total,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        try:
            ensure_finite_gradients(
                model,
                step=next_step,
                batch=last_micro_batch or [],
            )
        except RuntimeError as exc:
            nonfinite_skips_total += 1
            append_nonfinite_debug_log(
                nonfinite_debug_log_path,
                error=str(exc),
                step=next_step,
                micro_batch=last_micro_batch or [],
                grouped_candidates=None,
                rewards=None,
                advantages=None,
                include_reference_candidate=include_reference_candidate,
                decoder_cfg=decoder_cfg,
                include_first_token_reward=reward_tracks_first_token,
            )
            optimizer.zero_grad(set_to_none=True)
            print(
                json.dumps(
                    {
                        "nonfinite_skip_step": next_step,
                        "reason": "nonfinite_gradients",
                        "nonfinite_skips_total": nonfinite_skips_total,
                        "error": str(exc),
                        "nonfinite_debug_path": nonfinite_debug_log_path.as_posix(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if nonfinite_skips_total > max_nonfinite_skips:
                raise
            continue
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(train_cfg.get("max_grad_norm", 1.0)),
        )
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if should_debug_candidates and debug_snapshot is not None:
            payload = {
                "candidate_debug_step": global_step,
                "include_reference_candidate": include_reference_candidate,
                "samples": debug_snapshot,
            }
            with candidate_debug_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "candidate_debug_step": global_step,
                        "candidate_debug_examples": len(debug_snapshot),
                        "candidate_debug_path": candidate_debug_log_path.as_posix(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if global_step % logging_steps == 0 or global_step == 1:
            metrics = {
                "step": global_step,
                "loss": running_loss / max(running_micro_batches, 1),
                "rl_loss": running_rl_loss / max(running_micro_batches, 1),
                "sft_loss": running_sft_loss / max(running_micro_batches, 1),
                "reward_mean": running_reward / max(running_micro_batches, 1),
                "model_bleu_mean": running_model_bleu / max(running_micro_batches, 1),
                "copy_bleu_mean": running_copy_bleu / max(running_micro_batches, 1),
                "model_ter_mean": running_model_ter / max(running_micro_batches, 1),
                "copy_ter_mean": running_copy_ter / max(running_micro_batches, 1),
                "exact_copy_rate": running_exact_copy / max(running_micro_batches, 1),
                "distinct_candidate_rate": running_distinct / max(running_micro_batches, 1),
                "bleu_reward_component_mean": running_bleu_reward_component / max(running_micro_batches, 1),
                "ter_reward_component_mean": running_ter_reward_component / max(running_micro_batches, 1),
                "lr": scheduler.get_last_lr()[0],
            }
            if reward_has_first_token:
                metrics["first_token_reward_component_mean"] = (
                    running_first_token_reward_component / max(running_micro_batches, 1)
                )
            if reward_tracks_first_token:
                metrics["first_token_parse_rate"] = running_first_token_parse / max(running_micro_batches, 1)
                metrics["first_token_accuracy"] = running_first_token_accuracy / max(running_micro_batches, 1)
            if reward_hard_first_token_gate:
                metrics["first_token_gate_pass_rate"] = (
                    running_first_token_gate_pass / max(running_micro_batches, 1)
                )
            if reward_has_classifier:
                metrics["classifier_reward_component_mean"] = (
                    running_classifier_reward_component / max(running_micro_batches, 1)
                )
            if reward_has_token_conservatism:
                metrics["token_conservatism_reward_component_mean"] = (
                    running_token_conservatism_reward_component / max(running_micro_batches, 1)
                )
            metrics["examples_seen_total"] = int(examples_seen["total"])
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            with metrics_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            write_stage_c_example_counts(example_counts_path, global_step, examples_seen)
            print(
                "STAGE_C_EXAMPLE_COUNTS "
                + json.dumps(stage_c_example_counts_payload(global_step, examples_seen), ensure_ascii=False),
                flush=True,
            )
            running_loss = 0.0
            running_rl_loss = 0.0
            running_sft_loss = 0.0
            running_reward = 0.0
            running_model_bleu = 0.0
            running_copy_bleu = 0.0
            running_model_ter = 0.0
            running_copy_ter = 0.0
            running_exact_copy = 0.0
            running_distinct = 0.0
            running_bleu_reward_component = 0.0
            running_ter_reward_component = 0.0
            running_first_token_reward_component = 0.0
            running_first_token_parse = 0.0
            running_first_token_accuracy = 0.0
            running_first_token_gate_pass = 0.0
            running_classifier_reward_component = 0.0
            running_token_conservatism_reward_component = 0.0
            running_micro_batches = 0

        if global_step % save_steps == 0 or global_step == max_steps:
            save_checkpoint(
                model,
                tokenizer,
                out_dir,
                global_step,
                metrics_log_path,
                save_total_limit=save_total_limit,
            )

    model.save_pretrained(out_dir.as_posix())
    tokenizer.save_pretrained(out_dir.as_posix())
    write_stage_c_example_counts(example_counts_path, global_step, examples_seen)
    print(f"Saved final Stage C model to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
