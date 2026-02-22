#!/usr/bin/env python3
"""
Evaluate a saved classification-head checkpoint on a classification split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Evaluate classification head on test split.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "encoder_decoder" / "t5gemma2" / "classification_head_boilerplate.yaml",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "classification_test.jsonl",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory containing classifier_state_dict.pt and tokenizer files.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-length", type=int, default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=False,
        help="Do not contact Hugging Face Hub; use only local cache/files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "eval_results" / "encoder_decoder",
    )
    return parser.parse_args()


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_encoder_classifier_class() -> type:
    step4b_path = Path(__file__).resolve().parents[1] / "t5gemma2" / "step4b_classification_head_boilerplate.py"
    spec = importlib.util.spec_from_file_location("step4b_cls", step4b_path.as_posix())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {step4b_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EncoderClassifier


def to_id2label(meta_or_cfg: dict[str, Any]) -> dict[int, str]:
    raw = meta_or_cfg.get("id2label", {})
    id2label = {}
    for k, v in raw.items():
        try:
            id2label[int(k)] = str(v)
        except Exception:
            continue
    return id2label


def macro_f1(gold: list[int], pred: list[int], labels: list[int]) -> float:
    f1s = []
    g = np.asarray(gold, dtype=np.int64)
    p = np.asarray(pred, dtype=np.int64)
    for lab in labels:
        tp = int(((p == lab) & (g == lab)).sum())
        fp = int(((p == lab) & (g != lab)).sum())
        fn = int(((p != lab) & (g == lab)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


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
    train_cfg = cfg["training"]

    model_dir = args.model_dir or Path(train_cfg["output_dir"])
    state_path = model_dir / "classifier_state_dict.pt"
    meta_path = model_dir / "classifier_meta.json"
    if not state_path.exists():
        raise SystemExit(f"Missing classifier state dict: {state_path}")
    if not args.dataset_path.exists():
        raise SystemExit(f"Missing dataset path: {args.dataset_path}")

    print("Classification-head eval")
    print(f"  config: {args.config}")
    print(f"  model dir: {model_dir}")
    print(f"  dataset: {args.dataset_path}")

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    id2label = to_id2label(meta) or to_id2label(model_cfg)
    label2id = {v: k for k, v in id2label.items()}

    tokenizer_candidates: list[str] = []
    # Prefer tokenizer from the base model for consistency with training setup.
    tokenizer_candidates.append(str(model_cfg["base_model"]))
    if (model_dir / "tokenizer_config.json").exists():
        tokenizer_candidates.append(model_dir.as_posix())

    tok = None
    tok_errors = []
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    trust_options = [trust_remote_code]
    if trust_remote_code:
        trust_options.append(False)

    for cand in tokenizer_candidates:
        for trust_opt in trust_options:
            try:
                tok = AutoTokenizer.from_pretrained(
                    cand,
                    use_fast=True,
                    trust_remote_code=trust_opt,
                    local_files_only=args.local_files_only,
                )
                print(f"Tokenizer loaded from: {cand} (trust_remote_code={trust_opt})")
                break
            except Exception as e:
                tok_errors.append((cand, trust_opt, e))
        if tok is not None:
            break

    if tok is None:
        msgs = []
        for cand, trust_opt, err in tok_errors[-4:]:
            msgs.append(f"- {cand} (trust_remote_code={trust_opt}): {err}")
        raise RuntimeError(
            "Unable to load tokenizer from model-dir/base-model candidates.\n"
            + "\n".join(msgs)
        )

    EncoderClassifier = load_encoder_classifier_class()
    model = EncoderClassifier(
        model_cfg["base_model"],
        num_labels=int(model_cfg["num_labels"]),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        local_files_only=args.local_files_only,
        freeze_decoder=bool(model_cfg.get("freeze_decoder", True)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    state = torch.load(state_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as first_error:
        remapped_state = _remap_state_dict_keys_for_layout(state, model)
        load_result = model.load_state_dict(remapped_state, strict=False)
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        critical_missing = [k for k in missing if k.startswith("classifier.")]
        if critical_missing:
            raise RuntimeError(
                "State dict load failed; classifier head weights are missing after remap.\n"
                f"Original error: {first_error}\n"
                f"Critical missing keys: {critical_missing[:10]}"
            ) from first_error
        print(
            "Warning: non-strict state_dict load after key remap. "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    max_len = int(args.max_source_length or model_cfg.get("max_source_length", 512))
    ds = load_dataset("json", data_files={"test": args.dataset_path.as_posix()})["test"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = args.output_dir / f"{run_id}_classification_head_test_predictions.jsonl"
    summary_path = args.output_dir / f"{run_id}_classification_head_test_summary.json"

    all_gold: list[int] = []
    all_pred: list[int] = []
    cm: Counter[tuple[int, int]] = Counter()

    with pred_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(ds), args.batch_size):
            batch = ds[start : start + args.batch_size]
            texts = batch["input_text"]
            if "label_id" in batch:
                gold_ids = [int(x) for x in batch["label_id"]]
            else:
                gold_ids = [int(label2id[str(x)]) for x in batch["label"]]

            enc = tok(
                texts,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask")).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

            for text, gold_id, pred_id in zip(texts, gold_ids, pred_ids):
                all_gold.append(gold_id)
                all_pred.append(pred_id)
                cm[(gold_id, pred_id)] += 1

                rec = {
                    "input_text": text,
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                    "gold_label": id2label.get(gold_id, str(gold_id)),
                    "pred_label": id2label.get(pred_id, str(pred_id)),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    labels = sorted(set(all_gold) | set(all_pred))
    accuracy = float(np.mean(np.asarray(all_gold) == np.asarray(all_pred))) if all_gold else 0.0
    f1_macro = macro_f1(all_gold, all_pred, labels)
    confusion = {
        f"{id2label.get(g, g)}->{id2label.get(p, p)}": int(cm[(g, p)])
        for g in labels
        for p in labels
    }

    summary = {
        "n": len(all_gold),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "labels": {str(k): v for k, v in sorted(id2label.items())},
        "confusion_matrix": confusion,
        "dataset_path": args.dataset_path.as_posix(),
        "model_dir": model_dir.as_posix(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved predictions: {pred_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
