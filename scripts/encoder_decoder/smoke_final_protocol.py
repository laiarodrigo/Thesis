#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RENDERER = (
    REPO_ROOT
    / "scripts"
    / "encoder_decoder"
    / "single_task_models"
    / "t5gemma2_4b"
    / "build_final_supervised_jsonl.py"
)
TRAINER = REPO_ROOT / "scripts" / "encoder_decoder" / "train_encdec_lora.py"
EVALUATOR = REPO_ROOT / "scripts" / "encoder_decoder" / "eval" / "evaluate_encdec.py"
SOURCE_TOKENS = ("<pt-br>", "<pt-pt>")
CONTROL_TOKENS = ("<cls>", "<pt-br>", "<pt-pt>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-check the final Portuguese-variety supervised protocol. "
            "By default this only validates rendered examples and token handling; "
            "--run-train-eval adds a tiny LoRA train/evaluate cycle."
        )
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/thesis_final_protocol_smoke"),
    )
    parser.add_argument(
        "--model-id",
        default="google/t5gemma-2-270m-270m",
        help="Model used for tokenizer and optional train/eval smoke checks.",
    )
    parser.add_argument("--run-train-eval", action="store_true")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 in the optional train/eval smoke. Keep disabled on CPUs.",
    )
    parser.add_argument(
        "--allow-cpu-train",
        action="store_true",
        help="Allow --run-train-eval without CUDA. This can be very slow.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Do not clear the work directory before writing smoke artifacts.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_command(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def make_raw_data(raw_dir: Path) -> dict[str, Path]:
    translation_rows = [
        {
            "id": "br2pt_lexical",
            "dataset": "smoke",
            "bucket": "lexical",
            "direction": "translate_br2pt",
            "input_text": "Você está no ônibus?",
            "target_text": "Tu estás no autocarro?",
            "is_equal_pair": False,
        },
        {
            "id": "pt2br_lexical",
            "dataset": "smoke",
            "bucket": "lexical",
            "direction": "translate_pt2br",
            "input_text": "Tu estás no autocarro?",
            "target_text": "Você está no ônibus?",
            "is_equal_pair": False,
        },
        {
            "id": "equal_br2pt",
            "dataset": "smoke",
            "bucket": "random",
            "direction": "translate_br2pt",
            "input_text": "A reunião começa amanhã.",
            "target_text": "A reunião começa amanhã.",
            "is_equal_pair": True,
        },
        {
            "id": "equal_pt2br",
            "dataset": "smoke",
            "bucket": "random",
            "direction": "translate_pt2br",
            "input_text": "A reunião começa amanhã.",
            "target_text": "A reunião começa amanhã.",
            "is_equal_pair": True,
        },
    ]
    classification_rows = [
        {
            "id": "cls_br",
            "dataset": "smoke",
            "bucket": "lexical",
            "input_text": "Você está no ônibus?",
            "target_text": "pt-br",
        },
        {
            "id": "cls_pt",
            "dataset": "smoke",
            "bucket": "lexical",
            "input_text": "Tu estás no autocarro?",
            "target_text": "pt-pt",
        },
        {
            "id": "cls_equal",
            "dataset": "smoke",
            "bucket": "random",
            "input_text": "A reunião começa amanhã.",
            "target_text": "equal",
            "is_equal_pair": True,
        },
    ]
    paths = {
        "translation_train": raw_dir / "translation_train.jsonl",
        "translation_valid": raw_dir / "translation_valid.jsonl",
        "classification_train": raw_dir / "classification_train.jsonl",
        "classification_valid": raw_dir / "classification_valid.jsonl",
    }
    write_jsonl(paths["translation_train"], translation_rows)
    write_jsonl(paths["translation_valid"], translation_rows)
    write_jsonl(paths["classification_train"], classification_rows)
    write_jsonl(paths["classification_valid"], classification_rows)
    return paths


def render_mode(mode: str, raw_paths: dict[str, Path], out_dir: Path) -> None:
    cmd = [
        sys.executable,
        RENDERER.as_posix(),
        "--mode",
        mode,
        "--translation-train",
        raw_paths["translation_train"].as_posix(),
        "--translation-valid",
        raw_paths["translation_valid"].as_posix(),
        "--classification-train",
        raw_paths["classification_train"].as_posix(),
        "--classification-valid",
        raw_paths["classification_valid"].as_posix(),
        "--out-dir",
        out_dir.as_posix(),
        "--valid-min-rows",
        "0",
    ]
    run_command(cmd)


def validate_rendered_dataset(mode: str, out_dir: Path) -> None:
    rows = read_jsonl(out_dir / "train.jsonl")
    counts = Counter(row["task"] for row in rows)
    if mode == "translation_only":
        require(counts == {"translation": 4}, f"{mode}: unexpected task counts {counts}")
    else:
        require(
            counts == {"translation": 4, "classification": 4},
            f"{mode}: unexpected task counts {counts}",
        )

    translation_rows = [row for row in rows if row["task"] == "translation"]
    classification_rows = [row for row in rows if row["task"] == "classification"]
    require(
        all(row.get("loss_on_first_token_only") is False for row in translation_rows),
        f"{mode}: translation rows must use full-sequence loss",
    )
    require(
        all("loss_mask_prefix_tokens" not in row for row in translation_rows),
        f"{mode}: final protocol must not mask the decoder source token",
    )
    if mode != "translation_only":
        require(
            all(row.get("loss_on_first_token_only") is True for row in classification_rows),
            f"{mode}: classification rows must use first-token loss",
        )
        require(
            Counter(row["target_text"] for row in classification_rows)
            == {"<pt-br>": 2, "<pt-pt>": 2},
            f"{mode}: equal classification rows were not duplicated into both classes",
        )

    if mode in {"translation_only", "encoder_unified"}:
        require(
            all(row["input_text"].startswith(SOURCE_TOKENS) for row in translation_rows),
            f"{mode}: translation inputs must start with source-language tokens",
        )
        require(
            all(not row["target_text"].startswith(SOURCE_TOKENS) for row in translation_rows),
            f"{mode}: encoder-token translation targets should not be prefixed",
        )
    if mode == "encoder_unified":
        require(
            all(row["input_text"].startswith("<cls> ") for row in classification_rows),
            "encoder_unified: classification inputs must start with <cls>",
        )
    if mode == "decoder_unified":
        require(
            all(not row["input_text"].startswith(CONTROL_TOKENS) for row in translation_rows),
            "decoder_unified: translation inputs should be plain source text",
        )
        require(
            all(row["target_text"].startswith(SOURCE_TOKENS) for row in translation_rows),
            "decoder_unified: translation targets must start with the source-language label",
        )

    print(f"Template check passed: mode={mode} counts={dict(counts)}")


def validate_balanced_sampler(rendered_dir: Path) -> None:
    sys.path.insert(0, (REPO_ROOT / "scripts" / "encoder_decoder").as_posix())
    from task_balanced_sampler import FixedTaskMixSampler

    rows = read_jsonl(rendered_dir / "train.jsonl")
    sampler = FixedTaskMixSampler(
        [row["task"] for row in rows],
        window_size=2,
        translation_rows_per_window=1,
        classification_rows_per_window=1,
        seed=123,
    )
    sampled = list(sampler)
    require(len(sampled) == 8, f"balanced sampler length mismatch: {len(sampled)}")
    for start in range(0, len(sampled), 2):
        window_tasks = Counter(rows[idx]["task"] for idx in sampled[start : start + 2])
        require(
            window_tasks == {"translation": 1, "classification": 1},
            f"balanced sampler window mismatch: {window_tasks}",
        )
    print("Balanced sampler check passed: one translation and one classification row per window")


def validate_tokenizer(model_id: str, work_dir: Path, *, required: bool) -> bool:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        if required:
            raise RuntimeError("transformers is required for --run-train-eval") from exc
        print(f"Tokenizer check skipped: transformers is unavailable ({exc})")
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as exc:
        if required:
            raise RuntimeError(f"Could not load tokenizer for {model_id!r}") from exc
        print(f"Tokenizer check skipped: could not load {model_id!r} ({exc})")
        return False

    before = {
        token: tokenizer.encode(token, add_special_tokens=False)
        for token in CONTROL_TOKENS
    }
    added = int(tokenizer.add_tokens(list(CONTROL_TOKENS), special_tokens=False))
    after: dict[str, list[int]] = {}
    token_ids: dict[str, int] = {}
    for token in CONTROL_TOKENS:
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        encoded = tokenizer.encode(token, add_special_tokens=False)
        require(encoded == [token_id], f"token is not atomic after add_tokens: {token} -> {encoded}")
        after[token] = encoded
        token_ids[token] = token_id

    save_dir = work_dir / "tokenizer_check"
    tokenizer.save_pretrained(save_dir.as_posix())
    reloaded = AutoTokenizer.from_pretrained(save_dir.as_posix(), use_fast=True)
    for token in CONTROL_TOKENS:
        token_id = int(reloaded.convert_tokens_to_ids(token))
        encoded = reloaded.encode(token, add_special_tokens=False)
        decoded = reloaded.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        require(encoded == [token_id], f"persisted token is not atomic: {token} -> {encoded}")
        require(decoded == token, f"persisted token does not decode cleanly: {token} -> {decoded}")

    print(
        "Tokenizer check passed:"
        f" newly_added={added}"
        f" before={before}"
        f" after={after}"
        f" ids={token_ids}"
    )
    return True


def write_eval_splits(rendered_dir: Path) -> dict[str, Path]:
    rows = read_jsonl(rendered_dir / "valid.jsonl")
    translation_rows = [row for row in rows if row["task"] == "translation"]
    classification_rows = [row for row in rows if row["task"] == "classification"]
    paths = {
        "translation": rendered_dir / "translation_eval.jsonl",
        "classification": rendered_dir / "classification_eval.jsonl",
    }
    write_jsonl(paths["translation"], translation_rows)
    write_jsonl(paths["classification"], classification_rows)
    return paths


def smoke_config(
    *,
    model_id: str,
    rendered_dir: Path,
    output_dir: Path,
    bf16: bool,
) -> dict[str, Any]:
    return {
        "seed": 123,
        "model": {
            "base_model": model_id,
            "trust_remote_code": True,
            "max_source_length": 64,
            "max_target_length": 64,
            "control_tokens": list(CONTROL_TOKENS),
            "restricted_classification_tokens": list(SOURCE_TOKENS),
        },
        "dataset": {
            "train_path": (rendered_dir / "train.jsonl").as_posix(),
            "valid_path": (rendered_dir / "valid.jsonl").as_posix(),
        },
        "lora": {
            "enabled": True,
            "r": 4,
            "alpha": 8,
            "dropout": 0.0,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "train_control_token_embeddings": True,
        },
        "training": {
            "output_dir": output_dir.as_posix(),
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "max_steps": 2,
            "logging_steps": 1,
            "eval_strategy": "steps",
            "eval_steps": 1,
            "save_steps": 1,
            "save_total_limit": 1,
            "early_stopping_patience": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "do_eval": True,
            "bf16": bool(bf16),
            "fp16": False,
            "require_bf16": bool(bf16),
            "require_early_stopping": True,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "save_safetensors": True,
            "final_evaluate": True,
        },
        "task_batching": {
            "enabled": True,
            "task_column": "task",
            "translation_fraction": 0.5,
            "seed": 123,
        },
    }


def run_train_eval_for_mode(
    *,
    mode: str,
    model_id: str,
    rendered_dir: Path,
    work_dir: Path,
    bf16: bool,
) -> None:
    output_dir = work_dir / f"train_output_{mode}"
    config_path = work_dir / f"train_{mode}.json"
    config = smoke_config(
        model_id=model_id,
        rendered_dir=rendered_dir,
        output_dir=output_dir,
        bf16=bf16,
    )
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    run_command([sys.executable, TRAINER.as_posix(), "--config", config_path.as_posix()])

    eval_paths = write_eval_splits(rendered_dir)
    eval_out = work_dir / f"eval_{mode}"
    run_command(
        [
            sys.executable,
            EVALUATOR.as_posix(),
            "--task",
            "classification",
            "--dataset-path",
            eval_paths["classification"].as_posix(),
            "--model-id",
            model_id,
            "--adapter-dir",
            output_dir.as_posix(),
            "--tokenizer-path",
            output_dir.as_posix(),
            "--classification-mode",
            "score-first-token",
            "--classification-candidates",
            "<pt-br>",
            "<pt-pt>",
            "--batch-size",
            "1",
            "--output-dir",
            eval_out.as_posix(),
        ]
    )
    run_command(
        [
            sys.executable,
            EVALUATOR.as_posix(),
            "--task",
            "translation",
            "--dataset-path",
            eval_paths["translation"].as_posix(),
            "--model-id",
            model_id,
            "--adapter-dir",
            output_dir.as_posix(),
            "--tokenizer-path",
            output_dir.as_posix(),
            "--batch-size",
            "1",
            "--max-new-tokens",
            "32",
            "--num-beams",
            "1",
            "--no-repeat-ngram-size",
            "0",
            "--repetition-penalty",
            "1.0",
            "--output-dir",
            eval_out.as_posix(),
        ]
    )
    print(f"Tiny train/eval smoke passed: mode={mode} output={output_dir}")


def ensure_train_runtime(allow_cpu_train: bool, *, bf16: bool) -> None:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for --run-train-eval") from exc
    if not torch.cuda.is_available() and not allow_cpu_train:
        raise RuntimeError(
            "--run-train-eval needs CUDA by default. Use --allow-cpu-train only "
            "if you intentionally want a slow CPU smoke run."
        )
    if bf16 and torch.cuda.is_available():
        is_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
        if not bool(is_supported()):
            name = torch.cuda.get_device_name(0)
            raise RuntimeError(
                "--bf16 was requested, but the allocated GPU does not support bf16: "
                f"{name}. Request a bf16-capable GPU, or omit --bf16 for a logic-only "
                "smoke run."
            )


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    if work_dir.exists() and not args.keep_work_dir:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = make_raw_data(work_dir / "raw")
    rendered = {
        "translation_only": work_dir / "rendered_translation_only",
        "encoder_unified": work_dir / "rendered_encoder_unified",
        "decoder_unified": work_dir / "rendered_decoder_unified",
    }
    for mode, rendered_dir in rendered.items():
        render_mode(mode, raw_paths, rendered_dir)
        validate_rendered_dataset(mode, rendered_dir)

    validate_balanced_sampler(rendered["encoder_unified"])
    tokenizer_ready = validate_tokenizer(
        args.model_id,
        work_dir,
        required=bool(args.run_train_eval),
    )
    if args.run_train_eval:
        require(tokenizer_ready, "tokenizer check is required before train/eval smoke")
        ensure_train_runtime(args.allow_cpu_train, bf16=bool(args.bf16))
        for mode in ("encoder_unified", "decoder_unified"):
            run_train_eval_for_mode(
                mode=mode,
                model_id=args.model_id,
                rendered_dir=rendered[mode],
                work_dir=work_dir,
                bf16=bool(args.bf16),
            )

    print(f"Final protocol smoke completed. work_dir={work_dir}")


if __name__ == "__main__":
    main()
