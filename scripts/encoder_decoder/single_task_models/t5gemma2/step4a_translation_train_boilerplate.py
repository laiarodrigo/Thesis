#!/usr/bin/env python3
"""
Step 4A (boilerplate): translation training launcher for T5Gemma2.

This file is intentionally lightweight: it validates config and then optionally
calls the existing generic trainer. Fill TODOs incrementally.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Boilerplate launcher for translation training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "encoder_decoder" / "t5gemma2" / "translation_lora_boilerplate.yaml",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually execute the training command (otherwise dry-run).",
    )
    return parser.parse_args()


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_cfg(cfg: dict[str, Any]) -> None:
    required = ["model", "dataset", "training"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"Config missing keys: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    validate_cfg(cfg)
    lora_cfg = cfg.get("lora", {})
    use_lora = bool(lora_cfg.get("enabled", True))

    trainer_script = Path("scripts/encoder_decoder/train_encdec_lora.py")
    if not trainer_script.exists():
        raise SystemExit(f"Missing trainer script: {trainer_script}")

    cmd = ["python3", trainer_script.as_posix(), "--config", args.config.as_posix()]

    print("Step 4A boilerplate")
    print(f"  config: {args.config}")
    print(f"  model: {cfg['model']['base_model']}")
    print(f"  train data: {cfg['dataset']['train_path']}")
    print(f"  valid data: {cfg['dataset']['valid_path']}")
    print(f"  output dir: {cfg['training']['output_dir']}")
    print(f"  mode: {'lora' if use_lora else 'full-finetune'}")
    print(f"  command: {' '.join(cmd)}")

    if not args.execute:
        print("\nDry-run mode. Add --execute when ready.")
        print("TODOs before execute:")
        print("1) Confirm gated model access and HF auth.")
        if use_lora:
            print("2) Confirm LoRA target_modules for this checkpoint.")
            print("3) Start with a short debug run (small max_steps).")
        else:
            print("2) Start with a short debug run (small max_steps).")
            print("3) Monitor memory and reduce batch size if needed.")
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
