#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VOCAB_KEYS = {"vocab_size", "encoder_vocab_size", "decoder_vocab_size"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch saved config.json vocab fields to a single consistent value."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--target-vocab",
        type=int,
        default=None,
        help="Optional explicit vocab size. If omitted, uses max vocab key found in config.",
    )
    return parser.parse_args()


def collect_vocab_values(obj: Any, out: list[int]) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in VOCAB_KEYS:
                try:
                    out.append(int(val))
                except Exception:
                    pass
            collect_vocab_values(val, out)
        return
    if isinstance(obj, list):
        for item in obj:
            collect_vocab_values(item, out)


def patch_vocab_values(obj: Any, target: int) -> int:
    updates = 0
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in VOCAB_KEYS:
                if int(val) != target:
                    obj[key] = int(target)
                    updates += 1
            else:
                updates += patch_vocab_values(val, target)
        return updates
    if isinstance(obj, list):
        for item in obj:
            updates += patch_vocab_values(item, target)
    return updates


def main() -> None:
    args = parse_args()
    cfg_path = args.model_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config file: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    before_vals: list[int] = []
    collect_vocab_values(cfg, before_vals)
    if not before_vals and args.target_vocab is None:
        raise SystemExit(
            "No vocab keys found in config and --target-vocab was not provided."
        )

    target = int(args.target_vocab) if args.target_vocab is not None else max(before_vals)
    updates = patch_vocab_values(cfg, target)
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    after_vals: list[int] = []
    collect_vocab_values(cfg, after_vals)
    before_unique = sorted(set(before_vals))
    after_unique = sorted(set(after_vals))
    print(f"Patched: {cfg_path}")
    print(f"target_vocab={target} updates={updates}")
    print(f"before_unique={before_unique}")
    print(f"after_unique={after_unique}")


if __name__ == "__main__":
    main()
