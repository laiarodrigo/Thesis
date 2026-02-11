#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Axolotl trial configs for decoder-only HPO.")
    parser.add_argument("--space-file", type=Path, default=Path("hpo/decoder_only/qwen3_search_space.yaml"))
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_nested(cfg: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def sample_value(rng: random.Random, spec: dict[str, Any]) -> Any:
    p_type = spec["type"]
    if p_type == "categorical":
        return rng.choice(spec["choices"])
    if p_type == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        return rng.randrange(low, high + 1, step)
    if p_type == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        if spec.get("log", False):
            # sample uniformly in log10 space
            import math

            return 10 ** rng.uniform(math.log10(low), math.log10(high))
        if "step" in spec:
            step = float(spec["step"])
            n = int(round((high - low) / step))
            return low + step * rng.randint(0, n)
        return rng.uniform(low, high)
    raise ValueError(f"Unsupported parameter type: {p_type}")


def main() -> None:
    args = parse_args()
    space_cfg = load_yaml(args.space_file)
    base_cfg = load_yaml(Path(space_cfg["base_config"]))

    exp_cfg = space_cfg["experiment"]
    n_trials = args.n_trials or int(exp_cfg["n_trials"])
    seed = args.seed if args.seed is not None else int(exp_cfg.get("seed", 123))
    rng = random.Random(seed)

    generated_dir = Path(space_cfg["generated_config_dir"])
    manifest_path = Path(space_cfg["manifest_path"])
    output_root = Path(space_cfg["output_root"])
    search_space = space_cfg["search_space"]
    fixed_overrides = space_cfg.get("fixed_overrides", {})

    generated_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for trial_idx in range(n_trials):
        cfg = json.loads(json.dumps(base_cfg))  # deep copy without extra imports

        for path, value in fixed_overrides.items():
            set_nested(cfg, path, value)

        sampled = {}
        for path, spec in search_space.items():
            value = sample_value(rng, spec)
            sampled[path] = value
            set_nested(cfg, path, value)

        trial_name = f"trial_{trial_idx:04d}"
        trial_output_dir = output_root / trial_name
        trial_prepared_path = Path("data/decoder_only/qwen3/prepared_hpo") / trial_name
        cfg["output_dir"] = trial_output_dir.as_posix()
        cfg["dataset_prepared_path"] = trial_prepared_path.as_posix()

        config_path = generated_dir / f"{trial_name}.yaml"
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=False)

        params_path = generated_dir / f"{trial_name}.params.json"
        with params_path.open("w", encoding="utf-8") as fh:
            json.dump(sampled, fh, ensure_ascii=False, indent=2)

        records.append(
            {
                "trial_id": trial_idx,
                "trial_name": trial_name,
                "config_path": config_path.as_posix(),
                "output_dir": trial_output_dir.as_posix(),
                "dataset_prepared_path": trial_prepared_path.as_posix(),
                "params_path": params_path.as_posix(),
            }
        )

    fieldnames = [
        "trial_id",
        "trial_name",
        "config_path",
        "output_dir",
        "dataset_prepared_path",
        "params_path",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} trial configs at {generated_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
