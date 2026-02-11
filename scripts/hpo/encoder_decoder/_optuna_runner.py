#!/usr/bin/env python3
import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import optuna
import yaml


def parse_args(default_space_file: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna HPO for encoder-decoder training.")
    parser.add_argument("--space-file", type=Path, default=Path(default_space_file))
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--metric-key", type=str, default=None)
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


def suggest_value(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    p_type = spec["type"]
    if p_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if p_type == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if p_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    raise ValueError(f"Unsupported parameter type '{p_type}' for {name}")


def find_best_metric(output_dir: Path, metric_key: str) -> tuple[float, int | None]:
    state_files = sorted(output_dir.glob("**/trainer_state.json"))
    best_val = None
    best_step = None
    for state_path in state_files:
        try:
            with state_path.open("r", encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception:
            continue

        for entry in state.get("log_history", []):
            if metric_key in entry:
                cur = float(entry[metric_key])
                step = entry.get("step")
                if best_val is None or cur < best_val:
                    best_val = cur
                    best_step = int(step) if step is not None else None

    if best_val is None:
        raise RuntimeError(f"No '{metric_key}' found in trainer_state.json files under {output_dir}")
    return best_val, best_step


def run_optuna(default_space_file: str) -> None:
    args = parse_args(default_space_file)
    space_cfg = load_yaml(args.space_file)
    base_cfg = load_yaml(Path(space_cfg["base_config"]))

    study_cfg = space_cfg["study"]
    metric_key = args.metric_key or study_cfg.get("metric", "eval_loss")
    study_name = args.study_name or study_cfg["name"]
    storage = args.storage or study_cfg["storage"]
    direction = study_cfg.get("direction", "minimize")
    n_trials = args.n_trials or int(study_cfg["n_trials"])
    timeout = args.timeout

    output_root = Path(space_cfg["output_root"])
    generated_dir = Path(space_cfg["generated_config_dir"])
    runner_script = Path(space_cfg["runner_script"])
    fixed_overrides = space_cfg.get("fixed_overrides", {})
    search_space = space_cfg["search_space"]

    output_root.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=int(study_cfg.get("seed", 123)))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)

        for path, value in fixed_overrides.items():
            set_nested(cfg, path, value)

        sampled = {}
        for path, spec in search_space.items():
            sampled[path] = suggest_value(trial, path, spec)
            set_nested(cfg, path, sampled[path])

        trial_dir = output_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        set_nested(cfg, "training.output_dir", trial_dir.as_posix())

        config_path = generated_dir / f"trial_{trial.number:04d}.yaml"
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=False)

        trial.set_user_attr("config_path", config_path.as_posix())
        trial.set_user_attr("output_dir", trial_dir.as_posix())
        trial.set_user_attr("sampled", sampled)

        cmd = ["python", runner_script.as_posix(), "--config", config_path.as_posix()]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            trial.set_user_attr("status", "train_failed")
            return float("inf")

        try:
            best_metric, best_step = find_best_metric(trial_dir, metric_key=metric_key)
            trial.set_user_attr("status", "ok")
            trial.set_user_attr("best_step", best_step)
            return best_metric
        except Exception as exc:
            trial.set_user_attr("status", f"metric_missing: {exc}")
            return float("inf")

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print(f"Study: {study.study_name}")
    print(f"Trials: {len(study.trials)}")
    print(f"Best value ({metric_key}): {study.best_value}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    raise RuntimeError("Use run_optuna_translation.py or run_optuna_classification.py")
