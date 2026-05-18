#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Stage C candidate-debug payloads from either "
            "stage_c_candidate_debug.jsonl or mixed SLURM stdout."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Candidate-debug JSONL files or Slurm stdout files containing candidate_debug JSON lines.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional stage_c_metrics.jsonl path for run-level RL/diversity aggregates.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many low-/high-signal examples to include in the output.",
    )
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(frac: float) -> float:
        idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * frac))))
        return float(ordered[idx])

    return {
        "mean": float(sum(ordered) / len(ordered)),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "max": float(ordered[-1]),
        "zeros": int(sum(v == 0.0 for v in ordered)),
    }


def extract_payloads(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if "candidate_debug_step" not in obj or "samples" not in obj:
                continue
            if not isinstance(obj["samples"], list):
                continue
            payloads.append(obj)
    return payloads


def exact_model_unique_count(sample: dict[str, Any], candidates: list[dict[str, Any]]) -> int:
    full_unique = int(sample.get("unique_candidates", len({c.get("text", "") for c in candidates})))
    model_candidates = [c for c in candidates if not c.get("is_reference_candidate", False)]
    has_reference = len(model_candidates) != len(candidates)
    if not has_reference:
        return full_unique
    any_model_matches_gold = any(bool(c.get("matches_gold", False)) for c in model_candidates)
    if any_model_matches_gold:
        return full_unique
    return max(full_unique - 1, 0)


def summarize_payloads(payloads: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    include_reference_counter: Counter[bool] = Counter()
    per_stage_bucket: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    per_dataset: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for payload in payloads:
        step = int(payload["candidate_debug_step"])
        include_reference = bool(payload.get("include_reference_candidate", False))
        include_reference_counter[include_reference] += 1

        for sample in payload["samples"]:
            candidates = list(sample.get("candidates", []))
            if not candidates:
                continue
            model_candidates = [c for c in candidates if not c.get("is_reference_candidate", False)]
            full_rewards = [float(c.get("reward", 0.0)) for c in candidates]
            model_rewards = [float(c.get("reward", 0.0)) for c in model_candidates]
            full_advantages = [float(c.get("advantage", 0.0)) for c in candidates]
            model_advantages = [float(c.get("advantage", 0.0)) for c in model_candidates]

            full_range = max(full_rewards) - min(full_rewards)
            model_range = (max(model_rewards) - min(model_rewards)) if model_rewards else 0.0
            full_std = pstdev(full_rewards) if len(full_rewards) > 1 else 0.0
            model_std = pstdev(model_rewards) if len(model_rewards) > 1 else 0.0
            model_unique = exact_model_unique_count(sample, candidates)
            full_unique = int(sample.get("unique_candidates", len({c.get("text", "") for c in candidates})))
            ref_only_signal = include_reference and math.isclose(model_range, 0.0) and full_range > 0.0

            row = {
                "step": step,
                "record_id": str(sample.get("record_id", "")),
                "dataset": str(sample.get("dataset", "unknown")),
                "bucket": str(sample.get("bucket", "n/a")),
                "stage_bucket": str(sample.get("stage_bucket", "n/a")),
                "group_size": int(sample.get("group_size", len(candidates))),
                "full_unique": full_unique,
                "model_unique": model_unique,
                "full_range": full_range,
                "model_range": model_range,
                "full_std": full_std,
                "model_std": model_std,
                "full_adv_zero": all(math.isclose(v, 0.0) for v in full_advantages),
                "model_adv_zero": all(math.isclose(v, 0.0) for v in model_advantages),
                "all_match_gold": all(bool(c.get("matches_gold", False)) for c in candidates),
                "model_all_match_gold": all(bool(c.get("matches_gold", False)) for c in model_candidates),
                "model_all_match_source": all(bool(c.get("matches_source", False)) for c in model_candidates),
                "reference_candidate_added": bool(sample.get("reference_candidate_added", False)),
                "ref_only_signal": ref_only_signal,
                "source_text": str(sample.get("source_text", "")),
                "target_text": str(sample.get("target_text", "")),
                "candidates": [
                    {
                        "idx": int(c.get("idx", 0)),
                        "reward": float(c.get("reward", 0.0)),
                        "advantage": float(c.get("advantage", 0.0)),
                        "is_reference_candidate": bool(c.get("is_reference_candidate", False)),
                        "matches_gold": bool(c.get("matches_gold", False)),
                        "matches_source": bool(c.get("matches_source", False)),
                        "text": str(c.get("text", "")),
                    }
                    for c in candidates
                ],
            }
            sample_rows.append(row)
            per_stage_bucket[row["stage_bucket"]].append(row)
            per_dataset[row["dataset"]].append(row)

    def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        full_unique_counter = Counter(r["full_unique"] for r in rows)
        model_unique_counter = Counter(r["model_unique"] for r in rows)
        full_ranges = [float(r["full_range"]) for r in rows]
        model_ranges = [float(r["model_range"]) for r in rows]
        full_stds = [float(r["full_std"]) for r in rows]
        model_stds = [float(r["model_std"]) for r in rows]
        return {
            "samples": len(rows),
            "full_unique_distribution": {str(k): v for k, v in sorted(full_unique_counter.items())},
            "model_unique_distribution": {str(k): v for k, v in sorted(model_unique_counter.items())},
            "full_range": quantiles(full_ranges),
            "model_range": quantiles(model_ranges),
            "full_std": quantiles(full_stds),
            "model_std": quantiles(model_stds),
            "all_match_gold": int(sum(r["all_match_gold"] for r in rows)),
            "model_all_match_gold": int(sum(r["model_all_match_gold"] for r in rows)),
            "model_all_match_source": int(sum(r["model_all_match_source"] for r in rows)),
            "full_adv_zero": int(sum(r["full_adv_zero"] for r in rows)),
            "model_adv_zero": int(sum(r["model_adv_zero"] for r in rows)),
            "reference_only_signal": int(sum(r["ref_only_signal"] for r in rows)),
        }

    lowest_signal = sorted(
        sample_rows,
        key=lambda r: (
            r["model_range"],
            r["full_range"],
            r["model_unique"],
            r["step"],
            r["record_id"],
        ),
    )[: max(top_k, 0)]
    highest_signal = sorted(
        sample_rows,
        key=lambda r: (
            r["model_range"],
            r["full_range"],
            r["model_unique"],
            r["step"],
            r["record_id"],
        ),
        reverse=True,
    )[: max(top_k, 0)]
    ref_only_examples = [r for r in sample_rows if r["ref_only_signal"]][: max(top_k, 0)]

    def compact_example(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "step": row["step"],
            "record_id": row["record_id"],
            "dataset": row["dataset"],
            "bucket": row["bucket"],
            "stage_bucket": row["stage_bucket"],
            "full_unique": row["full_unique"],
            "model_unique": row["model_unique"],
            "full_range": round(float(row["full_range"]), 6),
            "model_range": round(float(row["model_range"]), 6),
            "full_std": round(float(row["full_std"]), 6),
            "model_std": round(float(row["model_std"]), 6),
            "all_match_gold": row["all_match_gold"],
            "model_all_match_gold": row["model_all_match_gold"],
            "model_all_match_source": row["model_all_match_source"],
            "reference_candidate_added": row["reference_candidate_added"],
            "source_text": row["source_text"],
            "target_text": row["target_text"],
            "candidates": row["candidates"],
        }

    return {
        "payloads": len(payloads),
        "samples": len(sample_rows),
        "include_reference_candidate_distribution": {
            str(k).lower(): v for k, v in sorted(include_reference_counter.items())
        },
        "overall": summarize_rows(sample_rows),
        "by_stage_bucket": {
            key: summarize_rows(value)
            for key, value in sorted(per_stage_bucket.items())
        },
        "by_dataset": {
            key: summarize_rows(value)
            for key, value in sorted(per_dataset.items())
        },
        "lowest_signal_examples": [compact_example(row) for row in lowest_signal],
        "reference_only_signal_examples": [compact_example(row) for row in ref_only_examples],
        "highest_signal_examples": [compact_example(row) for row in highest_signal],
    }


def summarize_metrics(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or not line.startswith("{"):
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or "rl_loss" not in obj:
                continue
            rows.append(obj)
    if not rows:
        return {"rows": 0}

    def maybe_mean(key: str) -> float | None:
        values = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    def maybe_min(key: str) -> float | None:
        values = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if not values:
            return None
        return float(min(values))

    def maybe_max(key: str) -> float | None:
        values = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if not values:
            return None
        return float(max(values))

    rl_losses = [abs(float(r["rl_loss"])) for r in rows]
    return {
        "rows": len(rows),
        "step_min": min(int(r["step"]) for r in rows),
        "step_max": max(int(r["step"]) for r in rows),
        "reward_mean_mean": maybe_mean("reward_mean"),
        "rl_loss_mean": maybe_mean("rl_loss"),
        "rl_loss_abs_mean": float(sum(rl_losses) / len(rl_losses)),
        "sft_loss_mean": maybe_mean("sft_loss"),
        "distinct_candidate_rate_mean": maybe_mean("distinct_candidate_rate"),
        "distinct_candidate_rate_min": maybe_min("distinct_candidate_rate"),
        "distinct_candidate_rate_max": maybe_max("distinct_candidate_rate"),
        "exact_copy_rate_mean": maybe_mean("exact_copy_rate"),
        "exact_copy_rate_min": maybe_min("exact_copy_rate"),
        "exact_copy_rate_max": maybe_max("exact_copy_rate"),
        "first_token_accuracy_mean": maybe_mean("first_token_accuracy"),
        "first_token_parse_rate_mean": maybe_mean("first_token_parse_rate"),
    }


def main() -> None:
    args = parse_args()
    payloads: list[dict[str, Any]] = []
    for path in args.paths:
        if not path.exists():
            raise SystemExit(f"Missing input file: {path}")
        payloads.extend(extract_payloads(path))
    if not payloads:
        raise SystemExit("No candidate_debug payloads found in the provided files.")

    result = {
        "inputs": [p.as_posix() for p in args.paths],
        "candidate_debug": summarize_payloads(payloads, top_k=args.top_k),
    }
    if args.metrics is not None:
        if not args.metrics.exists():
            raise SystemExit(f"Missing metrics file: {args.metrics}")
        result["metrics"] = summarize_metrics(args.metrics)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
