#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from sacrebleu import corpus_bleu, sentence_bleu

try:
    from metrics_utils import corpus_ter, sentence_ter
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.metrics_utils import corpus_ter, sentence_ter


ENCODER_TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline diagnostics for existing translation prediction JSONLs. "
            "Computes BLEU/TER, reward proxies, and pairwise comparisons against a baseline."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec in the form name=/path/to/predictions.jsonl. Repeatable.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline run name for pairwise comparisons.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where aggregate and pairwise reports will be written.",
    )
    parser.add_argument(
        "--bleu-weight",
        type=float,
        default=0.5,
        help="Weight for BLEU proxy in the mixed reward.",
    )
    parser.add_argument(
        "--ter-weight",
        "--wer-weight",
        dest="ter_weight",
        type=float,
        default=0.5,
        help="Weight for TER proxy in the mixed reward.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-8,
        help="Small constant used in reward proxy formulas.",
    )
    parser.add_argument(
        "--top-suspects",
        type=int,
        default=50,
        help="Number of suspicious pairwise examples to emit per comparison.",
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("Provide at least one --run name=path entry.")
    return args


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ").replace("\r", " ")).strip()


def strip_encoder_task_prefix(text: str) -> str:
    raw = text or ""
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_text(raw)


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run spec {spec!r}; expected name=/path/to/file.jsonl")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name:
        raise ValueError(f"Invalid --run spec {spec!r}; empty name")
    return name, path


def reward_metrics(
    *,
    source_text: str,
    gold_text: str,
    pred_text: str,
    eps: float,
    bleu_weight: float,
    ter_weight: float,
) -> dict[str, float]:
    model_bleu = sentence_bleu(pred_text, [gold_text]).score
    copy_bleu = sentence_bleu(source_text, [gold_text]).score
    model_ter = sentence_ter(pred_text, gold_text)
    copy_ter = sentence_ter(source_text, gold_text)

    bleu_reward = model_bleu / (model_bleu + copy_bleu + eps)
    ter_reward = (copy_ter + eps) / (model_ter + copy_ter + (2.0 * eps))
    mix_reward = (bleu_weight * bleu_reward) + (ter_weight * ter_reward)

    return {
        "sentence_bleu": float(model_bleu),
        "sentence_copy_bleu": float(copy_bleu),
        "sentence_ter": float(model_ter),
        "sentence_copy_ter": float(copy_ter),
        "bleu_reward": float(bleu_reward),
        "ter_reward": float(ter_reward),
        "mix_reward": float(mix_reward),
    }


def load_run(
    *,
    name: str,
    path: Path,
    eps: float,
    bleu_weight: float,
    ter_weight: float,
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    hyps: list[str] = []
    refs: list[str] = []
    copy_hyps: list[str] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ex_id = str(row.get("id", line_no))
            source_text = strip_encoder_task_prefix(str(row.get("input_text", "")))
            gold_text = normalize_text(str(row["gold"]))
            pred_text = normalize_text(str(row.get("pred_raw", "")))
            metrics = reward_metrics(
                source_text=source_text,
                gold_text=gold_text,
                pred_text=pred_text,
                eps=eps,
                bleu_weight=bleu_weight,
                ter_weight=ter_weight,
            )
            by_id[ex_id] = {
                "id": ex_id,
                "source_text": source_text,
                "gold_text": gold_text,
                "pred_text": pred_text,
                "is_exact_input_copy": pred_text == source_text,
                **metrics,
            }
            hyps.append(pred_text)
            refs.append(gold_text)
            copy_hyps.append(source_text)

    if not by_id:
        raise ValueError(f"Prediction file is empty: {path}")

    rows = list(by_id.values())
    summary = {
        "run": name,
        "path": path.as_posix(),
        "n": len(rows),
        "bleu": float(corpus_bleu(hyps, [refs]).score),
        "copy_baseline_bleu": float(corpus_bleu(copy_hyps, [refs]).score),
        "ter": float(corpus_ter(hyps, refs)),
        "copy_baseline_ter": float(corpus_ter(copy_hyps, refs)),
        "mean_sentence_bleu": float(sum(x["sentence_bleu"] for x in rows) / len(rows)),
        "mean_sentence_ter": float(sum(x["sentence_ter"] for x in rows) / len(rows)),
        "mean_bleu_reward": float(sum(x["bleu_reward"] for x in rows) / len(rows)),
        "mean_ter_reward": float(sum(x["ter_reward"] for x in rows) / len(rows)),
        "mean_mix_reward": float(sum(x["mix_reward"] for x in rows) / len(rows)),
        "exact_input_copy_rate": float(sum(1.0 for x in rows if x["is_exact_input_copy"]) / len(rows)),
    }
    return {"name": name, "path": path, "by_id": by_id, "summary": summary}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("_") or "run"


def compare_runs(
    baseline_run: dict[str, Any],
    candidate_run: dict[str, Any],
    *,
    top_suspects: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_by_id = baseline_run["by_id"]
    candidate_by_id = candidate_run["by_id"]
    common_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    if not common_ids:
        raise ValueError(
            f"No shared example ids between {baseline_run['name']!r} and {candidate_run['name']!r}"
        )

    delta_sentence_bleu = 0.0
    delta_sentence_ter = 0.0
    delta_bleu_reward = 0.0
    delta_ter_reward = 0.0
    delta_mix_reward = 0.0
    candidate_better_bleu = 0
    candidate_better_ter = 0
    mix_reward_up_bleu_down = 0
    mix_reward_up_ter_up = 0
    bleu_reward_up_bleu_down = 0
    ter_reward_up_ter_up = 0
    copy_delta_total = 0
    suspects: list[dict[str, Any]] = []

    for ex_id in common_ids:
        base = baseline_by_id[ex_id]
        cand = candidate_by_id[ex_id]
        d_bleu = cand["sentence_bleu"] - base["sentence_bleu"]
        d_ter = cand["sentence_ter"] - base["sentence_ter"]
        d_bleu_reward = cand["bleu_reward"] - base["bleu_reward"]
        d_ter_reward = cand["ter_reward"] - base["ter_reward"]
        d_mix_reward = cand["mix_reward"] - base["mix_reward"]

        delta_sentence_bleu += d_bleu
        delta_sentence_ter += d_ter
        delta_bleu_reward += d_bleu_reward
        delta_ter_reward += d_ter_reward
        delta_mix_reward += d_mix_reward
        candidate_better_bleu += int(d_bleu > 0.0)
        candidate_better_ter += int(d_ter < 0.0)
        mix_reward_up_bleu_down += int(d_mix_reward > 0.0 and d_bleu < 0.0)
        mix_reward_up_ter_up += int(d_mix_reward > 0.0 and d_ter > 0.0)
        bleu_reward_up_bleu_down += int(d_bleu_reward > 0.0 and d_bleu < 0.0)
        ter_reward_up_ter_up += int(d_ter_reward > 0.0 and d_ter > 0.0)
        copy_delta_total += int(cand["is_exact_input_copy"]) - int(base["is_exact_input_copy"])

        suspect_score = max(0.0, d_mix_reward) + max(0.0, -d_bleu / 100.0) + max(0.0, d_ter)
        if d_mix_reward > 0.0 and (d_bleu < 0.0 or d_ter > 0.0):
            suspects.append(
                {
                    "id": ex_id,
                    "baseline_run": baseline_run["name"],
                    "candidate_run": candidate_run["name"],
                    "suspect_score": round(float(suspect_score), 6),
                    "delta_mix_reward": round(float(d_mix_reward), 6),
                    "delta_bleu_reward": round(float(d_bleu_reward), 6),
                    "delta_ter_reward": round(float(d_ter_reward), 6),
                    "delta_sentence_bleu": round(float(d_bleu), 6),
                    "delta_sentence_ter": round(float(d_ter), 6),
                    "source_text": base["source_text"],
                    "gold_text": base["gold_text"],
                    "baseline_pred": base["pred_text"],
                    "candidate_pred": cand["pred_text"],
                    "baseline_mix_reward": round(float(base["mix_reward"]), 6),
                    "candidate_mix_reward": round(float(cand["mix_reward"]), 6),
                    "baseline_sentence_bleu": round(float(base["sentence_bleu"]), 6),
                    "candidate_sentence_bleu": round(float(cand["sentence_bleu"]), 6),
                    "baseline_sentence_ter": round(float(base["sentence_ter"]), 6),
                    "candidate_sentence_ter": round(float(cand["sentence_ter"]), 6),
                }
            )

    n = len(common_ids)
    suspects.sort(key=lambda row: row["suspect_score"], reverse=True)
    summary = {
        "baseline": baseline_run["name"],
        "candidate": candidate_run["name"],
        "n_common": n,
        "mean_delta_sentence_bleu": float(delta_sentence_bleu / n),
        "mean_delta_sentence_ter": float(delta_sentence_ter / n),
        "mean_delta_bleu_reward": float(delta_bleu_reward / n),
        "mean_delta_ter_reward": float(delta_ter_reward / n),
        "mean_delta_mix_reward": float(delta_mix_reward / n),
        "candidate_better_sentence_bleu_rate": float(candidate_better_bleu / n),
        "candidate_better_sentence_ter_rate": float(candidate_better_ter / n),
        "mix_reward_up_bleu_down_rate": float(mix_reward_up_bleu_down / n),
        "mix_reward_up_ter_worse_rate": float(mix_reward_up_ter_up / n),
        "bleu_reward_up_bleu_down_rate": float(bleu_reward_up_bleu_down / n),
        "ter_reward_up_ter_worse_rate": float(ter_reward_up_ter_up / n),
        "delta_exact_input_copy_rate": float(copy_delta_total / n),
        "suspect_examples": min(top_suspects, len(suspects)),
    }
    return summary, suspects[:top_suspects]


def main() -> None:
    args = parse_args()
    total_weight = args.bleu_weight + args.ter_weight
    if total_weight <= 0.0:
        raise SystemExit("bleu-weight + ter-weight must be > 0")
    bleu_weight = args.bleu_weight / total_weight
    ter_weight = args.ter_weight / total_weight

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for spec in args.run:
        name, path = parse_run_spec(spec)
        if not path.exists():
            raise SystemExit(f"Missing predictions file for run {name!r}: {path}")
        run = load_run(
            name=name,
            path=path,
            eps=args.epsilon,
            bleu_weight=bleu_weight,
            ter_weight=ter_weight,
        )
        runs.append(run)

    aggregate_rows = [run["summary"] for run in runs]
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.baseline:
        baseline_run = next((run for run in runs if run["name"] == args.baseline), None)
        if baseline_run is None:
            raise SystemExit(f"Baseline run {args.baseline!r} was not provided in --run")

        pairwise_rows: list[dict[str, Any]] = []
        for run in runs:
            if run["name"] == args.baseline:
                continue
            summary, suspects = compare_runs(
                baseline_run,
                run,
                top_suspects=max(args.top_suspects, 0),
            )
            pairwise_rows.append(summary)
            suspects_path = output_dir / f"suspects_vs_{safe_slug(run['name'])}.jsonl"
            with suspects_path.open("w", encoding="utf-8") as fh:
                for row in suspects:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        write_csv(output_dir / "pairwise_vs_baseline.csv", pairwise_rows)
        (output_dir / "pairwise_vs_baseline.json").write_text(
            json.dumps(pairwise_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    manifest = {
        "output_dir": output_dir.as_posix(),
        "runs": [run["summary"]["run"] for run in runs],
        "baseline": args.baseline,
        "bleu_weight": bleu_weight,
        "ter_weight": ter_weight,
        "epsilon": args.epsilon,
        "top_suspects": args.top_suspects,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote diagnostics to {output_dir}")
    for row in aggregate_rows:
        print(
            f"  run={row['run']} n={row['n']} bleu={row['bleu']:.4f}"
            f" ter={row['ter']:.6f} mix_reward={row['mean_mix_reward']:.6f}"
        )


if __name__ == "__main__":
    main()
