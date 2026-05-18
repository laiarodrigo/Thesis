# scripts/encoder_decoder/multitask/build_multitask_jsonl.py
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator
from task_protocol import (
    build_translation_input_from_encoder_prefix,
    class_label_to_decoder_payload,
    normalize_class_label,
    strip_encoder_prefix,
)

def convert_translation(in_path: Path) -> Iterator[dict]:
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            prefix, clean_input = strip_encoder_prefix(ex["input_text"])
            if prefix == "id":
                continue
            yield {
                "input_text": build_translation_input_from_encoder_prefix(prefix or "", clean_input),
                "target_text": str(ex["target_text"] or "").strip(),
                "task": "translation",
                "dataset": ex.get("dataset"),
                "bucket": ex.get("bucket"),
            }

def read_classification_rows(in_path: Path) -> list[dict]:
    rows: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            _, clean_input = strip_encoder_prefix(ex["input_text"])
            raw_label = ex.get("target_text", ex.get("label", ex.get("gold")))
            if raw_label is None:
                raise KeyError("classification row missing one of: target_text, label, gold")
            label = normalize_class_label(raw_label)
            rows.append(
                {
                "input_text": clean_input,
                "target_text": class_label_to_decoder_payload(label),
                "task": "classification",
                "label": label,
                "dataset": ex.get("dataset"),
                "bucket": ex.get("bucket"),
                }
            )
    return rows

def iter_classification_rows(in_path: Path) -> Iterator[dict]:
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            _, clean_input = strip_encoder_prefix(ex["input_text"])
            raw_label = ex.get("target_text", ex.get("label", ex.get("gold")))
            if raw_label is None:
                raise KeyError("classification row missing one of: target_text, label, gold")
            label = normalize_class_label(raw_label)
            yield {
                "input_text": clean_input,
                "target_text": class_label_to_decoder_payload(label),
                "task": "classification",
                "label": label,
                "dataset": ex.get("dataset"),
                "bucket": ex.get("bucket"),
            }

def normalize_dataset_name(value: object) -> str:
    text = (str(value).strip().lower() if value is not None else "")
    return text or "unknown"

def parse_ratio(value: str) -> float:
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    return ratio

def parse_ratio_overrides(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    raw = (spec or "").strip()
    if not raw:
        return out
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid --equal-max-ratio-by-dataset item {part!r}. Expected dataset=ratio."
            )
        key, value = part.split("=", 1)
        ds_key = normalize_dataset_name(key)
        out[ds_key] = parse_ratio(value.strip())
    return out

def max_equal_count(non_equal_count: int, ratio: float) -> int:
    if ratio <= 0.0:
        return 0
    if ratio >= 1.0:
        return 10**18
    return int((ratio * non_equal_count) / (1.0 - ratio))

def filter_equal_rows(
    rows: list[dict],
    *,
    default_ratio: float,
    ratio_by_dataset: dict[str, float],
    seed: int,
) -> tuple[list[dict], dict[str, dict[str, int | float]]]:
    rng = random.Random(seed)
    by_dataset_idx: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_dataset_idx[normalize_dataset_name(row.get("dataset"))].append(idx)

    keep = [True] * len(rows)
    stats: dict[str, dict[str, int | float]] = {}
    for dataset, indices in sorted(by_dataset_idx.items()):
        ratio = ratio_by_dataset.get(dataset, default_ratio)
        equal_idx = [i for i in indices if rows[i]["label"] == "equal"]
        non_equal_count = len(indices) - len(equal_idx)
        equal_before = len(equal_idx)

        allowed_equal = min(equal_before, max_equal_count(non_equal_count, ratio))
        if allowed_equal < equal_before:
            keep_equal = set(rng.sample(equal_idx, k=allowed_equal))
            for i in equal_idx:
                if i not in keep_equal:
                    keep[i] = False

        equal_after = sum(1 for i in indices if keep[i] and rows[i]["label"] == "equal")
        total_after = sum(1 for i in indices if keep[i])
        stats[dataset] = {
            "ratio_target": ratio,
            "rows_before": len(indices),
            "rows_after": total_after,
            "equal_before": equal_before,
            "equal_after": equal_after,
            "non_equal_before": non_equal_count,
        }

    filtered = [row for i, row in enumerate(rows) if keep[i]]
    return filtered, stats

def write_jsonl(rows: Iterable[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as w:
        for r in rows:
            if r.get("task") == "classification":
                # `label` is useful for diagnostics but not required by training.
                r = dict(r)
                r.pop("label", None)
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def stream_build(
    *,
    translation_in: Path,
    classification_in: Path,
    out_path: Path,
    progress_every: int = 500_000,
) -> tuple[int, int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    translation_n = 0
    classification_n = 0
    with out_path.open("w", encoding="utf-8") as w:
        for r in convert_translation(translation_in):
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += 1
            translation_n += 1
            if progress_every and total % progress_every == 0:
                print(f"[multitask-build] wrote={total} translation={translation_n} classification={classification_n}")

        for r in iter_classification_rows(classification_in):
            r = dict(r)
            r.pop("label", None)
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += 1
            classification_n += 1
            if progress_every and total % progress_every == 0:
                print(f"[multitask-build] wrote={total} translation={translation_n} classification={classification_n}")

    return total, translation_n, classification_n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translation-in", type=Path, required=True)
    ap.add_argument("--classification-in", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--equal-max-ratio",
        type=float,
        default=1.0,
        help=(
            "Maximum share of `equal` rows among classification rows per dataset in [0,1]. "
            "Use 1.0 to disable filtering."
        ),
    )
    ap.add_argument(
        "--equal-max-ratio-by-dataset",
        default="",
        help=(
            "Optional per-dataset overrides, e.g. 'gpt=0.30,frmt=0.25,opensubs=0.10'. "
            "Dataset keys are matched case-insensitively."
        ),
    )
    ap.add_argument("--equal-seed", type=int, default=42)
    ap.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        help="Emit a progress line every N written rows in streaming mode. Use 0 to disable.",
    )
    args = ap.parse_args()

    default_ratio = parse_ratio(str(args.equal_max_ratio))
    ratio_overrides = parse_ratio_overrides(args.equal_max_ratio_by_dataset)

    if default_ratio >= 1.0 and not ratio_overrides:
        total, translation_n, classification_n = stream_build(
            translation_in=args.translation_in,
            classification_in=args.classification_in,
            out_path=args.out,
            progress_every=int(args.progress_every),
        )
        print(f"Wrote {total} rows -> {args.out}")
        print(
            f"Breakdown: translation={translation_n} classification={classification_n} "
            f"(equal_max_ratio={default_ratio}, equal_seed={args.equal_seed})"
        )
        return

    translation_rows = list(convert_translation(args.translation_in))
    classification_rows = read_classification_rows(args.classification_in)
    classification_rows, filter_stats = filter_equal_rows(
        classification_rows,
        default_ratio=default_ratio,
        ratio_by_dataset=ratio_overrides,
        seed=int(args.equal_seed),
    )

    total = write_jsonl([*translation_rows, *classification_rows], args.out)
    print(f"Wrote {total} rows -> {args.out}")
    print(
        f"Breakdown: translation={len(translation_rows)} classification={len(classification_rows)} "
        f"(equal_max_ratio={default_ratio}, equal_seed={args.equal_seed})"
    )
    for dataset, st in sorted(filter_stats.items()):
        print(
            f"  dataset={dataset} ratio_target={st['ratio_target']:.3f} "
            f"rows_before={st['rows_before']} rows_after={st['rows_after']} "
            f"equal_before={st['equal_before']} equal_after={st['equal_after']} "
            f"non_equal={st['non_equal_before']}"
        )

if __name__ == "__main__":
    main()
