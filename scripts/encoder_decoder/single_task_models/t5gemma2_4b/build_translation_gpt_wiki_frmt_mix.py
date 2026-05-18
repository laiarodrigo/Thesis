#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from frmt_stageb_filter import (
    FrmtFilterConfig,
    evaluate_frmt_translation_row,
    normalize_space,
    strip_task_prefix,
)


TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)
FILTER_REPORT_FIELDS = [
    "record_id",
    "source_path",
    "line_no",
    "bucket",
    "direction",
    "decision",
    "decision_reason",
    "keep_reason",
    "passes_changed_spans",
    "passes_edit_ratio",
    "passes_structural_overlap",
    "passes_paraphrase_score",
    "passes_non_marker_changed",
    "passes_non_marker_over_marker_gap",
    "non_marker_over_marker_gap",
    "src_words",
    "tgt_words",
    "changed_spans",
    "changed_word_tokens",
    "marker_changed_tokens",
    "non_marker_changed_tokens",
    "edit_ratio",
    "structural_overlap",
    "paraphrase_score",
    "marker_preview",
    "changed_spans_preview",
    "source_text",
    "target_text",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description=(
            "Build a Stage B Wikipedia+FRMT mix for translation and classification, "
            "applying the conservative Stage C row filter to FRMT train/valid rows."
        )
    )
    parser.add_argument(
        "--frmt-train",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "frmt_only" / "translation_train.jsonl",
    )
    parser.add_argument(
        "--frmt-valid",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "frmt_only" / "translation_valid.jsonl",
    )
    parser.add_argument(
        "--wiki-train",
        type=Path,
        default=repo_root
        / "data"
        / "encoder_decoder"
        / "t5gemma2"
        / "compare_staged_v2"
        / "stageB_gpt_wiki"
        / "translation_train.jsonl",
    )
    parser.add_argument(
        "--wiki-valid",
        type=Path,
        default=repo_root
        / "data"
        / "encoder_decoder"
        / "t5gemma2"
        / "compare_staged_v2"
        / "stageB_gpt_wiki"
        / "translation_valid.jsonl",
    )
    parser.add_argument(
        "--ptbrvarid-train",
        type=Path,
        default=None,
        help="Optional PtBrVId translation train JSONL built from translated canonical pairs.",
    )
    parser.add_argument(
        "--ptbrvarid-valid",
        type=Path,
        default=None,
        help="Optional PtBrVId translation valid JSONL built from translated canonical pairs.",
    )
    parser.add_argument(
        "--frmt-cls-train",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "frmt_only" / "classification_train.jsonl",
    )
    parser.add_argument(
        "--frmt-cls-valid",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "frmt_only" / "classification_valid.jsonl",
    )
    parser.add_argument(
        "--wiki-cls-train",
        type=Path,
        default=repo_root
        / "data"
        / "encoder_decoder"
        / "t5gemma2"
        / "compare_staged_v2"
        / "stageB_gpt_wiki"
        / "classification_train.jsonl",
    )
    parser.add_argument(
        "--wiki-cls-valid",
        type=Path,
        default=repo_root
        / "data"
        / "encoder_decoder"
        / "t5gemma2"
        / "compare_staged_v2"
        / "stageB_gpt_wiki"
        / "classification_valid.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageB_gpt_wiki_frmt_mix",
    )
    parser.add_argument("--max-changed-spans", type=int, default=4)
    parser.add_argument("--max-edit-ratio", type=float, default=0.30)
    parser.add_argument("--min-structural-overlap", type=float, default=0.72)
    parser.add_argument("--max-paraphrase-score", type=float, default=0.18)
    parser.add_argument("--max-non-marker-changed", type=int, default=6)
    parser.add_argument("--max-non-marker-over-marker-gap", type=int, default=2)
    parser.add_argument(
        "--frmt-filter-report-prefix",
        default="frmt_translation_filter",
        help="Prefix used for the FRMT train/valid filter CSV files.",
    )
    return parser.parse_args()


def extract_task_prefix(text: str) -> str | None:
    match = TASK_PREFIX_RE.match(text or "")
    if not match:
        return None
    return match.group(1).strip().lower()


def infer_direction(row: dict) -> str:
    for key in ("task", "direction"):
        value = normalize_space(str(row.get(key) or "")).casefold()
        if value in {"translate_br2pt", "br2pt"}:
            return "translate_br2pt"
        if value in {"translate_pt2br", "pt2br"}:
            return "translate_pt2br"
    prefix = extract_task_prefix(str(row.get("input_text") or ""))
    if prefix == "br-pt":
        return "translate_br2pt"
    if prefix == "pt-br":
        return "translate_pt2br"
    return "translation"


def infer_task(row: dict, *, default_task: str) -> str:
    value = normalize_space(str(row.get("task") or row.get("direction") or "")).casefold()
    if value in {"translate_br2pt", "br2pt"}:
        return "translate_br2pt"
    if value in {"translate_pt2br", "pt2br"}:
        return "translate_pt2br"
    if value == "classification":
        return "classification"
    return default_task


def normalize_dataset(raw: object, *, fallback: str) -> str:
    text = normalize_space(str(raw or "")).casefold()
    if text == "frmt":
        return "FRMT"
    if text in {"ptbrvid", "ptbrvarid", "liaad/ptbrvid"}:
        return "PtBrVId"
    if text in {"gpt", "wikipedia", "wiki", "gpt_wiki", "gpt-wiki"}:
        return "GPT"
    if text in {"unknown", ""}:
        return fallback
    return str(raw)


def normalize_row(row: dict, *, fallback_dataset: str, default_task: str) -> dict:
    out = dict(row)
    out["dataset"] = normalize_dataset(row.get("dataset"), fallback=fallback_dataset)
    out["direction"] = infer_direction(row)
    out["task"] = infer_task(row, default_task=default_task)
    return out


def build_filter_config(args: argparse.Namespace) -> FrmtFilterConfig:
    return FrmtFilterConfig(
        max_changed_spans=int(args.max_changed_spans),
        max_edit_ratio=float(args.max_edit_ratio),
        min_structural_overlap=float(args.min_structural_overlap),
        max_paraphrase_score=float(args.max_paraphrase_score),
        max_non_marker_changed=int(args.max_non_marker_changed),
        max_non_marker_over_marker_gap=int(args.max_non_marker_over_marker_gap),
    )


def write_filter_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FILTER_REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FILTER_REPORT_FIELDS})


def summarize_filter_rows(rows: list[dict]) -> dict:
    decisions = Counter(row["decision"] for row in rows)
    reasons = Counter(row["decision_reason"] for row in rows)
    kept_by_bucket = Counter(row["bucket"] for row in rows if row["decision"] == "keep")
    dropped_by_bucket = Counter(row["bucket"] for row in rows if row["decision"] != "keep")
    kept_by_direction = Counter(row["direction"] for row in rows if row["decision"] == "keep")
    dropped_by_direction = Counter(row["direction"] for row in rows if row["decision"] != "keep")
    return {
        "rows": len(rows),
        "kept": decisions.get("keep", 0),
        "dropped": decisions.get("drop", 0),
        "by_decision_reason": dict(sorted(reasons.items())),
        "kept_by_bucket": dict(sorted(kept_by_bucket.items())),
        "dropped_by_bucket": dict(sorted(dropped_by_bucket.items())),
        "kept_by_direction": dict(sorted(kept_by_direction.items())),
        "dropped_by_direction": dict(sorted(dropped_by_direction.items())),
    }


def write_mix(
    src_paths: list[tuple[Path, str]],
    out_path: Path,
    *,
    default_task: str,
    frmt_filter_config: FrmtFilterConfig | None = None,
    frmt_kept_texts: set[str] | None = None,
    frmt_filter_rows: list[dict] | None = None,
) -> tuple[dict, set[str]]:
    counts = Counter()
    kept_texts: set[str] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_fh:
        for path, fallback_dataset in src_paths:
            with path.open("r", encoding="utf-8") as in_fh:
                for line_no, line in enumerate(in_fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    normalized = normalize_row(
                        row,
                        fallback_dataset=fallback_dataset,
                        default_task=default_task,
                    )

                    if default_task == "translation" and fallback_dataset == "FRMT":
                        if frmt_filter_config is None:
                            raise ValueError("FRMT translation filtering requires a filter config.")
                        filter_row = evaluate_frmt_translation_row(
                            normalized,
                            source_path=path.as_posix(),
                            line_no=line_no,
                            config=frmt_filter_config,
                        )
                        if frmt_filter_rows is not None:
                            frmt_filter_rows.append(filter_row)
                        counts[f"frmt_filter:{filter_row['decision_reason']}"] += 1
                        if filter_row["decision"] != "keep":
                            counts["rows_filtered_out"] += 1
                            continue
                        kept_texts.add(filter_row["source_text"])
                        kept_texts.add(filter_row["target_text"])

                    if (
                        default_task == "classification"
                        and fallback_dataset == "FRMT"
                        and frmt_kept_texts is not None
                    ):
                        cls_text = normalize_space(strip_task_prefix(str(normalized.get("input_text") or "")))
                        if cls_text not in frmt_kept_texts:
                            counts["rows_filtered_out"] += 1
                            counts["frmt_classification:missing_kept_text_match"] += 1
                            continue

                    out_fh.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    counts["rows"] += 1
                    counts[f"dataset:{normalized['dataset']}"] += 1
                    counts[f"task:{normalized['task']}"] += 1
                    counts[f"direction:{normalized['direction']}"] += 1
            counts[f"source:{path.as_posix()}"] += 1
    return dict(counts), kept_texts


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    filter_config = build_filter_config(args)

    translation_train_sources = [(args.frmt_train, "FRMT"), (args.wiki_train, "GPT")]
    translation_valid_sources = [(args.frmt_valid, "FRMT"), (args.wiki_valid, "GPT")]
    if args.ptbrvarid_train is not None:
        translation_train_sources.append((args.ptbrvarid_train, "PtBrVId"))
    if args.ptbrvarid_valid is not None:
        translation_valid_sources.append((args.ptbrvarid_valid, "PtBrVId"))

    frmt_train_filter_rows: list[dict] = []
    frmt_valid_filter_rows: list[dict] = []

    translation_train_counts, frmt_train_kept_texts = write_mix(
        translation_train_sources,
        args.out_dir / "translation_train.jsonl",
        default_task="translation",
        frmt_filter_config=filter_config,
        frmt_filter_rows=frmt_train_filter_rows,
    )
    translation_valid_counts, frmt_valid_kept_texts = write_mix(
        translation_valid_sources,
        args.out_dir / "translation_valid.jsonl",
        default_task="translation",
        frmt_filter_config=filter_config,
        frmt_filter_rows=frmt_valid_filter_rows,
    )
    classification_train_counts, _ = write_mix(
        [(args.frmt_cls_train, "FRMT"), (args.wiki_cls_train, "GPT")],
        args.out_dir / "classification_train.jsonl",
        default_task="classification",
        frmt_kept_texts=frmt_train_kept_texts,
    )
    classification_valid_counts, _ = write_mix(
        [(args.frmt_cls_valid, "FRMT"), (args.wiki_cls_valid, "GPT")],
        args.out_dir / "classification_valid.jsonl",
        default_task="classification",
        frmt_kept_texts=frmt_valid_kept_texts,
    )

    train_filter_csv = args.out_dir / f"{args.frmt_filter_report_prefix}_train.csv"
    valid_filter_csv = args.out_dir / f"{args.frmt_filter_report_prefix}_valid.csv"
    write_filter_csv(train_filter_csv, frmt_train_filter_rows)
    write_filter_csv(valid_filter_csv, frmt_valid_filter_rows)

    report = {
        "translation_train": translation_train_counts,
        "translation_valid": translation_valid_counts,
        "classification_train": classification_train_counts,
        "classification_valid": classification_valid_counts,
        "frmt_filter": {
            "config": {
                "max_changed_spans": filter_config.max_changed_spans,
                "max_edit_ratio": filter_config.max_edit_ratio,
                "min_structural_overlap": filter_config.min_structural_overlap,
                "max_paraphrase_score": filter_config.max_paraphrase_score,
                "max_non_marker_changed": filter_config.max_non_marker_changed,
                "max_non_marker_over_marker_gap": filter_config.max_non_marker_over_marker_gap,
            },
            "train": summarize_filter_rows(frmt_train_filter_rows),
            "valid": summarize_filter_rows(frmt_valid_filter_rows),
            "report_files": {
                "train_csv": train_filter_csv.as_posix(),
                "valid_csv": valid_filter_csv.as_posix(),
            },
        },
        "out_dir": args.out_dir.as_posix(),
    }
    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote: {args.out_dir / 'translation_train.jsonl'}")
    print(f"Wrote: {args.out_dir / 'translation_valid.jsonl'}")
    print(f"Wrote: {args.out_dir / 'classification_train.jsonl'}")
    print(f"Wrote: {args.out_dir / 'classification_valid.jsonl'}")
    print(f"Wrote: {train_filter_csv}")
    print(f"Wrote: {valid_filter_csv}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
