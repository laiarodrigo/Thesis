#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description=(
            "Keep Stage A translation data OpenSubs-only and augment Stage A "
            "classification with leftover PtBrVId rows that were not sampled for Stage B translation."
        )
    )
    parser.add_argument(
        "--opensubs-translation-train",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageA_opensubs_only" / "translation_train.jsonl",
    )
    parser.add_argument(
        "--opensubs-translation-valid",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageA_opensubs_only" / "translation_valid.jsonl",
    )
    parser.add_argument(
        "--opensubs-classification-train",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageA_opensubs_only" / "classification_train.jsonl",
    )
    parser.add_argument(
        "--opensubs-classification-valid",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageA_opensubs_only" / "classification_valid.jsonl",
    )
    parser.add_argument(
        "--ptbrvarid-db",
        type=Path,
        default=repo_root / "data" / "duckdb" / "subs_ptbr_filtered.duckdb",
    )
    parser.add_argument("--ptbrvarid-dataset", default="PtBrVId")
    parser.add_argument(
        "--ptbrvarid-sampled-csv",
        type=Path,
        required=True,
        help="CSV produced by translate_ptbrvarid_batches.py containing sampled source rows.",
    )
    parser.add_argument(
        "--ptbrvarid-splits",
        default="train,valid",
        help="Comma-separated PtBrVId splits to include for classification.",
    )
    parser.add_argument("--classification-token", default="<id>")
    parser.add_argument(
        "--ptbrvarid-exclude-domains",
        default="",
        help="Comma-separated PtBrVId domains to exclude, e.g. 'social_media,web'.",
    )
    parser.add_argument(
        "--classification-balance-mode",
        choices=("none", "upsample_ptbrvarid", "downsample_opensubs"),
        default="none",
        help=(
            "How to balance Stage A classification rows. "
            "'none' keeps the raw mix, 'upsample_ptbrvarid' duplicates leftover PtBrVId rows, "
            "and 'downsample_opensubs' subsamples OpenSubs rows."
        ),
    )
    parser.add_argument(
        "--train-ptbrvarid-target-share",
        type=float,
        default=0.0,
        help=(
            "If > 0, enforce this PtBrVId share in the final classification mix using the "
            "selected --classification-balance-mode."
        ),
    )
    parser.add_argument(
        "--valid-ptbrvarid-target-share",
        type=float,
        default=0.0,
        help=(
            "If > 0, enforce this PtBrVId share in the final classification mix using the "
            "selected --classification-balance-mode."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200000,
        help="Emit a progress log every N processed rows while streaming large inputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageA_opensubs_only",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def parse_excluded_domains(raw: str) -> set[str]:
    return {
        normalize_space(part).casefold()
        for part in str(raw or "").split(",")
        if normalize_space(part)
    }


def canonical_label(text: str) -> str:
    value = normalize_space(text).casefold()
    if value in {"pt-br", "pt_br", "ptbr"}:
        return "pt-br"
    if value in {"pt-pt", "pt_pt", "ptpt"}:
        return "pt-pt"
    raise ValueError(f"Unsupported label: {text}")


def maybe_resolve_db_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parents[4] / "data" / "duckdb" / "subs_filtered_final.duckdb"
    if path.name == "subs_ptbr_filtered.duckdb" and fallback.exists():
        return fallback
    return path


def same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except FileNotFoundError:
        return a.absolute() == b.absolute()


def temp_output_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def validate_share(name: str, value: float) -> float:
    if value < 0.0 or value >= 1.0:
        raise SystemExit(f"{name} must be in [0, 1).")
    return value


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def log_progress(message: str) -> None:
    print(message, flush=True)


def required_ptbrvarid_count(opensubs_count: int, target_share: float) -> int:
    if target_share <= 0.0:
        return 0
    return int((target_share * opensubs_count) / (1.0 - target_share) + 0.999999)


def expand_ptbrvarid_rows(
    rows: list[dict[str, Any]],
    *,
    opensubs_count: int,
    target_share: float,
    seed: int,
    split_name: str,
) -> list[dict[str, Any]]:
    if not rows or target_share <= 0.0:
        return list(rows)
    needed = required_ptbrvarid_count(opensubs_count, target_share)
    if needed <= len(rows):
        return list(rows)

    rng = random.Random(f"{seed}:{split_name}:ptbrvarid")
    expanded = list(rows)
    pool = list(rows)
    while len(expanded) < needed:
        rng.shuffle(pool)
        remaining = needed - len(expanded)
        expanded.extend(pool[:remaining])
    return expanded


def allowed_opensubs_count(ptbrvarid_count: int, target_share: float) -> int:
    if target_share <= 0.0 or ptbrvarid_count <= 0:
        return 0
    return int((ptbrvarid_count * (1.0 - target_share)) / target_share)


def reservoir_sample_jsonl(
    path: Path,
    *,
    sample_size: int,
    seed: int,
    split_name: str,
    progress_every: int,
) -> tuple[list[str], int]:
    if sample_size <= 0:
        total_rows = sum(1 for _ in iter_jsonl(path))
        return [], total_rows

    rng = random.Random(f"{seed}:{split_name}:opensubs")
    sample: list[str] = []
    total_rows = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            total_rows += 1
            if progress_every > 0 and total_rows % progress_every == 0:
                log_progress(
                    f"[opensubs:{split_name}] sampled_rows={total_rows} kept={len(sample)} target={sample_size}"
                )
            if len(sample) < sample_size:
                sample.append(stripped)
                continue
            idx = rng.randrange(total_rows)
            if idx < sample_size:
                sample[idx] = stripped
    return sample, total_rows


def copy_jsonl(src_path: Path, dst_path: Path) -> dict[str, Any]:
    tmp_path = temp_output_path(dst_path)
    counts = Counter()
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8") as in_fh, tmp_path.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["rows"] += 1
            counts[f"dataset:{normalize_space(str(row.get('dataset') or 'UNKNOWN'))}"] += 1
            counts[f"task:{normalize_space(str(row.get('task') or 'UNKNOWN'))}"] += 1
    tmp_path.replace(dst_path)
    return dict(counts)


def load_sampled_exclusions(path: Path) -> set[tuple[str, str]]:
    exclusions: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"label", "source_text"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Sampled CSV missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            label = canonical_label(str(row.get("label") or ""))
            source_text = normalize_space(str(row.get("source_text") or ""))
            if source_text:
                exclusions.add((label, source_text))
    return exclusions


def source_text_from_ptbrvarid_row(row: dict[str, Any]) -> str:
    label = canonical_label(str(row["label"]))
    if label == "pt-br":
        return normalize_space(str(row.get("text_pt_br") or ""))
    return normalize_space(str(row.get("text_pt_pt") or ""))


def load_leftover_ptbrvarid_rows(
    *,
    db_path: Path,
    dataset: str,
    splits: list[str],
    exclusions: set[tuple[str, str]],
    excluded_domains: set[str],
    progress_every: int,
) -> dict[str, list[dict[str, Any]]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency 'duckdb'. Use an environment with duckdb installed.") from exc

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {name for (name,) in con.execute("SHOW TABLES").fetchall()}
        if "ptbrvarid" not in tables:
            raise SystemExit(f"Table 'ptbrvarid' not found in {db_path}")
        cur = con.execute(
            """
            SELECT
              COALESCE(split, '') AS split,
              COALESCE(domain, '') AS domain,
              label,
              text_pt_br,
              text_pt_pt
            FROM ptbrvarid
            WHERE dataset = ?
            """,
            [dataset],
        )

        split_filter = {normalize_space(split).casefold() for split in splits}
        leftovers: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
        seen_keys: set[tuple[str, str]] = set()
        processed_rows = 0

        while True:
            rows = cur.fetchmany(10000)
            if not rows:
                break
            for split, domain, label, text_pt_br, text_pt_pt in rows:
                processed_rows += 1
                if progress_every > 0 and processed_rows % progress_every == 0:
                    log_progress(
                        f"[ptbrvarid] scanned_rows={processed_rows} "
                        f"train_leftovers={len(leftovers['train'])} "
                        f"valid_leftovers={len(leftovers['valid'])}"
                    )

                split_norm = normalize_space(str(split or "")).casefold()
                if split_filter and split_norm not in split_filter:
                    continue
                if split_norm not in leftovers:
                    continue
                domain_norm = normalize_space(str(domain or "")).casefold()
                if domain_norm in excluded_domains:
                    continue

                label_norm = canonical_label(str(label or ""))
                row = {
                    "split": split_norm,
                    "domain": normalize_space(str(domain or "")),
                    "label": label_norm,
                    "text_pt_br": normalize_space(str(text_pt_br or "")),
                    "text_pt_pt": normalize_space(str(text_pt_pt or "")),
                }
                source_text = source_text_from_ptbrvarid_row(row)
                if not source_text:
                    continue
                key = (label_norm, source_text)
                if key in exclusions or key in seen_keys:
                    continue
                seen_keys.add(key)
                leftovers[split_norm].append(row)
        return leftovers
    
    finally:
        con.close()


def write_classification_mix(
    *,
    opensubs_path: Path,
    ptbrvarid_rows: list[dict[str, Any]],
    out_path: Path,
    classification_token: str,
    balance_mode: str,
    ptbrvarid_target_share: float,
    seed: int,
    split_name: str,
    progress_every: int,
) -> dict[str, Any]:
    tmp_path = temp_output_path(out_path)
    counts = Counter()
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    opensubs_rows: list[str] | None = None
    opensubs_total_rows = 0
    if balance_mode == "downsample_opensubs" and ptbrvarid_target_share > 0.0 and ptbrvarid_rows:
        max_opensubs_rows = allowed_opensubs_count(len(ptbrvarid_rows), ptbrvarid_target_share)
        log_progress(
            f"[opensubs:{split_name}] downsampling_to={max_opensubs_rows} "
            f"for_ptbrvarid_rows={len(ptbrvarid_rows)} target_share={ptbrvarid_target_share}"
        )
        opensubs_rows, opensubs_total_rows = reservoir_sample_jsonl(
            opensubs_path,
            sample_size=max_opensubs_rows,
            seed=seed,
            split_name=split_name,
            progress_every=progress_every,
        )
    with tmp_path.open("w", encoding="utf-8") as out_fh:
        if opensubs_rows is not None:
            for raw_line in opensubs_rows:
                row = json.loads(raw_line)
                out_fh.write(raw_line + "\n")
                counts["rows"] += 1
                counts["opensubs_rows"] += 1
                counts[f"dataset:{normalize_space(str(row.get('dataset') or 'UNKNOWN'))}"] += 1
                counts[f"label:{normalize_space(str(row.get('target_text') or row.get('label') or 'UNKNOWN'))}"] += 1
        else:
            processed_rows = 0
            for row in iter_jsonl(opensubs_path):
                processed_rows += 1
                if progress_every > 0 and processed_rows % progress_every == 0:
                    log_progress(f"[opensubs:{split_name}] copied_rows={processed_rows}")
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["rows"] += 1
                counts["opensubs_rows"] += 1
                counts[f"dataset:{normalize_space(str(row.get('dataset') or 'UNKNOWN'))}"] += 1
                counts[f"label:{normalize_space(str(row.get('target_text') or row.get('label') or 'UNKNOWN'))}"] += 1

        if opensubs_rows is None:
            opensubs_total_rows = int(counts["opensubs_rows"])

        if balance_mode == "upsample_ptbrvarid":
            expanded_ptbrvarid_rows = expand_ptbrvarid_rows(
                ptbrvarid_rows,
                opensubs_count=int(counts["opensubs_rows"]),
                target_share=ptbrvarid_target_share,
                seed=seed,
                split_name=split_name,
            )
        else:
            expanded_ptbrvarid_rows = list(ptbrvarid_rows)
        counts["ptbrvarid_unique_rows"] = len(ptbrvarid_rows)
        counts["ptbrvarid_written_rows"] = len(expanded_ptbrvarid_rows)
        counts["ptbrvarid_target_share"] = ptbrvarid_target_share
        counts["classification_balance_mode"] = balance_mode
        counts["opensubs_total_rows"] = opensubs_total_rows

        for row in expanded_ptbrvarid_rows:
            source_text = source_text_from_ptbrvarid_row(row)
            example = {
                "input_text": f"{classification_token} {source_text}",
                "target_text": row["label"],
                "task": "classify",
                "dataset": "PtBrVId",
                "domain": row["domain"],
                "direction": "classification",
            }
            out_fh.write(json.dumps(example, ensure_ascii=False) + "\n")
            counts["rows"] += 1
            counts["ptbrvarid_rows"] += 1
            counts["dataset:PtBrVId"] += 1
            counts[f"label:{row['label']}"] += 1
            counts[f"domain:{normalize_space(row['domain'] or 'unknown')}"] += 1

    total_rows = int(counts["rows"])
    if total_rows:
        counts["actual_ptbrvarid_share"] = round(float(counts["ptbrvarid_rows"]) / total_rows, 6)

    tmp_path.replace(out_path)
    return dict(counts)


def main() -> None:
    args = parse_args()
    args.train_ptbrvarid_target_share = validate_share(
        "--train-ptbrvarid-target-share",
        float(args.train_ptbrvarid_target_share),
    )
    args.valid_ptbrvarid_target_share = validate_share(
        "--valid-ptbrvarid-target-share",
        float(args.valid_ptbrvarid_target_share),
    )
    args.ptbrvarid_db = maybe_resolve_db_path(args.ptbrvarid_db)
    if not args.ptbrvarid_db.exists():
        raise SystemExit(f"PtBrVId DB not found: {args.ptbrvarid_db}")

    split_values = [item.strip() for item in args.ptbrvarid_splits.split(",") if item.strip()]
    excluded_domains = parse_excluded_domains(args.ptbrvarid_exclude_domains)
    exclusions = load_sampled_exclusions(args.ptbrvarid_sampled_csv)
    log_progress(
        f"[ptbrvarid] loading leftovers from {args.ptbrvarid_db} "
        f"excluding_domains={sorted(excluded_domains)}"
    )
    leftover_rows = load_leftover_ptbrvarid_rows(
        db_path=args.ptbrvarid_db,
        dataset=args.ptbrvarid_dataset,
        splits=split_values,
        exclusions=exclusions,
        excluded_domains=excluded_domains,
        progress_every=int(args.progress_every),
    )
    log_progress(
        f"[ptbrvarid] leftovers_loaded train={len(leftover_rows.get('train', []))} "
        f"valid={len(leftover_rows.get('valid', []))}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    translation_train_out = args.out_dir / "translation_train.jsonl"
    translation_valid_out = args.out_dir / "translation_valid.jsonl"
    classification_train_out = args.out_dir / "classification_train.jsonl"
    classification_valid_out = args.out_dir / "classification_valid.jsonl"

    translation_train_counts = copy_jsonl(args.opensubs_translation_train, translation_train_out)
    translation_valid_counts = copy_jsonl(args.opensubs_translation_valid, translation_valid_out)
    classification_train_counts = write_classification_mix(
        opensubs_path=args.opensubs_classification_train,
        ptbrvarid_rows=leftover_rows.get("train", []),
        out_path=classification_train_out,
        classification_token=args.classification_token,
        balance_mode=args.classification_balance_mode,
        ptbrvarid_target_share=float(args.train_ptbrvarid_target_share),
        seed=int(args.seed),
        split_name="train",
        progress_every=int(args.progress_every),
    )
    classification_valid_counts = write_classification_mix(
        opensubs_path=args.opensubs_classification_valid,
        ptbrvarid_rows=leftover_rows.get("valid", []),
        out_path=classification_valid_out,
        classification_token=args.classification_token,
        balance_mode=args.classification_balance_mode,
        ptbrvarid_target_share=float(args.valid_ptbrvarid_target_share),
        seed=int(args.seed),
        split_name="valid",
        progress_every=int(args.progress_every),
    )

    report = {
        "out_dir": args.out_dir.as_posix(),
        "ptbrvarid_db": args.ptbrvarid_db.as_posix(),
        "ptbrvarid_dataset": args.ptbrvarid_dataset,
        "ptbrvarid_sampled_csv": args.ptbrvarid_sampled_csv.as_posix(),
        "ptbrvarid_excluded_domains": sorted(excluded_domains),
        "ptbrvarid_exclusions": len(exclusions),
        "translation_train": translation_train_counts,
        "translation_valid": translation_valid_counts,
        "classification_train": classification_train_counts,
        "classification_valid": classification_valid_counts,
        "ptbrvarid_leftover_rows": {
            "train": len(leftover_rows.get("train", [])),
            "valid": len(leftover_rows.get("valid", [])),
        },
    }
    report_path = args.out_dir / "build_report_ptbrvarid_cls.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
