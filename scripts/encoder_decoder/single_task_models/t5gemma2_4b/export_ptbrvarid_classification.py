#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(
        description=(
            "Export PtBrVId rows into seq2seq-style classification JSONL splits. "
            "This is meant for no-OpenSubs experiments where PtBrVId provides "
            "classification supervision and another corpus provides translation."
        )
    )
    parser.add_argument(
        "--ptbrvarid-db",
        type=Path,
        default=repo_root / "data" / "duckdb" / "subs_ptbr_filtered.duckdb",
    )
    parser.add_argument("--ptbrvarid-dataset", default="PtBrVId")
    parser.add_argument(
        "--splits",
        default="train,valid",
        help="Comma-separated PtBrVId splits to export.",
    )
    parser.add_argument(
        "--classification-token",
        default="<id>",
        help="Encoder-side token used by the with-cls setup.",
    )
    parser.add_argument(
        "--exclude-domains",
        default="",
        help="Comma-separated PtBrVId domains to exclude, e.g. 'social_media,web'.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200000,
        help="Emit a progress line every N scanned rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root
        / "data"
        / "encoder_decoder"
        / "t5gemma2"
        / "compare_staged_v2"
        / "ptbrvarid_classification",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def canonical_label(text: str) -> str:
    value = normalize_space(text).casefold()
    if value in {"pt-br", "pt_br", "ptbr"}:
        return "pt-br"
    if value in {"pt-pt", "pt_pt", "ptpt"}:
        return "pt-pt"
    raise ValueError(f"Unsupported label: {text}")


def parse_excluded_domains(raw: str) -> set[str]:
    return {
        normalize_space(part).casefold()
        for part in str(raw or "").split(",")
        if normalize_space(part)
    }


def maybe_resolve_db_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parents[4] / "data" / "duckdb" / "subs_filtered_final.duckdb"
    if path.name == "subs_ptbr_filtered.duckdb" and fallback.exists():
        return fallback
    return path


def log_progress(message: str) -> None:
    print(message, flush=True)


def source_text_from_ptbrvarid_row(row: dict[str, Any]) -> str:
    label = canonical_label(str(row["label"]))
    if label == "pt-br":
        return normalize_space(str(row.get("text_pt_br") or ""))
    return normalize_space(str(row.get("text_pt_pt") or ""))


def stream_ptbrvarid_rows(
    *,
    db_path: Path,
    dataset: str,
    splits: set[str],
    excluded_domains: set[str],
    progress_every: int,
    classification_token: str,
    out_dir: Path,
) -> dict[str, dict[str, Any]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency 'duckdb'. Use an environment with duckdb installed.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        split: out_dir / f"classification_{split}.jsonl"
        for split in sorted(splits)
    }
    out_files = {
        split: path.open("w", encoding="utf-8")
        for split, path in out_paths.items()
    }
    counts_by_split: dict[str, Counter[str]] = {
        split: Counter()
        for split in sorted(splits)
    }

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

        processed_rows = 0

        while True:
            rows = cur.fetchmany(10000)
            if not rows:
                break
            for split, domain, label, text_pt_br, text_pt_pt in rows:
                processed_rows += 1
                source_split = normalize_space(str(split or "")).casefold()
                if source_split not in splits:
                    continue
                domain_norm = normalize_space(str(domain or "")).casefold()
                if domain_norm in excluded_domains:
                    continue

                label_norm = canonical_label(str(label or ""))
                row = {
                    "split": source_split,
                    "domain": normalize_space(str(domain or "")),
                    "label": label_norm,
                    "text_pt_br": normalize_space(str(text_pt_br or "")),
                    "text_pt_pt": normalize_space(str(text_pt_pt or "")),
                }
                source_text = source_text_from_ptbrvarid_row(row)
                if not source_text:
                    counts_by_split[source_split]["skipped_invalid"] += 1
                    continue

                example = {
                    "input_text": f"{classification_token} {source_text}".strip(),
                    "target_text": label_norm,
                    "task": "classification",
                    "dataset": "PtBrVId",
                    "domain": normalize_space(str(domain or "")),
                    "direction": "classification",
                }
                out_files[source_split].write(json.dumps(example, ensure_ascii=False) + "\n")
                counts_by_split[source_split]["rows"] += 1
                counts_by_split[source_split][f"label:{label_norm}"] += 1
                counts_by_split[source_split][
                    f"domain:{normalize_space(str(domain or 'unknown'))}"
                ] += 1

                if progress_every > 0 and processed_rows % progress_every == 0:
                    parts = " ".join(
                        f"{name}={counts_by_split[name].get('rows', 0)}"
                        for name in sorted(counts_by_split)
                    )
                    log_progress(f"[ptbrvarid] scanned_rows={processed_rows} kept={parts}")

        return {split: dict(counts) for split, counts in sorted(counts_by_split.items())}
    finally:
        con.close()
        for fh in out_files.values():
            fh.close()


def main() -> None:
    args = parse_args()
    args.ptbrvarid_db = maybe_resolve_db_path(args.ptbrvarid_db)
    if not args.ptbrvarid_db.exists():
        raise SystemExit(f"PtBrVId DB not found: {args.ptbrvarid_db}")

    split_values = {
        normalize_space(item).casefold()
        for item in str(args.splits or "").split(",")
        if normalize_space(item)
    }
    if not split_values:
        raise SystemExit("No PtBrVId splits were requested.")
    excluded_domains = parse_excluded_domains(args.exclude_domains)

    log_progress(
        f"[ptbrvarid] loading rows from {args.ptbrvarid_db} "
        f"splits={sorted(split_values)} excluded_domains={sorted(excluded_domains)}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_stats = stream_ptbrvarid_rows(
        db_path=args.ptbrvarid_db,
        dataset=args.ptbrvarid_dataset,
        splits=split_values,
        excluded_domains=excluded_domains,
        progress_every=int(args.progress_every),
        classification_token=args.classification_token,
        out_dir=args.out_dir,
    )
    report: dict[str, Any] = {
        "ptbrvarid_db": args.ptbrvarid_db.as_posix(),
        "ptbrvarid_dataset": args.ptbrvarid_dataset,
        "splits": sorted(split_values),
        "exclude_domains": sorted(excluded_domains),
        "classification_token": args.classification_token,
        "out_dir": args.out_dir.as_posix(),
        "split_stats": split_stats,
    }

    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
