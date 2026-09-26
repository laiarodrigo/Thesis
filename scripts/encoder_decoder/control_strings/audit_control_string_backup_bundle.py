#!/usr/bin/env python3
"""Estimate the backup size for the control-string protocol bundle.

The default bundle is intentionally narrow:
  - control-string data/eval JSONLs,
  - final control-string adapter files only,
  - control-string eval results,
  - small configs/scripts/report metadata needed to reproduce the runs.

Checkpoint directories and pre-rerun backup directories are excluded by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_PATHS = [
    "data/encoder_decoder/t5gemma2/control_string_protocol",
    "data/encoder_decoder/t5gemma2/control_string_eval",
]

DEFAULT_EVAL_PATHS = [
    "eval_results/encoder_decoder/control_strings",
]

THESIS_REPRO_SOURCE_DATA_PATHS = [
    # Main OpenSubtitles-derived source DBs. These are kept because the generated
    # JSONLs are rebuildable only if these inputs and export scripts remain sane.
    "data/duckdb/subs_filtered_final.duckdb",
    "data/duckdb/subs_ptbr_filtered.duckdb",
    "data/duckdb/subs_project.duckdb",
    "data/duckdb/subs.duckdb",
    # Dataset sources and held-out eval sets used by recovered/control-string runs.
    "data/encoder_decoder/t5gemma2/frmt_only",
    "data/encoder_decoder/t5gemma2/golden_collection",
    "data/wikipedia",
    "data/wikipedia_downloads",
    "data/wikipedia_pt_inspiration",
    "data/wikipedia_pt_variant_csv",
]

THESIS_REPRO_GENERATED_DATA_PATHS = [
    # Recovered May r48 views.
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_with_cls",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_label_first_with_cls_noequal",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_label_first_with_cls_equal",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_translation_plus_cls_noequal",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_label_first_with_cls_noequal",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_label_first_with_cls_equal",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_translation_plus_cls_noequal",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_noequal",
    "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_plus_ptbrvarid_equal",
    "data/encoder_decoder/stage_c_subset/frmt_only_data_loose",
    "data/encoder_decoder/stage_c_subset/frmt_gpt_wiki_data_legit",
    # Current control-string views.
    "data/encoder_decoder/t5gemma2/control_string_protocol",
    "data/encoder_decoder/t5gemma2/control_string_eval",
]

DEFAULT_METADATA_PATHS = [
    "configs/encoder_decoder/t5gemma2_4b/control_strings",
    "configs/encoder_decoder/t5gemma2_4b/comparison_staged",
    "scripts/encoder_decoder/control_strings",
    "scripts/encoder_decoder/single_task_models/t5gemma2_4b",
    "scripts/encoder_decoder/eval",
    "report/final_model_rerun_plan.md",
]

ADAPTER_ROOT = Path("outputs/encoder_decoder/control_strings")


@dataclass(frozen=True)
class FileRecord:
    category: str
    path: str
    bytes: int
    sha256: str | None = None


def iter_files(path: Path, *, include_checkpoints: bool, include_pre_rerun_backups: bool) -> Iterable[Path]:
    if not path.exists():
        return
    if path.is_file():
        yield path
        return

    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        kept_dirs = []
        for d in dirs:
            if not include_checkpoints and d.startswith("checkpoint-"):
                continue
            if not include_pre_rerun_backups and ".pre_" in d:
                continue
            if not include_pre_rerun_backups and "backup" in d.lower():
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs

        for name in files:
            file_path = root_path / name
            if not include_pre_rerun_backups and ".pre_" in str(file_path):
                continue
            yield file_path


def iter_final_adapter_files(repo_root: Path, *, include_pre_rerun_backups: bool) -> Iterable[Path]:
    root = repo_root / ADAPTER_ROOT
    if not root.exists():
        return

    for adapter in sorted(root.glob("*")):
        if not adapter.is_dir():
            continue
        if not include_pre_rerun_backups and (".pre_" in adapter.name or "backup" in adapter.name.lower()):
            continue
        if not (adapter / "adapter_model.safetensors").is_file():
            continue

        # Root files are sufficient for PEFT final adapters/tokenizers.
        for child in sorted(adapter.iterdir()):
            if child.is_file():
                yield child


def iter_adapter_dir_files(adapter: Path) -> Iterable[Path]:
    if not (adapter / "adapter_model.safetensors").is_file():
        return
    for child in sorted(adapter.iterdir()):
        if child.is_file():
            yield child


def control_string_adapter_dirs(repo_root: Path, adapter_root: Path, *, include_pre_rerun_backups: bool) -> list[str]:
    root = repo_root / adapter_root
    if not root.exists():
        return []

    dirs = []
    for adapter in sorted(root.glob("*")):
        if not adapter.is_dir():
            continue
        if not include_pre_rerun_backups and (".pre_" in adapter.name or "backup" in adapter.name.lower()):
            continue
        if (adapter / "adapter_model.safetensors").is_file():
            dirs.append(adapter.relative_to(repo_root).as_posix())
    return dirs


def recovered_paths_from_csvs(repo_root: Path) -> tuple[list[str], list[str], list[str]]:
    """Infer recovered adapter/eval paths from recovered_may_r48 CSVs."""
    adapter_dirs: set[str] = set()
    eval_dirs: set[str] = set()
    csv_files: set[str] = set()

    for csv_path in sorted((repo_root / "eval_results" / "encoder_decoder").glob("recovered_may_r48*.csv")):
        csv_files.add(csv_path.relative_to(repo_root).as_posix())
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                predictions_path = row.get("predictions_path") or ""
                if not predictions_path:
                    continue
                parts = Path(predictions_path).parts
                if "compare_staged" not in parts:
                    continue
                idx = parts.index("compare_staged")
                if len(parts) <= idx + 2:
                    continue
                model_dir = parts[idx + 2]
                adapter_dirs.add(f"outputs/encoder_decoder/compare_staged/{model_dir}")
                eval_dirs.add("/".join(parts[: idx + 3]))

    return sorted(adapter_dirs), sorted(eval_dirs), sorted(csv_files)


def selected_adapter_file_set(repo_root: Path, adapter_dirs: Iterable[str]) -> set[str]:
    selected = set()
    for rel in adapter_dirs:
        adapter = repo_root / rel
        if not adapter.exists():
            continue
        for file_path in iter_adapter_dir_files(adapter):
            selected.add(file_path.relative_to(repo_root).as_posix())
    return selected


def classify_repo_file(rel: str, selected_adapter_files: set[str]) -> str | None:
    """Return category or None if the file should be excluded from repo-pruned."""
    parts = Path(rel).parts
    if not parts:
        return None

    excluded_dirs = {
        ".git",
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".vendor",
        "__pycache__",
        "env",
        "thesis_g07",
        "thesis_t5",
        "venv",
    }
    if any(part in excluded_dirs for part in parts):
        return None
    if any(part.startswith("checkpoint-") for part in parts):
        return None
    if any(".pre_" in part or ".old_" in part or ".bad." in part or "backup" == part.lower() for part in parts):
        return None
    if rel.endswith((".pyc", ".pyo", ".tmp", ".corrupt", ".wal")):
        return None

    # Keep only explicitly selected final adapters under outputs/encoder_decoder.
    if parts[0] == "outputs" and len(parts) >= 2 and parts[1] == "encoder_decoder":
        return "selected_final_adapters" if rel in selected_adapter_files else None

    if parts[0] == "data":
        if len(parts) == 2 and parts[1].startswith("corpus"):
            return None
        return "repo_data"
    if parts[0] == "eval_results":
        return "repo_eval_results"
    if parts[0] == "logs":
        return "repo_logs"
    if parts[0] == "configs":
        return "repo_configs"
    if parts[0] == "scripts":
        return "repo_scripts"
    if parts[0] == "report":
        return "repo_report"
    return "repo_other"


def add_pruned_repo_records(
    records: list[FileRecord],
    repo_root: Path,
    *,
    selected_adapter_files: set[str],
    hash_files: bool,
) -> None:
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        rel_root = root_path.relative_to(repo_root).as_posix()

        kept_dirs = []
        for d in dirs:
            rel_dir = d if rel_root == "." else f"{rel_root}/{d}"
            rel_parts = Path(rel_dir).parts
            if d in {
                ".git",
                ".cache",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
                ".venv",
                ".vendor",
                "__pycache__",
                "env",
                "thesis_g07",
                "thesis_t5",
                "venv",
            }:
                continue
            if d.startswith("checkpoint-") or ".pre_" in d or ".old_" in d or ".bad." in d or "backup" == d.lower():
                continue
            if len(rel_parts) >= 2 and rel_parts[0] == "outputs" and rel_parts[1] == "encoder_decoder":
                # Descend only into dirs that may contain selected adapter files.
                prefix = rel_dir.rstrip("/") + "/"
                if not any(path.startswith(prefix) for path in selected_adapter_files):
                    continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs

        for name in files:
            file_path = root_path / name
            rel = file_path.relative_to(repo_root).as_posix()
            if ".old_" in rel or ".bad." in rel:
                continue
            category = classify_repo_file(rel, selected_adapter_files)
            if category is None:
                continue
            # Symlinks and files disappearing during the walk are not reliable
            # backup targets for this manifest.
            if file_path.is_symlink():
                continue
            try:
                size = file_path.stat().st_size
            except FileNotFoundError:
                continue
            digest = sha256_file(file_path) if hash_files else None
            records.append(FileRecord(category, rel, size, digest))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_records(
    records: list[FileRecord],
    category: str,
    paths: list[str],
    repo_root: Path,
    *,
    include_checkpoints: bool,
    include_pre_rerun_backups: bool,
    hash_files: bool,
) -> list[str]:
    missing = []
    for rel in paths:
        path = repo_root / rel
        if not path.exists():
            missing.append(rel)
            continue
        for file_path in iter_files(
            path,
            include_checkpoints=include_checkpoints,
            include_pre_rerun_backups=include_pre_rerun_backups,
        ):
            size = file_path.stat().st_size
            digest = sha256_file(file_path) if hash_files else None
            records.append(FileRecord(category, file_path.relative_to(repo_root).as_posix(), size, digest))
    return missing


def human_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def summarize(records: list[FileRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for rec in records:
        bucket = summary.setdefault(rec.category, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += rec.bytes
    return summary


def write_csv(records: list[FileRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "bytes", "sha256", "path"])
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "category": rec.category,
                    "bytes": rec.bytes,
                    "sha256": rec.sha256 or "",
                    "path": rec.path,
                }
            )


def write_file_list(records: list[FileRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.path + "\n")


def dedupe_records(records: list[FileRecord]) -> list[FileRecord]:
    seen: set[str] = set()
    deduped: list[FileRecord] = []
    for rec in records:
        if rec.path in seen:
            continue
        seen.add(rec.path)
        deduped.append(rec)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["control-string", "thesis-repro", "repo-pruned"],
        default="control-string",
        help=(
            "control-string backs up only the current control-string bundle. "
            "thesis-repro also includes recovered-CSV model data/adapters/evals and source datasets. "
            "repo-pruned walks the whole repo but keeps only selected final adapters under outputs/encoder_decoder."
        ),
    )
    parser.add_argument("--repo-root", default=".", help="Repository root. Default: current directory.")
    parser.add_argument("--budget-gib", type=float, default=25.0, help="Available backup budget in GiB.")
    parser.add_argument("--out-json", default="logs/control_strings/backup/control_string_backup_audit.json")
    parser.add_argument("--out-csv", default="logs/control_strings/backup/control_string_backup_manifest.csv")
    parser.add_argument("--out-file-list", default="logs/control_strings/backup/control_string_backup_file_list.txt")
    parser.add_argument("--hash", action="store_true", help="Compute SHA-256 for every selected file. Slow for large data.")
    parser.add_argument("--include-checkpoints", action="store_true", help="Include checkpoint-* directories.")
    parser.add_argument(
        "--include-pre-rerun-backups",
        action="store_true",
        help="Include .pre_* and backup directories.",
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        help="Extra data path to include. Can be repeated.",
    )
    parser.add_argument(
        "--adapter-root",
        default=str(ADAPTER_ROOT),
        help="Adapter root relative to repo. Default: outputs/encoder_decoder/control_strings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    adapter_root = Path(args.adapter_root)

    records: list[FileRecord] = []
    missing: dict[str, list[str]] = {}

    recovered_adapter_dirs, recovered_eval_dirs, recovered_csv_files = recovered_paths_from_csvs(repo_root)
    control_adapter_dirs = control_string_adapter_dirs(
        repo_root,
        adapter_root,
        include_pre_rerun_backups=args.include_pre_rerun_backups,
    )

    source_data_paths: list[str] = []
    data_paths: list[str] = []
    eval_paths: list[str] = []
    metadata_paths: list[str] = []
    selected_adapter_dirs = sorted(set(control_adapter_dirs + recovered_adapter_dirs))

    if args.profile == "repo-pruned":
        selected_adapter_files = selected_adapter_file_set(repo_root, selected_adapter_dirs)
        add_pruned_repo_records(
            records,
            repo_root,
            selected_adapter_files=selected_adapter_files,
            hash_files=args.hash,
        )
        missing["selected_final_adapters"] = [
            rel for rel in selected_adapter_dirs if not (repo_root / rel / "adapter_model.safetensors").is_file()
        ]
    elif args.profile == "thesis-repro":
        source_data_paths = THESIS_REPRO_SOURCE_DATA_PATHS
        data_paths = THESIS_REPRO_GENERATED_DATA_PATHS + args.data_path
        eval_paths = DEFAULT_EVAL_PATHS + recovered_eval_dirs
        metadata_paths = DEFAULT_METADATA_PATHS + recovered_csv_files
    elif args.profile == "control-string":
        source_data_paths = []
        data_paths = DEFAULT_DATA_PATHS + args.data_path
        eval_paths = DEFAULT_EVAL_PATHS
        metadata_paths = DEFAULT_METADATA_PATHS

    if args.profile != "repo-pruned" and source_data_paths:
        missing["source_data"] = add_records(
            records,
            "source_data",
            source_data_paths,
            repo_root,
            include_checkpoints=args.include_checkpoints,
            include_pre_rerun_backups=args.include_pre_rerun_backups,
            hash_files=args.hash,
        )

    if args.profile != "repo-pruned":
        missing["data"] = add_records(
            records,
            "generated_data",
            data_paths,
            repo_root,
            include_checkpoints=args.include_checkpoints,
            include_pre_rerun_backups=args.include_pre_rerun_backups,
            hash_files=args.hash,
        )
        missing["eval_results"] = add_records(
            records,
            "eval_results",
            eval_paths,
            repo_root,
            include_checkpoints=args.include_checkpoints,
            include_pre_rerun_backups=args.include_pre_rerun_backups,
            hash_files=args.hash,
        )
        missing["metadata"] = add_records(
            records,
            "metadata",
            metadata_paths,
            repo_root,
            include_checkpoints=args.include_checkpoints,
            include_pre_rerun_backups=args.include_pre_rerun_backups,
            hash_files=args.hash,
        )

        adapter_abs_root = repo_root / adapter_root
        if not adapter_abs_root.exists():
            missing["final_adapters"] = [adapter_root.as_posix()]
        else:
            for file_path in iter_final_adapter_files(repo_root, include_pre_rerun_backups=args.include_pre_rerun_backups):
                size = file_path.stat().st_size
                digest = sha256_file(file_path) if args.hash else None
                records.append(
                    FileRecord("control_string_final_adapters", file_path.relative_to(repo_root).as_posix(), size, digest)
                )
            missing["final_adapters"] = []

    if args.profile == "thesis-repro":
        missing_recovered_adapters = []
        for rel in recovered_adapter_dirs:
            adapter = repo_root / rel
            if not adapter.exists():
                missing_recovered_adapters.append(rel)
                continue
            found = False
            for file_path in iter_adapter_dir_files(adapter):
                found = True
                size = file_path.stat().st_size
                digest = sha256_file(file_path) if args.hash else None
                records.append(
                    FileRecord("recovered_final_adapters", file_path.relative_to(repo_root).as_posix(), size, digest)
                )
            if not found:
                missing_recovered_adapters.append(rel)
        missing["recovered_final_adapters"] = missing_recovered_adapters

    records = dedupe_records(records)
    summary = summarize(records)
    total_bytes = sum(rec.bytes for rec in records)
    budget_bytes = int(args.budget_gib * (1024**3))
    fits_budget = total_bytes <= budget_bytes

    report = {
        "repo_root": str(repo_root),
        "budget_gib": args.budget_gib,
        "total_bytes": total_bytes,
        "total_gib": total_bytes / (1024**3),
        "fits_budget": fits_budget,
        "missing": missing,
        "summary": {
            category: {"files": item["files"], "bytes": item["bytes"], "gib": item["bytes"] / (1024**3)}
            for category, item in sorted(summary.items())
        },
        "options": {
            "profile": args.profile,
            "hash": args.hash,
            "include_checkpoints": args.include_checkpoints,
            "include_pre_rerun_backups": args.include_pre_rerun_backups,
            "adapter_root": adapter_root.as_posix(),
            "data_paths": data_paths,
            "source_data_paths": source_data_paths,
            "eval_paths": eval_paths,
            "metadata_paths": metadata_paths,
            "recovered_adapter_dirs_from_csv": recovered_adapter_dirs,
            "control_string_adapter_dirs": control_adapter_dirs,
            "selected_adapter_dirs": selected_adapter_dirs,
        },
    }

    out_json = repo_root / args.out_json
    out_csv = repo_root / args.out_csv
    out_file_list = repo_root / args.out_file_list
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(records, out_csv)
    write_file_list(records, out_file_list)

    print("Backup bundle audit")
    print(f"repo_root: {repo_root}")
    print(f"budget: {args.budget_gib:.2f} GiB")
    print(f"total: {human_gib(total_bytes)}")
    print(f"fits_budget: {fits_budget}")
    print()
    for category, item in sorted(summary.items()):
        print(f"{category:16s} {item['files']:8d} files {human_gib(item['bytes'])}")
    print()
    print(f"wrote: {out_json.relative_to(repo_root)}")
    print(f"wrote: {out_csv.relative_to(repo_root)}")
    print(f"wrote: {out_file_list.relative_to(repo_root)}")

    any_missing = {k: v for k, v in missing.items() if v}
    if any_missing:
        print()
        print("missing paths:")
        for category, paths in any_missing.items():
            for path in paths:
                print(f"  {category}: {path}")


if __name__ == "__main__":
    main()
