#!/usr/bin/env python3
"""Audit and publish the 34 dissertation models in one Hugging Face repository."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable

EXPECTED_MODELS = 34
DEFAULT_REPO_ID = "laiarodrigo/portuguese-variant-t5gemma2"
SHA256_RE = re.compile(r"[0-9a-f]{64}")

ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
FULL_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
OPTIONAL_FILES = (
    "training_example_counts.json",
    "stage_c_training_example_counts.json",
    "config_snapshot.yaml",
)


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        return list(reader.fieldnames), list(reader)


def manifest_bytes(
    fieldnames: list[str], rows: Iterable[dict[str, str]]
) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def write_manifest(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]
) -> None:
    path.write_bytes(manifest_bytes(fieldnames, rows))


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def base_model_id(row: dict[str, str]) -> str:
    if "4b" in row["architecture"].lower():
        return "google/t5gemma-2-4b-4b"
    return "google/t5gemma-2-270m-270m"


def required_names(row: dict[str, str]) -> tuple[str, ...]:
    if row["weight_type"] == "peft_lora_adapter":
        return ADAPTER_FILES
    if row["weight_type"] == "full_model":
        return FULL_MODEL_FILES
    raise ValueError(
        f"Unsupported weight type for {row['model_id']}: {row['weight_type']}"
    )


def model_files(
    root: Path, row: dict[str, str]
) -> list[tuple[Path, str]]:
    source_dir = root / row["output_dir"]
    config_path = root / row["config_path"]
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    files = []
    for name in required_names(row):
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        files.append((source, name))

    weight_path = source_dir / row["weight_file"]
    if weight_path.stat().st_size != int(row["size_bytes"]):
        raise ValueError(f"Weight size mismatch: {row['model_id']}")

    for name in OPTIONAL_FILES:
        source = source_dir / name
        if source.is_file():
            target = (
                "runtime_config_snapshot.yaml"
                if name == "config_snapshot.yaml"
                else name
            )
            files.append((source, target))

    files.append((config_path, "training_config.yaml"))
    return files


def validate_rows(
    root: Path, rows: list[dict[str, str]], selected: set[str]
) -> dict[str, list[tuple[Path, str]]]:
    if len(rows) != EXPECTED_MODELS:
        raise ValueError(f"Expected {EXPECTED_MODELS} model rows, found {len(rows)}")

    known_ids = {row["model_id"] for row in rows}
    unknown = selected - known_ids
    if unknown:
        raise ValueError(f"Unknown model IDs: {', '.join(sorted(unknown))}")

    plan = {}
    for row in rows:
        if selected and row["model_id"] not in selected:
            continue
        if row["status"] != "ready":
            raise ValueError(f"Model is not ready: {row['model_id']}")
        if not SHA256_RE.fullmatch(row["sha256"]):
            raise ValueError(f"Invalid SHA-256 for {row['model_id']}")
        plan[row["model_id"]] = model_files(root, row)
    return plan


def metadata(row: dict[str, str], revision: str) -> bytes:
    value = {
        "model_id": row["model_id"],
        "architecture": row["architecture"],
        "base_model": base_model_id(row),
        "stage": row["stage"],
        "control": row["control"],
        "supervision": row["supervision"],
        "variant": row["variant"],
        "weight_type": row["weight_type"],
        "weight_file": row["weight_file"],
        "weight_size_bytes": int(row["size_bytes"]),
        "weight_sha256": row["sha256"],
        "config_path": row["config_path"],
        "result_table_rows": row["result_table_rows"],
        "source_repository": "https://github.com/laiarodrigo/Thesis",
        "source_revision": revision,
    }
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def index_card(
    rows: list[dict[str, str]], repo_id: str, revision: str
) -> bytes:
    lines = [
        "---",
        "license: gemma",
        "library_name: transformers",
        "pipeline_tag: translation",
        "language:",
        "- pt",
        "tags:",
        "- portuguese",
        "- european-portuguese",
        "- brazilian-portuguese",
        "- text-classification",
        "- text2text-generation",
        "- peft",
        "- t5gemma2",
        "---",
        "",
        "# Portuguese Variety Identification and Rewriting with T5Gemma 2",
        "",
        "This repository contains the 34 models reported in the dissertation.",
        "The models support Portuguese variety identification and bidirectional",
        "rewriting between European and Brazilian Portuguese.",
        "",
        "Code, configurations, evaluation scripts, result tables, and artifact",
        "manifests are available at https://github.com/laiarodrigo/Thesis.",
        "",
        f"Source revision: {revision}",
        "",
        "Each model is stored in a folder named with its dissertation model ID.",
        "Every folder includes the exact training configuration, tokenizer files,",
        "artifact metadata, and either a PEFT LoRA adapter or a fully fine-tuned",
        "model. The artifact metadata records the SHA-256 value of the weight file.",
        "",
        "| Model | Stage | Control | Supervision | Variant | Artifact |",
        "|---|---:|---|---|---|---|",
    ]
    for row in rows:
        folder_url = f"https://huggingface.co/{repo_id}/tree/main/{row['model_id']}"
        lines.append(
            "| "
            f"[{row['model_id']}]({folder_url}) | {row['stage']} | "
            f"{row['control']} | {row['supervision']} | {row['variant']} | "
            f"{row['weight_type']} |"
        )

    lines.extend(
        [
            "",
            "## Loading",
            "",
            "Use the model folder as the subfolder argument when loading tokenizer",
            "and model files. The 4B entries are PEFT LoRA adapters for",
            "google/t5gemma-2-4b-4b. The 270M entries are standalone fully",
            "fine-tuned models based on google/t5gemma-2-270m-270m.",
            "",
            "## License",
            "",
            "These models are derivatives of T5Gemma 2 and are distributed under",
            "the Gemma Terms of Use: https://ai.google.dev/gemma/terms.",
            "Users are responsible for complying with those terms and with the",
            "licenses of the training and evaluation resources described in the",
            "dissertation.",
            "",
            "## Limitations",
            "",
            "The models were evaluated only under the dissertation protocols.",
            "Automatic BLEU, TER, and classification results do not establish",
            "performance for every Portuguese domain or every acceptable variety",
            "rewrite.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def audit(
    rows: list[dict[str, str]],
    plan: dict[str, list[tuple[Path, str]]],
    repo_id: str,
) -> None:
    total = 0
    for row in rows:
        if row["model_id"] not in plan:
            continue
        files = plan[row["model_id"]]
        size = sum(source.stat().st_size for source, _ in files)
        total += size
        names = ",".join(target for _, target in files)
        print(f"{row['model_id']}\t{repo_id}\t{size}\t{names}")
    print(f"models={len(plan)}\tselected_bytes={total}")


def upload(
    root: Path,
    manifest_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    plan: dict[str, list[tuple[Path, str]]],
    repo_id: str,
    revision: str,
    complete_release: bool,
) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    namespace = repo_id.split("/", 1)[0]
    identity = api.whoami()
    username = identity.get("name") or identity.get("user")
    if username != namespace:
        raise RuntimeError(
            f"Authenticated as {username!r}, expected namespace {namespace!r}"
        )

    api.create_repo(
        repo_id=repo_id, repo_type="model", private=False, exist_ok=True
    )
    root_url = f"https://huggingface.co/{repo_id}"

    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=index_card(rows, repo_id, revision),
            ),
            CommitOperationAdd(
                path_in_repo="model_manifest.csv",
                path_or_fileobj=manifest_bytes(fieldnames, rows),
            ),
        ],
        commit_message="Initialize dissertation model release",
    )
    remote_files = set(api.list_repo_files(repo_id, repo_type="model"))

    for row in rows:
        model_id = row["model_id"]
        if model_id not in plan:
            continue
        files = plan[model_id]
        expected = {f"{model_id}/{target}" for _, target in files}
        expected.add(f"{model_id}/artifact_metadata.json")
        if expected.issubset(remote_files):
            print(f"verified\t{model_id}\t{root_url}/tree/main/{model_id}")
        else:
            operations = [
                CommitOperationAdd(
                    path_in_repo=f"{model_id}/{target}",
                    path_or_fileobj=str(source),
                )
                for source, target in files
            ]
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{model_id}/artifact_metadata.json",
                    path_or_fileobj=metadata(row, revision),
                )
            )
            api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=operations,
                commit_message=f"Publish dissertation model {model_id}",
            )
            remote_files = set(api.list_repo_files(repo_id, repo_type="model"))
            missing = expected - remote_files
            if missing:
                raise RuntimeError(
                    f"Remote verification failed for {model_id}: {sorted(missing)}"
                )
            print(f"uploaded\t{model_id}\t{root_url}/tree/main/{model_id}")

        row["external_uri"] = f"{root_url}/tree/main/{model_id}"
        write_manifest(manifest_path, fieldnames, rows)

    if not complete_release:
        return

    missing_uris = [row["model_id"] for row in rows if not row["external_uri"]]
    if missing_uris:
        raise RuntimeError(
            "Cannot publish index; missing external URI for "
            + ", ".join(missing_uris)
        )

    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=index_card(rows, repo_id, revision),
            ),
            CommitOperationAdd(
                path_in_repo="model_manifest.csv",
                path_or_fileobj=manifest_bytes(fieldnames, rows),
            ),
        ],
        commit_message="Publish dissertation model index",
    )
    print(f"index\t{root_url}")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Audit and publish dissertation models to Hugging Face."
    )
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/dissertation/model_manifest.csv"),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help="Limit the operation to one model ID; repeat for multiple models.",
    )
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if not (args.audit or args.upload):
        parser.error("select --audit, --upload, or both")
    return args


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = root / args.manifest
    fieldnames, rows = read_manifest(manifest_path)
    selected = set(args.only)
    plan = validate_rows(root, rows, selected)
    revision = git_revision(root)

    if args.audit:
        audit(rows, plan, args.repo_id)
    if args.upload:
        upload(
            root=root,
            manifest_path=manifest_path,
            fieldnames=fieldnames,
            rows=rows,
            plan=plan,
            repo_id=args.repo_id,
            revision=revision,
            complete_release=not selected,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
