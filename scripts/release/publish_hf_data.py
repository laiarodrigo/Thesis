#!/usr/bin/env python3
"""Audit and privately publish curated dissertation data to Hugging Face."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import subprocess
from pathlib import Path

EXPECTED_ROWS = 26
DEFAULT_REPO_ID = "laiarodrigo/portuguese-variant-data"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RELEASE_FIELDS = [
    "path", "role", "source", "split", "size_bytes", "sha256", "status",
    "release_mode", "license", "source_uri", "external_uri", "notes",
]
FRMT_URI = "https://huggingface.co/datasets/hugosousa/frmt"
FRMT_PAPER = "https://aclanthology.org/2023.tacl-1.39/"
GOLDEN_URI = "https://huggingface.co/datasets/joaosanches/golden_collection"
GOLDEN_PAPER = "https://arxiv.org/abs/2408.07457"
OPENSUBS_URI = "https://opus.nlpl.eu/datasets/OpenSubtitles"
CODE_URI = "https://github.com/laiarodrigo/Thesis"
FRMT_FILENAMES = (
    "classification_test.jsonl",
    "classification_train.jsonl",
    "classification_valid.jsonl",
    "translation_test.jsonl",
    "translation_train.jsonl",
    "translation_valid.jsonl",
)
GOLDEN_FILENAMES = (
    "classification_test.jsonl",
    "translation_test.jsonl",
)
STALE_REMOTE_FRMT_FILENAMES = (
    "classification_test_label_first_noequal.jsonl",
    "classification_test_prefixed_noequal.jsonl",
    "classification_valid_unused.jsonl",
    "translation_test_label_first_noequal.jsonl",
    "translation_valid_unused.jsonl",
)


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def csv_content(fieldnames, rows):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_content(fieldnames, rows))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def validate_training_manifest(root, rows):
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    seen = set()
    for row in rows:
        relative = row["path"]
        if relative in seen:
            raise ValueError(f"Duplicate path: {relative}")
        seen.add(relative)
        path = root / relative
        if row["status"] != "ready":
            raise ValueError(f"Artifact is not ready: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Size mismatch: {relative}")
        if not SHA256_RE.fullmatch(row["sha256"]):
            raise ValueError(f"Invalid SHA-256: {relative}")


def training_release_rows(rows):
    output = []
    for row in rows:
        source = row["supervision"]
        if source == "opensubtitles":
            mode = "reconstruction_only"
            license_name = "upstream terms apply"
            source_uri = OPENSUBS_URI
            notes = (
                "Not hosted pending exact retrieval metadata and "
                "redistribution review."
            )
        elif source == "gpt_wikipedia":
            mode = "hosted"
            license_name = "mixed or source-specific; see dataset card"
            source_uri = CODE_URI
            notes = "Task-formatted GPT-Wikipedia supervision."
        elif source == "gpt_wikipedia_frmt":
            mode = "hosted"
            license_name = "mixed; FRMT component is CC BY 4.0"
            source_uri = FRMT_URI
            notes = "Task-formatted GPT-Wikipedia and FRMT supervision."
        else:
            raise ValueError(f"Unknown supervision source: {source}")
        output.append(
            {
                "path": row["path"],
                "role": "reward_training" if row["stage"] == "C" else "training",
                "source": source,
                "split": row["split"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "status": row["status"],
                "release_mode": mode,
                "license": license_name,
                "source_uri": source_uri,
                "external_uri": "",
                "notes": notes,
            }
        )
    return output


def infer_split(path):
    stem = path.stem.lower()
    for name in ("train", "valid", "validation", "dev", "test"):
        if name in stem:
            return "valid" if name in {"validation", "dev"} else name
    return "unspecified"


def discovered_rows(root):
    specs = [
        (
            root / "data/encoder_decoder/t5gemma2/frmt_only",
            FRMT_FILENAMES,
            "frmt", "hosted", "CC BY 4.0", FRMT_URI,
            "Processed FRMT split used by the dissertation.",
        ),
        (
            root / "data/encoder_decoder/t5gemma2/golden_collection",
            GOLDEN_FILENAMES,
            "golden_collection", "upstream_reference",
            "No explicit redistribution license recorded", GOLDEN_URI,
            "Hashed local evaluation artifact; text is not republished.",
        ),
    ]
    output = []
    for directory, filenames, source, mode, license_name, uri, notes in specs:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = [directory / name for name in filenames]
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(str(path) for path in missing))
        for path in files:
            split = infer_split(path)
            output.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "role": "evaluation" if split == "test" else "source_split",
                    "source": source,
                    "split": split,
                    "size_bytes": str(path.stat().st_size),
                    "sha256": file_sha256(path),
                    "status": "ready",
                    "release_mode": mode,
                    "license": license_name,
                    "source_uri": uri,
                    "external_uri": uri if mode == "upstream_reference" else "",
                    "notes": notes,
                }
            )
    return output


def build_plan(root, rows):
    plan = {}
    seen = set()
    for row in rows:
        relative = row["path"]
        if relative in seen:
            raise ValueError(f"Duplicate release path: {relative}")
        seen.add(relative)
        if row["release_mode"] != "hosted":
            continue
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Hosted size mismatch: {relative}")
        plan[relative] = path
    return plan


def card(rows, revision):
    hosted = [row for row in rows if row["release_mode"] == "hosted"]
    referenced = [row for row in rows if row["release_mode"] != "hosted"]
    hosted_bytes = sum(int(row["size_bytes"]) for row in hosted)
    lines = [
        "---",
        "pretty_name: Portuguese Variety Data for Identification and Rewriting",
        "language:",
        "- pt",
        "task_categories:",
        "- translation",
        "- text-classification",
        "license: other",
        "tags:",
        "- european-portuguese",
        "- brazilian-portuguese",
        "- portuguese-varieties",
        "---",
        "",
        "# Portuguese Variety Data for Identification and Rewriting",
        "",
        "This repository documents both training and evaluation resources used",
        "in the dissertation. It hosts the curated release subset and records",
        "other resources through exact hashes,",
        "upstream references, and deterministic reconstruction instructions.",
        "",
        f"Code and preprocessing: {CODE_URI}",
        f"Source revision: {revision}",
        f"Hosted files: {len(hosted)}",
        f"Hosted bytes: {hosted_bytes}",
        f"Reference or reconstruction entries: {len(referenced)}",
        "",
        "## Hosted material",
        "",
        "Hosted material includes task-formatted GPT-Wikipedia supervision,",
        "GPT-Wikipedia and FRMT mixtures, Stage C subsets, and processed FRMT",
        "splits. The FRMT component is CC BY 4.0. Its paper and attribution",
        f"information are available at {FRMT_PAPER}",
        "",
        "GPT-Wikipedia was constructed for this dissertation with GPT-4o",
        "through the IAedu API. Portuguese Wikipedia passages supplied topics,",
        "domains, and contexts rather than aligned examples. The model generated",
        "new near-literal European and Brazilian Portuguese pairs, which were",
        "then reviewed and cleaned to retain variety-relevant differences and",
        "remove unnecessary reformulation. The records and source repository",
        "should be used together to preserve provenance.",
        "",
        "## Licensing and attribution",
        "",
        "This is a compound release, so no single license applies to every",
        "entry. The repository therefore uses the Hugging Face `other` license",
        "tag. The `license`, `source_uri`, and `release_mode` columns in",
        "`release_manifest.csv` state the applicable terms and distribution",
        "decision for each artifact. FRMT-derived files retain CC BY 4.0",
        "attribution. Users remain responsible for the terms of upstream",
        "resources and generated material.",
        "",
        "## OpenSubtitles",
        "",
        "OpenSubtitles-derived training files are not hosted while exact",
        "retrieval metadata and redistribution terms are finalized. Their exact",
        "sizes and SHA-256 values remain in the release manifest. Source and",
        f"preprocessing information: {OPENSUBS_URI} and {CODE_URI}",
        "",
        "## Golden Collection",
        "",
        "Golden Collection evaluation text is not republished because no",
        "explicit redistribution license has been recorded. Exact local hashes",
        f"and the upstream references are provided: {GOLDEN_URI} and",
        f"{GOLDEN_PAPER}",
        "",
        "## Inventory",
        "",
        "The release_manifest.csv file is authoritative. Its release_mode field",
        "states whether each artifact is hosted, referenced upstream, or",
        "reconstructed from the cited source and preprocessing pipeline.",
        "",
        "## Intended use and limitations",
        "",
        "The hosted files support reproduction and analysis of the dissertation",
        "experiments on Portuguese variety identification and rewriting. They",
        "are not a comprehensive representation of either Portuguese variety.",
        "Generated and automatically aligned text can contain errors, and the",
        "resources should not be used as a sole authority on linguistic usage.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def audit(rows, plan, repo_id):
    counts = {}
    sizes = {}
    for row in rows:
        mode = row["release_mode"]
        counts[mode] = counts.get(mode, 0) + 1
        sizes[mode] = sizes.get(mode, 0) + int(row["size_bytes"])
    for mode in sorted(counts):
        print(f"{mode}\trows={counts[mode]}\tbytes={sizes[mode]}")
    print(f"hosted_files={len(plan)}\trepo={repo_id}")


def remote_inventory(api, repo_id):
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        recursive=True,
        expand=True,
    )
    return {entry.path: entry for entry in entries if hasattr(entry, "size")}


def remote_lfs_sha(entry):
    value = getattr(entry, "lfs", None)
    if isinstance(value, dict):
        return value.get("sha256")
    return getattr(value, "sha256", None) if value is not None else None


def set_external_uris(training_rows, release_rows, repo_id):
    base = f"https://huggingface.co/datasets/{repo_id}"
    by_path = {row["path"]: row for row in release_rows}
    for row in release_rows:
        if row["release_mode"] == "hosted":
            row["external_uri"] = f"{base}/blob/main/{row['path']}"
    for row in training_rows:
        row["external_uri"] = by_path[row["path"]]["external_uri"]


def verify_remote(api, repo_id, release_rows):
    inventory = remote_inventory(api, repo_id)
    errors = []
    for row in release_rows:
        if row["release_mode"] != "hosted":
            continue
        entry = inventory.get(row["path"])
        if entry is None:
            errors.append(f"missing {row['path']}")
            continue
        if int(entry.size) != int(row["size_bytes"]):
            errors.append(f"size mismatch {row['path']}")
            continue
        remote_sha = remote_lfs_sha(entry)
        if remote_sha and remote_sha != row["sha256"]:
            errors.append(f"SHA-256 mismatch {row['path']}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return inventory


def upload(
    root,
    training_path,
    training_fields,
    training_rows,
    release_path,
    release_rows,
    plan,
    repo_id,
    revision,
):
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = HfApi()
    identity = api.whoami()
    username = identity.get("name") or identity.get("user")
    namespace = repo_id.split("/", 1)[0]
    if username != namespace:
        raise RuntimeError(
            f"Authenticated as {username!r}, expected {namespace!r}"
        )

    api.create_repo(
        repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True
    )
    inventory = remote_inventory(api, repo_id)
    stale_paths = [
        "data/encoder_decoder/t5gemma2/frmt_only/" + name
        for name in STALE_REMOTE_FRMT_FILENAMES
    ]
    stale_paths = [path for path in stale_paths if path in inventory]
    if stale_paths:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[
                CommitOperationDelete(path_in_repo=path) for path in stale_paths
            ],
            commit_message="Remove noncanonical FRMT derivatives",
        )
        for path in stale_paths:
            print(f"removed\t{path}")
        inventory = remote_inventory(api, repo_id)
    for relative, source in plan.items():
        entry = inventory.get(relative)
        if entry is not None and int(entry.size) == source.stat().st_size:
            print(f"verified\t{relative}")
            continue
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(source),
            path_in_repo=relative,
            commit_message=f"Publish {relative}",
        )
        print(f"uploaded\t{relative}")
        inventory = remote_inventory(api, repo_id)

    set_external_uris(training_rows, release_rows, repo_id)
    write_csv(training_path, training_fields, training_rows)
    write_csv(release_path, RELEASE_FIELDS, release_rows)

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=card(release_rows, revision),
            ),
            CommitOperationAdd(
                path_in_repo="release_manifest.csv",
                path_or_fileobj=csv_content(RELEASE_FIELDS, release_rows),
            ),
            CommitOperationAdd(
                path_in_repo="dissertation_data_manifest.csv",
                path_or_fileobj=csv_content(training_fields, training_rows),
            ),
        ],
        commit_message="Publish dissertation data inventory",
    )

    inventory = verify_remote(api, repo_id, release_rows)
    required = {
        "README.md", "release_manifest.csv", "dissertation_data_manifest.csv"
    }
    missing = required - set(inventory)
    if missing:
        raise RuntimeError(f"Missing remote metadata: {sorted(missing)}")
    print(
        "HF_DATA_PRIVATE_UPLOAD_COMPLETED\t"
        f"https://huggingface.co/datasets/{repo_id}"
    )


def make_public(repo_id, release_rows):
    from huggingface_hub import HfApi

    api = HfApi()
    identity = api.whoami()
    username = identity.get("name") or identity.get("user")
    namespace = repo_id.split("/", 1)[0]
    if username != namespace:
        raise RuntimeError(
            f"Authenticated as {username!r}, expected {namespace!r}"
        )
    inventory = verify_remote(api, repo_id, release_rows)
    required = {
        "README.md", "release_manifest.csv", "dissertation_data_manifest.csv"
    }
    missing = required - set(inventory)
    if missing:
        raise RuntimeError(f"Missing remote metadata: {sorted(missing)}")
    api.update_repo_settings(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
    )
    print(
        "HF_DATA_PUBLICATION_COMPLETED\t"
        f"https://huggingface.co/datasets/{repo_id}"
    )


def parse_args():
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Audit and publish curated dissertation data."
    )
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/manifests/dissertation_data_manifest.csv"),
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("data/manifests/dissertation_dataset_release.csv"),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--make-public",
        action="store_true",
        help="Make the verified repository public after refreshing its metadata.",
    )
    args = parser.parse_args()
    if not (args.audit or args.upload or args.make_public):
        parser.error("select --audit, --upload, or --make-public")
    if args.make_public and not args.upload:
        parser.error("--make-public requires --upload")
    return args


def main():
    args = parse_args()
    root = args.root.resolve()
    training_path = root / args.training_manifest
    release_path = root / args.release_manifest
    training_fields, training_rows = read_csv(training_path)
    validate_training_manifest(root, training_rows)
    release_rows = training_release_rows(training_rows)
    release_rows.extend(discovered_rows(root))
    plan = build_plan(root, release_rows)
    revision = git_revision(root)
    write_csv(release_path, RELEASE_FIELDS, release_rows)

    if args.audit:
        audit(release_rows, plan, args.repo_id)
    if args.upload:
        upload(
            root,
            training_path,
            training_fields,
            training_rows,
            release_path,
            release_rows,
            plan,
            args.repo_id,
            revision,
        )
    if args.make_public:
        make_public(args.repo_id, release_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
