#!/usr/bin/env python3
"""Freeze and audit the leakage-controlled G0/G1 Gemma 31B DEV experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from compare_v0_passage import normalized_spans
from run_evaluator_v0 import INPUT_FIELDS, _candidate_occurrences


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "data/benchmark/dev_01_evaluator.json"
EVALUATION_PATH = PROJECT_ROOT / "data/benchmark/gemma31b_g0_g1_eval_96.jsonl"
DEMONSTRATIONS_PATH = (
    PROJECT_ROOT / "data/benchmark/gemma31b_g1_eight_shot_demonstrations.json"
)
MANIFEST_PATH = PROJECT_ROOT / "data/benchmark/gemma31b_g0_g1_manifest.json"
PROMPT_DIR = PROJECT_ROOT / "guidelines/v1"

SEED = 20261016
DEMONSTRATION_DOCUMENT_COUNT = 2
EVALUATION_DOCUMENT_COUNT = 8
GROUPS_PER_EVALUATION_DOCUMENT = 3
CANDIDATES_PER_GROUP = 4
PILOT_GROUPS = {
    ("ted2020-24360", "27"),
    ("ted2020-2609", "85"),
    ("ted2020-946", "45"),
}

# Selected only after the document split was fixed. Each row is from a distinct
# source sentence. Categories describe selection coverage and are never rendered.
DEMONSTRATIONS = (
    ("dev-v0-0045", "correct_ep", "error-free score-100 adaptation"),
    ("dev-v0-0134", "correct_ep", "error-free score-100 adaptation"),
    ("dev-v0-0037", "bp_retention_major", "important BP construction retained"),
    ("dev-v0-0121", "bp_retention_major", "important BP construction retained"),
    ("dev-v0-0042", "missing_content", "gold contains one Minor [MISSING]"),
    ("dev-v0-0142", "missing_content", "gold contains one Minor [MISSING]"),
    ("dev-v0-0067", "conservative_rewriting", "unnecessary structural rewrite"),
    ("dev-v0-0148", "conservative_rewriting", "localized Minor paraphrase"),
)


class ExperimentError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--audit-frozen", action="store_true")
    parser.add_argument("--canonical", type=Path, default=CANONICAL_PATH)
    parser.add_argument("--evaluation-input", type=Path, default=EVALUATION_PATH)
    parser.add_argument("--demonstrations", type=Path, default=DEMONSTRATIONS_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def stable_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ExperimentError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"could not read {path}: {exc}") from exc
    return rows


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def hash_rank(namespace: str, value: str) -> tuple[str, str]:
    return sha256_text(f"{SEED}:{namespace}:{value}"), value


def blind_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_id": row["evaluation_id"],
        "document_id": row["document_id"],
        "passage_id": row["passage_id"],
        "sentence_id": row["sentence_id"],
        "position": row["position"],
        "passage_context": row["passage_context"],
        "current_source_bp": row["current_source_bp"],
        "candidate_text": row["candidate"]["text"],
    }


def evaluation_data(row: dict[str, Any]) -> str:
    context = "\n".join(
        f"[{sentence['position']}] {sentence['source_bp']}"
        for sentence in row["passage_context"]
    )
    return (
        "PORTUGUESE PASSAGE CONTEXT (BP — REFERENCE ONLY):\n"
        f"{context}\n\n"
        f"CURRENT BP SOURCE SENTENCE (REFERENCE ONLY, position {row['position']}):\n"
        f"{row['current_source_bp']}\n\n"
        "EP CANDIDATE TO EVALUATE (ONLY TEXT TO ANNOTATE):\n"
        f"{row['candidate']['text']}"
    )


def demonstration_errors(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = row["candidate"]["text"]
    spans, error = normalized_spans(
        candidate,
        row["human_gold"]["error_spans"],
        f"{row['evaluation_id']} demonstration",
        human_inclusive_utf16=True,
    )
    if error:
        raise ExperimentError(error)
    errors: list[dict[str, Any]] = []
    for span in spans:
        item: dict[str, Any] = {
            "text": "[MISSING]" if span["missing"] else span["text"],
            "severity": span["severity"],
        }
        if not span["missing"]:
            occurrences = _candidate_occurrences(candidate, span["text"])
            if not occurrences or span["start"] not in occurrences:
                raise ExperimentError(
                    f"{row['evaluation_id']}: converted gold span does not align to candidate"
                )
            if len(occurrences) > 1:
                item["occurrence"] = occurrences.index(span["start"]) + 1
        errors.append(item)
    return errors


def demonstration_appendices(rows: list[dict[str, Any]]) -> tuple[str, str]:
    stage1 = ["# Human-labeled demonstrations"]
    stage2 = ["# Human-labeled demonstrations"]
    for index, row in enumerate(rows, 1):
        errors = {"errors": demonstration_errors(row)}
        data = evaluation_data(row)
        stage1.extend(
            [
                f"## Demonstration {index}",
                data,
                "CANONICAL HUMAN ERROR JSON:",
                json.dumps(errors, ensure_ascii=False, separators=(",", ":")),
            ]
        )
        stage2.extend(
            [
                f"## Demonstration {index}",
                data,
                "CANONICAL HUMAN ERROR JSON:",
                json.dumps(errors, ensure_ascii=False, separators=(",", ":")),
                "CANONICAL HUMAN SCORE JSON:",
                json.dumps(
                    {"score": row["human_gold"]["score"]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ]
        )
    return "\n\n".join(stage1), "\n\n".join(stage2)


def assert_frozen_or_write(path: Path, content: str) -> None:
    if path.exists():
        observed = path.read_text(encoding="utf-8")
        if observed != content:
            raise ExperimentError(f"frozen artifact differs from regenerated content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    if "test" in args.canonical.name.casefold():
        raise ExperimentError("refusing a canonical path whose filename contains TEST")
    canonical_bytes = args.canonical.read_bytes()
    canonical = json.loads(canonical_bytes)
    if canonical.get("split") != "DEV":
        raise ExperimentError("canonical input must explicitly declare split=DEV")
    records = canonical.get("records")
    if not isinstance(records, list) or len(records) != 388:
        raise ExperimentError("canonical DEV must contain exactly 388 records")
    by_id = {row["evaluation_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ExperimentError("canonical DEV contains duplicate evaluation IDs")

    document_order = list(dict.fromkeys(row["document_id"] for row in records))
    if len(document_order) != 10:
        raise ExperimentError(
            f"expected 10 original DEV documents, observed {len(document_order)}"
        )
    pilot_documents = {document_id for document_id, _ in PILOT_GROUPS}
    eligible_demo_documents = sorted(set(document_order) - pilot_documents)
    ranked_demo_documents = sorted(
        eligible_demo_documents,
        key=lambda value: (sha256_text(f"{SEED}:{value}"), value),
    )
    demonstration_documents = ranked_demo_documents[:DEMONSTRATION_DOCUMENT_COUNT]
    evaluation_documents = [
        value for value in document_order if value not in demonstration_documents
    ]
    if len(demonstration_documents) != 2 or len(evaluation_documents) != 8:
        raise ExperimentError("document grouping does not yield a safe 2/8 split")
    if set(demonstration_documents) & pilot_documents:
        raise ExperimentError("a previously inspected pilot document entered demonstrations")

    by_document_sentence: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        by_document_sentence[row["document_id"]][row["sentence_id"]].append(row)
    selected_groups: list[tuple[str, str]] = []
    group_manifest: list[dict[str, Any]] = []
    for document_id in evaluation_documents:
        groups = by_document_sentence[document_id]
        mandatory = [
            sentence_id
            for pilot_document, sentence_id in PILOT_GROUPS
            if pilot_document == document_id
        ]
        remaining = sorted(
            (sentence_id for sentence_id in groups if sentence_id not in mandatory),
            key=lambda sentence_id: hash_rank(
                "evaluation", f"{document_id}:{sentence_id}"
            ),
        )
        chosen = mandatory + remaining[: GROUPS_PER_EVALUATION_DOCUMENT - len(mandatory)]
        chosen.sort(key=lambda sentence_id: groups[sentence_id][0]["position"])
        if len(chosen) != GROUPS_PER_EVALUATION_DOCUMENT:
            raise ExperimentError(f"{document_id}: could not select three sentence groups")
        for sentence_id in chosen:
            group = groups[sentence_id]
            if len(group) != CANDIDATES_PER_GROUP:
                raise ExperimentError(
                    f"{document_id}/{sentence_id}: expected four candidate judgments"
                )
            selected_groups.append((document_id, sentence_id))
            group_manifest.append(
                {
                    "document_id": document_id,
                    "passage_id": group[0]["passage_id"],
                    "sentence_id": sentence_id,
                    "position": group[0]["position"],
                    "previously_inspected": (document_id, sentence_id) in PILOT_GROUPS,
                    "evaluation_ids": [row["evaluation_id"] for row in group],
                }
            )

    selected_group_set = set(selected_groups)
    evaluation_records = [
        row
        for row in records
        if (row["document_id"], row["sentence_id"]) in selected_group_set
    ]
    if len(evaluation_records) != 96:
        raise ExperimentError(f"expected 96 evaluation judgments, got {len(evaluation_records)}")
    pilot_ids = {
        row["evaluation_id"]
        for row in evaluation_records
        if (row["document_id"], row["sentence_id"]) in PILOT_GROUPS
    }
    if len(pilot_ids) != 12:
        raise ExperimentError(f"expected 12 previously inspected judgments, got {len(pilot_ids)}")

    selected_demo_ids = [value[0] for value in DEMONSTRATIONS]
    demonstration_records = [by_id[evaluation_id] for evaluation_id in selected_demo_ids]
    if [row["evaluation_id"] for row in demonstration_records] != selected_demo_ids:
        raise ExperimentError("demonstration order changed")
    if any(row["document_id"] not in demonstration_documents for row in demonstration_records):
        raise ExperimentError("a demonstration is outside the reserved documents")
    demo_sentence_keys = {
        (row["document_id"], row["sentence_id"]) for row in demonstration_records
    }
    if len(demo_sentence_keys) != len(demonstration_records):
        raise ExperimentError("demonstrations must come from eight distinct source sentences")

    blind_rows = [blind_row(row) for row in evaluation_records]
    if any(set(row) != INPUT_FIELDS for row in blind_rows):
        raise ExperimentError("evaluation input contains a non-blind field")
    evaluation_text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in blind_rows
    )

    stage1_appendix, stage2_appendix = demonstration_appendices(demonstration_records)
    demonstrations_payload = {
        "schema_version": "1.0",
        "demonstration_set_id": "gemma31b-g1-eight-shot-v1",
        "count": 8,
        "model_visible_fields_only": True,
        "stage1_appendix": stage1_appendix,
        "stage1_appendix_sha256": sha256_text(stage1_appendix),
        "stage2_appendix": stage2_appendix,
        "stage2_appendix_sha256": sha256_text(stage2_appendix),
    }
    demonstrations_text = stable_json(demonstrations_payload, pretty=True)

    demo_exact_hashes: set[str] = set()
    demo_normalized_hashes: set[str] = set()
    for row in demonstration_records:
        for text in (row["current_source_bp"], row["candidate"]["text"]):
            demo_exact_hashes.add(sha256_text(text))
            demo_normalized_hashes.add(sha256_text(normalized_text(text)))
    evaluation_exact_hashes = {
        sha256_text(text)
        for row in blind_rows
        for text in (row["current_source_bp"], row["candidate_text"])
    }
    evaluation_normalized_hashes = {
        sha256_text(normalized_text(text))
        for row in blind_rows
        for text in (row["current_source_bp"], row["candidate_text"])
    }
    exact_overlap = sorted(demo_exact_hashes & evaluation_exact_hashes)
    normalized_overlap = sorted(demo_normalized_hashes & evaluation_normalized_hashes)
    if exact_overlap or normalized_overlap:
        raise ExperimentError(
            "cross-pool text duplication detected: "
            f"exact={len(exact_overlap)} normalized={len(normalized_overlap)}"
        )

    prompt_hashes = {
        "stage1": sha256_bytes((PROMPT_DIR / "stage1_A.md").read_bytes()),
        "stage2": sha256_bytes((PROMPT_DIR / "stage2_score.md").read_bytes()),
    }
    demonstration_manifest = []
    for order, (evaluation_id, category, coverage) in enumerate(DEMONSTRATIONS, 1):
        row = by_id[evaluation_id]
        errors = demonstration_errors(row)
        demonstration_manifest.append(
            {
                "order": order,
                "evaluation_id": evaluation_id,
                "document_id": row["document_id"],
                "passage_id": row["passage_id"],
                "sentence_id": row["sentence_id"],
                "position": row["position"],
                "coverage_category": category,
                "coverage_note": coverage,
                "score": row["human_gold"]["score"],
                "minor_error_count": sum(e["severity"] == "Minor" for e in errors),
                "major_error_count": sum(e["severity"] == "Major" for e in errors),
                "missing_error_count": sum(e["text"] == "[MISSING]" for e in errors),
            }
        )

    manifest = {
        "schema_version": "1.0",
        "experiment_id": "gemma31b-g0-g1-dev-v1",
        "source_split": "DEV",
        "selection_seed": SEED,
        "selection_procedure": {
            "demonstration_documents": (
                "exclude documents containing the three inspected pilot groups, then sort "
                "eligible document IDs by SHA-256(seed:document_id)"
            ),
            "evaluation_groups": (
                "retain each inspected pilot group and fill to three groups per document by "
                "SHA-256(seed:evaluation:document_id:sentence_id); emit canonical order"
            ),
            "demonstrations": (
                "manual coverage selection from reserved documents only, frozen before inference"
            ),
        },
        "model": {
            "model_id": "google/gemma-4-31B-it",
            "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "dtype": "bfloat16",
            "quantization": None,
            "thinking": False,
            "deterministic": True,
        },
        "prompt_hashes": prompt_hashes,
        "demonstration_documents": demonstration_documents,
        "evaluation_documents": evaluation_documents,
        "demonstrations": demonstration_manifest,
        "evaluation_groups": group_manifest,
        "evaluation_ids": [row["evaluation_id"] for row in evaluation_records],
        "previously_inspected_evaluation_ids": sorted(pilot_ids),
        "new_evaluation_ids": [
            row["evaluation_id"]
            for row in evaluation_records
            if row["evaluation_id"] not in pilot_ids
        ],
        "counts": {
            "source_documents": len(document_order),
            "demonstration_documents": len(demonstration_documents),
            "evaluation_documents": len(evaluation_documents),
            "demonstrations": len(demonstration_records),
            "demonstration_source_sentences": len(demo_sentence_keys),
            "evaluation_source_sentences": len(selected_groups),
            "evaluation_judgments": len(evaluation_records),
            "previously_inspected_judgments": len(pilot_ids),
            "new_judgments": len(evaluation_records) - len(pilot_ids),
            "evaluation_by_system": dict(
                Counter(row["candidate"]["system_id"] for row in evaluation_records)
            ),
        },
        "artifact_hashes": {
            "canonical_dev_sha256": sha256_bytes(canonical_bytes),
            "blind_evaluation_input_sha256": sha256_text(evaluation_text),
            "demonstration_payload_sha256": sha256_text(demonstrations_text),
            "stage1_demonstration_appendix_sha256": sha256_text(stage1_appendix),
            "stage2_demonstration_appendix_sha256": sha256_text(stage2_appendix),
        },
        "demonstration_text_hashes": {
            "exact_sha256": sorted(demo_exact_hashes),
            "normalized_nfkc_casefold_whitespace_sha256": sorted(demo_normalized_hashes),
        },
        "leakage_audit": {
            "document_disjoint": "PASS",
            "demonstration_source_sentence_absent_from_evaluation": "PASS",
            "demonstration_candidate_absent_from_evaluation": "PASS",
            "exact_cross_pool_text_duplicates": 0,
            "normalized_cross_pool_text_duplicates": 0,
            "evaluation_gold_fields_in_blind_input": 0,
            "demonstration_source_sentences_unique": "PASS",
            "four_candidates_per_evaluation_sentence": "PASS",
            "shared_g0_g1_evaluation_input": "PASS",
            "test_data_loaded": False,
        },
    }
    manifest_text = stable_json(manifest, pretty=True)
    assert_frozen_or_write(args.evaluation_input, evaluation_text)
    assert_frozen_or_write(args.demonstrations, demonstrations_text)
    assert_frozen_or_write(args.manifest, manifest_text)
    print(
        "FROZEN split=DEV demo_docs="
        + ",".join(demonstration_documents)
        + " eval_docs="
        + ",".join(evaluation_documents)
    )
    print(
        f"FROZEN demonstrations=8 evaluation_groups=24 evaluation_rows=96 "
        f"pilot=12 new=84 seed={SEED}"
    )
    print(
        "LEAKAGE_AUDIT PASS document_overlap=0 exact_text_overlap=0 "
        "normalized_text_overlap=0 evaluation_gold_fields=0 TEST_loaded=false"
    )


def audit_frozen(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    demonstrations_bytes = args.demonstrations.read_bytes()
    evaluation_bytes = args.evaluation_input.read_bytes()
    rows = read_jsonl(args.evaluation_input)
    if manifest.get("source_split") != "DEV":
        raise ExperimentError("manifest is not DEV")
    if len(rows) != 96 or len({row.get("evaluation_id") for row in rows}) != 96:
        raise ExperimentError("blind evaluation input must contain 96 unique rows")
    if any(set(row) != INPUT_FIELDS for row in rows):
        raise ExperimentError("blind evaluation input contains extra or missing fields")
    if [row["evaluation_id"] for row in rows] != manifest.get("evaluation_ids"):
        raise ExperimentError("blind input IDs/order differ from frozen manifest")
    if {row["document_id"] for row in rows} != set(manifest["evaluation_documents"]):
        raise ExperimentError("blind input documents differ from frozen manifest")
    hashes = manifest.get("artifact_hashes", {})
    if sha256_bytes(evaluation_bytes) != hashes.get("blind_evaluation_input_sha256"):
        raise ExperimentError("blind evaluation input hash mismatch")
    if sha256_bytes(demonstrations_bytes) != hashes.get("demonstration_payload_sha256"):
        raise ExperimentError("demonstration payload hash mismatch")
    demonstrations = json.loads(demonstrations_bytes)
    if demonstrations.get("count") != 8 or demonstrations.get("model_visible_fields_only") is not True:
        raise ExperimentError("demonstration payload metadata invalid")
    for stage in ("stage1", "stage2"):
        appendix = demonstrations.get(f"{stage}_appendix")
        expected = demonstrations.get(f"{stage}_appendix_sha256")
        if not isinstance(appendix, str) or sha256_text(appendix) != expected:
            raise ExperimentError(f"{stage} demonstration appendix hash mismatch")
        if expected != hashes.get(f"{stage}_demonstration_appendix_sha256"):
            raise ExperimentError(f"{stage} demonstration manifest hash mismatch")
    if set(manifest["demonstration_documents"]) & set(manifest["evaluation_documents"]):
        raise ExperimentError("demonstration/evaluation document overlap")
    demo_exact = set(manifest["demonstration_text_hashes"]["exact_sha256"])
    demo_normalized = set(
        manifest["demonstration_text_hashes"][
            "normalized_nfkc_casefold_whitespace_sha256"
        ]
    )
    eval_exact = {
        sha256_text(text)
        for row in rows
        for text in (row["current_source_bp"], row["candidate_text"])
    }
    eval_normalized = {
        sha256_text(normalized_text(text))
        for row in rows
        for text in (row["current_source_bp"], row["candidate_text"])
    }
    if demo_exact & eval_exact or demo_normalized & eval_normalized:
        raise ExperimentError("frozen cross-pool text duplication detected")
    print(
        "LEAKAGE_AUDIT PASS demo_documents="
        + ",".join(manifest["demonstration_documents"])
        + " evaluation_documents="
        + ",".join(manifest["evaluation_documents"])
    )
    print(
        "LEAKAGE_AUDIT PASS demonstrations=8 evaluation_groups=24 evaluation_rows=96 "
        "document_overlap=0 exact_text_overlap=0 normalized_text_overlap=0 "
        "evaluation_gold_fields=0 shared_G0_G1_ids=96 TEST_loaded=false"
    )


def main() -> None:
    args = parse_args()
    try:
        if args.build:
            build(args)
        else:
            audit_frozen(args)
    except (ExperimentError, OSError, json.JSONDecodeError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
