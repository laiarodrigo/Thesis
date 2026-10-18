#!/usr/bin/env python3
"""Freeze and audit the leakage-controlled G2/G3/G4 Gemma 31B experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_gemma31b_g0_g1 import (
    ExperimentError,
    blind_row,
    demonstration_errors,
    evaluation_data,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_text,
    stable_json,
)
from run_evaluator_v0 import INPUT_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "data/benchmark/dev_01_evaluator.json"
G01_INPUT_PATH = PROJECT_ROOT / "data/benchmark/gemma31b_g0_g1_eval_96.jsonl"
G01_MANIFEST_PATH = PROJECT_ROOT / "data/benchmark/gemma31b_g0_g1_manifest.json"
EVALUATION_PATH = PROJECT_ROOT / "data/benchmark/gemma31b_g2_g4_eval_72.jsonl"
DEMONSTRATIONS_PATH = (
    PROJECT_ROOT / "data/benchmark/gemma31b_g2_g4_demonstrations.json"
)
MANIFEST_PATH = PROJECT_ROOT / "data/benchmark/gemma31b_g2_g4_manifest.json"
PROMPT_DIR = PROJECT_ROOT / "guidelines/v1"

STAGE1_IDS = (
    "dev-v0-0374", "dev-v0-0373", "dev-v0-0316", "dev-v0-0314",
    "dev-v0-0323", "dev-v0-0319", "dev-v0-0341", "dev-v0-0359",
)
STAGE2_IDS = (
    "dev-v0-0374", "dev-v0-0373", "dev-v0-0316", "dev-v0-0314",
    "dev-v0-0363", "dev-v0-0319", "dev-v0-0380", "dev-v0-0338",
)
SHARED_IDS = (
    "dev-v0-0374", "dev-v0-0373", "dev-v0-0316", "dev-v0-0314",
    "dev-v0-0319",
)
DEMONSTRATION_DOCUMENTS = ("ted2020-946", "ted2020-1886")
EVALUATION_DOCUMENTS = (
    "ted2020-18219", "ted2020-2433", "ted2020-24360",
    "ted2020-2609", "ted2020-1663", "ted2020-2860",
)
EVALUATION_RANGES = (
    (13, 20), (29, 32), (81, 84), (101, 104), (109, 112),
    (165, 172), (177, 180), (201, 208), (217, 220),
    (237, 240), (253, 256), (269, 272), (285, 288),
    (293, 296), (305, 308),
)
EXPECTED_EVALUATION_IDS = tuple(
    f"dev-v0-{number:04d}"
    for first, last in EVALUATION_RANGES
    for number in range(first, last + 1)
)
PILOT_IDS = tuple(
    [f"dev-v0-{number:04d}" for number in range(165, 169)]
    + [f"dev-v0-{number:04d}" for number in range(205, 209)]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--audit-frozen", action="store_true")
    parser.add_argument("--canonical", type=Path, default=CANONICAL_PATH)
    parser.add_argument("--g0-g1-input", type=Path, default=G01_INPUT_PATH)
    parser.add_argument("--g0-g1-manifest", type=Path, default=G01_MANIFEST_PATH)
    parser.add_argument("--evaluation-input", type=Path, default=EVALUATION_PATH)
    parser.add_argument("--demonstrations", type=Path, default=DEMONSTRATIONS_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def _assert_frozen_or_write(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ExperimentError(f"frozen artifact differs from regenerated content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _appendix(rows: list[dict[str, Any]], *, include_scores: bool) -> str:
    lines = ["# Human-labeled demonstrations"]
    for index, row in enumerate(rows, 1):
        errors = {"errors": demonstration_errors(row)}
        lines.extend(
            [
                f"## Demonstration {index}",
                evaluation_data(row),
                "CANONICAL HUMAN ERROR JSON:",
                json.dumps(errors, ensure_ascii=False, separators=(",", ":")),
            ]
        )
        if include_scores:
            lines.extend(
                [
                    "CANONICAL HUMAN SCORE JSON:",
                    json.dumps(
                        {"score": row["human_gold"]["score"]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ]
            )
    return "\n\n".join(lines)


def _load_sources(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if "test" in args.canonical.name.casefold():
        raise ExperimentError("refusing a canonical path whose filename contains TEST")
    canonical = read_json(args.canonical)
    if canonical.get("split") != "DEV":
        raise ExperimentError("canonical input must explicitly declare split=DEV")
    records = canonical.get("records")
    if not isinstance(records, list) or len(records) != 388:
        raise ExperimentError("canonical DEV must contain exactly 388 records")
    return canonical, read_jsonl(args.g0_g1_input), read_json(args.g0_g1_manifest)


def _materialize(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    canonical, frozen96, old_manifest = _load_sources(args)
    canonical_records = canonical["records"]
    by_id = {row["evaluation_id"]: row for row in canonical_records}
    if len(by_id) != 388:
        raise ExperimentError("canonical DEV contains duplicate evaluation IDs")
    frozen_by_id = {row["evaluation_id"]: row for row in frozen96}
    if len(frozen_by_id) != 96 or list(frozen_by_id) != old_manifest.get("evaluation_ids"):
        raise ExperimentError("frozen G0/G1 blind input does not match its manifest")
    if set(EXPECTED_EVALUATION_IDS) - set(frozen_by_id):
        raise ExperimentError("requested 72-ID subset is not contained in frozen G0/G1 input")
    evaluation_rows = [frozen_by_id[value] for value in EXPECTED_EVALUATION_IDS]
    if any(set(row) != INPUT_FIELDS for row in evaluation_rows):
        raise ExperimentError("72-row evaluation input is not blind")
    if tuple(dict.fromkeys(row["document_id"] for row in evaluation_rows)) != EVALUATION_DOCUMENTS:
        raise ExperimentError("72-row document order/provenance differs from the approved set")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in evaluation_rows:
        groups.setdefault((row["document_id"], row["sentence_id"]), []).append(row)
    if len(groups) != 18 or any(len(rows) != 4 for rows in groups.values()):
        raise ExperimentError("evaluation set must contain 18 complete four-candidate groups")
    for row in evaluation_rows:
        if blind_row(by_id[row["evaluation_id"]]) != row:
            raise ExperimentError(f"{row['evaluation_id']}: frozen blind row differs from canonical DEV")

    stage1_rows = [by_id[value] for value in STAGE1_IDS]
    stage2_rows = [by_id[value] for value in STAGE2_IDS]
    demo_docs = {row["document_id"] for row in stage1_rows + stage2_rows}
    if demo_docs != set(DEMONSTRATION_DOCUMENTS):
        raise ExperimentError("demonstrations do not come from exactly the approved documents")
    if demo_docs & set(EVALUATION_DOCUMENTS):
        raise ExperimentError("demonstration/evaluation document leakage")
    if set(STAGE1_IDS) & set(EXPECTED_EVALUATION_IDS) or set(STAGE2_IDS) & set(EXPECTED_EVALUATION_IDS):
        raise ExperimentError("target judgment appears in demonstrations")
    if tuple(value for value in STAGE1_IDS if value in STAGE2_IDS) != SHARED_IDS:
        raise ExperimentError("shared demonstration IDs differ from the approved five")
    for evaluation_id in SHARED_IDS:
        stage1_row = next(row for row in stage1_rows if row["evaluation_id"] == evaluation_id)
        stage2_row = next(row for row in stage2_rows if row["evaluation_id"] == evaluation_id)
        if demonstration_errors(stage1_row) != demonstration_errors(stage2_row):
            raise ExperimentError(f"{evaluation_id}: shared canonical annotations differ")

    stage1_appendix = _appendix(stage1_rows, include_scores=False)
    stage2_appendix = _appendix(stage2_rows, include_scores=True)
    if "CANONICAL HUMAN SCORE JSON:" in stage1_appendix or '"score"' in stage1_appendix:
        raise ExperimentError("Stage 1 appendix leaks human scores")
    expected_scores = [0, 10, 65, 70, 80, 90, 100, 100]
    if sorted(row["human_gold"]["score"] for row in stage2_rows) != expected_scores:
        raise ExperimentError("Stage 2 score distribution differs from the approved design")
    demo_payload = {
        "schema_version": "2.0-stage-specific",
        "demonstration_set_id": "gemma31b-g2-g4-stage-specific-eight-shot-v1",
        "model_visible_fields_only": True,
        "stage1_count": 8,
        "stage1_evaluation_ids": list(STAGE1_IDS),
        "stage1_appendix": stage1_appendix,
        "stage1_appendix_sha256": sha256_text(stage1_appendix),
        "stage2_count": 8,
        "stage2_evaluation_ids": list(STAGE2_IDS),
        "stage2_appendix": stage2_appendix,
        "stage2_appendix_sha256": sha256_text(stage2_appendix),
        "shared_evaluation_ids": list(SHARED_IDS),
    }
    evaluation_text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in evaluation_rows
    )
    demonstrations_text = stable_json(demo_payload, pretty=True)
    prompt_hashes = {
        "stage1": sha256_bytes((PROMPT_DIR / "stage1_A.md").read_bytes()),
        "stage2": sha256_bytes((PROMPT_DIR / "stage2_score.md").read_bytes()),
    }
    def provenance(row: dict[str, Any], *, include_score: bool) -> dict[str, Any]:
        errors = demonstration_errors(row)
        value = {
            "evaluation_id": row["evaluation_id"],
            "document_id": row["document_id"],
            "passage_id": row["passage_id"],
            "sentence_id": row["sentence_id"],
            "position": row["position"],
            "canonical_error_json_sha256": sha256_text(
                json.dumps({"errors": errors}, ensure_ascii=False, separators=(",", ":"))
            ),
        }
        if include_score:
            value["score"] = row["human_gold"]["score"]
        return value

    manifest = {
        "schema_version": "1.0",
        "experiment_id": "gemma31b-g2-g4-dev-v1",
        "source_split": "DEV",
        "model": old_manifest["model"],
        "prompt_hashes": prompt_hashes,
        "demonstration_documents": list(DEMONSTRATION_DOCUMENTS),
        "evaluation_documents": list(EVALUATION_DOCUMENTS),
        "stage1_demonstration_ids": list(STAGE1_IDS),
        "stage2_demonstration_ids": list(STAGE2_IDS),
        "shared_demonstration_ids": list(SHARED_IDS),
        "stage1_demonstrations": [
            provenance(row, include_score=False) for row in stage1_rows
        ],
        "stage2_demonstrations": [
            provenance(row, include_score=True) for row in stage2_rows
        ],
        "evaluation_ids": list(EXPECTED_EVALUATION_IDS),
        "previously_inspected_evaluation_ids": list(PILOT_IDS),
        "new_evaluation_ids": [value for value in EXPECTED_EVALUATION_IDS if value not in PILOT_IDS],
        "evaluation_groups": [
            {
                "document_id": document_id,
                "sentence_id": sentence_id,
                "evaluation_ids": [row["evaluation_id"] for row in rows],
                "previously_inspected": all(row["evaluation_id"] in PILOT_IDS for row in rows),
            }
            for (document_id, sentence_id), rows in groups.items()
        ],
        "counts": {
            "evaluation_documents": 6,
            "demonstration_documents": 2,
            "evaluation_source_sentences": 18,
            "evaluation_judgments": 72,
            "previously_inspected_judgments": 8,
            "new_judgments": 64,
            "stage1_demonstrations": 8,
            "stage2_demonstrations": 8,
            "shared_demonstrations": 5,
        },
        "artifact_hashes": {
            "canonical_dev_sha256": sha256_bytes(args.canonical.read_bytes()),
            "source_g0_g1_input_sha256": sha256_bytes(args.g0_g1_input.read_bytes()),
            "source_g0_g1_manifest_sha256": sha256_bytes(args.g0_g1_manifest.read_bytes()),
            "blind_evaluation_input_sha256": sha256_text(evaluation_text),
            "demonstration_payload_sha256": sha256_text(demonstrations_text),
            "stage1_demonstration_appendix_sha256": sha256_text(stage1_appendix),
            "stage2_demonstration_appendix_sha256": sha256_text(stage2_appendix),
        },
        "leakage_audit": {
            "document_disjoint": "PASS",
            "target_ids_absent_from_demonstrations": "PASS",
            "evaluation_gold_fields_in_blind_input": 0,
            "stage1_score_fields": 0,
            "system_identity_fields": 0,
            "human_comment_fields": 0,
            "test_data_loaded": False,
        },
    }
    return evaluation_text, demonstrations_text, manifest


def build(args: argparse.Namespace) -> None:
    evaluation_text, demonstrations_text, manifest = _materialize(args)
    _assert_frozen_or_write(args.evaluation_input, evaluation_text)
    _assert_frozen_or_write(args.demonstrations, demonstrations_text)
    _assert_frozen_or_write(args.manifest, stable_json(manifest, pretty=True))
    print("FROZEN G2/G3/G4 evaluation_rows=72 groups=18 pilot=8 new=64")
    print("LEAKAGE_AUDIT PASS demonstration_docs=2 evaluation_docs=6 TEST_loaded=false")


def audit(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    rows = read_jsonl(args.evaluation_input)
    demo_bytes = args.demonstrations.read_bytes()
    demonstrations = json.loads(demo_bytes)
    hashes = manifest.get("artifact_hashes", {})
    if manifest.get("source_split") != "DEV":
        raise ExperimentError("frozen manifest is not DEV")
    if len(rows) != 72 or len({row.get("evaluation_id") for row in rows}) != 72:
        raise ExperimentError("blind evaluation input must contain 72 unique rows")
    if [row["evaluation_id"] for row in rows] != manifest.get("evaluation_ids"):
        raise ExperimentError("blind input IDs/order differ from frozen manifest")
    if any(set(row) != INPUT_FIELDS for row in rows):
        raise ExperimentError("blind evaluation input contains extra or missing fields")
    if sha256_bytes(args.evaluation_input.read_bytes()) != hashes.get("blind_evaluation_input_sha256"):
        raise ExperimentError("blind evaluation input hash mismatch")
    if sha256_bytes(demo_bytes) != hashes.get("demonstration_payload_sha256"):
        raise ExperimentError("demonstration payload hash mismatch")
    if demonstrations.get("stage1_evaluation_ids") != manifest.get("stage1_demonstration_ids"):
        raise ExperimentError("Stage 1 demonstration IDs differ from manifest")
    if demonstrations.get("stage2_evaluation_ids") != manifest.get("stage2_demonstration_ids"):
        raise ExperimentError("Stage 2 demonstration IDs differ from manifest")
    if demonstrations.get("shared_evaluation_ids") != manifest.get("shared_demonstration_ids"):
        raise ExperimentError("shared demonstration IDs differ from manifest")
    for stage in ("stage1", "stage2"):
        appendix = demonstrations.get(f"{stage}_appendix")
        expected = demonstrations.get(f"{stage}_appendix_sha256")
        if not isinstance(appendix, str) or sha256_text(appendix) != expected:
            raise ExperimentError(f"{stage} demonstration appendix hash mismatch")
        if expected != hashes.get(f"{stage}_demonstration_appendix_sha256"):
            raise ExperimentError(f"{stage} appendix differs from manifest")
    if "CANONICAL HUMAN SCORE JSON:" in demonstrations["stage1_appendix"] or '"score"' in demonstrations["stage1_appendix"]:
        raise ExperimentError("Stage 1 appendix leaks scores")
    if set(manifest["demonstration_documents"]) & set(manifest["evaluation_documents"]):
        raise ExperimentError("demonstration/evaluation document overlap")
    if {row["document_id"] for row in rows} != set(manifest["evaluation_documents"]):
        raise ExperimentError("blind input document provenance differs from manifest")
    groups = {(row["document_id"], row["sentence_id"]) for row in rows}
    if len(groups) != 18:
        raise ExperimentError("blind input does not contain 18 source-sentence groups")
    print("LEAKAGE_AUDIT PASS rows=72 groups=18 stage1_demos=8 stage2_demos=8 shared=5")
    print("LEAKAGE_AUDIT PASS document_overlap=0 target_gold=0 comments=0 system_identity=0 TEST_loaded=false")


def main() -> None:
    args = parse_args()
    try:
        build(args) if args.build else audit(args)
    except (ExperimentError, OSError, json.JSONDecodeError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
