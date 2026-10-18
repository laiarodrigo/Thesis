#!/usr/bin/env python3
"""Run strict deterministic two-stage V1 evaluator variants."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

from run_evaluator_v0 import (
    INPUT_FIELDS,
    MODEL_ID,
    MODEL_REVISION,
    _candidate_occurrences,
    _has_stoppable_complete_json_object,
    _json_safe,
    _load_completed,
    _normalize_json_representation,
    _read_jsonl,
    _validate_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_DIR = PROJECT_ROOT / "guidelines/v1"
PROMPT_STRUCTURE_VERSION = "1.0-two-stage-original-input-validated-errors"
OUTPUT_SCHEMA_VERSION = "1.0-v1-two-stage"
OUTPUT_PARSER_VERSION = "1.0-v1-strict"
STAGE1_STANDARD_SCHEMA_VERSION = "1.0-candidate-errors"
STAGE1_OMISSION_SCHEMA_VERSION = "1.0-candidate-errors-source-omissions"
STAGE2_SCHEMA_VERSION = "1.0-score-only"
CHANNEL_OUTPUT_PARSER_VERSION = "1.2-v1-channel-aware-prefix"
LARGER_MODEL_ID = "google/gemma-4-31B-it"
LARGER_MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
FEW_SHOT_PROMPT_STRUCTURE_VERSION = (
    "1.1-two-stage-original-input-validated-errors-eight-shot"
)
STAGE_SPECIFIC_PROMPT_STRUCTURE_VERSION = (
    "1.2-two-stage-original-input-validated-errors-stage-specific-eight-shot"
)
PROMPT_FILES = {
    "A": {"stage1": "stage1_A.md", "stage2": "stage2_score.md"},
    "B": {"stage1": "stage1_B.md", "stage2": "stage2_score.md"},
    "C": {"stage1": "stage1_C.md", "stage2": "stage2_score.md"},
    "E": {
        "stage1_accuracy": "stage1_E_accuracy.md",
        "stage1_fluency": "stage1_E_fluency.md",
        "stage2": "stage2_score.md",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-id",
        choices=("V1", "F1", "F2", "G0", "G1", "G24S1", "G2", "G3", "G4"),
        default="V1",
    )
    parser.add_argument("--variant", choices=sorted(PROMPT_FILES), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--demonstrations", type=Path)
    parser.add_argument(
        "--stage1-source",
        type=Path,
        help="validated predictions whose Stage 1 artifact is reused verbatim",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="run and save Stage 1 only (used once for the shared G2/G4 dependency)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--stage1-max-new-tokens", type=int)
    parser.add_argument("--stage2-max-new-tokens", type=int)
    parser.add_argument("--dtype", choices=("auto", "bfloat16"), default="auto")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--evaluation-id",
        action="append",
        default=[],
        help="run only this evaluation ID; repeat for a smoke subset",
    )
    parser.add_argument("--print-protocol", action="store_true")
    return parser.parse_args()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_prompts(prompt_dir: Path, variant: str) -> tuple[dict[str, str], dict[str, str]]:
    prompts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for stage_name, filename in PROMPT_FILES[variant].items():
        path = prompt_dir / filename
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"ERROR: could not read {stage_name} prompt {path}: {exc}") from exc
        if not text.strip():
            raise SystemExit(f"ERROR: empty {stage_name} prompt: {path}")
        prompts[stage_name] = text
        hashes[stage_name] = _sha256_text(text)
    return prompts, hashes


def _load_demonstrations(path: Path) -> tuple[dict[str, str], dict[str, str], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: could not read demonstrations {path}: {exc}") from exc
    g1_keys = {
        "schema_version",
        "demonstration_set_id",
        "count",
        "model_visible_fields_only",
        "stage1_appendix",
        "stage1_appendix_sha256",
        "stage2_appendix",
        "stage2_appendix_sha256",
    }
    stage_specific_keys = {
        "schema_version",
        "demonstration_set_id",
        "model_visible_fields_only",
        "stage1_count",
        "stage1_evaluation_ids",
        "stage1_appendix",
        "stage1_appendix_sha256",
        "stage2_count",
        "stage2_evaluation_ids",
        "stage2_appendix",
        "stage2_appendix_sha256",
        "shared_evaluation_ids",
    }
    if not isinstance(payload, dict) or set(payload) not in (g1_keys, stage_specific_keys):
        raise SystemExit("ERROR: demonstration payload schema is not frozen")
    if payload.get("model_visible_fields_only") is not True:
        raise SystemExit("ERROR: demonstrations must contain model-visible fields only")
    if set(payload) == g1_keys:
        if payload.get("count") != 8:
            raise SystemExit("ERROR: G1 requires exactly eight demonstrations")
    else:
        if payload.get("stage1_count") != 8 or payload.get("stage2_count") != 8:
            raise SystemExit("ERROR: stage-specific experiments require eight demonstrations per stage")
        stage1_ids = payload.get("stage1_evaluation_ids")
        stage2_ids = payload.get("stage2_evaluation_ids")
        shared_ids = payload.get("shared_evaluation_ids")
        if (
            not isinstance(stage1_ids, list)
            or not isinstance(stage2_ids, list)
            or not isinstance(shared_ids, list)
            or len(stage1_ids) != 8
            or len(stage2_ids) != 8
            or len(shared_ids) != 5
            or set(stage1_ids) & set(stage2_ids) != set(shared_ids)
        ):
            raise SystemExit("ERROR: stage-specific demonstration ID provenance is invalid")
    appendices: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for stage in ("stage1", "stage2"):
        appendix = payload.get(f"{stage}_appendix")
        expected_hash = payload.get(f"{stage}_appendix_sha256")
        if not isinstance(appendix, str) or not appendix.strip():
            raise SystemExit(f"ERROR: missing {stage} demonstration appendix")
        observed_hash = _sha256_text(appendix)
        if observed_hash != expected_hash:
            raise SystemExit(f"ERROR: {stage} demonstration appendix hash mismatch")
        appendices[stage] = appendix
        hashes[stage] = observed_hash
    return appendices, hashes, hashlib.sha256(raw).hexdigest()


def _evaluation_data(row: dict[str, Any]) -> str:
    context = "\n".join(
        f"[{sentence['position']}] {sentence['source_bp']}"
        for sentence in row["passage_context"]
    )
    return (
        "# Evaluation data\n\n"
        "PORTUGUESE PASSAGE CONTEXT (BP — REFERENCE ONLY):\n"
        f"{context}\n\n"
        f"CURRENT BP SOURCE SENTENCE (REFERENCE ONLY, position {row['position']}):\n"
        f"{row['current_source_bp']}\n\n"
        "EP CANDIDATE TO EVALUATE (ONLY TEXT TO ANNOTATE):\n"
        f"{row['candidate_text']}"
    )


def _render_stage1(
    prompt: str, row: dict[str, Any], demonstration_appendix: str | None = None
) -> str:
    parts = [prompt.rstrip()]
    if demonstration_appendix is not None:
        parts.append(demonstration_appendix.rstrip())
    parts.append(_evaluation_data(row))
    return "\n\n".join(parts)


def _stage2_annotations(error_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for span in error_spans:
        annotation = {"text": span["text"], "severity": span["severity"]}
        if "occurrence" in span:
            annotation["occurrence"] = span["occurrence"]
        annotations.append(annotation)
    return annotations


def _render_stage2(
    prompt: str,
    row: dict[str, Any],
    error_spans: list[dict[str, Any]],
    demonstration_appendix: str | None = None,
) -> str:
    annotations = {"errors": _stage2_annotations(error_spans)}
    parts = [prompt.rstrip()]
    if demonstration_appendix is not None:
        parts.append(demonstration_appendix.rstrip())
    parts.extend(
        [
            _evaluation_data(row),
            "# Validated Stage 1 candidate-side annotations\n\n"
            f"{json.dumps(annotations, ensure_ascii=False, separators=(',', ':'))}",
        ]
    )
    return "\n\n".join(parts)


def _decode_object(
    raw: str, *, stopped_after_complete_json_object: bool
) -> tuple[dict[str, Any] | None, list[str], list[str], dict[str, Any]]:
    text, format_status, format_errors = _normalize_json_representation(
        raw,
        stopped_after_complete_json_object=stopped_after_complete_json_object,
    )
    warnings = (
        ["markdown_wrapper_normalized"]
        if format_status["markdown_wrapper_normalized"]
        else []
    )
    if format_errors or text is None:
        return None, format_errors, warnings, format_status
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"invalid_json: {exc}"], warnings, format_status
    if not isinstance(value, dict):
        return None, ["top_level_must_be_object"], warnings, format_status
    return value, [], warnings, format_status


def _parse_candidate_error_list(
    raw_spans: Any,
    candidate: str,
    *,
    allow_missing: bool,
    label: str = "errors",
    allow_multiple_missing: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_spans, list):
        return [], [f"{label}_must_be_list"]
    for index, span in enumerate(raw_spans):
        prefix = f"{label}[{index}]"
        if not isinstance(span, dict):
            errors.append(f"{prefix}_must_be_object")
            continue
        if set(span) not in ({"text", "severity"}, {"text", "severity", "occurrence"}):
            errors.append(f"{prefix}_keys_invalid")
        marked_text = span.get("text")
        severity = span.get("severity")
        has_occurrence = "occurrence" in span
        occurrence = span.get("occurrence")
        if severity not in {"Minor", "Major"}:
            errors.append(f"{prefix}_severity_invalid")
        occurrence_valid = not has_occurrence or (
            isinstance(occurrence, int)
            and not isinstance(occurrence, bool)
            and occurrence >= 1
        )
        if not occurrence_valid:
            errors.append(f"{prefix}_occurrence_must_be_positive_integer")
        start: int | None = None
        end: int | None = None
        missing = marked_text == "[MISSING]"
        if not isinstance(marked_text, str) or not marked_text:
            errors.append(f"{prefix}_text_must_be_nonempty_string")
        elif missing:
            if not allow_missing:
                errors.append(f"{prefix}_missing_not_allowed")
            elif has_occurrence:
                errors.append(f"{prefix}_occurrence_not_allowed_for_missing")
            else:
                start = end = len(candidate)
        else:
            occurrences = _candidate_occurrences(candidate, marked_text)
            if not occurrences:
                errors.append(f"{prefix}_text_not_in_candidate")
            elif has_occurrence:
                if occurrence_valid:
                    if occurrence > len(occurrences):
                        errors.append(f"{prefix}_occurrence_out_of_range")
                    else:
                        start = occurrences[occurrence - 1]
                        end = start + len(marked_text)
            elif len(occurrences) > 1:
                errors.append(f"{prefix}_text_ambiguous_in_candidate")
            else:
                start = occurrences[0]
                end = start + len(marked_text)
        if (
            start is not None
            and end is not None
            and severity in {"Minor", "Major"}
            and isinstance(marked_text, str)
        ):
            item: dict[str, Any] = {
                "start": start,
                "end": end,
                "text": marked_text,
                "severity": severity,
                "missing": missing,
            }
            if has_occurrence:
                item["occurrence"] = occurrence
            normalized.append(item)

    ordered = sorted(normalized, key=lambda item: (item["start"], item["end"]))
    for previous, current in zip(ordered, ordered[1:]):
        same_missing_anchor = (
            previous["missing"]
            and current["missing"]
            and previous["start"] == current["start"]
        )
        if same_missing_anchor and allow_multiple_missing:
            continue
        if current["start"] < previous["end"] or (
            current["start"] == previous["start"]
            and current["end"] == previous["end"]
        ):
            errors.append(f"{label}_overlap_or_duplicate")
            break
    return normalized, errors


def _parse_stage1_standard(
    raw: str,
    candidate: str,
    *,
    stopped_after_complete_json_object: bool,
    allow_missing: bool,
) -> tuple[dict[str, Any] | None, list[str], list[str], dict[str, Any]]:
    value, errors, warnings, format_status = _decode_object(
        raw,
        stopped_after_complete_json_object=stopped_after_complete_json_object,
    )
    if value is None:
        return None, errors, warnings, format_status
    if set(value) != {"errors"}:
        errors.append("top_level_keys_must_be_exactly_errors")
    spans, span_errors = _parse_candidate_error_list(
        value.get("errors"), candidate, allow_missing=allow_missing
    )
    errors.extend(span_errors)
    if errors:
        return None, errors, warnings, format_status
    return {"error_spans": spans}, [], warnings, format_status


def _parse_source_omissions(
    raw_omissions: Any, source: str, candidate: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    omissions: list[dict[str, Any]] = []
    missing_spans: list[dict[str, Any]] = []
    if not isinstance(raw_omissions, list):
        return [], [], ["omissions_must_be_list"]
    for index, omission in enumerate(raw_omissions):
        prefix = f"omissions[{index}]"
        if not isinstance(omission, dict):
            errors.append(f"{prefix}_must_be_object")
            continue
        if set(omission) not in (
            {"source_text", "severity"},
            {"source_text", "severity", "occurrence"},
        ):
            errors.append(f"{prefix}_keys_invalid")
        source_text = omission.get("source_text")
        severity = omission.get("severity")
        has_occurrence = "occurrence" in omission
        occurrence = omission.get("occurrence")
        if severity not in {"Minor", "Major"}:
            errors.append(f"{prefix}_severity_invalid")
        occurrence_valid = not has_occurrence or (
            isinstance(occurrence, int)
            and not isinstance(occurrence, bool)
            and occurrence >= 1
        )
        if not occurrence_valid:
            errors.append(f"{prefix}_occurrence_must_be_positive_integer")
        start: int | None = None
        end: int | None = None
        if not isinstance(source_text, str) or not source_text:
            errors.append(f"{prefix}_source_text_must_be_nonempty_string")
        else:
            occurrences = _candidate_occurrences(source, source_text)
            if not occurrences:
                errors.append(f"{prefix}_source_text_not_in_current_source")
            elif has_occurrence:
                if occurrence_valid:
                    if occurrence > len(occurrences):
                        errors.append(f"{prefix}_occurrence_out_of_range")
                    else:
                        start = occurrences[occurrence - 1]
                        end = start + len(source_text)
            elif len(occurrences) > 1:
                errors.append(f"{prefix}_source_text_ambiguous_in_current_source")
            else:
                start = occurrences[0]
                end = start + len(source_text)
        if (
            start is not None
            and end is not None
            and severity in {"Minor", "Major"}
            and isinstance(source_text, str)
        ):
            normalized: dict[str, Any] = {
                "source_start": start,
                "source_end": end,
                "source_text": source_text,
                "severity": severity,
                "candidate_anchor": len(candidate),
            }
            if has_occurrence:
                normalized["source_occurrence"] = occurrence
            omissions.append(normalized)
            missing_spans.append(
                {
                    "start": len(candidate),
                    "end": len(candidate),
                    "text": "[MISSING]",
                    "severity": severity,
                    "missing": True,
                    "source_start": start,
                    "source_end": end,
                    "source_text": source_text,
                    **(
                        {"source_occurrence": occurrence}
                        if has_occurrence
                        else {}
                    ),
                }
            )
    ordered = sorted(omissions, key=lambda item: (item["source_start"], item["source_end"]))
    for previous, current in zip(ordered, ordered[1:]):
        if current["source_start"] < previous["source_end"] or (
            current["source_start"] == previous["source_start"]
            and current["source_end"] == previous["source_end"]
        ):
            errors.append("omissions_overlap_or_duplicate")
            break
    return omissions, missing_spans, errors


def _parse_stage1_c(
    raw: str,
    candidate: str,
    source: str,
    *,
    stopped_after_complete_json_object: bool,
) -> tuple[dict[str, Any] | None, list[str], list[str], dict[str, Any]]:
    value, errors, warnings, format_status = _decode_object(
        raw,
        stopped_after_complete_json_object=stopped_after_complete_json_object,
    )
    if value is None:
        return None, errors, warnings, format_status
    if set(value) != {"errors", "omissions"}:
        errors.append("top_level_keys_must_be_exactly_errors_and_omissions")
    candidate_spans, candidate_errors = _parse_candidate_error_list(
        value.get("errors"), candidate, allow_missing=False
    )
    omissions, missing_spans, omission_errors = _parse_source_omissions(
        value.get("omissions"), source, candidate
    )
    errors.extend(candidate_errors)
    errors.extend(omission_errors)
    if errors:
        return None, errors, warnings, format_status
    return {
        "candidate_error_spans": candidate_spans,
        "omissions": omissions,
        "error_spans": candidate_spans + missing_spans,
        "multiple_omission_representation": (
            "one preserved [MISSING] span per verified source omission at the candidate-end anchor"
        ),
    }, [], warnings, format_status


def _combine_e_passes(
    accuracy_spans: list[dict[str, Any]], fluency_spans: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    for accuracy in accuracy_spans:
        for fluency in fluency_spans:
            same = accuracy["start"] == fluency["start"] and accuracy["end"] == fluency["end"]
            overlaps = (
                max(accuracy["start"], fluency["start"])
                < min(accuracy["end"], fluency["end"])
            )
            same_missing = accuracy["missing"] and fluency["missing"] and same
            if same or same_missing:
                kind = (
                    "cross_pass_duplicate"
                    if accuracy["severity"] == fluency["severity"]
                    else "cross_pass_severity_conflict"
                )
                errors.append(kind)
            elif overlaps:
                errors.append("cross_pass_overlap")
    if errors:
        return None, sorted(set(errors))
    combined = [
        {**span, "origin_pass": "accuracy_variety"} for span in accuracy_spans
    ] + [
        {**span, "origin_pass": "fluency_conservatism"} for span in fluency_spans
    ]
    combined.sort(
        key=lambda span: (
            span["start"], span["end"],
            0 if span["origin_pass"] == "accuracy_variety" else 1,
        )
    )
    return {"error_spans": combined}, []


def _parse_stage2(
    raw: str,
    error_spans: list[dict[str, Any]],
    *,
    stopped_after_complete_json_object: bool,
) -> tuple[dict[str, Any] | None, list[str], list[str], dict[str, Any]]:
    value, errors, warnings, format_status = _decode_object(
        raw,
        stopped_after_complete_json_object=stopped_after_complete_json_object,
    )
    if value is None:
        return None, errors, warnings, format_status
    if set(value) != {"score"}:
        errors.append("top_level_keys_must_be_exactly_score")
    score = value.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        errors.append("score_must_be_integer_0_to_100")
    elif error_spans and score == 100:
        errors.append("score_must_be_below_100_when_errors_present")
    elif not error_spans and score != 100:
        errors.append("score_must_be_100_when_errors_empty")
    if errors:
        return None, errors, warnings, format_status
    return {"score": score}, [], warnings, format_status


def _stage_with_parse(
    generation: dict[str, Any],
    parsed: tuple[dict[str, Any] | None, list[str], list[str], dict[str, Any]],
    prompt_sha256: str,
) -> dict[str, Any]:
    prediction, errors, warnings, format_status = parsed
    return {
        "prompt_sha256": prompt_sha256,
        **generation,
        **format_status,
        "prediction": prediction,
        "parse_valid": not errors,
        "parse_errors": errors,
        "parse_warnings": warnings,
    }


def _prefixed(prefix: str, values: list[str]) -> list[str]:
    return [f"{prefix}:{value}" for value in values]


def _validate_experiment(args: argparse.Namespace) -> tuple[int, int, str]:
    stage1_max = args.stage1_max_new_tokens or args.max_new_tokens
    stage2_max = args.stage2_max_new_tokens or args.max_new_tokens
    if stage1_max < 1 or stage2_max < 1:
        raise SystemExit("ERROR: stage token budgets must be positive")
    if args.experiment_id == "V1":
        if args.model_id != MODEL_ID or args.revision != MODEL_REVISION:
            raise SystemExit(
                f"ERROR: V1 only permits {MODEL_ID} at revision {MODEL_REVISION}"
            )
        if args.enable_thinking:
            raise SystemExit("ERROR: the frozen V1 configurations have thinking disabled")
        output_parser_version = OUTPUT_PARSER_VERSION
    elif args.experiment_id == "F1":
        if args.variant != "A":
            raise SystemExit("ERROR: F1 must use the frozen V1-A prompt variant")
        if args.model_id != MODEL_ID or args.revision != MODEL_REVISION:
            raise SystemExit(
                f"ERROR: F1 requires {MODEL_ID} at revision {MODEL_REVISION}"
            )
        if not args.enable_thinking:
            raise SystemExit("ERROR: F1 requires --enable-thinking")
        if args.dtype != "auto":
            raise SystemExit("ERROR: F1 preserves the E4B baseline dtype=auto loading")
        output_parser_version = CHANNEL_OUTPUT_PARSER_VERSION
    elif args.experiment_id in {"F2", "G0", "G1", "G24S1", "G2", "G3", "G4"}:
        if args.variant != "A":
            raise SystemExit(
                f"ERROR: {args.experiment_id} must use the frozen V1-A prompt variant"
            )
        if args.model_id != LARGER_MODEL_ID or args.revision != LARGER_MODEL_REVISION:
            raise SystemExit(
                f"ERROR: {args.experiment_id} requires {LARGER_MODEL_ID} at revision "
                f"{LARGER_MODEL_REVISION}"
            )
        if args.enable_thinking:
            raise SystemExit(
                f"ERROR: {args.experiment_id} requires thinking to remain disabled"
            )
        if args.dtype != "bfloat16":
            raise SystemExit(
                f"ERROR: {args.experiment_id} requires unquantized BF16 inference"
            )
        output_parser_version = CHANNEL_OUTPUT_PARSER_VERSION
    else:
        raise SystemExit(f"ERROR: unsupported experiment {args.experiment_id}")
    if args.experiment_id in {"G1", "G24S1", "G2", "G3", "G4"} and args.demonstrations is None:
        raise SystemExit(f"ERROR: {args.experiment_id} requires --demonstrations")
    if args.experiment_id not in {"G1", "G24S1", "G2", "G3", "G4"} and args.demonstrations is not None:
        raise SystemExit("ERROR: demonstrations are not permitted for this experiment")
    if args.experiment_id == "G24S1":
        if not args.stage1_only or args.stage1_source is not None:
            raise SystemExit("ERROR: G24S1 requires --stage1-only and forbids --stage1-source")
    elif args.stage1_only:
        raise SystemExit("ERROR: --stage1-only is reserved for G24S1")
    if args.experiment_id in {"G2", "G3", "G4"} and args.stage1_source is None:
        raise SystemExit(f"ERROR: {args.experiment_id} requires --stage1-source")
    if args.experiment_id not in {"G2", "G3", "G4"} and args.stage1_source is not None:
        raise SystemExit("ERROR: --stage1-source is permitted only for G2/G3/G4")
    return stage1_max, stage2_max, output_parser_version


def _final_segment_for_stopping(raw_special: str, *, thinking_enabled: bool) -> str | None:
    thought_open = "<|channel>thought\n"
    thought_close = "<channel|>"
    if thinking_enabled or thought_open in raw_special:
        if thought_close not in raw_special:
            return None
        return raw_special.split(thought_close, 1)[1]
    return raw_special


def _parse_model_response(
    processor: Any,
    raw_special: str,
    prefix: Any,
    *,
    thinking_enabled: bool,
) -> tuple[str | None, str | None, list[str], list[str], Any]:
    errors: list[str] = []
    warnings: list[str] = []
    thought_open = "<|channel>thought\n"
    thought_close = "<channel|>"
    open_count = raw_special.count(thought_open)
    close_count = raw_special.count(thought_close)
    if thinking_enabled:
        if open_count != 1:
            errors.append(f"thinking_open_delimiter_count_{open_count}")
        if close_count != 1:
            errors.append(f"thinking_close_delimiter_count_{close_count}")
        if open_count == 1 and close_count == 1:
            if raw_special.index(thought_open) > raw_special.index(thought_close):
                errors.append("thinking_delimiters_out_of_order")
    elif open_count or close_count:
        warnings.append("unexpected_thinking_channel_while_disabled")
        if open_count != close_count:
            errors.append("malformed_unexpected_thinking_channel")

    parsed: Any = None
    try:
        parsed = processor.parse_response(raw_special, prefix=prefix)
    except Exception as exc:  # exact processor failure is retained for diagnosis
        errors.append(f"processor_parse_response_failed:{type(exc).__name__}:{exc}")
    content: str | None = None
    thinking: str | None = None
    if isinstance(parsed, dict):
        content_value = parsed.get("content")
        thinking_value = parsed.get("thinking")
        if isinstance(content_value, str) and content_value.strip():
            content = content_value.strip()
        else:
            errors.append("final_answer_missing_or_empty")
        if thinking_value is None:
            thinking = None
        elif isinstance(thinking_value, str):
            thinking = thinking_value
        else:
            errors.append("thinking_output_not_string")
    elif parsed is not None:
        errors.append("processor_parse_response_not_object")
    if thinking_enabled and (thinking is None or not thinking.strip()):
        errors.append("thinking_output_missing_or_empty")
    if not thinking_enabled and isinstance(thinking, str) and thinking.strip():
        warnings.append("nonempty_thinking_output_while_disabled")
    if content is not None and any(
        token in content for token in ("<|channel>", "<channel|>", "<|turn>", "<turn|>")
    ):
        errors.append("special_channel_token_in_final_answer")
    return content, thinking, errors, warnings, parsed


def _response_parse_failure(
    generation: dict[str, Any],
) -> tuple[None, list[str], list[str], dict[str, Any]]:
    return (
        None,
        _prefixed("response", generation["response_parse_errors"]),
        _prefixed("response", generation["response_parse_warnings"]),
        {
            "raw_format_compliant": False,
            "markdown_wrapper_normalized": False,
            "recognized_output_representation": None,
        },
    )


def _load_reusable_stage1(
    path: Path,
    inputs: list[dict[str, Any]],
    *,
    expected_source_experiment: str,
    expected_prompt_sha256: str,
    expected_messages: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load an explicit Stage-1 dependency and prove target/prompt compatibility."""
    rows = _read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        evaluation_id = row.get("evaluation_id")
        if not isinstance(evaluation_id, str):
            raise SystemExit(f"ERROR: Stage 1 source has a row without evaluation_id: {path}")
        if evaluation_id in by_id:
            raise SystemExit(f"ERROR: duplicate Stage 1 source ID {evaluation_id}: {path}")
        by_id[evaluation_id] = row
    required = {row["evaluation_id"] for row in inputs}
    missing = sorted(required - set(by_id))
    if missing:
        raise SystemExit(f"ERROR: Stage 1 source is missing target IDs: {missing}")
    if expected_source_experiment == "G24S1" and set(by_id) != required:
        raise SystemExit("ERROR: shared improved Stage 1 artifact must contain exactly the target IDs")
    for input_row in inputs:
        evaluation_id = input_row["evaluation_id"]
        source_row = by_id[evaluation_id]
        if source_row.get("experiment_id") != expected_source_experiment:
            raise SystemExit(
                f"ERROR: {evaluation_id}: expected Stage 1 source experiment "
                f"{expected_source_experiment}, got {source_row.get('experiment_id')}"
            )
        if source_row.get("variant") != "A":
            raise SystemExit(f"ERROR: {evaluation_id}: reusable Stage 1 is not variant A")
        if source_row.get("output_parser_version") != CHANNEL_OUTPUT_PARSER_VERSION:
            raise SystemExit(f"ERROR: {evaluation_id}: reusable Stage 1 parser is incompatible")
        if source_row.get("stage1_schema_version") != STAGE1_STANDARD_SCHEMA_VERSION:
            raise SystemExit(f"ERROR: {evaluation_id}: reusable Stage 1 schema is incompatible")
        model = source_row.get("model", {})
        if (
            model.get("model_id") != LARGER_MODEL_ID
            or model.get("requested_revision") != LARGER_MODEL_REVISION
        ):
            raise SystemExit(f"ERROR: {evaluation_id}: reusable Stage 1 model/revision differs")
        generation = source_row.get("generation_config", {})
        if (
            generation.get("deterministic") is not True
            or generation.get("do_sample") is not False
            or generation.get("enable_thinking") is not False
            or generation.get("stage1_max_new_tokens") != 512
        ):
            raise SystemExit(f"ERROR: {evaluation_id}: reusable Stage 1 generation differs")
        stage = source_row.get("stages", {}).get("stage1")
        if not isinstance(stage, dict):
            raise SystemExit(f"ERROR: {evaluation_id}: reusable Stage 1 record is absent")
        if stage.get("prompt_sha256") != expected_prompt_sha256:
            raise SystemExit(f"ERROR: {evaluation_id}: reusable Stage 1 prompt hash differs")
        messages = stage.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or messages[0] != {"role": "user", "content": expected_messages[evaluation_id]}
        ):
            raise SystemExit(
                f"ERROR: {evaluation_id}: reusable Stage 1 rendered target/prompt differs"
            )
        prediction = stage.get("prediction")
        if stage.get("parse_valid") is True:
            if not isinstance(prediction, dict) or not isinstance(prediction.get("error_spans"), list):
                raise SystemExit(f"ERROR: {evaluation_id}: valid reusable Stage 1 lacks spans")
        elif prediction is not None:
            raise SystemExit(f"ERROR: {evaluation_id}: invalid reusable Stage 1 has a prediction")
    return {value: by_id[value] for value in required}, hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("ERROR: --max-new-tokens must be positive")
    stage1_max_new_tokens, stage2_max_new_tokens, output_parser_version = (
        _validate_experiment(args)
    )

    all_inputs = _read_jsonl(args.input)
    _validate_inputs(all_inputs)
    if any(set(row) != INPUT_FIELDS for row in all_inputs):
        raise SystemExit("ERROR: V1 input does not match the frozen blind-row schema")
    all_ids = {row["evaluation_id"] for row in all_inputs}
    requested_ids = args.evaluation_id
    if len(requested_ids) != len(set(requested_ids)):
        raise SystemExit("ERROR: duplicate --evaluation-id selection")
    unknown_ids = sorted(set(requested_ids) - all_ids)
    if unknown_ids:
        raise SystemExit(f"ERROR: unknown --evaluation-id values: {unknown_ids}")
    selected_ids = set(requested_ids) if requested_ids else all_ids
    inputs = [row for row in all_inputs if row["evaluation_id"] in selected_ids]
    prompts, prompt_hashes = _load_prompts(args.prompt_dir, args.variant)
    demonstration_appendices: dict[str, str] = {}
    demonstration_hashes: dict[str, str] = {}
    demonstration_payload_sha256: str | None = None
    if args.experiment_id in {"G1", "G24S1", "G2", "G3", "G4"}:
        (
            demonstration_appendices,
            demonstration_hashes,
            demonstration_payload_sha256,
        ) = _load_demonstrations(args.demonstrations)
    if args.experiment_id == "G1":
        prompt_structure_version = FEW_SHOT_PROMPT_STRUCTURE_VERSION
    elif args.experiment_id in {"G24S1", "G2", "G3", "G4"}:
        prompt_structure_version = STAGE_SPECIFIC_PROMPT_STRUCTURE_VERSION
    else:
        prompt_structure_version = PROMPT_STRUCTURE_VERSION
    stage1_demo = args.experiment_id in {"G24S1", "G2", "G4"}
    stage2_demo = args.experiment_id in {"G3", "G4"}
    input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    selection_sha256 = _sha256_text(
        json.dumps(
            [row["evaluation_id"] for row in inputs],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    stage1_schema = (
        STAGE1_OMISSION_SCHEMA_VERSION
        if args.variant == "C"
        else STAGE1_STANDARD_SCHEMA_VERSION
    )
    protocol_payload = {
        "variant": args.variant,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "prompt_hashes": prompt_hashes,
        "prompt_structure_version": prompt_structure_version,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "output_parser_version": output_parser_version,
        "stage1_schema_version": stage1_schema,
        "stage2_schema_version": STAGE2_SCHEMA_VERSION,
        "max_new_tokens": args.max_new_tokens,
        "deterministic": True,
        "enable_thinking": args.enable_thinking,
    }
    if args.experiment_id != "V1":
        protocol_payload.update(
            {
                "experiment_id": args.experiment_id,
                "dtype": args.dtype,
                "stage1_max_new_tokens": stage1_max_new_tokens,
                "stage2_max_new_tokens": stage2_max_new_tokens,
                "selection_sha256": selection_sha256,
            }
        )
    if args.experiment_id == "G1":
        protocol_payload.update(
            {
                "demonstration_count": 8,
                "demonstration_payload_sha256": demonstration_payload_sha256,
                "demonstration_appendix_hashes": demonstration_hashes,
            }
        )
    elif args.experiment_id in {"G24S1", "G2", "G3", "G4"}:
        protocol_payload.update(
            {
                "stage1_demonstrations_enabled": stage1_demo,
                "stage2_demonstrations_enabled": stage2_demo,
                "demonstration_payload_sha256": demonstration_payload_sha256,
                "demonstration_appendix_hashes": demonstration_hashes,
                "stage1_only": args.stage1_only,
            }
        )

    reusable_stage1: dict[str, dict[str, Any]] = {}
    stage1_source_sha256: str | None = None
    if args.stage1_source is not None:
        source_experiment = "G0" if args.experiment_id == "G3" else "G24S1"
        source_appendix = demonstration_appendices.get("stage1") if source_experiment == "G24S1" else None
        expected_messages = {
            row["evaluation_id"]: _render_stage1(
                prompts["stage1"], row, source_appendix
            )
            for row in inputs
        }
        reusable_stage1, stage1_source_sha256 = _load_reusable_stage1(
            args.stage1_source,
            inputs,
            expected_source_experiment=source_experiment,
            expected_prompt_sha256=prompt_hashes["stage1"],
            expected_messages=expected_messages,
        )
        protocol_payload.update(
            {
                "stage1_source_experiment": source_experiment,
                "stage1_source_sha256": stage1_source_sha256,
            }
        )
    protocol_sha256 = _sha256_text(
        json.dumps(
            protocol_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    print(
        f"experiment={args.experiment_id} variant={args.variant} "
        f"input_rows={len(all_inputs)} selected_rows={len(inputs)} "
        f"input_sha256={input_sha256} selection_sha256={selection_sha256} "
        f"protocol_sha256={protocol_sha256} prompt_hashes={prompt_hashes}"
    )
    if args.print_protocol:
        print(json.dumps(protocol_payload, ensure_ascii=False, sort_keys=True, indent=2))
        return

    completed = _load_completed(args.output)
    stale = sorted(
        evaluation_id
        for evaluation_id, row in completed.items()
        if row.get("variant") != args.variant
        or row.get("experiment_id", "V1") != args.experiment_id
        or row.get("protocol_sha256") != protocol_sha256
        or row.get("input_sha256") != input_sha256
    )
    if stale:
        raise SystemExit(
            "ERROR: output contains rows from a different V1 protocol or input; "
            f"refusing to resume: {stale}"
        )
    expected_ids = {row["evaluation_id"] for row in inputs}
    unexpected = sorted(set(completed) - expected_ids)
    if unexpected:
        raise SystemExit(f"ERROR: output contains unexpected evaluation IDs: {unexpected}")
    pending = [row for row in inputs if row["evaluation_id"] not in completed]
    print(
        f"{args.experiment_id}/{args.variant} input={len(inputs)} completed={len(completed)} "
        f"pending={len(pending)} input_sha256={input_sha256} "
        f"protocol_sha256={protocol_sha256} prompt_hashes={prompt_hashes}"
    )
    if not pending:
        print("No pending candidates; existing predictions were left unchanged.")
        return

    import torch
    import transformers
    import accelerate
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is required; refusing to run Gemma 4 on CPU")

    def bf16_supported(device_index: int) -> bool:
        with torch.cuda.device(device_index):
            return bool(torch.cuda.is_bf16_supported())

    if args.dtype == "bfloat16" and not all(
        bf16_supported(device_index)
        for device_index in range(torch.cuda.device_count())
    ):
        raise SystemExit("ERROR: BF16 was requested but is not supported on every visible GPU")
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    load_dtype: Any = "auto" if args.dtype == "auto" else torch.bfloat16
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=load_dtype,
        device_map="auto",
    )
    model.eval()
    device_map = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    if args.experiment_id in {"F2", "G0", "G1", "G24S1", "G2", "G3", "G4"}:
        placements = set(device_map.values())
        if any(value in {"cpu", "disk"} for value in placements):
            raise SystemExit(
                f"ERROR: {args.experiment_id} model spilled outside GPU memory: "
                f"{sorted(placements)}"
            )
        gpu_placements = {
            value.removeprefix("cuda:")
            for value in placements
            if value.isdigit() or value.startswith("cuda:")
        }
        if gpu_placements != {"0", "1"}:
            raise SystemExit(
                f"ERROR: {args.experiment_id} must be sharded over exactly the two "
                "authorized visible GPUs; "
                f"observed placements={sorted(placements)}"
            )
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    context_window: int | None = None
    for config in (getattr(model.config, "text_config", None), model.config):
        value = getattr(config, "max_position_embeddings", None)
        if isinstance(value, int) and value > 0:
            context_window = value
            break
    if context_window is None:
        raise SystemExit("ERROR: model context-window size could not be established")
    generation_kwargs: dict[str, Any] = {
        "do_sample": False,
        "use_cache": True,
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_token_id)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = int(eos_token_id)

    def rendered_input_tokens(user_message: str) -> int:
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        return int(processor(text=rendered, return_tensors="pt")["input_ids"].shape[-1])

    if args.variant == "E":
        stage1_messages = [
            rendered
            for row in inputs
            for rendered in (
                _render_stage1(prompts["stage1_accuracy"], row),
                _render_stage1(prompts["stage1_fluency"], row),
            )
        ]
    else:
        stage1_messages = [
            _render_stage1(
                prompts["stage1"],
                row,
                demonstration_appendices.get("stage1") if stage1_demo else None,
            )
            for row in inputs
        ]
    stage2_messages = [
        _render_stage2(
            prompts["stage2"],
            row,
            [],
            demonstration_appendices.get("stage2") if stage2_demo else None,
        )
        for row in inputs
    ]
    stage1_preflight_max = max(map(rendered_input_tokens, stage1_messages))
    stage2_empty_preflight_max = (
        0 if args.stage1_only else max(map(rendered_input_tokens, stage2_messages))
    )
    # The validated Stage-1 JSON inserted into Stage 2 cannot be longer than the
    # Stage-1 generation budget. Reserving that full budget is conservative.
    stage2_preflight_upper_bound = stage2_empty_preflight_max + stage1_max_new_tokens
    if stage1_preflight_max + stage1_max_new_tokens > context_window:
        raise SystemExit("ERROR: Stage 1 prompt/output preflight exceeds context window")
    if not args.stage1_only and stage2_preflight_upper_bound + stage2_max_new_tokens > context_window:
        raise SystemExit("ERROR: Stage 2 prompt/output preflight exceeds context window")
    context_preflight = {
        "stage1_max_input_tokens": stage1_preflight_max,
        "stage2_empty_max_input_tokens": stage2_empty_preflight_max,
        "stage2_conservative_max_input_tokens": stage2_preflight_upper_bound,
        "context_window_tokens": context_window,
        "stage1_output_budget": stage1_max_new_tokens,
        "stage2_output_budget": stage2_max_new_tokens,
        "passed": True,
    }
    print(
        "CONTEXT_AUDIT PASS "
        f"stage1_input_max={stage1_preflight_max} "
        f"stage2_conservative_input_max={stage2_preflight_upper_bound} "
        f"stage1_budget={stage1_max_new_tokens} stage2_budget={stage2_max_new_tokens} "
        f"context_window={context_window}"
    )

    class StopAfterCompleteJSONObject(transformers.StoppingCriteria):
        def __init__(self, prompt_tokens: int) -> None:
            self.prompt_tokens = prompt_tokens
            self.triggered = False

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            raw_special = processor.decode(
                input_ids[0, self.prompt_tokens :], skip_special_tokens=False
            )
            final_segment = _final_segment_for_stopping(
                raw_special, thinking_enabled=args.enable_thinking
            )
            complete = final_segment is not None and _has_stoppable_complete_json_object(
                final_segment
            )
            if complete:
                self.triggered = True
            return torch.full(
                (input_ids.shape[0],),
                complete,
                device=input_ids.device,
                dtype=torch.bool,
            )

    def generate(user_message: str, max_new_tokens: int) -> dict[str, Any]:
        messages = [{"role": "user", "content": user_message}]
        rendered_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        model_inputs = processor(text=rendered_prompt, return_tensors="pt")
        model_inputs = model_inputs.to(next(model.parameters()).device)
        input_tokens = int(model_inputs["input_ids"].shape[-1])
        if input_tokens + max_new_tokens > context_window:
            raise SystemExit(
                "ERROR: rendered prompt plus output budget exceeds model context window: "
                f"input={input_tokens} output_budget={max_new_tokens} "
                f"context_window={context_window}"
            )
        stopper = StopAfterCompleteJSONObject(input_tokens)
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                **generation_kwargs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=transformers.StoppingCriteriaList([stopper]),
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        continuation = generated[0, input_tokens:]
        output_tokens = int(continuation.shape[-1])
        raw_special = processor.decode(continuation, skip_special_tokens=False)
        raw_plain = processor.decode(continuation, skip_special_tokens=True)
        response_parse_errors: list[str] = []
        response_parse_warnings: list[str] = []
        parsed_response: Any = None
        thinking_output: str | None = None
        final_answer: str | None = raw_plain
        if args.experiment_id in {"F1", "F2", "G0", "G1", "G24S1", "G2", "G3", "G4"}:
            (
                final_answer,
                thinking_output,
                response_parse_errors,
                response_parse_warnings,
                parsed_response,
            ) = _parse_model_response(
                processor,
                raw_special,
                model_inputs["input_ids"][0],
                thinking_enabled=args.enable_thinking,
            )
        return {
            "messages": messages,
            "rendered_prompt": rendered_prompt,
            "raw_model_output": raw_plain,
            "raw_model_output_with_special_tokens": raw_special,
            "thinking_output": thinking_output,
            "final_answer": final_answer,
            "processor_parsed_response": _json_safe(parsed_response),
            "response_parse_valid": not response_parse_errors,
            "response_parse_errors": response_parse_errors,
            "response_parse_warnings": response_parse_warnings,
            "generation_stopped_after_complete_json_object": stopper.triggered,
            "input_token_count": input_tokens,
            "output_token_count": output_tokens,
            "max_new_tokens": max_new_tokens,
            "hit_max_new_tokens": output_tokens >= max_new_tokens,
            "wall_time_seconds": elapsed,
            "cuda_peak_allocated_bytes": [
                torch.cuda.max_memory_allocated(device_index)
                for device_index in range(torch.cuda.device_count())
            ],
            "cuda_peak_reserved_bytes": [
                torch.cuda.max_memory_reserved(device_index)
                for device_index in range(torch.cuda.device_count())
            ],
        }

    runtime = {
        "hostname": platform.node(),
        "process_id": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "cuda_devices": [
            {
                "index": device_index,
                "name": torch.cuda.get_device_name(device_index),
                "total_memory_bytes": torch.cuda.get_device_properties(
                    device_index
                ).total_memory,
                "bf16_supported": bf16_supported(device_index),
            }
            for device_index in range(torch.cuda.device_count())
        ],
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_device": str(next(model.parameters()).device),
        "model_dtype": str(next(model.parameters()).dtype),
        "requested_dtype": args.dtype,
        "context_window_tokens": context_window,
        "quantization": None,
        "hf_device_map": device_map,
        "generation_defaults": {
            key: _json_safe(value)
            for key, value in model.generation_config.to_dict().items()
        },
        "context_preflight": context_preflight,
    }
    model_record = {
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "architecture": model.__class__.__name__,
        "processor": processor.__class__.__name__,
    }
    generation_config = {
        "deterministic": True,
        "do_sample": False,
        "stage1_max_new_tokens": stage1_max_new_tokens,
        "stage2_max_new_tokens": stage2_max_new_tokens,
        "use_cache": True,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "enable_thinking": args.enable_thinking,
        "stop_after_complete_json_object": True,
        "stopper_scope": "final_answer_channel_only",
        "context_window_tokens": context_window,
        "context_window_enforced": True,
    }
    if args.experiment_id == "G1":
        generation_config.update(
            {
                "demonstration_count": 8,
                "demonstration_payload_sha256": demonstration_payload_sha256,
                "demonstration_appendix_hashes": demonstration_hashes,
            }
        )
    elif args.experiment_id in {"G24S1", "G2", "G3", "G4"}:
        generation_config.update(
            {
                "stage1_demonstrations_enabled": stage1_demo,
                "stage2_demonstrations_enabled": stage2_demo,
                "demonstration_payload_sha256": demonstration_payload_sha256,
                "demonstration_appendix_hashes": demonstration_hashes,
                "stage1_only": args.stage1_only,
                "stage1_source_sha256": stage1_source_sha256,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for row in pending:
            stages: dict[str, Any] = {}
            overall_errors: list[str] = []
            overall_warnings: list[str] = []
            stage1_prediction: dict[str, Any] | None = None

            stage1_reuse: dict[str, Any] | None = None
            if reusable_stage1:
                source_row = reusable_stage1[row["evaluation_id"]]
                stages["stage1"] = copy.deepcopy(source_row["stages"]["stage1"])
                stage1_prediction = stages["stage1"].get("prediction")
                overall_errors.extend(
                    _prefixed("stage1", stages["stage1"].get("parse_errors", []))
                )
                overall_warnings.extend(
                    _prefixed("stage1", stages["stage1"].get("parse_warnings", []))
                )
                stage1_reuse = {
                    "source_path": str(args.stage1_source),
                    "source_sha256": stage1_source_sha256,
                    "source_experiment_id": source_row["experiment_id"],
                    "source_protocol_sha256": source_row["protocol_sha256"],
                    "source_evaluation_id": row["evaluation_id"],
                    "compatibility_validation": "PASS",
                }
            elif args.variant == "E":
                accuracy_generation = generate(
                    _render_stage1(prompts["stage1_accuracy"], row),
                    stage1_max_new_tokens,
                )
                accuracy_parsed = (
                    _response_parse_failure(accuracy_generation)
                    if accuracy_generation["response_parse_errors"]
                    else _parse_stage1_standard(
                        accuracy_generation["final_answer"],
                        row["candidate_text"],
                        stopped_after_complete_json_object=accuracy_generation[
                            "generation_stopped_after_complete_json_object"
                        ],
                        allow_missing=True,
                    )
                )
                stages["stage1_accuracy"] = _stage_with_parse(
                    accuracy_generation,
                    accuracy_parsed,
                    prompt_hashes["stage1_accuracy"],
                )
                fluency_generation = generate(
                    _render_stage1(prompts["stage1_fluency"], row),
                    stage1_max_new_tokens,
                )
                fluency_parsed = (
                    _response_parse_failure(fluency_generation)
                    if fluency_generation["response_parse_errors"]
                    else _parse_stage1_standard(
                        fluency_generation["final_answer"],
                        row["candidate_text"],
                        stopped_after_complete_json_object=fluency_generation[
                            "generation_stopped_after_complete_json_object"
                        ],
                        allow_missing=False,
                    )
                )
                stages["stage1_fluency"] = _stage_with_parse(
                    fluency_generation,
                    fluency_parsed,
                    prompt_hashes["stage1_fluency"],
                )
                overall_errors.extend(
                    _prefixed(
                        "stage1_accuracy", stages["stage1_accuracy"]["parse_errors"]
                    )
                )
                overall_errors.extend(
                    _prefixed(
                        "stage1_fluency", stages["stage1_fluency"]["parse_errors"]
                    )
                )
                overall_warnings.extend(
                    _prefixed(
                        "stage1_accuracy", stages["stage1_accuracy"]["parse_warnings"]
                    )
                )
                overall_warnings.extend(
                    _prefixed(
                        "stage1_fluency", stages["stage1_fluency"]["parse_warnings"]
                    )
                )
                if not overall_errors:
                    stage1_prediction, combine_errors = _combine_e_passes(
                        stages["stage1_accuracy"]["prediction"]["error_spans"],
                        stages["stage1_fluency"]["prediction"]["error_spans"],
                    )
                    stages["stage1_combined"] = {
                        "prediction": stage1_prediction,
                        "parse_valid": not combine_errors,
                        "parse_errors": combine_errors,
                    }
                    overall_errors.extend(
                        _prefixed("stage1_combined", combine_errors)
                    )
                else:
                    stages["stage1_combined"] = {
                        "prediction": None,
                        "parse_valid": False,
                        "parse_errors": ["component_stage_invalid"],
                    }
            else:
                stage1_generation = generate(
                    _render_stage1(
                        prompts["stage1"],
                        row,
                        demonstration_appendices.get("stage1") if stage1_demo else None,
                    ),
                    stage1_max_new_tokens,
                )
                if stage1_generation["response_parse_errors"]:
                    parsed_stage1 = _response_parse_failure(stage1_generation)
                elif args.variant == "C":
                    parsed_stage1 = _parse_stage1_c(
                        stage1_generation["final_answer"],
                        row["candidate_text"],
                        row["current_source_bp"],
                        stopped_after_complete_json_object=stage1_generation[
                            "generation_stopped_after_complete_json_object"
                        ],
                    )
                else:
                    parsed_stage1 = _parse_stage1_standard(
                        stage1_generation["final_answer"],
                        row["candidate_text"],
                        stopped_after_complete_json_object=stage1_generation[
                            "generation_stopped_after_complete_json_object"
                        ],
                        allow_missing=True,
                    )
                stages["stage1"] = _stage_with_parse(
                    stage1_generation, parsed_stage1, prompt_hashes["stage1"]
                )
                stage1_prediction = stages["stage1"]["prediction"]
                overall_errors.extend(
                    _prefixed("stage1", stages["stage1"]["parse_errors"])
                )
                overall_warnings.extend(
                    _prefixed("stage1", stages["stage1"]["parse_warnings"])
                )

            stage2_prediction: dict[str, Any] | None = None
            if args.stage1_only:
                stages["stage2"] = {
                    "status": "not_run",
                    "reason": "shared_stage1_artifact",
                }
            elif stage1_prediction is not None and not overall_errors:
                error_spans = stage1_prediction["error_spans"]
                stage2_generation = generate(
                    _render_stage2(
                        prompts["stage2"],
                        row,
                        error_spans,
                        demonstration_appendices.get("stage2") if stage2_demo else None,
                    ),
                    stage2_max_new_tokens,
                )
                parsed_stage2 = (
                    _response_parse_failure(stage2_generation)
                    if stage2_generation["response_parse_errors"]
                    else _parse_stage2(
                        stage2_generation["final_answer"],
                        error_spans,
                        stopped_after_complete_json_object=stage2_generation[
                            "generation_stopped_after_complete_json_object"
                        ],
                    )
                )
                stages["stage2"] = _stage_with_parse(
                    stage2_generation, parsed_stage2, prompt_hashes["stage2"]
                )
                stage2_prediction = stages["stage2"]["prediction"]
                overall_errors.extend(
                    _prefixed("stage2", stages["stage2"]["parse_errors"])
                )
                overall_warnings.extend(
                    _prefixed("stage2", stages["stage2"]["parse_warnings"])
                )
            else:
                stages["stage2"] = {
                    "status": "skipped",
                    "reason": "stage1_not_strictly_valid",
                }

            prediction = None
            if args.stage1_only and not overall_errors and stage1_prediction is not None:
                prediction = {"error_spans": stage1_prediction["error_spans"]}
            elif (
                not overall_errors
                and stage1_prediction is not None
                and stage2_prediction is not None
            ):
                prediction = {
                    "error_spans": stage1_prediction["error_spans"],
                    "score": stage2_prediction["score"],
                }
            output_row = {
                "schema_version": (
                    "1.2" if args.experiment_id in {"G24S1", "G2", "G3", "G4"}
                    else "1.1" if args.experiment_id != "V1" else "1.0"
                ),
                "evaluator_version": (
                    f"V1-{args.variant}"
                    if args.experiment_id == "V1"
                    else args.experiment_id
                ),
                "experiment_id": args.experiment_id,
                "variant": args.variant,
                "prompt_structure_version": prompt_structure_version,
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "output_parser_version": output_parser_version,
                "stage1_schema_version": stage1_schema,
                "stage2_schema_version": STAGE2_SCHEMA_VERSION,
                "evaluation_id": row["evaluation_id"],
                "input_sha256": input_sha256,
                "selection_sha256": selection_sha256,
                "prompt_hashes": prompt_hashes,
                "demonstration_appendix_hashes": demonstration_hashes,
                "demonstration_payload_sha256": demonstration_payload_sha256,
                "protocol_sha256": protocol_sha256,
                "stage1_reuse": stage1_reuse,
                "model": model_record,
                "stages": stages,
                "prediction": prediction,
                "parse_valid": prediction is not None,
                "parse_errors": overall_errors,
                "parse_warnings": overall_warnings,
                "generation_config": generation_config,
                "runtime": runtime,
            }
            handle.write(
                json.dumps(output_row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            completed[row["evaluation_id"]] = output_row
            print(
                f"[{len(completed)}/{len(inputs)}] {row['evaluation_id']} "
                f"parse_valid={output_row['parse_valid']}"
            )
    valid = sum(row.get("parse_valid") is True for row in completed.values())
    print(
        f"{args.experiment_id}/{args.variant} complete: predictions={len(completed)} "
        f"parse_valid={valid} parse_invalid={len(completed) - valid}"
    )


if __name__ == "__main__":
    main()
