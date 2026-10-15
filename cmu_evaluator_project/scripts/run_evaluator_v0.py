#!/usr/bin/env python3
"""Run deterministic zero-shot Gemma 4 evaluation over blind DEV inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any


MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
PROMPT_STRUCTURE_VERSION = "2.0-single-user-evaluation-data"
MODEL_OUTPUT_SCHEMA_VERSION = "2.1-substring-occurrence"
OUTPUT_PARSER_VERSION = "1.1-strict-markdown-wrapper"
INPUT_FIELDS = {
    "evaluation_id",
    "document_id",
    "passage_id",
    "sentence_id",
    "position",
    "passage_context",
    "current_source_bp",
    "candidate_text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"ERROR: invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"ERROR: expected object at {path}:{line_number}")
                rows.append(row)
    except OSError as exc:
        raise SystemExit(f"ERROR: could not read {path}: {exc}") from exc
    return rows


def _validate_inputs(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit("ERROR: V0 input contains no rows")
    seen_evaluation: set[str] = set()
    for index, row in enumerate(rows, 1):
        if set(row) != INPUT_FIELDS:
            raise SystemExit(
                f"ERROR: input row {index} fields differ from blind V0 schema: "
                f"observed={sorted(row)}, expected={sorted(INPUT_FIELDS)}"
            )
        for field in (
            "evaluation_id",
            "document_id",
            "passage_id",
            "sentence_id",
            "current_source_bp",
            "candidate_text",
        ):
            if not isinstance(row[field], str) or not row[field]:
                raise SystemExit(f"ERROR: input row {index} has invalid {field}")
        if row["evaluation_id"] in seen_evaluation:
            raise SystemExit(f"ERROR: duplicate V0 input identity at row {index}")
        seen_evaluation.add(row["evaluation_id"])
        context = row["passage_context"]
        if not isinstance(context, list) or not context:
            raise SystemExit(f"ERROR: input row {index} has empty passage context")
        expected_context_fields = {"sentence_id", "position", "source_bp"}
        for sentence in context:
            if not isinstance(sentence, dict) or set(sentence) != expected_context_fields:
                raise SystemExit(f"ERROR: input row {index} has invalid context schema")
            if not isinstance(sentence["source_bp"], str) or not sentence["source_bp"]:
                raise SystemExit(f"ERROR: input row {index} has invalid context text")
        matches = [
            sentence
            for sentence in context
            if sentence["sentence_id"] == row["sentence_id"]
            and sentence["position"] == row["position"]
        ]
        if len(matches) != 1 or matches[0]["source_bp"] != row["current_source_bp"]:
            raise SystemExit(f"ERROR: input row {index} current source/context mismatch")


def _render_user_message(instructions: str, row: dict[str, Any]) -> str:
    context = "\n".join(
        f"[{sentence['position']}] {sentence['source_bp']}"
        for sentence in row["passage_context"]
    )
    return (
        f"{instructions.rstrip()}\n\n"
        "# Evaluation data\n\n"
        "PORTUGUESE PASSAGE CONTEXT (BP — REFERENCE ONLY):\n"
        f"{context}\n\n"
        f"CURRENT BP SOURCE SENTENCE (REFERENCE ONLY, position {row['position']}):\n"
        f"{row['current_source_bp']}\n\n"
        "EP CANDIDATE TO EVALUATE (ONLY TEXT TO ANNOTATE):\n"
        f"{row['candidate_text']}"
    )


def _complete_json_object_end(text: str) -> int | None:
    """Return the end of a complete leading JSON object, without parsing it."""
    start = len(text) - len(text.lstrip())
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                return None
    return None


def _has_stoppable_complete_json_object(text: str) -> bool:
    """Recognize a completed leading object in a plain or permitted fenced response."""
    stripped = text.lstrip()
    for opening in ("```json\n", "```\n"):
        if stripped.startswith(opening):
            body = stripped[len(opening) :]
            end = _complete_json_object_end(body)
            if end is None:
                return False
            return body[end:].strip() in {"", "```"}
    return _complete_json_object_end(stripped) is not None


def _normalize_json_representation(
    raw: str, *, stopped_after_complete_json_object: bool
) -> tuple[str | None, dict[str, Any], list[str]]:
    """Remove only an explicitly permitted Markdown wrapper around one JSON value."""
    text = raw.strip()
    status = {
        "raw_format_compliant": False,
        "markdown_wrapper_normalized": False,
        "output_representation": "unrecognized",
    }

    try:
        json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        status["raw_format_compliant"] = True
        status["output_representation"] = "plain_json"
        return text, status, []

    opening = next(
        (prefix for prefix in ("```json\n", "```\n") if text.startswith(prefix)),
        None,
    )
    if opening is None:
        return None, status, ["output_format_not_allowed"]

    fenced_body = text[len(opening) :]
    if fenced_body.endswith("\n```"):
        json_text = fenced_body[: -len("\n```")]
        representation = "markdown_fence_complete"
    elif stopped_after_complete_json_object and "```" not in fenced_body:
        json_text = fenced_body
        representation = "markdown_fence_open_stopped_at_json"
    else:
        return None, status, ["markdown_fence_incomplete_or_has_trailing_content"]

    try:
        json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, status, [f"invalid_json_in_markdown_fence: {exc}"]

    status["markdown_wrapper_normalized"] = True
    status["output_representation"] = representation
    return json_text, status, []


def _candidate_occurrences(candidate: str, substring: str) -> list[int]:
    occurrences: list[int] = []
    cursor = 0
    while True:
        start = candidate.find(substring, cursor)
        if start < 0:
            return occurrences
        occurrences.append(start)
        cursor = start + 1


def _parse_prediction(
    raw: str,
    candidate: str,
    *,
    stopped_after_complete_json_object: bool = False,
) -> tuple[dict[str, Any] | None, list[str], list[str], dict[str, Any]]:
    warnings: list[str] = []
    errors: list[str] = []
    text, format_status, format_errors = _normalize_json_representation(
        raw,
        stopped_after_complete_json_object=stopped_after_complete_json_object,
    )
    if format_errors or text is None:
        return None, format_errors, warnings, format_status
    if format_status["markdown_wrapper_normalized"]:
        warnings.append("markdown_wrapper_normalized")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"invalid_json: {exc}"], warnings, format_status
    if not isinstance(value, dict):
        return None, ["top_level_must_be_object"], warnings, format_status
    if set(value) != {"errors", "score"}:
        errors.append("top_level_keys_must_be_exactly_errors_and_score")
    score = value.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        errors.append("score_must_be_integer_0_to_100")
    raw_spans = value.get("errors")
    normalized_spans: list[dict[str, Any]] = []
    if not isinstance(raw_spans, list):
        errors.append("errors_must_be_list")
    else:
        for index, span in enumerate(raw_spans):
            prefix = f"errors[{index}]"
            if not isinstance(span, dict):
                errors.append(f"{prefix}_must_be_object")
                continue
            span_keys = set(span)
            if span_keys not in ({"text", "severity"}, {"text", "severity", "occurrence"}):
                errors.append(f"{prefix}_keys_invalid")
            severity, marked_text = span.get("severity"), span.get("text")
            occurrence = span.get("occurrence")
            has_occurrence = "occurrence" in span
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
                if has_occurrence:
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
                normalized_span = {
                    "start": start,
                    "end": end,
                    "text": marked_text,
                    "severity": severity,
                    "missing": missing,
                }
                if has_occurrence:
                    normalized_span["occurrence"] = occurrence
                normalized_spans.append(normalized_span)
    ordered_spans = sorted(normalized_spans, key=lambda span: (span["start"], span["end"]))
    for previous, current in zip(ordered_spans, ordered_spans[1:]):
        if current["start"] < previous["end"] or (
            current["start"] == previous["start"]
            and current["end"] == previous["end"]
        ):
            errors.append("error_spans_overlap_or_duplicate")
            break
    if isinstance(raw_spans, list) and isinstance(score, int) and not isinstance(score, bool):
        if raw_spans and score == 100:
            errors.append("score_must_be_below_100_when_errors_present")
        elif not raw_spans and score != 100:
            errors.append("score_must_be_100_when_errors_empty")
    if errors:
        return None, errors, warnings, format_status
    return {"error_spans": normalized_spans, "score": score}, [], warnings, format_status


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(_read_jsonl(path), 1):
        evaluation_id = row.get("evaluation_id")
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise SystemExit(f"ERROR: completed output line {line_number} has no evaluation_id")
        if evaluation_id in completed:
            raise SystemExit(f"ERROR: duplicate completed evaluation_id {evaluation_id}")
        completed[evaluation_id] = row
    return completed


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def main() -> None:
    args = parse_args()
    if args.model_id != MODEL_ID:
        raise SystemExit(f"ERROR: V0 only permits model {MODEL_ID}")
    if args.revision != MODEL_REVISION:
        raise SystemExit(f"ERROR: V0 only permits revision {MODEL_REVISION}")
    if args.max_new_tokens < 1:
        raise SystemExit("ERROR: --max-new-tokens must be positive")
    inputs = _read_jsonl(args.input)
    _validate_inputs(inputs)
    try:
        instructions = args.instructions.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"ERROR: could not read instructions: {exc}") from exc
    if not instructions.strip():
        raise SystemExit("ERROR: evaluator instructions are empty")
    instructions_sha256 = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
    prompt_schema_sha256 = hashlib.sha256(
        json.dumps(
            {
                "instructions_sha256": instructions_sha256,
                "prompt_structure_version": PROMPT_STRUCTURE_VERSION,
                "model_output_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    completed = _load_completed(args.output)
    stale_protocol_ids = sorted(
        evaluation_id
        for evaluation_id, row in completed.items()
        if row.get("prompt_schema_sha256") != prompt_schema_sha256
        or row.get("output_parser_version") != OUTPUT_PARSER_VERSION
    )
    if stale_protocol_ids:
        raise SystemExit(
            "ERROR: output contains predictions from a different prompt/schema protocol; "
            f"refusing to resume: {stale_protocol_ids}"
        )
    expected_ids = {row["evaluation_id"] for row in inputs}
    unexpected_completed = set(completed) - expected_ids
    if unexpected_completed:
        raise SystemExit(f"ERROR: output contains unexpected evaluation IDs: {sorted(unexpected_completed)}")
    pending = [row for row in inputs if row["evaluation_id"] not in completed]
    print(
        f"V0 input rows={len(inputs)} completed={len(completed)} pending={len(pending)} "
        f"instructions_sha256={instructions_sha256} "
        f"prompt_schema_sha256={prompt_schema_sha256}"
    )
    if not pending:
        print("No pending candidates; existing predictions were left unchanged.")
        return

    # Heavy imports happen only after all input/resume validation.
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is required; refusing to run Gemma 4 on CPU")
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    generation_kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_token_id)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = int(eos_token_id)

    class StopAfterCompleteJSONObject(transformers.StoppingCriteria):
        def __init__(self, prompt_tokens: int) -> None:
            self.prompt_tokens = prompt_tokens
            self.triggered = False

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            continuation_text = tokenizer.decode(
                input_ids[0, self.prompt_tokens :], skip_special_tokens=True
            )
            complete = _has_stoppable_complete_json_object(continuation_text)
            if complete:
                self.triggered = True
            return torch.full(
                (input_ids.shape[0],), complete, device=input_ids.device, dtype=torch.bool
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for pending_index, row in enumerate(pending, 1):
            user_message = _render_user_message(instructions, row)
            messages = [{"role": "user", "content": user_message}]
            rendered_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = processor(text=rendered_prompt, return_tensors="pt")
            input_device = next(model.parameters()).device
            model_inputs = model_inputs.to(input_device)
            input_tokens = int(model_inputs["input_ids"].shape[-1])
            json_stopper = StopAfterCompleteJSONObject(input_tokens)
            row_generation_kwargs = {
                **generation_kwargs,
                "stopping_criteria": transformers.StoppingCriteriaList(
                    [json_stopper]
                ),
            }
            with torch.inference_mode():
                generated = model.generate(**model_inputs, **row_generation_kwargs)
            continuation = generated[0, input_tokens:]
            output_tokens = int(continuation.shape[-1])
            raw_response = tokenizer.decode(continuation, skip_special_tokens=True)
            prediction, parse_errors, parse_warnings, format_status = _parse_prediction(
                raw_response,
                row["candidate_text"],
                stopped_after_complete_json_object=json_stopper.triggered,
            )
            output_row = {
                "schema_version": "1.3",
                "evaluator_version": "V0",
                "prompt_structure_version": PROMPT_STRUCTURE_VERSION,
                "model_output_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
                "output_parser_version": OUTPUT_PARSER_VERSION,
                "evaluation_id": row["evaluation_id"],
                "model": {
                    "model_id": args.model_id,
                    "requested_revision": args.revision,
                    "resolved_revision": getattr(model.config, "_commit_hash", None),
                    "architecture": model.__class__.__name__,
                    "processor": processor.__class__.__name__,
                },
                "instructions_sha256": instructions_sha256,
                "prompt_schema_sha256": prompt_schema_sha256,
                "messages": messages,
                "rendered_prompt": rendered_prompt,
                "raw_model_output": raw_response,
                **format_status,
                "generation_stopped_after_complete_json_object": json_stopper.triggered,
                "prediction": prediction,
                "parse_valid": not parse_errors,
                "parse_errors": parse_errors,
                "parse_warnings": parse_warnings,
                "input_token_count": input_tokens,
                "output_token_count": output_tokens,
                "hit_max_new_tokens": output_tokens >= args.max_new_tokens,
                "generation_config": {
                    "deterministic": True,
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                    "use_cache": True,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": eos_token_id,
                    "enable_thinking": False,
                    "stop_after_complete_json_object": True,
                },
                "runtime": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_device_count": torch.cuda.device_count(),
                    "model_device": str(next(model.parameters()).device),
                    "model_dtype": str(next(model.parameters()).dtype),
                    "hf_device_map": {
                        str(key): str(value)
                        for key, value in getattr(model, "hf_device_map", {}).items()
                    },
                    "generation_defaults": {
                        key: _json_safe(value)
                        for key, value in model.generation_config.to_dict().items()
                    },
                },
            }
            handle.write(json.dumps(output_row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed[row["evaluation_id"]] = output_row
            print(
                f"[{len(completed)}/{len(inputs)}] {row['evaluation_id']} "
                f"parse_valid={output_row['parse_valid']}"
            )
    valid = sum(bool(row.get("parse_valid")) for row in completed.values())
    print(f"V0 complete: predictions={len(completed)} parse_valid={valid} parse_invalid={len(completed)-valid}")


if __name__ == "__main__":
    main()
