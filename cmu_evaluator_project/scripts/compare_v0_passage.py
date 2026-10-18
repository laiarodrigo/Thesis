#!/usr/bin/env python3
"""Render qualitative HTML comparisons for V0 passages or the V0/V1 pilot."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "data/benchmark/dev_01_evaluator.json"
BLIND_INPUT_PATH = PROJECT_ROOT / "data/benchmark/dev_01_v0_input.jsonl"
PREDICTIONS_PATH = PROJECT_ROOT / "runtime/evaluator_dev/v0_predictions.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "comparisons/v0"
PILOT_INPUT_PATH = PROJECT_ROOT / "data/benchmark/v1_37_input.jsonl"
V0_OFFICIAL_PATH = PROJECT_ROOT / "runtime/evaluator_dev/v0_official_dev_predictions.jsonl"
V0_SYNTHETIC_PATH = (
    PROJECT_ROOT
    / "diagnostics/v0_source_sensitivity/v0_source_sensitivity_predictions.jsonl"
)
V1_PREDICTION_PATHS = OrderedDict(
    (
        ("V1-A", PROJECT_ROOT / "runtime/evaluator_v1/v1_a_37_predictions.jsonl"),
        ("V1-B", PROJECT_ROOT / "runtime/evaluator_v1/v1_b_37_predictions.jsonl"),
        ("V1-C", PROJECT_ROOT / "runtime/evaluator_v1/v1_c_37_predictions.jsonl"),
        ("V1-E", PROJECT_ROOT / "runtime/evaluator_v1/v1_e_37_predictions.jsonl"),
    )
)
V1_COMPARISON_PATH = PROJECT_ROOT / "comparisons/v1/v0_v1_37_comparison.html"
REASONING_PREDICTION_PATHS = OrderedDict(
    (
        (
            "F1 thinking",
            (
                PROJECT_ROOT
                / "runtime/evaluator_model_reasoning/f1_e4b_thinking_37_predictions.jsonl",
                "F1",
            ),
        ),
        (
            "F2 31B",
            (
                PROJECT_ROOT
                / "runtime/evaluator_model_reasoning/f2_31b_no_thinking_37_predictions.jsonl",
                "F2",
            ),
        ),
    )
)
REASONING_COMPARISON_PATH = (
    PROJECT_ROOT / "comparisons/model_reasoning/v1a_f1_f2_37_comparison.html"
)
G_INPUT_PATH = PROJECT_ROOT / "data/benchmark/gemma31b_g0_g1_eval_96.jsonl"
G_MANIFEST_PATH = PROJECT_ROOT / "data/benchmark/gemma31b_g0_g1_manifest.json"
G_PREDICTION_PATHS = OrderedDict(
    (
        ("G0 zero-shot", PROJECT_ROOT / "runtime/evaluator_gemma31b/g0_31b_zero_shot_96_predictions.jsonl"),
        ("G1 eight-shot", PROJECT_ROOT / "runtime/evaluator_gemma31b/g1_31b_eight_shot_96_predictions.jsonl"),
    )
)
G_COMPARISON_PATH = PROJECT_ROOT / "comparisons/gemma31b/g0_g1_96_comparison.html"
G24_INPUT_PATH = PROJECT_ROOT / "data/benchmark/gemma31b_g2_g4_eval_72.jsonl"
G24_MANIFEST_PATH = PROJECT_ROOT / "data/benchmark/gemma31b_g2_g4_manifest.json"
G24_PREDICTION_PATHS = OrderedDict(
    (
        *G_PREDICTION_PATHS.items(),
        ("G2 improved Stage 1", PROJECT_ROOT / "runtime/evaluator_gemma31b/g2_31b_improved_stage1_72_predictions.jsonl"),
        ("G3 improved Stage 2", PROJECT_ROOT / "runtime/evaluator_gemma31b/g3_31b_improved_stage2_72_predictions.jsonl"),
        ("G4 improved both", PROJECT_ROOT / "runtime/evaluator_gemma31b/g4_31b_improved_both_72_predictions.jsonl"),
    )
)
G24_COMPARISON_PATH = PROJECT_ROOT / "comparisons/gemma31b/g0_g4_72_comparison.html"


class ComparisonError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--list-passages", action="store_true")
    choice.add_argument("--passage-id")
    choice.add_argument(
        "--v1-pilot",
        action="store_true",
        help="render the single V0/V1 comparison for the shared 37-row pilot",
    )
    choice.add_argument(
        "--reasoning-pilot",
        action="store_true",
        help="render human/V1-A/F1/F2 comparison for the shared 37-row pilot",
    )
    choice.add_argument(
        "--g0-g1",
        action="store_true",
        help="render the frozen 96-judgment G0/G1 DEV comparison",
    )
    choice.add_argument(
        "--g0-g4",
        action="store_true",
        help="render the common 72-judgment G0/G1/G2/G3/G4 DEV comparison",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ComparisonError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ComparisonError(f"Expected an object at {path}:{line_number}")
                rows.append(row)
    except OSError as exc:
        raise ComparisonError(f"Could not read {path}: {exc}") from exc
    return rows


def index_unique(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        evaluation_id = row.get("evaluation_id")
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise ComparisonError(f"{name} row {row_number} has no evaluation_id")
        if evaluation_id in indexed:
            raise ComparisonError(f"Duplicate {name} evaluation_id: {evaluation_id}")
        indexed[evaluation_id] = row
    return indexed


def load_and_match(
    predictions_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    canonical = read_json(CANONICAL_PATH)
    canonical_rows = canonical.get("records")
    if not isinstance(canonical_rows, list) or not canonical_rows:
        raise ComparisonError(f"Canonical records are missing from {CANONICAL_PATH}")
    if not all(isinstance(row, dict) for row in canonical_rows):
        raise ComparisonError("Canonical records must all be objects")

    canonical_by_id = index_unique(canonical_rows, "canonical")
    blind_by_id = index_unique(read_jsonl(BLIND_INPUT_PATH), "blind input")
    predictions_by_id = index_unique(read_jsonl(predictions_path), "prediction")
    canonical_ids = set(canonical_by_id)
    missing_blind = sorted(canonical_ids - set(blind_by_id))
    unmatched_blind = sorted(set(blind_by_id) - canonical_ids)
    if missing_blind or unmatched_blind:
        raise ComparisonError(
            "blind input ID mismatch: "
            f"missing={missing_blind or 'none'}, unmatched={unmatched_blind or 'none'}"
        )
    unmatched_predictions = sorted(set(predictions_by_id) - canonical_ids)
    if unmatched_predictions:
        raise ComparisonError(
            f"predictions contain unmatched evaluation IDs: {unmatched_predictions}"
        )
    if not predictions_by_id:
        raise ComparisonError("predictions contain no evaluation records")

    system_order: list[str] = []
    passage_sentence_systems: dict[tuple[str, str], set[str]] = {}
    for row in canonical_rows:
        evaluation_id = row["evaluation_id"]
        blind = blind_by_id[evaluation_id]
        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            raise ComparisonError(f"{evaluation_id}: canonical candidate is missing")
        system_id = candidate.get("system_id")
        if not isinstance(system_id, str) or not system_id:
            raise ComparisonError(f"{evaluation_id}: system_id is missing")
        if system_id not in system_order:
            system_order.append(system_id)
        for canonical_field, blind_field in (
            ("document_id", "document_id"),
            ("passage_id", "passage_id"),
            ("sentence_id", "sentence_id"),
            ("position", "position"),
            ("current_source_bp", "current_source_bp"),
        ):
            if row.get(canonical_field) != blind.get(blind_field):
                raise ComparisonError(
                    f"{evaluation_id}: canonical/blind mismatch in {canonical_field}"
                )
        if candidate.get("text") != blind.get("candidate_text"):
            raise ComparisonError(f"{evaluation_id}: canonical/blind candidate text mismatch")
        sentence_key = (row["passage_id"], row["sentence_id"])
        seen_systems = passage_sentence_systems.setdefault(sentence_key, set())
        if system_id in seen_systems:
            raise ComparisonError(f"{sentence_key}: duplicate system {system_id}")
        seen_systems.add(system_id)

    if len(system_order) != 4:
        raise ComparisonError(f"Expected four candidate systems, observed {system_order}")
    expected_systems = set(system_order)
    for sentence_key, observed in passage_sentence_systems.items():
        if observed != expected_systems:
            raise ComparisonError(
                f"{sentence_key}: expected systems {system_order}, observed {sorted(observed)}"
            )

    joined = [
        {
            **row,
            "blind_input": blind_by_id[row["evaluation_id"]],
            "v0_prediction": predictions_by_id[row["evaluation_id"]],
        }
        for row in canonical_rows
        if row["evaluation_id"] in predictions_by_id
    ]
    return joined, predictions_by_id, system_order


def passages_in_order(rows: list[dict[str, Any]]) -> OrderedDict[str, dict[str, Any]]:
    passages: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        passage = passages.setdefault(
            row["passage_id"],
            {"document_id": row["document_id"], "sentences": OrderedDict()},
        )
        sentence = passage["sentences"].setdefault(
            row["sentence_id"],
            {
                "sentence_id": row["sentence_id"],
                "position": row["position"],
                "source": row["current_source_bp"],
                "candidates": [],
            },
        )
        if sentence["position"] != row["position"] or sentence["source"] != row["current_source_bp"]:
            raise ComparisonError(f"{row['passage_id']}/{row['sentence_id']}: inconsistent source data")
        sentence["candidates"].append(row)
    for passage_id, passage in passages.items():
        ordered = sorted(passage["sentences"].values(), key=lambda row: row["position"])
        positions = [row["position"] for row in ordered]
        if len(set(positions)) != len(positions):
            raise ComparisonError(f"{passage_id}: duplicate sentence position")
        passage["sentences"] = ordered
    return passages


def utf16_offset_to_python(text: str, offset: int) -> int | None:
    """Map a JavaScript UTF-16 code-unit boundary to a Python string boundary."""
    units = 0
    if offset == 0:
        return 0
    for index, character in enumerate(text, 1):
        units += 2 if ord(character) > 0xFFFF else 1
        if units == offset:
            return index
        if units > offset:
            return None
    return len(text) if units == offset else None


def normalized_spans(
    text: str,
    spans: Any,
    label: str,
    *,
    human_inclusive_utf16: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(spans, list):
        return [], f"{label}: spans are not a list"
    result: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            return [], f"{label}: span {index} is not an object"
        start, end = span.get("start"), span.get("end")
        severity = span.get("severity")
        marked_text = span.get("text")
        missing = span.get("missing") is True or marked_text == "[MISSING]"
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start <= end
        ):
            return [], f"{label}: span {index} has invalid offsets"
        if severity not in {"Minor", "Major"}:
            return [], f"{label}: span {index} has invalid severity"

        if human_inclusive_utf16:
            # Pearmut indexes its per-character DOM spans in JavaScript UTF-16
            # code units and stores both endpoints inclusively. Its synthetic
            # [MISSING] token is instead a zero-width anchor and must not gain 1.
            if missing:
                if start != end:
                    return [], f"{label}: span {index} has an invalid [MISSING] anchor"
                python_start = utf16_offset_to_python(text, start)
                python_end = python_start
            else:
                python_start = utf16_offset_to_python(text, start)
                python_end = utf16_offset_to_python(text, end + 1)
            if python_start is None or python_end is None:
                return [], (
                    f"{label}: span {index} does not align to Unicode character boundaries"
                )
            expected_text = "[MISSING]" if missing else text[python_start:python_end]
            convention = "inclusive UTF-16"
        else:
            if end > len(text):
                return [], f"{label}: span {index} has invalid offsets"
            python_start, python_end = start, end
            expected_text = "[MISSING]" if missing else text[start:end]
            convention = "exclusive"
            if missing != (start == end):
                return [], f"{label}: span {index} has inconsistent [MISSING] offsets"
            if marked_text != expected_text:
                return [], f"{label}: span {index} text does not match its offsets"
        result.append(
            {
                "start": python_start,
                "end": python_end,
                "display_start": start,
                "display_end": end,
                "text": expected_text,
                "severity": severity,
                "missing": missing,
                "convention": convention,
            }
        )
    result.sort(key=lambda span: (span["start"], span["end"], not span["missing"]))
    cursor = 0
    for span in result:
        if span["start"] < cursor:
            return [], f"{label}: overlapping spans cannot be highlighted safely"
        if span["start"] != span["end"]:
            cursor = span["end"]
    return result, None


def render_highlighted(
    text: str, spans: Any, origin: str, *, human_inclusive_utf16: bool
) -> tuple[str, str]:
    safe_spans, warning = normalized_spans(
        text, spans, origin, human_inclusive_utf16=human_inclusive_utf16
    )
    if warning:
        return f'<div class="annotated-text">{html.escape(text)}</div>', (
            f'<div class="warning">{html.escape(warning)}</div>'
        )
    pieces: list[str] = []
    recovered_text: list[str] = []
    cursor = 0
    for span in safe_spans:
        start, end = span["start"], span["end"]
        pieces.append(html.escape(text[cursor:start]))
        recovered_text.append(text[cursor:start])
        css = f"span {origin.lower()} {span['severity'].lower()}"
        offset_label = f'[{span["display_start"]}:{span["display_end"]}] {span["convention"]}'
        if span["missing"]:
            pieces.append(
                f'<mark class="{css} missing" data-annotation-marker="missing" '
                f'title="{origin} {span["severity"]} at {html.escape(offset_label)}">'
                "[MISSING]</mark>"
            )
            cursor = start
        else:
            pieces.append(
                f'<mark class="{css}" title="{origin} {span["severity"]} '
                f'{html.escape(offset_label)}">{html.escape(text[start:end])}</mark>'
            )
            recovered_text.append(text[start:end])
            cursor = end
    pieces.append(html.escape(text[cursor:]))
    recovered_text.append(text[cursor:])
    if "".join(recovered_text) != text:
        return f'<div class="annotated-text">{html.escape(text)}</div>', (
            f'<div class="warning">{html.escape(origin)}: annotation rendering did not '
            "preserve the candidate text; highlighting was suppressed.</div>"
        )
    details = "".join(
        f'<li><strong>{span["severity"]}</strong> '
        f'[{span["display_start"]}:{span["display_end"]}] {span["convention"]} '
        f'{html.escape(span["text"])}</li>'
        for span in safe_spans
    )
    span_list = (
        f'<ul class="span-list">{details}</ul>'
        if details
        else '<div class="no-errors">No errors.</div>'
    )
    return f'<div class="annotated-text">{"".join(pieces)}</div>', span_list


def raw_fields(raw_response: Any) -> tuple[Any, Any]:
    if not isinstance(raw_response, str):
        return None, None
    try:
        value = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        return None, None
    if not isinstance(value, dict):
        return None, None
    return value.get("score"), value.get("errors")


def render_candidate(row: dict[str, Any]) -> str:
    evaluation_id = row["evaluation_id"]
    candidate = row["candidate"]
    text = candidate["text"]
    gold = row["human_gold"]
    prediction_row = row["v0_prediction"]
    human_text, human_spans = render_highlighted(
        text, gold.get("error_spans"), "Human", human_inclusive_utf16=True
    )

    parse_valid = prediction_row.get("parse_valid") is True
    prediction = prediction_row.get("prediction")
    if parse_valid and isinstance(prediction, dict):
        predicted_text, predicted_spans = render_highlighted(
            text,
            prediction.get("error_spans"),
            "Gemma",
            human_inclusive_utf16=False,
        )
        v0_body = (
            '<span class="status valid">Valid parsed prediction</span>'
            f'<div class="score">Score: <strong>{html.escape(str(prediction.get("score")))}</strong></div>'
            f"{predicted_text}{predicted_spans}"
        )
    else:
        raw_score, raw_errors = raw_fields(prediction_row.get("raw_model_output"))
        parse_errors = prediction_row.get("parse_errors")
        if not isinstance(parse_errors, list):
            parse_errors = ["parse_errors field is unavailable"]
        error_items = "".join(f"<li>{html.escape(str(error))}</li>" for error in parse_errors)
        raw_errors_text = (
            json.dumps(raw_errors, ensure_ascii=False, indent=2)
            if raw_errors is not None
            else "Raw error objects could not be read as exact JSON."
        )
        score_text = html.escape(str(raw_score)) if raw_score is not None else "unavailable"
        v0_body = (
            '<span class="status invalid">Invalid / unvalidated prediction</span>'
            f'<div class="score unvalidated">Raw score (unvalidated): <strong>{score_text}</strong></div>'
            f'<div class="parse-errors"><strong>Parsing errors:</strong><ul>{error_items}</ul></div>'
            '<div><strong>Raw error objects (not repaired or highlighted):</strong></div>'
            f'<pre>{html.escape(raw_errors_text)}</pre>'
        )

    raw_response = prediction_row.get("raw_model_output")
    raw_display = raw_response if isinstance(raw_response, str) else json.dumps(raw_response, ensure_ascii=False)
    return f"""
      <section class="candidate-card">
        <header><span class="system">{html.escape(candidate['system_id'])}</span><code>{html.escape(evaluation_id)}</code></header>
        <div class="candidate-full"><strong>Candidate:</strong> {html.escape(text)}</div>
        <div class="comparison-grid">
          <div class="judgment human-panel">
            <h4>Human gold</h4>
            <div class="score">Score: <strong>{html.escape(str(gold.get('score')))}</strong></div>
            {human_text}{human_spans}
          </div>
          <div class="judgment gemma-panel">
            <h4>Gemma V0</h4>
            {v0_body}
          </div>
        </div>
        <details><summary>Raw V0 response</summary><pre>{html.escape(raw_display)}</pre></details>
      </section>
    """


def render_report(
    passage_id: str,
    passage: dict[str, Any],
    system_order: list[str],
) -> str:
    order = {system: index for index, system in enumerate(system_order)}
    rows: list[str] = []
    invalid_count = 0
    for sentence in passage["sentences"]:
        candidates = sorted(
            sentence["candidates"], key=lambda row: order[row["candidate"]["system_id"]]
        )
        if len(candidates) != 4:
            raise ComparisonError(
                f"{passage_id}/{sentence['sentence_id']}: expected four candidates, observed {len(candidates)}"
            )
        invalid_count += sum(row["v0_prediction"].get("parse_valid") is not True for row in candidates)
        candidate_html = "".join(render_candidate(row) for row in candidates)
        rows.append(
            f"""
            <tr>
              <th scope="row">
                <div class="sentence-number">Sentence {html.escape(sentence['sentence_id'])}</div>
                <div class="position">Passage position {sentence['position']}</div>
                <p lang="pt-BR">{html.escape(sentence['source'])}</p>
              </th>
              <td><div class="candidate-stack">{candidate_html}</div></td>
            </tr>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V0 DEV comparison — {html.escape(passage_id)}</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #172033; background: #f4f6f9; }}
    body {{ margin: 0 auto; max-width: 1500px; padding: 24px; }}
    h1 {{ margin-bottom: 6px; }} .summary {{ color: #526078; margin-top: 0; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0; font-size: 14px; }}
    .legend span {{ padding: 4px 9px; border-radius: 4px; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0 14px; }}
    th[scope=row] {{ width: 25%; min-width: 240px; text-align: left; vertical-align: top; background: white; padding: 16px; border-radius: 8px 0 0 8px; }}
    td {{ background: white; padding: 12px; border-radius: 0 8px 8px 0; }}
    .sentence-number {{ font-weight: 750; }} .position {{ color: #6b7280; font-size: 13px; }}
    .candidate-stack {{ display: grid; gap: 12px; }}
    .candidate-card {{ border: 1px solid #d8dee9; border-radius: 7px; padding: 12px; background: #fbfcfe; }}
    .candidate-card header {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }}
    .system {{ font-weight: 750; }} code {{ color: #687386; }}
    .candidate-full {{ margin: 9px 0; line-height: 1.45; }}
    .comparison-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .judgment {{ border-radius: 6px; padding: 10px; min-width: 0; }}
    .human-panel {{ background: #fff8eb; border: 1px solid #f2d49b; }}
    .gemma-panel {{ background: #eef8ff; border: 1px solid #b9dcf3; }}
    h4 {{ margin: 0 0 5px; }} .score {{ margin: 4px 0 7px; }}
    .annotated-text {{ background: white; padding: 8px; border-radius: 4px; line-height: 1.65; overflow-wrap: anywhere; }}
    mark.span {{ padding: 1px 2px; border-radius: 3px; }}
    mark.human.minor, .legend .human-minor {{ background: #ffe49b; }}
    mark.human.major, .legend .human-major {{ background: #ffaaa5; }}
    mark.gemma.minor, .legend .gemma-minor {{ background: #9de5f3; }}
    mark.gemma.major, .legend .gemma-major {{ background: #d1b3ff; }}
    mark.missing {{ border: 1px dashed #202938; font-weight: 700; }}
    .no-errors {{ color: #55805d; font-size: 13px; white-space: nowrap; }}
    .span-list {{ margin: 6px 0 0; padding-left: 20px; font-size: 13px; }}
    .status {{ display: inline-block; font-weight: 700; font-size: 13px; padding: 3px 7px; border-radius: 4px; }}
    .status.valid {{ background: #ccebd2; color: #185928; }}
    .status.invalid {{ background: #ffd2cf; color: #8b1e18; }}
    .unvalidated {{ color: #8b1e18; }} .parse-errors {{ color: #8b1e18; }}
    .warning {{ color: #8b1e18; font-weight: 650; margin-top: 5px; }}
    details {{ margin-top: 8px; }} summary {{ cursor: pointer; color: #334e75; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #172033; color: #f5f7fa; padding: 10px; border-radius: 5px; font-size: 12px; }}
    @media (max-width: 900px) {{ .comparison-grid {{ grid-template-columns: 1fr; }} th[scope=row] {{ width: auto; min-width: 180px; }} }}
  </style>
</head>
<body>
  <h1>Gemma V0 DEV comparison</h1>
  <p class="summary"><strong>Passage:</strong> {html.escape(passage_id)} · <strong>Document:</strong> {html.escape(passage['document_id'])} · <strong>Sentences:</strong> {len(passage['sentences'])} · <strong>Candidate judgments:</strong> {len(passage['sentences']) * 4} · <strong>Invalid V0 outputs:</strong> {invalid_count}</p>
  <div class="legend">
    <span class="human-minor">Human Minor</span><span class="human-major">Human Major</span>
    <span class="gemma-minor">Gemma Minor</span><span class="gemma-major">Gemma Major</span>
  </div>
  <table>
    <thead><tr><th>BP source sentence</th><th>Candidate comparisons</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


def _assert_prompt_identity(
    evaluation_id: str,
    input_row: dict[str, Any],
    prediction_row: dict[str, Any],
    evaluator: str,
) -> None:
    """Confirm every saved model prompt contains the joined pilot input."""
    context = "\n".join(
        f"[{sentence['position']}] {sentence['source_bp']}"
        for sentence in input_row["passage_context"]
    )
    required = (
        "PORTUGUESE PASSAGE CONTEXT (BP — REFERENCE ONLY):\n" + context,
        "CURRENT BP SOURCE SENTENCE (REFERENCE ONLY, position "
        f"{input_row['position']}):\n{input_row['current_source_bp']}",
        "EP CANDIDATE TO EVALUATE (ONLY TEXT TO ANNOTATE):\n"
        + input_row["candidate_text"],
    )
    containers: list[tuple[str, dict[str, Any]]] = []
    stages = prediction_row.get("stages")
    if isinstance(stages, dict):
        containers.extend(
            (stage_name, stage)
            for stage_name, stage in stages.items()
            if isinstance(stage, dict) and isinstance(stage.get("messages"), list)
        )
    elif isinstance(prediction_row.get("messages"), list):
        containers.append(("generation", prediction_row))
    if not containers:
        raise ComparisonError(f"{evaluation_id}/{evaluator}: no saved model prompt")
    for stage_name, container in containers:
        messages = container["messages"]
        if (
            len(messages) != 1
            or not isinstance(messages[0], dict)
            or not isinstance(messages[0].get("content"), str)
        ):
            raise ComparisonError(
                f"{evaluation_id}/{evaluator}/{stage_name}: unexpected saved messages"
            )
        content = messages[0]["content"]
        missing = [fragment for fragment in required if fragment not in content]
        if missing:
            raise ComparisonError(
                f"{evaluation_id}/{evaluator}/{stage_name}: prompt/input mismatch"
            )


def load_v1_pilot() -> tuple[
    list[dict[str, Any]],
    OrderedDict[str, dict[str, Any]],
    OrderedDict[str, list[dict[str, Any]]],
]:
    inputs = read_jsonl(PILOT_INPUT_PATH)
    input_by_id = index_unique(inputs, "V1 pilot input")
    input_ids = [row["evaluation_id"] for row in inputs]
    synthetic_ids = [value for value in input_ids if value.startswith("diag-v0-")]
    dev_ids = [value for value in input_ids if value.startswith("dev-v0-")]
    if len(inputs) != 37 or len(synthetic_ids) != 25 or len(dev_ids) != 12:
        raise ComparisonError(
            "V1 pilot must contain exactly 37 unique rows split into 25 synthetic "
            f"and 12 DEV rows; observed {len(inputs)}/{len(synthetic_ids)}/{len(dev_ids)}"
        )

    canonical_rows = read_json(CANONICAL_PATH).get("records")
    if not isinstance(canonical_rows, list):
        raise ComparisonError("Canonical DEV records are missing")
    canonical_by_id = index_unique(canonical_rows, "canonical DEV")

    v0_dev = index_unique(read_jsonl(V0_OFFICIAL_PATH), "official V0 DEV")
    v0_synthetic = index_unique(read_jsonl(V0_SYNTHETIC_PATH), "V0 synthetic")
    v0_predictions = {**v0_dev, **v0_synthetic}
    prediction_sets: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict(
        (("V0", v0_predictions),)
    )
    expected_ids = set(input_ids)
    for evaluator, path in V1_PREDICTION_PATHS.items():
        predictions = index_unique(read_jsonl(path), evaluator)
        missing = sorted(expected_ids - set(predictions))
        unexpected = sorted(set(predictions) - expected_ids)
        if missing or unexpected:
            raise ComparisonError(
                f"{evaluator} ID mismatch: missing={missing or 'none'}, "
                f"unexpected={unexpected or 'none'}"
            )
        expected_variant = evaluator.removeprefix("V1-")
        wrong_variant = sorted(
            evaluation_id
            for evaluation_id, row in predictions.items()
            if row.get("variant") != expected_variant
        )
        if wrong_variant:
            raise ComparisonError(f"{evaluator}: wrong variant IDs {wrong_variant}")
        prediction_sets[evaluator] = predictions

    pilot_rows: list[dict[str, Any]] = []
    dev_groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    synthetic_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    expected_conditions = {
        "correct",
        "meaning-change",
        "major-omission",
        "unrelated",
        "source-conflict",
    }
    for input_row in inputs:
        evaluation_id = input_row["evaluation_id"]
        predictions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for evaluator, indexed in prediction_sets.items():
            if evaluation_id not in indexed:
                raise ComparisonError(f"{evaluator} is missing {evaluation_id}")
            prediction_row = indexed[evaluation_id]
            _assert_prompt_identity(evaluation_id, input_row, prediction_row, evaluator)
            predictions[evaluator] = prediction_row

        pilot_row = {"input": input_row, "predictions": predictions}
        if evaluation_id.startswith("dev-v0-"):
            canonical = canonical_by_id.get(evaluation_id)
            if canonical is None:
                raise ComparisonError(f"Canonical DEV is missing {evaluation_id}")
            candidate = canonical.get("candidate")
            if not isinstance(candidate, dict):
                raise ComparisonError(f"{evaluation_id}: missing canonical candidate")
            checks = {
                "passage_id": canonical.get("passage_id"),
                "sentence_id": canonical.get("sentence_id"),
                "position": canonical.get("position"),
                "current_source_bp": canonical.get("current_source_bp"),
                "candidate_text": candidate.get("text"),
            }
            mismatches = [
                key for key, value in checks.items() if input_row.get(key) != value
            ]
            if mismatches:
                raise ComparisonError(
                    f"{evaluation_id}: pilot/canonical mismatch in {mismatches}"
                )
            pilot_row["canonical"] = canonical
            group_key = f"{canonical['passage_id']}::{canonical['sentence_id']}"
            group = dev_groups.setdefault(
                group_key,
                {
                    "passage_id": canonical["passage_id"],
                    "sentence_id": canonical["sentence_id"],
                    "position": canonical["position"],
                    "source": canonical["current_source_bp"],
                    "rows": [],
                },
            )
            if (
                group["position"] != canonical["position"]
                or group["source"] != canonical["current_source_bp"]
            ):
                raise ComparisonError(f"{group_key}: inconsistent DEV sentence data")
            group["rows"].append(pilot_row)
        else:
            match = re.fullmatch(r"diag-v0-(s\d+)-(.+)", evaluation_id)
            if match is None or match.group(2) not in expected_conditions:
                raise ComparisonError(f"Unexpected synthetic evaluation ID: {evaluation_id}")
            pilot_row["synthetic_group"] = match.group(1)
            pilot_row["condition"] = match.group(2)
            synthetic_groups.setdefault(match.group(1), []).append(pilot_row)
        pilot_rows.append(pilot_row)

    for group_key, group in dev_groups.items():
        systems = [row["canonical"]["candidate"]["system_id"] for row in group["rows"]]
        if len(group["rows"]) != 4 or len(set(systems)) != 4:
            raise ComparisonError(f"{group_key}: expected four distinct candidate systems")
    for group_key, rows in synthetic_groups.items():
        conditions = {row["condition"] for row in rows}
        if len(rows) != 5 or conditions != expected_conditions:
            raise ComparisonError(
                f"{group_key}: expected the five diagnostic conditions, observed {conditions}"
            )
    if len(dev_groups) != 3 or len(synthetic_groups) != 5:
        raise ComparisonError(
            f"Expected 3 DEV and 5 synthetic groups, observed "
            f"{len(dev_groups)} and {len(synthetic_groups)}"
        )
    return pilot_rows, dev_groups, synthetic_groups


def _span_relation(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    if left["start"] == left["end"] or right["start"] == right["end"]:
        if (
            left["start"] == left["end"] == right["start"] == right["end"]
        ):
            return "exact"
        return None
    if left["start"] == right["start"] and left["end"] == right["end"]:
        return "exact"
    if max(left["start"], right["start"]) < min(left["end"], right["end"]):
        return "overlap"
    return None


def _span_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["start"] == left["end"] or right["start"] == right["end"]:
        return (
            1.0
            if left["start"] == left["end"] == right["start"] == right["end"]
            else 0.0
        )
    intersection = max(
        0, min(left["end"], right["end"]) - max(left["start"], right["start"])
    )
    union = max(left["end"], right["end"]) - min(left["start"], right["start"])
    return intersection / union if intersection else 0.0


def _one_to_one_overlap_pairs(
    human_spans: list[dict[str, Any]], predicted_spans: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Mirror the quantitative maximum-cardinality, maximum-IoU matching."""

    @lru_cache(maxsize=None)
    def solve(
        human_index: int, used_mask: int
    ) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if human_index == len(human_spans):
            return 0, 0.0, ()
        best = solve(human_index + 1, used_mask)
        for predicted_index, predicted in enumerate(predicted_spans):
            if used_mask & (1 << predicted_index):
                continue
            weight = _span_iou(human_spans[human_index], predicted)
            if weight <= 0:
                continue
            count, total_weight, pairs = solve(
                human_index + 1, used_mask | (1 << predicted_index)
            )
            candidate = (
                count + 1,
                total_weight + weight,
                ((human_index, predicted_index),) + pairs,
            )
            if (candidate[0], candidate[1]) > (best[0], best[1]):
                best = candidate
        return best

    return list(solve(0, 0)[2])


def _prediction_details(evaluator: str, row: dict[str, Any]) -> str:
    blocks: list[str] = []
    stages = row.get("stages")
    if isinstance(stages, dict):
        for stage_name, stage in stages.items():
            if not isinstance(stage, dict):
                continue
            status = stage.get("status")
            parse_valid = stage.get("parse_valid")
            status_text = (
                f"status={status}; reason={stage.get('reason')}"
                if status is not None
                else f"parse_valid={parse_valid}"
            )
            prediction_text = json.dumps(
                stage.get("prediction"), ensure_ascii=False, indent=2
            )
            raw = stage.get("raw_model_output")
            raw_text = raw if isinstance(raw, str) else "No model continuation (stage skipped)."
            raw_special = stage.get("raw_model_output_with_special_tokens")
            thinking = stage.get("thinking_output")
            final_answer = stage.get("final_answer")
            errors = json.dumps(stage.get("parse_errors", []), ensure_ascii=False)
            blocks.append(
                f"<h5>{html.escape(stage_name)}</h5>"
                f"<div class=\"stage-status\">{html.escape(status_text)}; "
                f"errors={html.escape(errors)}</div>"
                f"<div class=\"detail-label\">Parsed stage prediction</div>"
                f"<pre>{html.escape(prediction_text)}</pre>"
                f"<div class=\"detail-label\">Exact raw continuation</div>"
                f"<pre>{html.escape(raw_text)}</pre>"
                + (
                    '<div class="detail-label">Raw continuation with channel tokens</div>'
                    f'<pre>{html.escape(str(raw_special))}</pre>'
                    '<div class="detail-label">Extracted thinking</div>'
                    f'<pre>{html.escape(str(thinking))}</pre>'
                    '<div class="detail-label">Extracted final answer</div>'
                    f'<pre>{html.escape(str(final_answer))}</pre>'
                    if raw_special is not None
                    else ""
                )
            )
    else:
        raw = row.get("raw_model_output")
        raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        blocks.append(
            '<div class="detail-label">Parsed prediction</div>'
            f'<pre>{html.escape(json.dumps(row.get("prediction"), ensure_ascii=False, indent=2))}</pre>'
            '<div class="detail-label">Exact raw continuation</div>'
            f"<pre>{html.escape(raw_text)}</pre>"
        )
    return (
        f"<details><summary>{html.escape(evaluator)} raw and stage details</summary>"
        f"{''.join(blocks)}</details>"
    )


def _render_prediction_text(
    candidate: str,
    spans: list[dict[str, Any]],
    human_spans: list[dict[str, Any]] | None,
) -> tuple[str, str, list[dict[str, Any]]]:
    pairs = (
        _one_to_one_overlap_pairs(human_spans, spans)
        if human_spans is not None
        else []
    )
    matched_human = {human_index for human_index, _ in pairs}
    predicted_to_human = {
        predicted_index: human_index for human_index, predicted_index in pairs
    }
    pieces: list[str] = []
    cursor = 0
    for predicted_index, span in enumerate(spans):
        start, end = span["start"], span["end"]
        pieces.append(html.escape(candidate[cursor:start]))
        human_index = predicted_to_human.get(predicted_index)
        relation = (
            _span_relation(span, human_spans[human_index])
            if human_spans is not None and human_index is not None
            else None
        )
        agreement_class = "matches-gold" if relation else "additional"
        relation_label = (
            "exact human span"
            if relation == "exact"
            else "overlaps human span"
            if relation == "overlap"
            else "additional prediction"
            if human_spans is not None
            else "diagnostic prediction"
        )
        marker = "[MISSING]" if span["missing"] else candidate[start:end]
        pieces.append(
            f'<mark class="model-span {span["severity"].lower()} {agreement_class}" '
            f'title="{html.escape(span["severity"] + "; " + relation_label)}">'
            f"{html.escape(marker)}</mark>"
        )
        if not span["missing"]:
            cursor = end
        else:
            cursor = start
    pieces.append(html.escape(candidate[cursor:]))

    items: list[str] = []
    for predicted_index, span in enumerate(spans):
        human_index = predicted_to_human.get(predicted_index)
        relation = (
            _span_relation(span, human_spans[human_index])
            if human_spans is not None and human_index is not None
            else None
        )
        label = (
            "exact human span"
            if relation == "exact"
            else "overlaps human span"
            if relation == "overlap"
            else "additional prediction"
            if human_spans is not None
            else "prediction"
        )
        items.append(
            f'<li><strong>{span["severity"]}</strong> {html.escape(span["text"])} '
            f'<span class="relation {"matched" if relation else "extra"}">{label}</span></li>'
        )
    listing = (
        f'<ul class="span-list">{"".join(items)}</ul>'
        if items
        else '<div class="no-errors">No predicted errors.</div>'
    )
    missed = []
    if human_spans is not None:
        missed = [
            human
            for human_index, human in enumerate(human_spans)
            if human_index not in matched_human
        ]
    return f'<div class="annotated-text">{"".join(pieces)}</div>', listing, missed


def render_pilot_prediction(
    evaluator: str,
    prediction_row: dict[str, Any],
    candidate: str,
    *,
    human_spans: list[dict[str, Any]] | None,
    baseline_score: int | None = None,
    is_baseline_condition: bool = False,
) -> str:
    parse_valid = prediction_row.get("parse_valid") is True
    prediction = prediction_row.get("prediction")
    body: list[str] = []
    if parse_valid and isinstance(prediction, dict):
        safe_spans, warning = normalized_spans(
            candidate,
            prediction.get("error_spans"),
            f"{evaluator} prediction",
            human_inclusive_utf16=False,
        )
        score = prediction.get("score")
        if warning:
            body.extend(
                [
                    '<span class="status invalid">Display validation failed</span>',
                    f'<div class="warning">{html.escape(warning)}</div>',
                    f'<div class="annotated-text">{html.escape(candidate)}</div>',
                ]
            )
        else:
            annotated, listing, missed = _render_prediction_text(
                candidate, safe_spans, human_spans
            )
            delta_text = ""
            if baseline_score is not None and isinstance(score, int):
                delta = score - baseline_score
                delta_text = (
                    '<span class="delta baseline">correct baseline</span>'
                    if is_baseline_condition
                    else f'<span class="delta">Δ vs correct: {delta:+d}</span>'
                )
            body.extend(
                [
                    '<span class="status valid">Valid</span>',
                    f'<div class="score">Score: <strong>{html.escape(str(score))}</strong>{delta_text}</div>',
                    annotated,
                    listing,
                ]
            )
            if human_spans is not None:
                if missed:
                    missed_items = "".join(
                        f'<li><strong>{span["severity"]}</strong> {html.escape(span["text"])}</li>'
                        for span in missed
                    )
                    body.append(
                        f'<div class="missed"><strong>Missed human errors:</strong><ul>{missed_items}</ul></div>'
                    )
                else:
                    body.append('<div class="matched-all">No human span missed.</div>')
    else:
        errors = prediction_row.get("parse_errors")
        if not isinstance(errors, list):
            errors = ["parse_errors unavailable"]
        error_items = "".join(f"<li>{html.escape(str(error))}</li>" for error in errors)
        body.extend(
            [
                '<span class="status invalid">Invalid / no validated score</span>',
                f'<div class="parse-errors"><strong>Reasons:</strong><ul>{error_items}</ul></div>',
            ]
        )
        if any("text_not_in_candidate" in str(error) for error in errors):
            body.append(
                '<div class="source-span-alert">Candidate-substring violation: inspect the '
                "raw stage output for text copied from the BP source.</div>"
            )
        stages = prediction_row.get("stages")
        if isinstance(stages, dict):
            skipped = [
                f"{name}: {stage.get('reason', 'skipped')}"
                for name, stage in stages.items()
                if isinstance(stage, dict) and stage.get("status") == "skipped"
            ]
            if skipped:
                body.append(
                    '<div class="skipped"><strong>Skipped:</strong> '
                    + html.escape("; ".join(skipped))
                    + "</div>"
                )
    return (
        f'<section class="model-card model-{html.escape(evaluator.lower())}">'
        f"<h4>{html.escape(evaluator)}</h4>{''.join(body)}"
        f"{_prediction_details(evaluator, prediction_row)}</section>"
    )


def _valid_score(row: dict[str, Any]) -> int | None:
    prediction = row.get("prediction")
    if row.get("parse_valid") is True and isinstance(prediction, dict):
        score = prediction.get("score")
        if isinstance(score, int) and not isinstance(score, bool):
            return score
    return None


def render_v1_pilot_report(
    *,
    extra_prediction_paths: OrderedDict[str, tuple[Path, str]] | None = None,
    evaluators: tuple[str, ...] | None = None,
    title: str = "V0/V1 37-input pilot comparison",
) -> str:
    pilot_rows, dev_groups, synthetic_groups = load_v1_pilot()
    input_by_id = {
        row["input"]["evaluation_id"]: row for row in pilot_rows
    }
    if extra_prediction_paths:
        expected_ids = set(input_by_id)
        expected_prompt_hashes = {
            "stage1": "fde2e0a7224edd694c66268d40d4a5f448e5e617a362b0bdb69e3803650f90ed",
            "stage2": "d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd",
        }
        for label, (path, experiment_id) in extra_prediction_paths.items():
            indexed = index_unique(read_jsonl(path), label)
            missing = sorted(expected_ids - set(indexed))
            unexpected = sorted(set(indexed) - expected_ids)
            if missing or unexpected:
                raise ComparisonError(
                    f"{label} ID mismatch: missing={missing or 'none'}, "
                    f"unexpected={unexpected or 'none'}"
                )
            for evaluation_id, prediction_row in indexed.items():
                if prediction_row.get("experiment_id") != experiment_id:
                    raise ComparisonError(
                        f"{evaluation_id}/{label}: wrong experiment identifier"
                    )
                if prediction_row.get("prompt_hashes") != expected_prompt_hashes:
                    raise ComparisonError(
                        f"{evaluation_id}/{label}: V1-A prompt hashes do not match"
                    )
                pilot_row = input_by_id[evaluation_id]
                _assert_prompt_identity(
                    evaluation_id, pilot_row["input"], prediction_row, label
                )
                pilot_row["predictions"][label] = prediction_row
    if evaluators is None:
        evaluators = ("V0", *V1_PREDICTION_PATHS.keys())
    dev_sections: list[str] = []
    invalid_counts = {evaluator: 0 for evaluator in evaluators}
    for group in dev_groups.values():
        candidates: list[str] = []
        for pilot_row in group["rows"]:
            canonical = pilot_row["canonical"]
            evaluation_id = canonical["evaluation_id"]
            candidate = canonical["candidate"]
            text = candidate["text"]
            gold = canonical["human_gold"]
            human_safe, human_warning = normalized_spans(
                text,
                gold.get("error_spans"),
                f"{evaluation_id} human",
                human_inclusive_utf16=True,
            )
            if human_warning:
                raise ComparisonError(human_warning)
            human_text, human_listing = render_highlighted(
                text, gold.get("error_spans"), "Human", human_inclusive_utf16=True
            )
            score_strip = [f"Human {gold.get('score')}"]
            model_panels: list[str] = []
            for evaluator in evaluators:
                prediction_row = pilot_row["predictions"][evaluator]
                score = _valid_score(prediction_row)
                score_strip.append(f"{evaluator} {score if score is not None else 'invalid'}")
                invalid_counts[evaluator] += score is None
                model_panels.append(
                    render_pilot_prediction(
                        evaluator,
                        prediction_row,
                        text,
                        human_spans=human_safe,
                    )
                )
            candidates.append(
                f"""
                <article class="candidate-card">
                  <header><strong>{html.escape(candidate['system_id'])}</strong><code>{html.escape(evaluation_id)}</code></header>
                  <div class="candidate-full" lang="pt-PT">{html.escape(text)}</div>
                  <div class="score-strip">{html.escape(' · '.join(score_strip))}</div>
                  <div class="judgment-grid">
                    <section class="model-card human-panel">
                      <h4>Human gold</h4>
                      <div class="score">Score: <strong>{html.escape(str(gold.get('score')))}</strong></div>
                      {human_text}{human_listing}
                    </section>
                    {''.join(model_panels)}
                  </div>
                </article>
                """
            )
        dev_sections.append(
            f"""
            <section class="sentence-group">
              <h3>{html.escape(group['passage_id'])} · sentence {html.escape(str(group['sentence_id']))}</h3>
              <div class="source"><span>BP source · position {group['position']}</span><p lang="pt-BR">{html.escape(group['source'])}</p></div>
              {''.join(candidates)}
            </section>
            """
        )

    synthetic_sections: list[str] = []
    for group_id, rows in synthetic_groups.items():
        correct_row = next(row for row in rows if row["condition"] == "correct")
        baselines = {
            evaluator: _valid_score(correct_row["predictions"][evaluator])
            for evaluator in evaluators
        }
        condition_cards: list[str] = []
        for pilot_row in rows:
            input_row = pilot_row["input"]
            condition = pilot_row["condition"]
            candidate = input_row["candidate_text"]
            model_panels = []
            for evaluator in evaluators:
                model_panels.append(
                    render_pilot_prediction(
                        evaluator,
                        pilot_row["predictions"][evaluator],
                        candidate,
                        human_spans=None,
                        baseline_score=baselines[evaluator],
                        is_baseline_condition=condition == "correct",
                    )
                )
            condition_cards.append(
                f"""
                <article class="diagnostic-card">
                  <h4>{html.escape(condition)}</h4>
                  <div class="diagnostic-pair"><strong>BP source:</strong> <span lang="pt-BR">{html.escape(input_row['current_source_bp'])}</span></div>
                  <div class="diagnostic-pair"><strong>EP candidate:</strong> <span lang="pt-PT">{html.escape(candidate)}</span></div>
                  <code>{html.escape(input_row['evaluation_id'])}</code>
                  <div class="judgment-grid">{''.join(model_panels)}</div>
                </article>
                """
            )
        synthetic_sections.append(
            f"""
            <details class="diagnostic-group">
              <summary>Synthetic group {html.escape(group_id)} · five conditions</summary>
              {''.join(condition_cards)}
            </details>
            """
        )

    invalid_summary = " · ".join(
        f"{evaluator}: {count}/12 invalid" for evaluator, count in invalid_counts.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #172033; background: #f3f5f8; }}
    * {{ box-sizing: border-box; }}
    body {{ max-width: 1500px; margin: auto; padding: 24px; }}
    h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 32px; }} h3 {{ margin: 0 0 12px; }}
    .summary {{ color: #526078; margin-top: 0; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; font-size: 13px; }}
    .legend span, .relation, .delta {{ border-radius: 4px; padding: 2px 6px; }}
    .human-minor, mark.human.minor {{ background: #ffe49b; }}
    .human-major, mark.human.major {{ background: #ffaaa5; }}
    .pred-minor, mark.model-span.minor {{ background: #aee8f1; }}
    .pred-major, mark.model-span.major {{ background: #d6bcfa; }}
    .match-key, mark.matches-gold {{ outline: 2px solid #2d8a51; }}
    .extra-key, mark.additional {{ border-bottom: 3px solid #c2413b; }}
    .sentence-group, .diagnostic-group {{ display: block; background: white; border: 1px solid #d8dee9; border-radius: 10px; padding: 16px; margin: 16px 0; }}
    .source {{ background: #eef2f7; border-radius: 7px; padding: 10px 12px; margin-bottom: 12px; }}
    .source span {{ color: #5e6a7d; font-size: 13px; font-weight: 700; }} .source p {{ margin: 5px 0 0; }}
    .candidate-card, .diagnostic-card {{ border: 1px solid #dbe1ea; border-radius: 8px; padding: 12px; margin-top: 12px; background: #fbfcfe; }}
    .candidate-card > header {{ display: flex; justify-content: space-between; gap: 12px; }}
    code {{ color: #657186; font-size: 12px; }}
    .candidate-full {{ font-size: 15px; line-height: 1.45; margin: 8px 0; }}
    .score-strip {{ color: #445166; font-size: 13px; margin-bottom: 10px; }}
    .judgment-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); gap: 9px; margin-top: 10px; }}
    .model-card {{ min-width: 0; background: #edf7ff; border: 1px solid #bdd9ed; border-radius: 7px; padding: 9px; }}
    .human-panel {{ background: #fff8eb; border-color: #f1d39a; }}
    .model-card h4 {{ margin: 0 0 6px; }} .score {{ margin: 5px 0; }}
    .annotated-text {{ background: white; border-radius: 4px; line-height: 1.7; overflow-wrap: anywhere; padding: 7px; }}
    mark {{ padding: 1px 2px; border-radius: 3px; }} mark.model-span {{ color: inherit; }}
    mark.missing, mark.model-span:empty {{ border: 1px dashed #263243; font-weight: 700; }}
    .span-list, .missed ul, .parse-errors ul {{ margin: 6px 0 0; padding-left: 19px; font-size: 12px; }}
    .no-errors, .matched-all {{ color: #397049; font-size: 12px; margin-top: 5px; }}
    .missed {{ color: #9d2d28; font-size: 12px; margin-top: 7px; }}
    .relation {{ font-size: 10px; white-space: nowrap; }} .relation.matched {{ background: #ccebd2; }} .relation.extra {{ background: #ffd8d4; }}
    .status {{ display: inline-block; font-size: 12px; font-weight: 750; padding: 2px 6px; border-radius: 4px; }}
    .status.valid {{ background: #ccebd2; color: #185928; }} .status.invalid {{ background: #ffd2cf; color: #8b1e18; }}
    .delta {{ background: #e6eaf0; margin-left: 7px; font-size: 11px; }} .delta.baseline {{ color: #526078; }}
    .parse-errors, .warning, .skipped {{ color: #8b1e18; font-size: 12px; margin-top: 6px; }}
    .source-span-alert {{ color: #7c2d12; background: #ffedd5; border: 1px solid #fb923c; border-radius: 4px; padding: 5px; margin-top: 6px; font-size: 12px; font-weight: 700; }}
    details {{ margin-top: 8px; }} summary {{ cursor: pointer; color: #334e75; font-weight: 650; }}
    .diagnostic-group > summary {{ font-size: 17px; }} .diagnostic-card h4 {{ margin: 0 0 6px; }}
    .diagnostic-pair {{ line-height: 1.45; margin: 4px 0; }}
    .detail-label, .stage-status {{ font-size: 11px; font-weight: 700; margin: 7px 0 3px; overflow-wrap: anywhere; }}
    h5 {{ margin: 10px 0 2px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; max-height: 260px; overflow-y: auto; background: #172033; color: #f5f7fa; padding: 8px; border-radius: 5px; font-size: 11px; }}
    @media (max-width: 600px) {{ body {{ padding: 12px; }} .judgment-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="summary">Three formal DEV sentences (12 candidate judgments) plus five synthetic diagnostic groups (25 judgments). {html.escape(invalid_summary)}</p>
  <div class="legend">
    <span class="human-minor">Human Minor</span><span class="human-major">Human Major</span>
    <span class="pred-minor">Predicted Minor</span><span class="pred-major">Predicted Major</span>
    <span class="match-key">Prediction overlaps human</span><span class="extra-key">Additional prediction</span>
  </div>
  <h2>Formal DEV comparison</h2>
  <p>Green-outlined predictions are one-to-one overlap matches with human spans; red-underlined predictions are additional. Each valid evaluator panel explicitly lists gold spans left unmatched by the same maximum-cardinality, maximum-IoU protocol used quantitatively. Human offsets use verified inclusive UTF-16 conversion and model offsets are exclusive Python indexes.</p>
  {''.join(dev_sections)}
  <details class="diagnostic-group">
    <summary>Synthetic source-sensitivity comparison (diagnostic expectations, not formal gold)</summary>
    <p>Score deltas are relative to the same evaluator's valid score for the correct-candidate condition in that sentence group.</p>
    {''.join(synthetic_sections)}
  </details>
</body>
</html>
"""


def _comparison_signal(
    candidate: str,
    human_score: int,
    human_spans: list[dict[str, Any]],
    g0: dict[str, Any],
    g1: dict[str, Any],
) -> str:
    values: list[tuple[int, int]] = []
    for label, row in (("G0", g0), ("G1", g1)):
        prediction = row.get("prediction")
        if row.get("parse_valid") is not True or not isinstance(prediction, dict):
            return '<span class="signal neutral">G0/G1 comparison unavailable: invalid output</span>'
        spans, warning = normalized_spans(
            candidate,
            prediction.get("error_spans"),
            f"{label} prediction",
            human_inclusive_utf16=False,
        )
        score = prediction.get("score")
        if warning or not isinstance(score, int):
            return '<span class="signal neutral">G0/G1 comparison unavailable: display validation failed</span>'
        values.append((abs(score - human_score), len(_one_to_one_overlap_pairs(human_spans, spans))))
    (g0_distance, g0_matches), (g1_distance, g1_matches) = values
    score_direction = "closer" if g1_distance < g0_distance else "farther" if g1_distance > g0_distance else "same score distance"
    span_direction = "more" if g1_matches > g0_matches else "fewer" if g1_matches < g0_matches else "same number of"
    css_class = (
        "improved"
        if g1_distance <= g0_distance and g1_matches >= g0_matches and (g1_distance, g1_matches) != (g0_distance, g0_matches)
        else "regressed"
        if g1_distance >= g0_distance and g1_matches <= g0_matches and (g1_distance, g1_matches) != (g0_distance, g0_matches)
        else "neutral"
    )
    return (
        f'<span class="signal {css_class}">G1: {score_direction} to human score; '
        f'{span_direction} matched human spans ({g0_matches}→{g1_matches})</span>'
    )


def render_g0_g1_report(*, five_way: bool = False) -> str:
    manifest_path = G24_MANIFEST_PATH if five_way else G_MANIFEST_PATH
    input_path = G24_INPUT_PATH if five_way else G_INPUT_PATH
    prediction_paths = G24_PREDICTION_PATHS if five_way else G_PREDICTION_PATHS
    expected_count = 72 if five_way else 96
    expected_groups = 18 if five_way else 24
    manifest = read_json(manifest_path)
    inputs = read_jsonl(input_path)
    input_by_id = index_unique(inputs, "G0/G4 blind input" if five_way else "G0/G1 blind input")
    expected_ids = manifest.get("evaluation_ids")
    if not isinstance(expected_ids, list) or len(expected_ids) != expected_count:
        raise ComparisonError(f"Frozen manifest does not contain {expected_count} evaluation IDs")
    if [row["evaluation_id"] for row in inputs] != expected_ids:
        raise ComparisonError("G0/G1 blind input IDs/order differ from frozen manifest")
    canonical = read_json(CANONICAL_PATH).get("records")
    if not isinstance(canonical, list):
        raise ComparisonError("Canonical DEV records are unavailable")
    canonical_by_id = index_unique(canonical, "canonical DEV")
    expected_hashes = {
        "stage1": "fde2e0a7224edd694c66268d40d4a5f448e5e617a362b0bdb69e3803650f90ed",
        "stage2": "d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd",
    }
    predictions: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
    for label, path in prediction_paths.items():
        indexed = index_unique(read_jsonl(path), label)
        if not set(expected_ids) <= set(indexed):
            raise ComparisonError(f"{label}: prediction IDs do not cover frozen evaluation IDs")
        indexed = {evaluation_id: indexed[evaluation_id] for evaluation_id in expected_ids}
        experiment_id = label.split()[0]
        for evaluation_id, row in indexed.items():
            if row.get("experiment_id") != experiment_id or row.get("prompt_hashes") != expected_hashes:
                raise ComparisonError(f"{evaluation_id}/{label}: frozen configuration mismatch")
            _assert_prompt_identity(evaluation_id, input_by_id[evaluation_id], row, label)
        predictions[label] = indexed

    pilot_ids = set(manifest.get("previously_inspected_evaluation_ids", []))
    grouped: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
    for evaluation_id in expected_ids:
        row = canonical_by_id.get(evaluation_id)
        input_row = input_by_id[evaluation_id]
        if row is None:
            raise ComparisonError(f"{evaluation_id}: absent from canonical DEV")
        if (
            row["candidate"]["text"] != input_row["candidate_text"]
            or row["current_source_bp"] != input_row["current_source_bp"]
            or row["document_id"] != input_row["document_id"]
        ):
            raise ComparisonError(f"{evaluation_id}: canonical/blind text or document mismatch")
        grouped.setdefault((row["passage_id"], row["sentence_id"]), []).append(evaluation_id)
    if len(grouped) != expected_groups or any(len(values) != 4 for values in grouped.values()):
        raise ComparisonError(f"Expected {expected_groups} complete four-candidate sentence groups")

    invalid_counts = {label: 0 for label in predictions}
    sections: list[str] = []
    for (passage_id, sentence_id), group_ids in grouped.items():
        first = canonical_by_id[group_ids[0]]
        inspected = set(group_ids) <= pilot_ids
        cards: list[str] = []
        for evaluation_id in group_ids:
            row = canonical_by_id[evaluation_id]
            candidate = row["candidate"]["text"]
            gold = row["human_gold"]
            human_spans, warning = normalized_spans(
                candidate,
                gold.get("error_spans"),
                f"{evaluation_id} human",
                human_inclusive_utf16=True,
            )
            if warning:
                raise ComparisonError(warning)
            human_text, human_listing = render_highlighted(
                candidate,
                gold.get("error_spans"),
                "Human",
                human_inclusive_utf16=True,
            )
            model_panels: list[str] = []
            scores = [f"Human {gold['score']}"]
            for label, indexed in predictions.items():
                prediction_row = indexed[evaluation_id]
                score = _valid_score(prediction_row)
                invalid_counts[label] += score is None
                scores.append(f"{label} {score if score is not None else 'invalid'}")
                model_panels.append(
                    render_pilot_prediction(
                        label,
                        prediction_row,
                        candidate,
                        human_spans=human_spans,
                    )
                )
            signal = (
                ""
                if five_way
                else _comparison_signal(
                    candidate,
                    gold["score"],
                    human_spans,
                    predictions["G0 zero-shot"][evaluation_id],
                    predictions["G1 eight-shot"][evaluation_id],
                )
            )
            cards.append(
                f'<article class="candidate-card"><header><strong>{html.escape(row["candidate"]["system_id"])}</strong>'
                f'<code>{html.escape(evaluation_id)}</code></header><div class="candidate-full" lang="pt-PT">{html.escape(candidate)}</div>'
                f'<div class="score-strip">{html.escape(" · ".join(scores))}</div>{signal}'
                f'<div class="judgment-grid"><section class="model-card human-panel"><h4>Human gold</h4>'
                f'<div class="score">Score: <strong>{gold["score"]}</strong></div>{human_text}{human_listing}</section>'
                f'{"".join(model_panels)}</div></article>'
            )
        sample_label = "Previously inspected pilot" if inspected else "Newly selected"
        sections.append(
            f'<section class="sentence-group {"pilot" if inspected else "new"}"><h3>{html.escape(passage_id)} · sentence {html.escape(str(sentence_id))} '
            f'<span class="sample-label">{sample_label}</span></h3><div class="source"><span>BP source · position {first["position"]}</span>'
            f'<p lang="pt-BR">{html.escape(first["current_source_bp"])}</p></div>{"".join(cards)}</section>'
        )
    invalid_summary = " · ".join(f"{label}: {count}/{expected_count} invalid" for label, count in invalid_counts.items())
    title = "Gemma 31B G0–G4 common DEV comparison" if five_way else "Gemma 31B G0/G1 paired DEV comparison"
    heading = "Gemma 31B G0/G1/G2/G3/G4" if five_way else "Gemma 31B G0 zero-shot vs G1 eight-shot"
    sample_summary = (
        "18 sentence groups, 72 candidate judgments (64 newly selected, 8 previously inspected)"
        if five_way else
        "24 sentence groups, 96 candidate judgments (84 newly selected, 12 previously inspected)"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>
:root {{ font-family: system-ui,sans-serif; color:#172033; background:#f3f5f8 }} * {{ box-sizing:border-box }}
body {{ max-width:1500px; margin:auto; padding:24px }} .summary {{ color:#526078 }}
.sentence-group {{ background:#fff; border:1px solid #d8dee9; border-radius:10px; padding:16px; margin:16px 0 }}
.sentence-group.pilot {{ border-left:7px solid #7c3aed }} .sentence-group.new {{ border-left:7px solid #2563eb }}
.sample-label,.signal,.status {{ display:inline-block; border-radius:4px; padding:2px 6px; font-size:12px }}
.sample-label {{ background:#e8edf5; vertical-align:middle }} .signal {{ margin:0 0 8px }} .signal.improved {{ background:#ccebd2 }}
.signal.regressed {{ background:#ffd2cf }} .signal.neutral {{ background:#e6eaf0 }}
.source {{ background:#eef2f7; border-radius:7px; padding:10px 12px; margin-bottom:12px }} .source span {{ color:#5e6a7d; font-size:13px; font-weight:700 }} .source p {{ margin:5px 0 0 }}
.candidate-card {{ border:1px solid #dbe1ea; border-radius:8px; padding:12px; margin-top:12px; background:#fbfcfe }}
.candidate-card>header {{ display:flex; justify-content:space-between; gap:12px }} code {{ color:#657186; font-size:12px }}
.candidate-full {{ margin:8px 0; line-height:1.45 }} .score-strip {{ color:#445166; font-size:13px; margin-bottom:7px }}
.judgment-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:9px; margin-top:8px }}
.model-card {{ min-width:0; background:#edf7ff; border:1px solid #bdd9ed; border-radius:7px; padding:9px }} .human-panel {{ background:#fff8eb; border-color:#f1d39a }}
.model-card h4 {{ margin:0 0 6px }} .annotated-text {{ background:#fff; border-radius:4px; line-height:1.7; overflow-wrap:anywhere; padding:7px }}
mark {{ padding:1px 2px; border-radius:3px }} mark.human.minor {{ background:#ffe49b }} mark.human.major {{ background:#ffaaa5 }} mark.model-span.minor {{ background:#aee8f1 }} mark.model-span.major {{ background:#d6bcfa }}
mark.matches-gold {{ outline:2px solid #2d8a51 }} mark.additional {{ border-bottom:3px solid #c2413b }}
.span-list,.missed ul,.parse-errors ul {{ margin:6px 0 0; padding-left:19px; font-size:12px }} .missed,.parse-errors,.warning,.skipped {{ color:#9d2d28; font-size:12px }}
.status.valid {{ background:#ccebd2; color:#185928 }} .status.invalid {{ background:#ffd2cf; color:#8b1e18 }} .no-errors,.matched-all {{ color:#397049; font-size:12px }}
.relation {{ font-size:10px; border-radius:4px; padding:2px 6px }} .relation.matched {{ background:#ccebd2 }} .relation.extra {{ background:#ffd8d4 }}
details {{ margin-top:8px }} summary {{ cursor:pointer; color:#334e75; font-weight:650 }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; max-height:260px; overflow:auto; background:#172033; color:#f5f7fa; padding:8px; border-radius:5px; font-size:11px }}
.detail-label,.stage-status {{ font-size:11px; font-weight:700; margin:7px 0 3px; overflow-wrap:anywhere }} h5 {{ margin:10px 0 2px }}
@media(max-width:600px) {{ body {{ padding:12px }} .judgment-grid {{ grid-template-columns:1fr }} }}
</style></head><body><h1>{html.escape(heading)}</h1>
<p class="summary">Frozen paired DEV subset: {html.escape(sample_summary)}. {html.escape(invalid_summary)}</p>
<p>Human spans use verified inclusive UTF-16 conversion; evaluator spans use exclusive Python offsets. Green outlines are one-to-one overlap matches, red underlines are additional predictions. Unmatched spans show mechanical gold overlap only and require human linguistic interpretation.</p>
{''.join(sections)}</body></html>"""


def main() -> None:
    args = parse_args()
    try:
        if args.v1_pilot:
            V1_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
            V1_COMPARISON_PATH.write_text(render_v1_pilot_report(), encoding="utf-8")
            print(V1_COMPARISON_PATH)
            return
        if args.reasoning_pilot:
            REASONING_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
            REASONING_COMPARISON_PATH.write_text(
                render_v1_pilot_report(
                    extra_prediction_paths=REASONING_PREDICTION_PATHS,
                    evaluators=("V1-A", "F1 thinking", "F2 31B"),
                    title="V1-A vs thinking vs larger-model comparison",
                ),
                encoding="utf-8",
            )
            print(REASONING_COMPARISON_PATH)
            return
        if args.g0_g1:
            G_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
            G_COMPARISON_PATH.write_text(render_g0_g1_report(), encoding="utf-8")
            print(G_COMPARISON_PATH)
            return
        if args.g0_g4:
            G24_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
            G24_COMPARISON_PATH.write_text(
                render_g0_g1_report(five_way=True), encoding="utf-8"
            )
            print(G24_COMPARISON_PATH)
            return
        joined, _, system_order = load_and_match(args.predictions)
        passages = passages_in_order(joined)
        if args.list_passages:
            for passage_id, passage in passages.items():
                print(f"{passage_id}\t{len(passage['sentences'])} sentences\t{passage['document_id']}")
            return
        passage_id = args.passage_id
        if passage_id not in passages:
            raise ComparisonError(
                f"Unknown DEV passage {passage_id!r}. Use --list-passages to see valid IDs."
            )
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", passage_id)
        output_path = OUTPUT_DIR / f"v0_comparison_{safe_name}.html"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            render_report(passage_id, passages[passage_id], system_order), encoding="utf-8"
        )
        print(output_path)
    except ComparisonError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
