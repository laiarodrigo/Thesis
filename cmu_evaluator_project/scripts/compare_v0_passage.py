#!/usr/bin/env python3
"""Render one-passage qualitative comparisons of human DEV gold and V0."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "data/benchmark/dev_01_evaluator.json"
BLIND_INPUT_PATH = PROJECT_ROOT / "data/benchmark/dev_01_v0_input.jsonl"
PREDICTIONS_PATH = PROJECT_ROOT / "runtime/evaluator_dev/v0_predictions.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "comparisons/v0"


class ComparisonError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--list-passages", action="store_true")
    choice.add_argument("--passage-id")
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


def main() -> None:
    args = parse_args()
    try:
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
