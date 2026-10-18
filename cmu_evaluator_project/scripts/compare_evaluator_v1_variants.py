#!/usr/bin/env python3
"""Compare four 37-row V1 pilots with frozen V0 on identical examples."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from compare_v0_passage import (
    CANONICAL_PATH,
    ComparisonError,
    index_unique,
    normalized_spans,
    read_json,
    read_jsonl,
)
from evaluate_evaluator_v0 import (
    error_metrics,
    one_to_one_matches,
    overlap_match,
    pearson,
    ranking_metrics,
    safe_div,
    score_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/benchmark/v1_37_input.jsonl"
DEFAULT_V0_DEV = PROJECT_ROOT / "runtime/evaluator_dev/v0_official_dev_predictions.jsonl"
DEFAULT_V0_SYNTHETIC = (
    PROJECT_ROOT
    / "diagnostics/v0_source_sensitivity/v0_source_sensitivity_predictions.jsonl"
)
DEFAULT_V1_DIR = PROJECT_ROOT / "runtime/evaluator_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "comparisons/v1/v1_37_comparison.md"
DEFAULT_REASONING_DIR = PROJECT_ROOT / "runtime/evaluator_model_reasoning"
DEFAULT_REASONING_OUTPUT = (
    PROJECT_ROOT / "comparisons/model_reasoning/v1a_f1_f2_37_results.md"
)
DEFAULT_G_INPUT = PROJECT_ROOT / "data/benchmark/gemma31b_g0_g1_eval_96.jsonl"
DEFAULT_G_MANIFEST = PROJECT_ROOT / "data/benchmark/gemma31b_g0_g1_manifest.json"
DEFAULT_G_DIR = PROJECT_ROOT / "runtime/evaluator_gemma31b"
DEFAULT_G_OUTPUT = PROJECT_ROOT / "comparisons/gemma31b/g0_g1_96_comparison.md"
DEFAULT_G24_INPUT = PROJECT_ROOT / "data/benchmark/gemma31b_g2_g4_eval_72.jsonl"
DEFAULT_G24_MANIFEST = PROJECT_ROOT / "data/benchmark/gemma31b_g2_g4_manifest.json"
DEFAULT_G24_OUTPUT = PROJECT_ROOT / "comparisons/gemma31b/g0_g4_72_comparison.md"
VARIANTS = ("A", "B", "C", "E")
SYSTEM_ORDER = ("copy", "B4-D-G-PTBR", "B4-D-GF-PTBR", "gemma-4-E4B-it")
CONDITIONS = (
    "correct",
    "meaning-change",
    "major-omission",
    "unrelated",
    "source-conflict",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--v0-dev", type=Path, default=DEFAULT_V0_DEV)
    parser.add_argument("--v0-synthetic", type=Path, default=DEFAULT_V0_SYNTHETIC)
    parser.add_argument("--v1-a", type=Path, default=DEFAULT_V1_DIR / "v1_a_37_predictions.jsonl")
    parser.add_argument("--v1-b", type=Path, default=DEFAULT_V1_DIR / "v1_b_37_predictions.jsonl")
    parser.add_argument("--v1-c", type=Path, default=DEFAULT_V1_DIR / "v1_c_37_predictions.jsonl")
    parser.add_argument("--v1-e", type=Path, default=DEFAULT_V1_DIR / "v1_e_37_predictions.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reasoning-comparison", action="store_true")
    parser.add_argument(
        "--f1",
        type=Path,
        default=DEFAULT_REASONING_DIR / "f1_e4b_thinking_37_predictions.jsonl",
    )
    parser.add_argument(
        "--f2",
        type=Path,
        default=DEFAULT_REASONING_DIR / "f2_31b_no_thinking_37_predictions.jsonl",
    )
    parser.add_argument("--reasoning-output", type=Path, default=DEFAULT_REASONING_OUTPUT)
    parser.add_argument("--g0-g1-comparison", action="store_true")
    parser.add_argument("--g-input", type=Path, default=DEFAULT_G_INPUT)
    parser.add_argument("--g-manifest", type=Path, default=DEFAULT_G_MANIFEST)
    parser.add_argument(
        "--g0",
        type=Path,
        default=DEFAULT_G_DIR / "g0_31b_zero_shot_96_predictions.jsonl",
    )
    parser.add_argument(
        "--g1",
        type=Path,
        default=DEFAULT_G_DIR / "g1_31b_eight_shot_96_predictions.jsonl",
    )
    parser.add_argument("--g-output", type=Path, default=DEFAULT_G_OUTPUT)
    parser.add_argument("--g0-g4-comparison", action="store_true")
    parser.add_argument("--g24-input", type=Path, default=DEFAULT_G24_INPUT)
    parser.add_argument("--g24-manifest", type=Path, default=DEFAULT_G24_MANIFEST)
    parser.add_argument(
        "--g2", type=Path,
        default=DEFAULT_G_DIR / "g2_31b_improved_stage1_72_predictions.jsonl",
    )
    parser.add_argument(
        "--g3", type=Path,
        default=DEFAULT_G_DIR / "g3_31b_improved_stage2_72_predictions.jsonl",
    )
    parser.add_argument(
        "--g4", type=Path,
        default=DEFAULT_G_DIR / "g4_31b_improved_both_72_predictions.jsonl",
    )
    parser.add_argument("--g24-output", type=Path, default=DEFAULT_G24_OUTPUT)
    return parser.parse_args()


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def md(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(md(cell) for cell in row) + " |" for row in rows),
    ]


def require_exact_predictions(
    path: Path,
    expected_ids: set[str],
    *,
    variant: str | None = None,
) -> dict[str, dict[str, Any]]:
    indexed = index_unique(read_jsonl(path), f"prediction {path}")
    missing = sorted(expected_ids - set(indexed))
    unexpected = sorted(set(indexed) - expected_ids)
    if missing or unexpected:
        raise ComparisonError(
            f"{path}: ID mismatch missing={missing or 'none'} unexpected={unexpected or 'none'}"
        )
    if variant is not None:
        wrong = sorted(
            evaluation_id
            for evaluation_id, row in indexed.items()
            if row.get("variant") != variant
        )
        if wrong:
            raise ComparisonError(f"{path}: wrong variant for IDs {wrong}")
        protocols = {row.get("protocol_sha256") for row in indexed.values()}
        prompt_hashes = {
            json.dumps(row.get("prompt_hashes"), sort_keys=True)
            for row in indexed.values()
        }
        if len(protocols) != 1 or None in protocols or len(prompt_hashes) != 1:
            raise ComparisonError(f"{path}: inconsistent protocol or prompt hashes")
    return indexed


def prediction_value(row: dict[str, Any]) -> dict[str, Any] | None:
    value = row.get("prediction")
    if row.get("parse_valid") is not True or not isinstance(value, dict):
        return None
    score, spans = value.get("score"), value.get("error_spans")
    if not isinstance(score, int) or isinstance(score, bool) or not isinstance(spans, list):
        return None
    return value


def synthetic_identity(evaluation_id: str) -> tuple[str, str]:
    prefix = "diag-v0-"
    if not evaluation_id.startswith(prefix):
        raise ComparisonError(f"Unexpected synthetic ID: {evaluation_id}")
    rest = evaluation_id[len(prefix) :]
    group, separator, condition = rest.partition("-")
    if not separator or condition not in CONDITIONS:
        raise ComparisonError(f"Could not parse synthetic condition: {evaluation_id}")
    return group, condition


def paired_condition(
    predictions: dict[str, dict[str, Any]],
    synthetic_ids: list[str],
    condition: str,
) -> tuple[list[dict[str, Any]], list[float]]:
    by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for evaluation_id in synthetic_ids:
        group, item_condition = synthetic_identity(evaluation_id)
        by_group[group][item_condition] = predictions[evaluation_id]
    valid_rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for group in sorted(by_group):
        target = prediction_value(by_group[group][condition])
        correct = prediction_value(by_group[group]["correct"])
        if target is not None:
            valid_rows.append(target)
        if target is not None and correct is not None:
            deltas.append(float(target["score"] - correct["score"]))
    return valid_rows, deltas


def synthetic_metrics(
    predictions: dict[str, dict[str, Any]], synthetic_ids: list[str]
) -> dict[str, Any]:
    all_valid = sum(prediction_value(predictions[evaluation_id]) is not None for evaluation_id in synthetic_ids)
    correct, _ = paired_condition(predictions, synthetic_ids, "correct")
    meaning, meaning_delta = paired_condition(predictions, synthetic_ids, "meaning-change")
    omission, _ = paired_condition(predictions, synthetic_ids, "major-omission")
    unrelated, unrelated_delta = paired_condition(predictions, synthetic_ids, "unrelated")
    conflict, conflict_delta = paired_condition(predictions, synthetic_ids, "source-conflict")

    def any_major(row: dict[str, Any]) -> bool:
        return any(span.get("severity") == "Major" for span in row["error_spans"])

    def any_missing(row: dict[str, Any]) -> bool:
        return any(
            span.get("missing") is True or span.get("text") == "[MISSING]"
            for span in row["error_spans"]
        )

    def mean(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    return {
        "valid": all_valid,
        "correct_valid": len(correct),
        "correct_false_positive": sum(bool(row["error_spans"]) for row in correct),
        "meaning_valid": len(meaning),
        "meaning_major": sum(any_major(row) for row in meaning),
        "meaning_paired": len(meaning_delta),
        "meaning_lower": sum(value < 0 for value in meaning_delta),
        "meaning_delta_mean": mean(meaning_delta),
        "omission_valid": len(omission),
        "omission_missing": sum(any_missing(row) for row in omission),
        "omission_major": sum(any_major(row) for row in omission),
        "unrelated_valid": len(unrelated),
        "unrelated_major": sum(any_major(row) for row in unrelated),
        "unrelated_paired": len(unrelated_delta),
        "unrelated_delta_mean": mean(unrelated_delta),
        "conflict_valid": len(conflict),
        "conflict_paired": len(conflict_delta),
        "conflict_lower": sum(value < 0 for value in conflict_delta),
        "conflict_delta_mean": mean(conflict_delta),
    }


def build_dev_records(
    dev_ids: list[str], predictions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    canonical = read_json(CANONICAL_PATH)
    rows = canonical.get("records")
    if not isinstance(rows, list):
        raise ComparisonError("Canonical DEV records are missing")
    canonical_by_id = index_unique(rows, "canonical DEV")
    records: list[dict[str, Any]] = []
    for evaluation_id in dev_ids:
        if evaluation_id not in canonical_by_id:
            raise ComparisonError(f"DEV ID absent from canonical gold: {evaluation_id}")
        row = canonical_by_id[evaluation_id]
        candidate = row["candidate"]["text"]
        human_spans, human_error = normalized_spans(
            candidate,
            row["human_gold"]["error_spans"],
            f"{evaluation_id} human",
            human_inclusive_utf16=True,
        )
        if human_error:
            raise ComparisonError(human_error)
        prediction_row = predictions[evaluation_id]
        value = prediction_value(prediction_row)
        predicted_spans: list[dict[str, Any]] = []
        predicted_score: int | None = None
        parse_errors = prediction_row.get("parse_errors", [])
        if value is not None:
            predicted_spans, predicted_error = normalized_spans(
                candidate,
                value["error_spans"],
                f"{evaluation_id} prediction",
                human_inclusive_utf16=False,
            )
            if predicted_error:
                raise ComparisonError(predicted_error)
            predicted_score = value["score"]
        records.append(
            {
                "evaluation_id": evaluation_id,
                "passage_id": row["passage_id"],
                "sentence_id": row["sentence_id"],
                "position": row["position"],
                "system_id": row["candidate"]["system_id"],
                "candidate_text": candidate,
                "human_score": row["human_gold"]["score"],
                "human_spans": human_spans,
                "prediction": value,
                "predicted_score": predicted_score,
                "predicted_spans": predicted_spans,
                "parse_errors": parse_errors,
                "development_comment": None,
            }
        )
    return records


def dev_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = score_metrics(records)
    errors = error_metrics(records)
    rankings = ranking_metrics(records, list(SYSTEM_ORDER))
    error_free_valid = [
        record
        for record in records
        if record["prediction"] is not None and not record["human_spans"]
    ]
    return {
        "scores": scores,
        "errors": errors,
        "rankings": rankings,
        "error_free_valid": len(error_free_valid),
        "error_free_false_positive": sum(
            bool(record["predicted_spans"]) for record in error_free_valid
        ),
    }


def delta(value: float | None, baseline: float | None) -> float | None:
    return value - baseline if value is not None and baseline is not None else None


def require_experiment_predictions(
    path: Path,
    expected_ids: set[str],
    experiment_id: str,
) -> dict[str, dict[str, Any]]:
    indexed = require_exact_predictions(path, expected_ids, variant="A")
    wrong = sorted(
        evaluation_id
        for evaluation_id, row in indexed.items()
        if row.get("experiment_id") != experiment_id
    )
    if wrong:
        raise ComparisonError(f"{path}: wrong experiment ID for {wrong}")
    expected_hashes = {
        "stage1": "fde2e0a7224edd694c66268d40d4a5f448e5e617a362b0bdb69e3803650f90ed",
        "stage2": "d4405349a285ccabfadfb94565a0cd382509c98685aa5cb2a8d46a9b846e12fd",
    }
    if any(row.get("prompt_hashes") != expected_hashes for row in indexed.values()):
        raise ComparisonError(f"{path}: prompt hashes differ from frozen V1-A")
    return indexed


def _json_from_final_answer(stage: dict[str, Any]) -> dict[str, Any] | None:
    text = stage.get("final_answer")
    if not isinstance(text, str):
        # Frozen V1-A predates explicit channel separation. Its saved continuation
        # remains the authoritative raw response for this offline diagnostic.
        text = stage.get("raw_model_output")
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if stripped.startswith("```json\n") and stripped.endswith("```"):
        stripped = stripped[len("```json\n") : -3].strip()
    elif stripped.startswith("```\n") and stripped.endswith("```"):
        stripped = stripped[len("```\n") : -3].strip()
    elif stripped.startswith("```json\n"):
        stripped = stripped[len("```json\n") :].strip()
    elif stripped.startswith("```\n"):
        stripped = stripped[len("```\n") :].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def source_side_span_mistakes(
    predictions: dict[str, dict[str, Any]],
    input_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    violation_records = 0
    violation_spans = 0
    confirmed_source_records = 0
    confirmed_source_spans = 0
    for evaluation_id, row in predictions.items():
        errors = [str(value) for value in row.get("parse_errors", [])]
        if not any("text_not_in_candidate" in value for value in errors):
            continue
        violation_records += 1
        source = input_by_id[evaluation_id]["current_source_bp"]
        candidate = input_by_id[evaluation_id]["candidate_text"]
        record_confirmed = False
        stages = row.get("stages", {})
        if not isinstance(stages, dict):
            continue
        for stage in stages.values():
            if not isinstance(stage, dict):
                continue
            value = _json_from_final_answer(stage)
            if value is None or not isinstance(value.get("errors"), list):
                continue
            for error in value["errors"]:
                if not isinstance(error, dict):
                    continue
                marked = error.get("text")
                if (
                    isinstance(marked, str)
                    and marked != "[MISSING]"
                    and marked not in candidate
                ):
                    violation_spans += 1
                    if marked in source:
                        confirmed_source_spans += 1
                        record_confirmed = True
        confirmed_source_records += record_confirmed
    return {
        "violation_records": violation_records,
        "violation_spans": violation_spans,
        "confirmed_source_records": confirmed_source_records,
        "confirmed_source_spans": confirmed_source_spans,
    }


def failure_stage_counts(predictions: dict[str, dict[str, Any]]) -> dict[str, int]:
    invalid = [row for row in predictions.values() if prediction_value(row) is None]
    return {
        "invalid": len(invalid),
        "stage1": sum(
            any(str(error).startswith("stage1:") for error in row.get("parse_errors", []))
            for row in invalid
        ),
        "stage2": sum(
            any(str(error).startswith("stage2:") for error in row.get("parse_errors", []))
            for row in invalid
        ),
        "truncated": sum(
            any(
                isinstance(stage, dict) and stage.get("hit_max_new_tokens") is True
                for stage in row.get("stages", {}).values()
            )
            for row in predictions.values()
            if isinstance(row.get("stages"), dict)
        ),
    }


def prediction_summary(row: dict[str, Any]) -> str:
    value = prediction_value(row)
    if value is None:
        errors = ", ".join(str(error) for error in row.get("parse_errors", []))
        return f"invalid ({errors or 'unknown reason'})"
    spans = ", ".join(
        f"{span.get('text')} [{span.get('severity')}]" for span in value["error_spans"]
    )
    return f"{value['score']}: {spans or 'no errors'}"


def run_reasoning_comparison(args: argparse.Namespace) -> None:
    inputs = read_jsonl(args.input)
    input_by_id = index_unique(inputs, "shared 37-row input")
    input_ids = [row["evaluation_id"] for row in inputs]
    synthetic_ids = [value for value in input_ids if value.startswith("diag-v0-")]
    dev_ids = [value for value in input_ids if value.startswith("dev-v0-")]
    if len(input_ids) != 37 or len(set(input_ids)) != 37:
        raise ComparisonError("Reasoning comparison requires the unchanged 37-row input")
    expected_ids = set(input_ids)
    baseline = require_exact_predictions(args.v1_a, expected_ids, variant="A")
    f1 = require_experiment_predictions(args.f1, expected_ids, "F1")
    f2 = require_experiment_predictions(args.f2, expected_ids, "F2")
    predictions = {"V1-A": baseline, "F1": f1, "F2": f2}

    synthetic_results = {
        label: synthetic_metrics(rows, synthetic_ids) for label, rows in predictions.items()
    }
    dev_records = {
        label: build_dev_records(dev_ids, rows) for label, rows in predictions.items()
    }
    dev_results = {label: dev_metrics(rows) for label, rows in dev_records.items()}
    source_mistakes = {
        label: source_side_span_mistakes(rows, input_by_id)
        for label, rows in predictions.items()
    }
    failures = {
        label: failure_stage_counts(rows) for label, rows in predictions.items()
    }

    lines = [
        "# V1-A reasoning and model-capacity comparison",
        "",
        "Configurations: frozen V1-A E4B without thinking, F1 E4B with thinking, and F2 31B without thinking. All use the identical 37 inputs and byte-identical V1-A Stage 1/Stage 2 prompts.",
        "",
        "This is an exploratory pilot: formal DEV metrics use only 12 judgments from three complete four-candidate sentence groups.",
        "",
        "## Configuration and coverage",
        "",
    ]
    config_rows = []
    for label, rows in predictions.items():
        first = rows[input_ids[0]]
        model = first.get("model", {})
        generation = first.get("generation_config", {})
        config_rows.append(
            [
                label,
                model.get("model_id"),
                model.get("requested_revision"),
                generation.get("enable_thinking"),
                first.get("runtime", {}).get("requested_dtype", "auto"),
                first.get("protocol_sha256"),
                f"{sum(prediction_value(row) is not None for row in rows.values())}/37",
            ]
        )
    lines.extend(
        table(
            ["Configuration", "Model", "Revision", "Thinking", "Dtype", "Protocol", "Valid"],
            config_rows,
        )
    )

    lines.extend(["", "## Synthetic diagnostics", ""])
    synthetic_rows = []
    for label in predictions:
        result = synthetic_results[label]
        synthetic_rows.append(
            [
                label,
                f"{result['valid']}/25",
                f"{result['correct_false_positive']}/{result['correct_valid']}",
                f"Major {result['meaning_major']}/{result['meaning_valid']}; lower {result['meaning_lower']}/{result['meaning_paired']}; mean Δ {fmt(result['meaning_delta_mean'])}",
                f"[MISSING] {result['omission_missing']}/{result['omission_valid']}; Major {result['omission_major']}/{result['omission_valid']}",
                f"Major {result['unrelated_major']}/{result['unrelated_valid']}; mean Δ {fmt(result['unrelated_delta_mean'])}",
                f"lower {result['conflict_lower']}/{result['conflict_paired']}; mean Δ {fmt(result['conflict_delta_mean'])}",
            ]
        )
    lines.extend(
        table(
            ["Configuration", "Valid", "Correct FP", "Meaning change", "Omission", "Unrelated", "Source conflict"],
            synthetic_rows,
        )
    )

    lines.extend(["", "## Formal DEV agreement", ""])
    dev_rows_out = []
    for label in predictions:
        result = dev_results[label]
        scores, errors, ranks = result["scores"], result["errors"], result["rankings"]
        dev_rows_out.append(
            [
                label,
                f"{scores['n']}/{scores['total']}",
                fmt(scores["mae"]),
                fmt(scores["mean_signed_difference"]),
                fmt(scores["pearson"]),
                fmt(scores["spearman"]),
                f"{fmt(errors['exact']['f1'])} ({errors['exact']['matched']}/{errors['exact']['predicted']}/{errors['exact']['gold']})",
                f"{fmt(errors['overlap']['f1'])} ({errors['overlap']['matched']}/{errors['overlap']['predicted']}/{errors['overlap']['gold']})",
                f"{fmt(errors['major_overlap']['recall'])} ({errors['major_overlap']['matched']}/{errors['major_overlap']['gold']})",
                f"{result['error_free_valid']-result['error_free_false_positive']}/{result['error_free_valid']}",
                f"{ranks['top_set_exact']}/{ranks['sentences_complete']} exact; {ranks['top_set_any_overlap']}/{ranks['sentences_complete']} overlap",
            ]
        )
    lines.extend(
        table(
            ["Configuration", "Valid", "MAE", "Signed Δ", "Pearson", "Spearman", "Exact F1 (M/P/G)", "Overlap F1 (M/P/G)", "Major recall", "Error-free TN", "Top-set agreement"],
            dev_rows_out,
        )
    )

    lines.extend(["", "## Failure attribution and source-side span mistakes", ""])
    failure_rows = []
    for label in predictions:
        failure = failures[label]
        source = source_mistakes[label]
        failure_rows.append(
            [
                label,
                f"{failure['invalid']}/37",
                failure["stage1"],
                failure["stage2"],
                failure["truncated"],
                f"{source['violation_records']} records / {source['violation_spans']} spans",
                f"{source['confirmed_source_records']} records / {source['confirmed_source_spans']} spans",
            ]
        )
    lines.extend(
        table(
            ["Configuration", "Invalid", "Stage 1 failures", "Stage 2 failures", "Truncated records", "Absent candidate spans", "Confirmed BP-source copies"],
            failure_rows,
        )
    )

    lines.extend(["", "## The 12 DEV judgments", ""])
    canonical = index_unique(read_json(CANONICAL_PATH)["records"], "canonical DEV")
    normalized_gold_by_id = {
        record["evaluation_id"]: record for record in dev_records["V1-A"]
    }
    detail_rows = []
    for evaluation_id in dev_ids:
        gold = canonical[evaluation_id]
        human_spans = ", ".join(
            f"{span.get('text')} [{span.get('severity')}]"
            for span in normalized_gold_by_id[evaluation_id]["human_spans"]
        )
        detail_rows.append(
            [
                f"{gold['passage_id']} / {gold['sentence_id']}",
                evaluation_id,
                gold["candidate"]["system_id"],
                f"{gold['human_gold']['score']}: {human_spans or 'no errors'}",
                prediction_summary(baseline[evaluation_id]),
                prediction_summary(f1[evaluation_id]),
                prediction_summary(f2[evaluation_id]),
            ]
        )
    lines.extend(
        table(
            ["Passage / sentence", "ID", "System", "Human", "V1-A", "F1", "F2"],
            detail_rows,
        )
    )
    lines.extend(
        [
            "",
            "Stage attribution is structural: a Stage 1 failure prevents Stage 2 from running; a Stage 2 failure occurs after validated Stage 1 annotations. Source-copy counts require the invalid reported substring to be absent from the candidate and present verbatim in the current BP source.",
            "",
        ]
    )
    args.reasoning_output.parent.mkdir(parents=True, exist_ok=True)
    args.reasoning_output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.reasoning_output.resolve()}")


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float | None:
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denominator if denominator else None


def error_free_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["prediction"] is not None]
    tp = tn = fp = fn = 0
    for record in valid:
        human_free = not record["human_spans"]
        predicted_free = not record["predicted_spans"]
        if human_free and predicted_free:
            tp += 1
        elif not human_free and not predicted_free:
            tn += 1
        elif not human_free and predicted_free:
            fp += 1
        else:
            fn += 1
    return {
        "n": len(valid),
        "human_error_free": tp + fn,
        "true_error_free": tp,
        "false_error_free": fp,
        "false_positive_on_error_free": fn,
        "mcc": _mcc(tp, tn, fp, fn),
    }


def missing_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["prediction"] is not None]
    tp = fp = fn = 0
    predicted_markers = well_formed = 0
    for record in valid:
        gold_missing = any(span.get("missing") is True for span in record["human_spans"])
        predicted = [
            span for span in record["predicted_spans"] if span.get("missing") is True
        ]
        predicted_missing = bool(predicted)
        tp += gold_missing and predicted_missing
        fp += (not gold_missing) and predicted_missing
        fn += gold_missing and (not predicted_missing)
        predicted_markers += len(predicted)
        well_formed += sum(
            span.get("text") == "[MISSING]"
            and span.get("start") == span.get("end") == len(record["candidate_text"])
            for span in predicted
        )
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (0.0 if precision == 0 or recall == 0 else None)
    )
    return {
        "gold_positive": tp + fn,
        "predicted_positive": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "markers": predicted_markers,
        "well_formed_markers": well_formed,
    }


def official_mpp(records: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from span_mt_metrics_eval import compute
    except ImportError:
        return {
            "n": sum(record["prediction"] is not None for record in records),
            "precision": None,
            "recall": None,
            "f1": None,
            "available": False,
            "reason": "official span-mt-metrics-eval dependency not installed",
            "missing_markers_excluded": True,
        }
    valid = [record for record in records if record["prediction"] is not None]

    def converted(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "start": span["start"],
                "end": span["end"],
                "side": "target",
                "severity": span["severity"].lower(),
            }
            for span in spans
            if not span.get("missing") and span["start"] < span["end"]
        ]

    result = compute(
        [converted(record["predicted_spans"]) for record in valid],
        [converted(record["human_spans"]) for record in valid],
        measure="MPP",
        matching="one_to_one",
        matching_algorithm="optimal",
        averaging="micro",
        severity_weights={"minor": 0.25, "major": 1.0},
    )
    return {
        "n": len(valid),
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f_score,
        "available": True,
        "reason": None,
        "missing_markers_excluded": True,
    }


def pdp_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["passage_id"], record["sentence_id"])].append(record)
    complete = [
        group
        for group in grouped.values()
        if len(group) == 4 and all(record["prediction"] is not None for record in group)
    ]
    human_differences: list[float] = []
    predicted_differences: list[float] = []
    for group in complete:
        ordered = sorted(group, key=lambda record: SYSTEM_ORDER.index(record["system_id"]))
        for left in range(4):
            for right in range(left + 1, 4):
                human_difference = float(ordered[left]["human_score"] - ordered[right]["human_score"])
                predicted_difference = float(
                    ordered[left]["predicted_score"] - ordered[right]["predicted_score"]
                )
                human_differences.extend((human_difference, -human_difference))
                predicted_differences.extend((predicted_difference, -predicted_difference))
    value = pearson(human_differences, predicted_differences)
    if human_differences and value is None:
        value = 0.0
    return {
        "complete_groups": len(complete),
        "pairwise_differences": len(human_differences),
        "pdp": value,
    }


def g_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scores": score_metrics(records),
        "errors": error_metrics(records),
        "mpp": official_mpp(records),
        "error_free": error_free_metrics(records),
        "missing": missing_metrics(records),
        "ranking": ranking_metrics(records, list(SYSTEM_ORDER)),
        "pdp": pdp_metrics(records),
    }


def run_g0_g1_comparison(args: argparse.Namespace) -> None:
    manifest = read_json(args.g_manifest)
    inputs = read_jsonl(args.g_input)
    input_by_id = index_unique(inputs, "G0/G1 blind input")
    ids = [row["evaluation_id"] for row in inputs]
    if len(ids) != 96 or len(set(ids)) != 96 or ids != manifest.get("evaluation_ids"):
        raise ComparisonError("G0/G1 input does not match the frozen 96-ID manifest")
    expected_ids = set(ids)
    predictions = {
        "G0": require_experiment_predictions(args.g0, expected_ids, "G0"),
        "G1": require_experiment_predictions(args.g1, expected_ids, "G1"),
    }
    if any(
        predictions["G0"][evaluation_id].get("input_sha256")
        != predictions["G1"][evaluation_id].get("input_sha256")
        for evaluation_id in ids
    ):
        raise ComparisonError("G0 and G1 do not share the same frozen input hash")
    pilot_ids = set(manifest["previously_inspected_evaluation_ids"])
    new_ids = set(manifest["new_evaluation_ids"])
    slices = {
        "All 96": set(ids),
        "New 84": new_ids,
        "Prior pilot 12": pilot_ids,
    }
    records = {
        label: build_dev_records(ids, rows) for label, rows in predictions.items()
    }
    by_label_id = {
        label: {record["evaluation_id"]: record for record in values}
        for label, values in records.items()
    }
    hardware = {}
    for label in ("G0", "G1"):
        runtime = predictions[label][ids[0]].get("runtime", {})
        devices = runtime.get("cuda_devices", [])
        names = sorted(
            {
                device.get("name")
                for device in devices
                if isinstance(device, dict) and isinstance(device.get("name"), str)
            }
        )
        hardware[label] = (
            f"{runtime.get('hostname', 'unknown host')} physical GPUs "
            f"{runtime.get('cuda_visible_devices', 'unknown')} "
            f"({', '.join(names) or 'unknown device'})"
        )

    lines = [
        "# Gemma 31B G0 zero-shot versus G1 eight-shot",
        "",
        "Paired exploratory comparison on the frozen 96-judgment DEV subset. The two reserved demonstration documents are excluded from evaluation; the 84 newly selected judgments are reported separately from the 12 previously inspected pilot judgments.",
        "",
        "## Frozen design and coverage",
        "",
        f"- Demonstration documents: `{', '.join(manifest['demonstration_documents'])}`.",
        f"- Evaluation documents: `{', '.join(manifest['evaluation_documents'])}`.",
        "- Evaluation sample: 24 complete sentence groups × 4 candidates = 96 judgments.",
        f"- Execution hardware: G0 used {hardware['G0']}; G1 used {hardware['G1']}. The frozen model and deterministic BF16 settings are identical, but different GPU architectures are a minor numerical reproducibility confound.",
        "- MPP uses the official `span-mt-metrics-eval` implementation with severity weights Minor=0.25 and Major=1.0 when that optional dependency is installed. Otherwise MPP is reported as unavailable without substitution. `[MISSING]` is evaluated separately because it is not a candidate-side non-empty span.",
        "- PDP is the published Pairwise Difference Pearson calculation over both directions of every within-sentence candidate pair.",
        "",
    ]
    coverage_rows = []
    for slice_name, slice_ids in slices.items():
        common = {
            evaluation_id
            for evaluation_id in slice_ids
            if prediction_value(predictions["G0"][evaluation_id]) is not None
            and prediction_value(predictions["G1"][evaluation_id]) is not None
        }
        coverage_rows.append(
            [
                slice_name,
                len(slice_ids),
                sum(prediction_value(predictions["G0"][value]) is not None for value in slice_ids),
                sum(prediction_value(predictions["G1"][value]) is not None for value in slice_ids),
                len(common),
            ]
        )
    lines.extend(table(["Slice", "Expected", "G0 valid", "G1 valid", "Common valid"], coverage_rows))

    for slice_name, slice_ids in slices.items():
        common_ids = {
            evaluation_id
            for evaluation_id in slice_ids
            if prediction_value(predictions["G0"][evaluation_id]) is not None
            and prediction_value(predictions["G1"][evaluation_id]) is not None
        }
        lines.extend(["", f"## {slice_name}: common-valid paired results", ""])
        metric_rows = []
        for label in ("G0", "G1"):
            selected = [by_label_id[label][value] for value in ids if value in common_ids]
            result = g_metrics(selected)
            score = result["scores"]
            error = result["errors"]
            ef = result["error_free"]
            missing = result["missing"]
            ranking = result["ranking"]
            pdp = result["pdp"]
            mpp = result["mpp"]
            metric_rows.append(
                [
                    label,
                    score["n"],
                    fmt(score["mae"]),
                    fmt(score["mean_signed_difference"]),
                    fmt(score["pearson"]),
                    fmt(score["spearman"]),
                    fmt(mpp["f1"]) if mpp["available"] else "unavailable",
                    fmt(error["exact"]["f1"]),
                    fmt(error["overlap"]["f1"]),
                    fmt(error["major_overlap"]["recall"]),
                    f"{ef['false_positive_on_error_free']}/{ef['human_error_free']}; MCC {fmt(ef['mcc'])}",
                    f"R {fmt(missing['recall'])}; P {fmt(missing['precision'])}",
                    f"{fmt(pdp['pdp'])} ({pdp['complete_groups']} groups)",
                    f"{ranking['top_set_exact']}/{ranking['sentences_complete']} exact; {ranking['top_set_any_overlap']}/{ranking['sentences_complete']} overlap",
                ]
            )
        lines.extend(
            table(
                ["Evaluator", "N", "MAE", "Bias", "Pearson", "Spearman", "MPP F1", "Exact F1", "Overlap F1", "Major recall", "Error-free FP; MCC", "Missing", "PDP", "Top-set"],
                metric_rows,
            )
        )

    lines.extend(["", "## Structural and stage-specific failures", ""])
    failure_rows = []
    for label in ("G0", "G1"):
        failures = failure_stage_counts(predictions[label])
        reasons = Counter(
            error
            for row in predictions[label].values()
            for error in row.get("parse_errors", [])
        )
        source = source_side_span_mistakes(predictions[label], input_by_id)
        failure_rows.append(
            [
                label,
                failures["invalid"],
                failures["stage1"],
                failures["stage2"],
                failures["truncated"],
                f"{source['confirmed_source_records']} records / {source['confirmed_source_spans']} spans",
                "; ".join(f"{key}: {value}" for key, value in reasons.most_common()) or "—",
            ]
        )
    lines.extend(table(["Evaluator", "Invalid", "Stage 1", "Stage 2", "Truncated", "Confirmed source-only", "Reasons"], failure_rows))

    lines.extend(["", "## Per-system paired results on all common-valid judgments", ""])
    common_all = {
        evaluation_id
        for evaluation_id in ids
        if prediction_value(predictions["G0"][evaluation_id]) is not None
        and prediction_value(predictions["G1"][evaluation_id]) is not None
    }
    per_system_rows = []
    for system in SYSTEM_ORDER:
        for label in ("G0", "G1"):
            selected = [
                by_label_id[label][value]
                for value in ids
                if value in common_all and by_label_id[label][value]["system_id"] == system
            ]
            score = score_metrics(selected)
            error = error_metrics(selected)
            per_system_rows.append(
                [system, label, score["n"], fmt(score["mae"]), fmt(score["mean_signed_difference"]), fmt(error["overlap"]["f1"]), fmt(error["major_overlap"]["recall"])]
            )
    lines.extend(table(["Candidate system", "Evaluator", "N", "MAE", "Bias", "Overlap F1", "Major recall"], per_system_rows))
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Formal gold contains spans, severity, and score, but no error-category labels. BP-retention, proper-name/meaning-change, and unnecessary-rewrite findings therefore cannot be separated reliably without a later qualitative review; private development comments are not used here. Missing-content behavior and human-error-free acceptance are reported directly. Valid Stage 2 judgments are structurally consistent with validated Stage 1 annotations because the frozen parser rejects a score of 100 when errors exist and rejects a non-100 score when Stage 1 is empty.",
            "",
        ]
    )
    args.g_output.parent.mkdir(parents=True, exist_ok=True)
    args.g_output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.g_output.resolve()}")


def _require_prediction_subset(
    path: Path, expected_ids: set[str], experiment: str
) -> dict[str, dict[str, Any]]:
    indexed = index_unique(read_jsonl(path), f"{experiment} predictions")
    missing = sorted(expected_ids - set(indexed))
    if missing:
        raise ComparisonError(f"{path}: missing {experiment} IDs {missing}")
    selected = {evaluation_id: indexed[evaluation_id] for evaluation_id in expected_ids}
    wrong = sorted(
        evaluation_id
        for evaluation_id, row in selected.items()
        if row.get("experiment_id") != experiment or row.get("variant") != "A"
    )
    if wrong:
        raise ComparisonError(f"{path}: wrong {experiment}/A identity for {wrong}")
    return selected


def run_g0_g4_comparison(args: argparse.Namespace) -> None:
    manifest = read_json(args.g24_manifest)
    inputs = read_jsonl(args.g24_input)
    input_by_id = index_unique(inputs, "G0/G4 common blind input")
    ids = [row["evaluation_id"] for row in inputs]
    if len(ids) != 72 or len(set(ids)) != 72 or ids != manifest.get("evaluation_ids"):
        raise ComparisonError("G0/G4 input does not match the frozen 72-ID manifest")
    expected = set(ids)
    paths = {"G0": args.g0, "G1": args.g1, "G2": args.g2, "G3": args.g3, "G4": args.g4}
    predictions = {
        label: _require_prediction_subset(path, expected, label)
        for label, path in paths.items()
    }
    records = {
        label: build_dev_records(ids, rows) for label, rows in predictions.items()
    }
    indexed_records = {
        label: {record["evaluation_id"]: record for record in values}
        for label, values in records.items()
    }
    pilot = set(manifest["previously_inspected_evaluation_ids"])
    new = set(manifest["new_evaluation_ids"])
    slices = {"All 72": expected, "New 64": new, "Prior pilot 8": pilot}
    common_valid = {
        evaluation_id
        for evaluation_id in ids
        if all(prediction_value(predictions[label][evaluation_id]) is not None for label in paths)
    }
    lines = [
        "# Gemma 31B G0–G4 comparison on the common 72-judgment DEV subset",
        "",
        "Controlled DEV comparison of zero-shot, original demonstrations, and the stage-specific improved demonstrations. This is development analysis, not held-out TEST evaluation.",
        "",
        "## Frozen design and coverage",
        "",
        f"- Demonstration documents for G2–G4: `{', '.join(manifest['demonstration_documents'])}`.",
        f"- Evaluation documents: `{', '.join(manifest['evaluation_documents'])}`.",
        "- Evaluation sample: 18 complete source-sentence groups × 4 candidates = 72 judgments.",
        "- G2 isolates improved Stage 1; G3 isolates improved Stage 2; G4 applies both.",
        "- Primary paired metrics use only IDs valid for all five configurations.",
        "- MPP is computed only with the official optional package; otherwise it is unavailable without substitution.",
        "",
    ]
    coverage_rows = []
    for slice_name, slice_ids in slices.items():
        row = [slice_name, len(slice_ids)]
        for label in paths:
            row.append(sum(prediction_value(predictions[label][value]) is not None for value in slice_ids))
        row.append(sum(value in common_valid for value in slice_ids))
        coverage_rows.append(row)
    lines.extend(table(["Slice", "Expected", *[f"{label} valid" for label in paths], "All-five valid"], coverage_rows))

    for slice_name, slice_ids in slices.items():
        paired_ids = common_valid & slice_ids
        lines.extend(["", f"## {slice_name}: all-five common-valid results", ""])
        metric_rows = []
        for label in paths:
            selected = [indexed_records[label][value] for value in ids if value in paired_ids]
            result = g_metrics(selected)
            score, error = result["scores"], result["errors"]
            ef, missing = result["error_free"], result["missing"]
            ranking, pdp, mpp = result["ranking"], result["pdp"], result["mpp"]
            metric_rows.append([
                label, score["n"], fmt(score["mae"]), fmt(score["mean_signed_difference"]),
                fmt(score["pearson"]), fmt(score["spearman"]),
                fmt(mpp["f1"]) if mpp["available"] else "unavailable",
                fmt(error["exact"]["f1"]), fmt(error["overlap"]["f1"]),
                fmt(error["major_overlap"]["recall"]),
                f"{ef['false_positive_on_error_free']}/{ef['human_error_free']}; MCC {fmt(ef['mcc'])}",
                f"P {fmt(missing['precision'])} ({missing['tp']}/{missing['tp'] + missing['fp']}); R {fmt(missing['recall'])} ({missing['tp']}/{missing['tp'] + missing['fn']})",
                f"{fmt(pdp['pdp'])} ({pdp['complete_groups']} groups)",
                f"{ranking['top_set_exact']}/{ranking['sentences_complete']} exact; {ranking['top_set_any_overlap']}/{ranking['sentences_complete']} overlap",
            ])
        lines.extend(table(
            ["Evaluator", "N", "MAE", "Bias", "Pearson", "Spearman", "MPP F1", "Exact F1", "Overlap F1", "Major recall", "Error-free FP; MCC", "[MISSING] P/R", "PDP", "Top-set"],
            metric_rows,
        ))

    lines.extend(["", "## Isolated-factor comparisons on all-five common-valid IDs", ""])
    factor_rows = []
    for left, right, factor in (
        ("G0", "G2", "Improved Stage 1 with zero-shot Stage 2"),
        ("G0", "G3", "Improved Stage 2 with frozen zero-shot Stage 1"),
        ("G2", "G4", "Improved Stage 2 given improved Stage 1"),
        ("G3", "G4", "Improved Stage 1 given improved Stage 2"),
    ):
        left_metrics = g_metrics([indexed_records[left][value] for value in ids if value in common_valid])
        right_metrics = g_metrics([indexed_records[right][value] for value in ids if value in common_valid])
        factor_rows.append([
            f"{left} → {right}", factor,
            fmt(right_metrics["scores"]["mae"] - left_metrics["scores"]["mae"]),
            fmt(right_metrics["errors"]["overlap"]["f1"] - left_metrics["errors"]["overlap"]["f1"]),
            fmt(right_metrics["errors"]["major_overlap"]["recall"] - left_metrics["errors"]["major_overlap"]["recall"]),
        ])
    lines.extend(table(["Contrast", "Isolated factor", "Δ MAE", "Δ overlap F1", "Δ Major recall"], factor_rows))

    lines.extend(["", "## Structural and stage-specific failures", ""])
    failure_rows = []
    for label in paths:
        failures = failure_stage_counts(predictions[label])
        reasons = Counter(error for row in predictions[label].values() for error in row.get("parse_errors", []))
        source = source_side_span_mistakes(predictions[label], input_by_id)
        failure_rows.append([
            label, failures["invalid"], failures["stage1"], failures["stage2"], failures["truncated"],
            f"{source['confirmed_source_records']} records / {source['confirmed_source_spans']} spans",
            "; ".join(f"{key}: {value}" for key, value in reasons.most_common()) or "—",
        ])
    lines.extend(table(["Evaluator", "Invalid", "Stage 1", "Stage 2", "Truncated", "Confirmed source-only", "Reasons"], failure_rows))

    lines.extend(["", "## Per-candidate-system results on all-five common-valid IDs", ""])
    system_rows = []
    for system in SYSTEM_ORDER:
        for label in paths:
            selected = [indexed_records[label][value] for value in ids if value in common_valid and indexed_records[label][value]["system_id"] == system]
            score, error = score_metrics(selected), error_metrics(selected)
            system_rows.append([system, label, score["n"], fmt(score["mae"]), fmt(score["mean_signed_difference"]), fmt(error["overlap"]["f1"]), fmt(error["major_overlap"]["recall"])])
    lines.extend(table(["Candidate system", "Evaluator", "N", "MAE", "Bias", "Overlap F1", "Major recall"], system_rows))
    lines.extend(["", "Excluded IDs remain visible in the coverage and structural-failure tables. Formal gold has no error categories, so unmatched spans are mechanical disagreements and are not automatically described as hallucinations.", ""])
    args.g24_output.parent.mkdir(parents=True, exist_ok=True)
    args.g24_output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.g24_output.resolve()}")


def main() -> None:
    args = parse_args()
    if args.g0_g4_comparison:
        run_g0_g4_comparison(args)
        return
    if args.g0_g1_comparison:
        run_g0_g1_comparison(args)
        return
    if args.reasoning_comparison:
        run_reasoning_comparison(args)
        return
    inputs = read_jsonl(args.input)
    input_by_id = index_unique(inputs, "V1 shared input")
    input_ids = [row["evaluation_id"] for row in inputs]
    synthetic_ids = [evaluation_id for evaluation_id in input_ids if evaluation_id.startswith("diag-v0-")]
    dev_ids = [evaluation_id for evaluation_id in input_ids if evaluation_id.startswith("dev-v0-")]
    if len(inputs) != 37 or len(synthetic_ids) != 25 or len(dev_ids) != 12:
        raise ComparisonError(
            f"Expected 37 inputs split 25 synthetic/12 DEV, got {len(inputs)}/"
            f"{len(synthetic_ids)}/{len(dev_ids)}"
        )
    if set(input_ids) != set(input_by_id):
        raise ComparisonError("Duplicate shared-input IDs")

    v0_synthetic = index_unique(read_jsonl(args.v0_synthetic), "V0 synthetic")
    v0_dev_all = index_unique(read_jsonl(args.v0_dev), "V0 DEV")
    if set(synthetic_ids) - set(v0_synthetic) or set(dev_ids) - set(v0_dev_all):
        raise ComparisonError("Frozen V0 predictions do not cover all shared examples")
    predictions: dict[str, dict[str, dict[str, Any]]] = {
        "V0": {
            **{evaluation_id: v0_synthetic[evaluation_id] for evaluation_id in synthetic_ids},
            **{evaluation_id: v0_dev_all[evaluation_id] for evaluation_id in dev_ids},
        }
    }
    variant_paths = {
        "A": args.v1_a,
        "B": args.v1_b,
        "C": args.v1_c,
        "E": args.v1_e,
    }
    expected_ids = set(input_ids)
    for variant, path in variant_paths.items():
        predictions[variant] = require_exact_predictions(
            path, expected_ids, variant=variant
        )

    synthetic_results = {
        label: synthetic_metrics(rows, synthetic_ids)
        for label, rows in predictions.items()
    }
    dev_records = {
        label: build_dev_records(dev_ids, rows)
        for label, rows in predictions.items()
    }
    dev_results = {
        label: dev_metrics(records) for label, records in dev_records.items()
    }

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    canonical_by_id = index_unique(read_json(CANONICAL_PATH)["records"], "canonical DEV")
    for evaluation_id in dev_ids:
        row = canonical_by_id[evaluation_id]
        groups[(row["passage_id"], row["sentence_id"])].append(evaluation_id)

    lines = [
        "# V1 four-variant 37-example comparison",
        "",
        "Frozen baseline: V0 at commit `2ff1d575`.",
        "",
        "This pilot is exploratory: it contains 25 synthetic diagnostics and only 12 DEV judgments from three complete four-candidate sentence groups. It cannot establish generalizable superiority.",
        "",
        "## Coverage and selected DEV groups",
        "",
        f"- Shared input: **{len(inputs)}** unique rows in identical order for all variants.",
        f"- Synthetic: **{len(synthetic_ids)}** rows.",
        f"- DEV: **{len(dev_ids)}** rows from **{len(groups)}** passages/sentence groups.",
        "",
    ]
    lines.extend(
        table(
            ["Passage / sentence", "Evaluation IDs"],
            [
                [f"{passage} / {sentence}", ", ".join(ids)]
                for (passage, sentence), ids in groups.items()
            ],
        )
    )
    lines.extend(["", "## Prompt configurations", ""])
    prompt_rows = []
    for variant in VARIANTS:
        first = predictions[variant][input_ids[0]]
        hashes = first["prompt_hashes"]
        prompt_rows.append(
            [
                f"V1-{variant}",
                first["protocol_sha256"],
                "; ".join(f"{stage}={value}" for stage, value in sorted(hashes.items())),
            ]
        )
    lines.extend(table(["Variant", "Protocol SHA-256", "Prompt SHA-256 values"], prompt_rows))

    lines.extend(["", "## Synthetic source-sensitivity results", ""])
    synthetic_rows = []
    for label in ("V0", *VARIANTS):
        result = synthetic_results[label]
        synthetic_rows.append(
            [
                label,
                f"{result['valid']}/25",
                f"{result['correct_false_positive']}/{result['correct_valid']}",
                f"Major {result['meaning_major']}/{result['meaning_valid']}; lower {result['meaning_lower']}/{result['meaning_paired']}; mean Δ {fmt(result['meaning_delta_mean'])}",
                f"valid {result['omission_valid']}/5; [MISSING] {result['omission_missing']}/{result['omission_valid']}; any Major {result['omission_major']}/{result['omission_valid']}",
                f"Major {result['unrelated_major']}/{result['unrelated_valid']}; mean Δ {fmt(result['unrelated_delta_mean'])} (N={result['unrelated_paired']})",
                f"lower {result['conflict_lower']}/{result['conflict_paired']}; mean Δ {fmt(result['conflict_delta_mean'])}",
            ]
        )
    lines.extend(
        table(
            [
                "Evaluator",
                "Valid",
                "Correct-adaptation FP",
                "Meaning change",
                "Major omission",
                "Unrelated",
                "Source conflict",
            ],
            synthetic_rows,
        )
    )
    lines.extend(
        [
            "",
            "Score differences (Δ) are condition score minus the paired correct-candidate score; negative values are the expected direction.",
            "",
            "## DEV agreement results",
            "",
        ]
    )
    dev_rows = []
    for label in ("V0", *VARIANTS):
        result = dev_results[label]
        scores, errors, ranks = result["scores"], result["errors"], result["rankings"]
        major = errors["major_overlap"]
        dev_rows.append(
            [
                label,
                f"{scores['n']}/{scores['total']}",
                fmt(scores["mae"]),
                fmt(scores["mean_signed_difference"]),
                fmt(scores["pearson"]),
                fmt(scores["spearman"]),
                f"{fmt(errors['exact']['f1'])} ({errors['exact']['matched']}/{errors['exact']['predicted']}/{errors['exact']['gold']})",
                f"{fmt(errors['overlap']['f1'])} ({errors['overlap']['matched']}/{errors['overlap']['predicted']}/{errors['overlap']['gold']})",
                f"{fmt(major['recall'])} ({major['matched']}/{major['gold']})",
                f"{result['error_free_false_positive']}/{result['error_free_valid']}",
                f"{ranks['top_set_exact']}/{ranks['sentences_complete']} exact; {ranks['top_set_any_overlap']}/{ranks['sentences_complete']} overlap",
            ]
        )
    lines.extend(
        table(
            [
                "Evaluator",
                "Valid",
                "MAE",
                "Signed Δ",
                "Pearson",
                "Spearman",
                "Exact F1 (M/P/G)",
                "Overlap F1 (M/P/G)",
                "Major recall",
                "FP on error-free",
                "Top-set agreement",
            ],
            dev_rows,
        )
    )
    lines.extend(
        [
            "",
            "Span cells show matched/predicted/gold counts. Ranking uses only complete four-candidate groups. Correlations and all DEV differences are exploratory because N≤12.",
            "",
            "## Intervention differences relative to V1-A",
            "",
        ]
    )
    a = dev_results["A"]
    comparison_rows = []
    for variant in ("B", "C", "E"):
        result = dev_results[variant]
        comparison_rows.append(
            [
                variant,
                result["scores"]["n"] - a["scores"]["n"],
                fmt(delta(result["scores"]["mae"], a["scores"]["mae"])),
                fmt(delta(result["errors"]["overlap"]["f1"], a["errors"]["overlap"]["f1"])),
                fmt(delta(result["errors"]["major_overlap"]["recall"], a["errors"]["major_overlap"]["recall"])),
                result["error_free_false_positive"] - a["error_free_false_positive"],
                result["rankings"]["top_set_exact"] - a["rankings"]["top_set_exact"],
            ]
        )
    lines.extend(
        table(
            [
                "Variant vs A",
                "Δ valid N",
                "Δ MAE",
                "Δ overlap F1",
                "Δ Major recall",
                "Δ error-free FP",
                "Δ exact top sets",
            ],
            comparison_rows,
        )
    )
    lines.extend(
        [
            "",
            "Negative Δ MAE and error-free false positives, and positive Δ overlap F1, Major recall, or exact top sets, are favorable directions. Interpret every difference as a small-pilot diagnostic, not a general result.",
            "",
            "## Structural failures",
            "",
        ]
    )
    failure_rows = []
    for label in ("V0", *VARIANTS):
        counts = Counter(
            error
            for evaluation_id in input_ids
            for error in predictions[label][evaluation_id].get("parse_errors", [])
        )
        invalid = sum(
            prediction_value(predictions[label][evaluation_id]) is None
            for evaluation_id in input_ids
        )
        failure_rows.append(
            [label, invalid, "; ".join(f"{key}: {value}" for key, value in counts.items()) or "—"]
        )
    lines.extend(table(["Evaluator", "Invalid / 37", "Failure reasons"], failure_rows))
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except ComparisonError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
