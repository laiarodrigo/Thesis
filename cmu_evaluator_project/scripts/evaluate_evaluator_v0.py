#!/usr/bin/env python3
"""Compare official V0 predictions with canonical human DEV annotations."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from compare_v0_passage import ComparisonError, load_and_match, normalized_spans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "runtime/evaluator_dev/v0_official_dev_predictions.jsonl"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "comparisons/v0/v0_dev_analysis.md"
GEMMA_SYSTEM = "gemma-4-E4B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def harmonic(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            result[index] = average
        cursor = end
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(ranks(left), ranks(right))


def percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "sd": statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None),
        "min": min(values) if values else None,
        "q1": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "q3": percentile(values, 0.75),
        "max": max(values) if values else None,
        "bands": {
            "0–20": sum(0 <= value <= 20 for value in values),
            "21–40": sum(21 <= value <= 40 for value in values),
            "41–60": sum(41 <= value <= 60 for value in values),
            "61–80": sum(61 <= value <= 80 for value in values),
            "81–100": sum(81 <= value <= 100 for value in values),
        },
    }


def score_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["prediction"] is not None]
    human = [float(record["human_score"]) for record in valid]
    predicted = [float(record["predicted_score"]) for record in valid]
    differences = [pred - gold for gold, pred in zip(human, predicted)]
    return {
        "total": len(records),
        "n": len(valid),
        "excluded": len(records) - len(valid),
        "mae": statistics.fmean(abs(value) for value in differences) if differences else None,
        "mean_signed_difference": statistics.fmean(differences) if differences else None,
        "pearson": pearson(human, predicted),
        "spearman": spearman(human, predicted),
        "human_distribution": distribution(human),
        "predicted_distribution": distribution(predicted),
    }


def span_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start, left_end = left["start"], left["end"]
    right_start, right_end = right["start"], right["end"]
    if left_start == left_end or right_start == right_end:
        return 1.0 if left_start == left_end == right_start == right_end else 0.0
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return intersection / union if intersection else 0.0


def exact_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["start"] == right["start"] and left["end"] == right["end"]


def overlap_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return span_iou(left, right) > 0


def one_to_one_matches(
    gold: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    compatible: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> list[tuple[int, int]]:
    """Maximum-cardinality matching, then maximum total IoU, with stable tie-breaking."""

    @lru_cache(maxsize=None)
    def solve(gold_index: int, used_mask: int) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if gold_index == len(gold):
            return 0, 0.0, ()
        best = solve(gold_index + 1, used_mask)
        for predicted_index, predicted_span in enumerate(predicted):
            if used_mask & (1 << predicted_index):
                continue
            if not compatible(gold[gold_index], predicted_span):
                continue
            count, weight, pairs = solve(
                gold_index + 1, used_mask | (1 << predicted_index)
            )
            candidate = (
                count + 1,
                weight + span_iou(gold[gold_index], predicted_span),
                ((gold_index, predicted_index),) + pairs,
            )
            if (candidate[0], candidate[1]) > (best[0], best[1]):
                best = candidate
        return best

    return list(solve(0, 0)[2])


def span_prf(
    records: list[dict[str, Any]],
    matcher: Callable[[dict[str, Any], dict[str, Any]], bool],
    *,
    severity: str | None = None,
) -> dict[str, Any]:
    gold_total = predicted_total = matched_total = 0
    for record in records:
        if record["prediction"] is None:
            continue
        gold = record["human_spans"]
        predicted = record["predicted_spans"]
        if severity is not None:
            gold = [span for span in gold if span["severity"] == severity]
            predicted = [span for span in predicted if span["severity"] == severity]
        pairs = one_to_one_matches(gold, predicted, matcher)
        gold_total += len(gold)
        predicted_total += len(predicted)
        matched_total += len(pairs)
    precision = safe_div(matched_total, predicted_total)
    recall = safe_div(matched_total, gold_total)
    return {
        "matched": matched_total,
        "gold": gold_total,
        "predicted": predicted_total,
        "precision": precision,
        "recall": recall,
        "f1": harmonic(precision, recall),
    }


def severity_aware_matcher(
    matcher: Callable[[dict[str, Any], dict[str, Any]], bool]
) -> Callable[[dict[str, Any], dict[str, Any]], bool]:
    return lambda gold, predicted: (
        gold["severity"] == predicted["severity"] and matcher(gold, predicted)
    )


def error_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["prediction"] is not None]
    overlap_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unmatched_gold = unmatched_predicted = 0
    false_positive_records = 0
    for record in valid:
        gold = record["human_spans"]
        predicted = record["predicted_spans"]
        pairs = one_to_one_matches(gold, predicted, overlap_match)
        overlap_pairs.extend((gold[i], predicted[j]) for i, j in pairs)
        unmatched_gold += len(gold) - len(pairs)
        unmatched_predicted += len(predicted) - len(pairs)
        false_positive_records += bool(predicted) and not gold

    severity_confusion = Counter(
        (gold["severity"], predicted["severity"])
        for gold, predicted in overlap_pairs
    )
    severity_correct = sum(gold == predicted for gold, predicted in severity_confusion.elements())
    gold_major_total = sum(
        span["severity"] == "Major" for record in valid for span in record["human_spans"]
    )
    predicted_major_total = sum(
        span["severity"] == "Major"
        for record in valid
        for span in record["predicted_spans"]
    )
    major_matches = span_prf(valid, overlap_match, severity="Major")
    return {
        "records_total": len(records),
        "records_valid": len(valid),
        "records_excluded": len(records) - len(valid),
        "exact": span_prf(valid, exact_match),
        "exact_with_severity": span_prf(valid, severity_aware_matcher(exact_match)),
        "overlap": span_prf(valid, overlap_match),
        "overlap_with_severity": span_prf(valid, severity_aware_matcher(overlap_match)),
        "minor_overlap": span_prf(valid, overlap_match, severity="Minor"),
        "major_overlap": major_matches,
        "severity_agreement_on_overlap": {
            "matched_pairs": len(overlap_pairs),
            "correct": severity_correct,
            "accuracy": safe_div(severity_correct, len(overlap_pairs)),
            "confusion": severity_confusion,
        },
        "missed_major_spans": gold_major_total - major_matches["matched"],
        "gold_major_spans": gold_major_total,
        "false_positive_major_spans": predicted_major_total - major_matches["matched"],
        "predicted_major_spans": predicted_major_total,
        "unmatched_gold_spans_any_severity": unmatched_gold,
        "unmatched_predicted_spans_any_severity": unmatched_predicted,
        "human_error_free_but_predicted_error_records": false_positive_records,
    }


def compare_relation(left: float, right: float) -> int:
    return (left > right) - (left < right)


def ranking_metrics(records: list[dict[str, Any]], systems: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["passage_id"], record["sentence_id"])].append(record)
    complete = [
        group
        for group in grouped.values()
        if len(group) == len(systems) and all(record["prediction"] is not None for record in group)
    ]

    top_exact = top_overlap = 0
    top_jaccard: list[float] = []
    human_ties = predicted_ties = 0
    sentence_spearman: list[float] = []
    undefined_spearman = 0
    pairwise_correct = pairwise_total = 0
    gemma_human_top = gemma_predicted_top = 0
    gemma_human_sole_top = gemma_predicted_sole_top = 0
    gemma_human_credit = gemma_predicted_credit = 0.0
    paired_residual_gaps: list[float] = []

    for group in complete:
        by_system = {record["system_id"]: record for record in group}
        human_scores = [float(by_system[system]["human_score"]) for system in systems]
        predicted_scores = [float(by_system[system]["predicted_score"]) for system in systems]
        human_max, predicted_max = max(human_scores), max(predicted_scores)
        human_top = {
            system for system, score in zip(systems, human_scores) if score == human_max
        }
        predicted_top = {
            system for system, score in zip(systems, predicted_scores) if score == predicted_max
        }
        top_exact += human_top == predicted_top
        top_overlap += bool(human_top & predicted_top)
        top_jaccard.append(len(human_top & predicted_top) / len(human_top | predicted_top))
        human_ties += len(human_top) > 1
        predicted_ties += len(predicted_top) > 1

        correlation = spearman(human_scores, predicted_scores)
        if correlation is None:
            undefined_spearman += 1
        else:
            sentence_spearman.append(correlation)

        for left in range(len(systems)):
            for right in range(left + 1, len(systems)):
                pairwise_total += 1
                pairwise_correct += compare_relation(
                    human_scores[left], human_scores[right]
                ) == compare_relation(predicted_scores[left], predicted_scores[right])

        if GEMMA_SYSTEM in human_top:
            gemma_human_top += 1
            gemma_human_credit += 1 / len(human_top)
        if GEMMA_SYSTEM in predicted_top:
            gemma_predicted_top += 1
            gemma_predicted_credit += 1 / len(predicted_top)
        gemma_human_sole_top += human_top == {GEMMA_SYSTEM}
        gemma_predicted_sole_top += predicted_top == {GEMMA_SYSTEM}

        residuals = {
            system: float(by_system[system]["predicted_score"])
            - float(by_system[system]["human_score"])
            for system in systems
        }
        paired_residual_gaps.append(
            residuals[GEMMA_SYSTEM]
            - statistics.fmean(
                residual for system, residual in residuals.items() if system != GEMMA_SYSTEM
            )
        )

    return {
        "sentences_total": len(grouped),
        "sentences_complete": len(complete),
        "sentences_excluded": len(grouped) - len(complete),
        "candidates_in_complete_sentences": len(complete) * len(systems),
        "top_set_exact": top_exact,
        "top_set_any_overlap": top_overlap,
        "top_set_mean_jaccard": statistics.fmean(top_jaccard) if top_jaccard else None,
        "human_top_ties": human_ties,
        "predicted_top_ties": predicted_ties,
        "mean_sentence_spearman": (
            statistics.fmean(sentence_spearman) if sentence_spearman else None
        ),
        "sentence_spearman_defined": len(sentence_spearman),
        "sentence_spearman_undefined": undefined_spearman,
        "pairwise_relation_correct": pairwise_correct,
        "pairwise_relation_total": pairwise_total,
        "gemma_human_top_membership": gemma_human_top,
        "gemma_predicted_top_membership": gemma_predicted_top,
        "gemma_human_sole_top": gemma_human_sole_top,
        "gemma_predicted_sole_top": gemma_predicted_sole_top,
        "gemma_human_tie_credit": gemma_human_credit,
        "gemma_predicted_tie_credit": gemma_predicted_credit,
        "paired_gemma_residual_gap_mean": (
            statistics.fmean(paired_residual_gaps) if paired_residual_gaps else None
        ),
        "paired_gemma_residual_gap_median": percentile(paired_residual_gaps, 0.5),
        "paired_gemma_residual_gap_positive": sum(value > 0 for value in paired_residual_gaps),
        "paired_gemma_residual_gap_values": paired_residual_gaps,
    }


def missed_major_details(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record["prediction"] is None:
        return []
    gold = [span for span in record["human_spans"] if span["severity"] == "Major"]
    predicted = [span for span in record["predicted_spans"] if span["severity"] == "Major"]
    pairs = one_to_one_matches(gold, predicted, overlap_match)
    matched = {gold_index for gold_index, _ in pairs}
    return [span for index, span in enumerate(gold) if index not in matched]


def md(value: Any, limit: int | None = None) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    if limit is not None and len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def record_label(record: dict[str, Any]) -> str:
    return f"{record['passage_id']} / pos. {record['position']} / {record['system_id']}"


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(md(cell) for cell in row) + " |" for row in rows),
    ]


def render_report(
    records: list[dict[str, Any]], systems: list[str], predictions_path: Path
) -> str:
    valid = [record for record in records if record["prediction"] is not None]
    overall_scores = score_metrics(records)
    overall_errors = error_metrics(records)
    ranking = ranking_metrics(records, systems)
    by_system_scores = {
        system: score_metrics([record for record in records if record["system_id"] == system])
        for system in systems
    }
    by_system_errors = {
        system: error_metrics([record for record in records if record["system_id"] == system])
        for system in systems
    }
    invalid_reasons = Counter(
        error
        for record in records
        if record["prediction"] is None
        for error in record["parse_errors"]
    )

    largest = sorted(
        valid,
        key=lambda record: (
            abs(record["predicted_score"] - record["human_score"]),
            record["evaluation_id"],
        ),
        reverse=True,
    )[:15]
    major_failures = [
        (record, missed_major_details(record))
        for record in valid
        if missed_major_details(record)
    ]
    major_failures.sort(
        key=lambda item: (
            len(item[1]),
            item[0]["predicted_score"] - item[0]["human_score"],
        ),
        reverse=True,
    )
    human_accepted_penalized = sorted(
        [
            record
            for record in valid
            if not record["human_spans"] and record["predicted_spans"]
        ],
        key=lambda record: record["human_score"] - record["predicted_score"],
        reverse=True,
    )
    unnecessary = [
        record
        for record in valid
        if re.search(r"\bunnecessar", record["development_comment"] or "", re.IGNORECASE)
    ]
    unnecessary_accepted = sorted(
        [record for record in unnecessary if record["human_spans"] and not record["predicted_spans"]],
        key=lambda record: record["predicted_score"] - record["human_score"],
        reverse=True,
    )
    unnecessary_without_overlap = [
        record
        for record in unnecessary
        if not one_to_one_matches(
            record["human_spans"], record["predicted_spans"], overlap_match
        )
    ]
    unnecessary_overestimated = [
        record
        for record in unnecessary
        if record["predicted_score"] - record["human_score"] >= 30
    ]
    unnecessary_examples = sorted(
        unnecessary,
        key=lambda record: record["predicted_score"] - record["human_score"],
        reverse=True,
    )

    gemma_scores = by_system_scores[GEMMA_SYSTEM]
    non_gemma_valid = [record for record in valid if record["system_id"] != GEMMA_SYSTEM]
    non_gemma_residuals = [
        record["predicted_score"] - record["human_score"] for record in non_gemma_valid
    ]
    gemma_residuals = [
        record["predicted_score"] - record["human_score"]
        for record in valid
        if record["system_id"] == GEMMA_SYSTEM
    ]
    pooled_gap = (
        statistics.fmean(gemma_residuals) - statistics.fmean(non_gemma_residuals)
        if gemma_residuals and non_gemma_residuals
        else None
    )

    lines = [
        "# Official V0 DEV analysis",
        "",
        f"Predictions: `{predictions_path}`",
        "",
        "## Coverage and scope",
        "",
        f"- Canonical DEV judgments: **{len(records)}** (97 source sentences × 4 systems).",
        f"- Structurally valid predictions used for candidate-level agreement: **{len(valid)}/{len(records)}**; excluded: **{len(records) - len(valid)}**.",
        f"- Four-candidate ranking coverage: **{ranking['sentences_complete']}/{ranking['sentences_total']} sentences** ({ranking['candidates_in_complete_sentences']} candidates); excluded sentences: **{ranking['sentences_excluded']}**.",
        "- Human Pearmut spans were converted from inclusive UTF-16 offsets to exclusive Python code-point offsets with the comparison viewer's verified converter. Gemma spans were validated as exclusive Python offsets.",
        "",
        "All agreement metrics below exclude structurally invalid predictions. Denominators are shown explicitly.",
        "",
        "## Score agreement",
        "",
        *table(
            ["Slice", "Valid / total", "MAE", "Mean Gemma−human", "Pearson", "Spearman", "Human mean", "Gemma mean"],
            [[
                "Overall",
                f"{overall_scores['n']} / {overall_scores['total']}",
                fmt(overall_scores["mae"]),
                fmt(overall_scores["mean_signed_difference"]),
                fmt(overall_scores["pearson"]),
                fmt(overall_scores["spearman"]),
                fmt(overall_scores["human_distribution"]["mean"]),
                fmt(overall_scores["predicted_distribution"]["mean"]),
            ]] + [[
                system,
                f"{by_system_scores[system]['n']} / {by_system_scores[system]['total']}",
                fmt(by_system_scores[system]["mae"]),
                fmt(by_system_scores[system]["mean_signed_difference"]),
                fmt(by_system_scores[system]["pearson"]),
                fmt(by_system_scores[system]["spearman"]),
                fmt(by_system_scores[system]["human_distribution"]["mean"]),
                fmt(by_system_scores[system]["predicted_distribution"]["mean"]),
            ] for system in systems],
        ),
        "",
        "### Paired score distributions",
        "",
        *table(
            ["Scores", "N", "Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"],
            [[
                label,
                values["n"],
                fmt(values["mean"]),
                fmt(values["sd"]),
                fmt(values["min"]),
                fmt(values["q1"]),
                fmt(values["median"]),
                fmt(values["q3"]),
                fmt(values["max"]),
            ] for label, values in (
                ("Human", overall_scores["human_distribution"]),
                ("Gemma V0", overall_scores["predicted_distribution"]),
            )],
        ),
        "",
        *table(
            ["Scores", "0–20", "21–40", "41–60", "61–80", "81–100", "Denominator"],
            [[
                label,
                *[values["bands"][band] for band in ("0–20", "21–40", "41–60", "61–80", "81–100")],
                values["n"],
            ] for label, values in (
                ("Human", overall_scores["human_distribution"]),
                ("Gemma V0", overall_scores["predicted_distribution"]),
            )],
        ),
        "",
        "## Error-span agreement",
        "",
        "Exact matching requires identical exclusive boundaries. Overlap matching requires at least one shared code point; two `[MISSING]` spans match only at the same anchor. Every result uses maximum-cardinality one-to-one matching, with total IoU as the secondary criterion.",
        "",
        *table(
            ["Matching", "Matched", "Predicted spans", "Human spans", "Precision", "Recall", "F1"],
            [[
                label,
                metrics["matched"],
                metrics["predicted"],
                metrics["gold"],
                fmt(metrics["precision"]),
                fmt(metrics["recall"]),
                fmt(metrics["f1"]),
            ] for label, metrics in (
                ("Exact boundary", overall_errors["exact"]),
                ("Exact boundary + severity", overall_errors["exact_with_severity"]),
                ("Any overlap", overall_errors["overlap"]),
                ("Any overlap + severity", overall_errors["overlap_with_severity"]),
            )],
        ),
        "",
        "### Minor/Major agreement",
        "",
        *table(
            ["Severity", "Matched overlap", "Predicted", "Human", "Precision", "Recall", "F1"],
            [[
                severity,
                metrics["matched"],
                metrics["predicted"],
                metrics["gold"],
                fmt(metrics["precision"]),
                fmt(metrics["recall"]),
                fmt(metrics["f1"]),
            ] for severity, metrics in (
                ("Minor", overall_errors["minor_overlap"]),
                ("Major", overall_errors["major_overlap"]),
            )],
        ),
        "",
        f"- Severity agreement among the **{overall_errors['severity_agreement_on_overlap']['matched_pairs']}** severity-agnostic overlap matches: **{fmt(overall_errors['severity_agreement_on_overlap']['accuracy'])}** ({overall_errors['severity_agreement_on_overlap']['correct']}/{overall_errors['severity_agreement_on_overlap']['matched_pairs']}).",
        "",
        *table(
            ["Human severity", "Predicted Minor", "Predicted Major", "Matched denominator"],
            [[
                human_severity,
                overall_errors["severity_agreement_on_overlap"]["confusion"].get((human_severity, "Minor"), 0),
                overall_errors["severity_agreement_on_overlap"]["confusion"].get((human_severity, "Major"), 0),
                sum(overall_errors["severity_agreement_on_overlap"]["confusion"].get((human_severity, predicted_severity), 0) for predicted_severity in ("Minor", "Major")),
            ] for human_severity in ("Minor", "Major")],
        ),
        "",
        f"- Missed human Major spans: **{overall_errors['missed_major_spans']}/{overall_errors['gold_major_spans']}**. False-positive predicted Major spans: **{overall_errors['false_positive_major_spans']}/{overall_errors['predicted_major_spans']}**.",
        f"- Unmatched predicted spans of either severity: **{overall_errors['unmatched_predicted_spans_any_severity']}/{overall_errors['overlap']['predicted']}**. Human-error-free candidates with at least one predicted error: **{overall_errors['human_error_free_but_predicted_error_records']}/{sum(not record['human_spans'] for record in valid)}**.",
        "",
        "### Error agreement by candidate system",
        "",
        *table(
            ["System", "Valid / total", "Exact F1", "Overlap F1", "Minor F1", "Major F1", "Missed Major", "False-positive spans"],
            [[
                system,
                f"{by_system_errors[system]['records_valid']} / {by_system_errors[system]['records_total']}",
                fmt(by_system_errors[system]["exact"]["f1"]),
                fmt(by_system_errors[system]["overlap"]["f1"]),
                fmt(by_system_errors[system]["minor_overlap"]["f1"]),
                fmt(by_system_errors[system]["major_overlap"]["f1"]),
                f"{by_system_errors[system]['missed_major_spans']} / {by_system_errors[system]['gold_major_spans']}",
                f"{by_system_errors[system]['unmatched_predicted_spans_any_severity']} / {by_system_errors[system]['overlap']['predicted']}",
            ] for system in systems],
        ),
        "",
        "## Four-candidate rankings and Gemma self-preference",
        "",
        f"Ranking metrics use only **{ranking['sentences_complete']}/{ranking['sentences_total']}** sentences with all four predictions valid; **{ranking['sentences_excluded']}** sentences are excluded.",
        "",
        *table(
            ["Ranking measure", "Result", "Denominator"],
            [
                ["Exact equality of tied top sets", ranking["top_set_exact"], ranking["sentences_complete"]],
                ["Any overlap between tied top sets", ranking["top_set_any_overlap"], ranking["sentences_complete"]],
                ["Mean top-set Jaccard", fmt(ranking["top_set_mean_jaccard"]), ranking["sentences_complete"]],
                ["Pairwise order/tie relation agreement", ranking["pairwise_relation_correct"], ranking["pairwise_relation_total"]],
                ["Mean within-sentence Spearman", fmt(ranking["mean_sentence_spearman"]), f"{ranking['sentence_spearman_defined']} defined; {ranking['sentence_spearman_undefined']} undefined"],
                ["Human top-set ties", ranking["human_top_ties"], ranking["sentences_complete"]],
                ["Gemma top-set ties", ranking["predicted_top_ties"], ranking["sentences_complete"]],
            ],
        ),
        "",
        *table(
            ["Gemma-generated candidate", "Human ranking", "Evaluator ranking", "Denominator"],
            [
                ["Member of top set", ranking["gemma_human_top_membership"], ranking["gemma_predicted_top_membership"], ranking["sentences_complete"]],
                ["Sole top candidate", ranking["gemma_human_sole_top"], ranking["gemma_predicted_sole_top"], ranking["sentences_complete"]],
                ["Tie-adjusted top credit", fmt(ranking["gemma_human_tie_credit"]), fmt(ranking["gemma_predicted_tie_credit"]), ranking["sentences_complete"]],
            ],
        ),
        "",
        f"Across all valid candidates, the Gemma-generated system's mean score residual is **{fmt(gemma_scores['mean_signed_difference'])}** points (N={gemma_scores['n']}); the pooled non-Gemma residual is **{fmt(statistics.fmean(non_gemma_residuals) if non_gemma_residuals else None)}** (N={len(non_gemma_residuals)}), a Gemma-minus-other gap of **{fmt(pooled_gap)}** points.",
        f"Within the {ranking['sentences_complete']} complete sentences, Gemma's residual minus the mean residual of the other three systems averages **{fmt(ranking['paired_gemma_residual_gap_mean'])}** points (median {fmt(ranking['paired_gemma_residual_gap_median'])}); it is positive in **{ranking['paired_gemma_residual_gap_positive']}/{ranking['sentences_complete']}** sentences.",
        "",
        "These are descriptive self-preference diagnostics, not a causal or significance claim.",
        "",
        "## Failure analysis",
        "",
        f"Structurally invalid predictions excluded from agreement metrics: **{len(records) - len(valid)}/{len(records)}**.",
        "",
        *table(
            ["Invalid prediction reason", "Error occurrences"],
            [[reason, count] for reason, count in invalid_reasons.most_common()],
        ),
        "",
        "### Largest score disagreements",
        "",
        *table(
            ["Evaluation", "Passage / position / system", "Human", "Gemma", "Gemma−human", "Candidate"],
            [[
                record["evaluation_id"],
                record_label(record),
                record["human_score"],
                record["predicted_score"],
                f"{record['predicted_score'] - record['human_score']:+d}",
                md(record["candidate_text"], 120),
            ] for record in largest],
        ),
        "",
        "### Missed Major errors",
        "",
        f"At least one human Major span lacks an overlapping predicted Major span in **{len(major_failures)}** valid candidates. The table shows up to 15, ordered by number of missed spans and score overestimation.",
        "",
        *table(
            ["Evaluation", "Passage / position / system", "Scores H/G", "Missed human Major spans"],
            [[
                record["evaluation_id"],
                record_label(record),
                f"{record['human_score']} / {record['predicted_score']}",
                "; ".join(span["text"] for span in spans),
            ] for record, spans in major_failures[:15]],
        ),
        "",
        "### Human-accepted candidates penalized by Gemma",
        "",
        f"Gemma predicts at least one error for **{len(human_accepted_penalized)}/{sum(not record['human_spans'] for record in valid)}** valid candidates that have no human error spans. The table shows the ten largest score penalties.",
        "",
        *table(
            ["Evaluation", "Passage / position / system", "Scores H/G", "Predicted error spans"],
            [[
                record["evaluation_id"],
                record_label(record),
                f"{record['human_score']} / {record['predicted_score']}",
                "; ".join(f"{span['severity']}: {span['text']}" for span in record["predicted_spans"]),
            ] for record in human_accepted_penalized[:10]],
        ),
        "",
        "### Explicit unnecessary-rewrite notes",
        "",
        f"Rodrigo's English comments are development notes, not formal gold. Using only comments that explicitly contain `unnecessary`, **{len(unnecessary)}** valid candidates are identifiable as unnecessary-rewrite cases. Gemma emits no errors for **{len(unnecessary_accepted)}/{len(unnecessary)}**, has no span overlapping the human annotations for **{len(unnecessary_without_overlap)}/{len(unnecessary)}**, and scores **{len(unnecessary_overestimated)}/{len(unnecessary)}** at least 30 points above the human score. The ten largest score overestimates are shown.",
        "",
        *table(
            ["Evaluation", "Passage / position / system", "Scores H/G", "Overlapping spans", "Development note"],
            [[
                record["evaluation_id"],
                record_label(record),
                f"{record['human_score']} / {record['predicted_score']}",
                len(one_to_one_matches(record["human_spans"], record["predicted_spans"], overlap_match)),
                md(record["development_comment"], 140),
            ] for record in unnecessary_examples[:10]],
        ),
        "",
        "### Failure counts by system",
        "",
        *table(
            ["System", "Invalid", "Large score errors ≥30", "Candidates with missed Major", "Human-error-free false positives", "Unnecessary-note score overestimates ≥30"],
            [[
                system,
                sum(record["prediction"] is None for record in records if record["system_id"] == system),
                sum(record["prediction"] is not None and abs(record["predicted_score"] - record["human_score"]) >= 30 for record in records if record["system_id"] == system),
                sum(bool(missed_major_details(record)) for record in records if record["system_id"] == system and record["prediction"] is not None),
                sum(record["prediction"] is not None and not record["human_spans"] and bool(record["predicted_spans"]) for record in records if record["system_id"] == system),
                sum(record in unnecessary_overestimated for record in records if record["system_id"] == system),
            ] for system in systems],
        ),
        "",
        "## Limitations",
        "",
        "- The 23 structurally invalid predictions are excluded rather than repaired; this can change slice coverage and removes any sentence containing one of them from four-way ranking metrics.",
        "- Overlap means any shared code point, so it is more permissive than exact-boundary matching. One-to-one matching prevents duplicate credit.",
        "- The evaluator emits no error categories or rationales. Unnecessary-rewrite analysis is therefore limited to explicit English development-note keywords and human-error-free false positives; those comments are not formal gold.",
        "- Self-preference results are descriptive residual and ranking comparisons on DEV, not a causal test.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_records(
    predictions_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    joined, predictions_by_id, systems = load_and_match(predictions_path)
    if len(joined) != 388 or len(predictions_by_id) != 388:
        raise ComparisonError(
            f"official DEV requires 388 matched predictions; "
            f"matched={len(joined)}, predictions={len(predictions_by_id)}"
        )
    records: list[dict[str, Any]] = []
    for row in joined:
        evaluation_id = row["evaluation_id"]
        candidate_text = row["candidate"]["text"]
        human_spans, human_warning = normalized_spans(
            candidate_text,
            row["human_gold"]["error_spans"],
            f"{evaluation_id} human",
            human_inclusive_utf16=True,
        )
        if human_warning:
            raise ComparisonError(human_warning)

        prediction_row = row["v0_prediction"]
        prediction = None
        predicted_spans: list[dict[str, Any]] = []
        predicted_score = None
        if prediction_row.get("parse_valid") is True:
            prediction = prediction_row.get("prediction")
            if not isinstance(prediction, dict):
                raise ComparisonError(f"{evaluation_id}: parse-valid prediction is missing")
            predicted_spans, predicted_warning = normalized_spans(
                candidate_text,
                prediction.get("error_spans"),
                f"{evaluation_id} Gemma",
                human_inclusive_utf16=False,
            )
            if predicted_warning:
                raise ComparisonError(predicted_warning)
            predicted_score = prediction.get("score")
            if not isinstance(predicted_score, int) or isinstance(predicted_score, bool):
                raise ComparisonError(f"{evaluation_id}: invalid predicted score")

        development_notes = row.get("development_notes")
        comment = (
            development_notes.get("english_pearmut_comment")
            if isinstance(development_notes, dict)
            else None
        )
        records.append(
            {
                "evaluation_id": evaluation_id,
                "candidate_id": row["candidate_id"],
                "passage_id": row["passage_id"],
                "sentence_id": row["sentence_id"],
                "position": row["position"],
                "system_id": row["candidate"]["system_id"],
                "candidate_text": candidate_text,
                "human_score": row["human_gold"]["score"],
                "human_spans": human_spans,
                "prediction": prediction,
                "predicted_score": predicted_score,
                "predicted_spans": predicted_spans,
                "parse_errors": prediction_row.get("parse_errors", []),
                "development_comment": comment if isinstance(comment, str) else None,
            }
        )
    return records, systems


def main() -> None:
    args = parse_args()
    try:
        records, systems = build_records(args.predictions)
        report = render_report(records, systems, args.predictions)
    except ComparisonError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    valid = sum(record["prediction"] is not None for record in records)
    print(
        f"V0 DEV analysis: records={len(records)} valid={valid} "
        f"invalid={len(records) - valid} systems={len(systems)}"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
