#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Sequence


PT_VARIANT_LABELS: tuple[str, str] = ("pt-br", "pt-pt")
ALL_VARIANT_LABELS: tuple[str, str, str] = ("pt-br", "pt-pt", "equal")

_SHIFT_MAX_LEN = 10
_SHIFT_MAX_DIST = 50
_BEAM_RADIUS = 25
_CACHE_LIMIT = 10_000
_SHIFT_CANDIDATE_BUDGET = 1_000
_INF = 10**15

_OP_NOP = " "
_OP_SUB = "s"
_OP_INS = "i"
_OP_DEL = "d"
_OP_UNKNOWN = "x"
_FLIP_INS_DEL = str.maketrans(_OP_INS + _OP_DEL, _OP_DEL + _OP_INS)


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").replace("\r", " ").split())


def macro_f1_from_labels(
    gold_labels: Sequence[str],
    pred_labels: Sequence[str],
    labels: Sequence[str],
) -> float:
    if not labels:
        return 0.0

    score_sum = 0.0
    for label in labels:
        tp = fp = fn = 0
        for gold, pred in zip(gold_labels, pred_labels):
            if pred == label and gold == label:
                tp += 1
            elif pred == label and gold != label:
                fp += 1
            elif pred != label and gold == label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        score_sum += (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
    return score_sum / len(labels)


def word_edit_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> int:
    if not ref_tokens:
        return len(hyp_tokens)
    if not hyp_tokens:
        return len(ref_tokens)

    prev = list(range(len(hyp_tokens) + 1))
    for row_idx, ref_tok in enumerate(ref_tokens, start=1):
        curr = [row_idx]
        for col_idx, hyp_tok in enumerate(hyp_tokens, start=1):
            sub_cost = 0 if ref_tok == hyp_tok else 1
            curr.append(
                min(
                    prev[col_idx] + 1,
                    curr[col_idx - 1] + 1,
                    prev[col_idx - 1] + sub_cost,
                )
            )
        prev = curr
    return prev[-1]


def word_error_rate(hyp_text: str, ref_text: str) -> float:
    ref_tokens = normalize_whitespace(ref_text).split()
    hyp_tokens = normalize_whitespace(hyp_text).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return word_edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def corpus_word_error_rate(hyps: Sequence[str], refs: Sequence[str]) -> float:
    total_edits = 0
    total_ref_tokens = 0
    for hyp_text, ref_text in zip(hyps, refs):
        ref_tokens = normalize_whitespace(ref_text).split()
        hyp_tokens = normalize_whitespace(hyp_text).split()
        if not ref_tokens:
            total_edits += len(hyp_tokens)
            continue
        total_edits += word_edit_distance(ref_tokens, hyp_tokens)
        total_ref_tokens += len(ref_tokens)

    if total_ref_tokens == 0:
        return 0.0 if total_edits == 0 else 1.0
    return total_edits / total_ref_tokens


def sentence_ter(hyp_text: str, ref_text: str) -> float:
    hyp_tokens = normalize_whitespace(hyp_text).split()
    ref_tokens = normalize_whitespace(ref_text).split()
    edits, ref_len = ter_edit_count(hyp_tokens, ref_tokens)
    if ref_len == 0:
        return 0.0 if edits == 0 else 1.0
    return edits / ref_len


def corpus_ter(hyps: Sequence[str], refs: Sequence[str]) -> float:
    total_edits = 0
    total_ref_tokens = 0
    for hyp_text, ref_text in zip(hyps, refs):
        hyp_tokens = normalize_whitespace(hyp_text).split()
        ref_tokens = normalize_whitespace(ref_text).split()
        edits, ref_len = ter_edit_count(hyp_tokens, ref_tokens)
        total_edits += edits
        total_ref_tokens += ref_len

    if total_ref_tokens == 0:
        return 0.0 if total_edits == 0 else 1.0
    return total_edits / total_ref_tokens


def ter_edit_count(hyp_tokens: list[str], ref_tokens: list[str]) -> tuple[int, int]:
    if not ref_tokens:
        return len(hyp_tokens), 0

    cached_distance = _BeamTraceEditDistance(ref_tokens)
    shifted = list(hyp_tokens)
    shift_count = 0
    checked = 0

    while True:
        gain, shifted_candidate, checked = _best_shift(
            shifted,
            ref_tokens,
            cached_distance,
            checked,
        )
        if checked >= _SHIFT_CANDIDATE_BUDGET or gain <= 0:
            break
        shift_count += 1
        shifted = shifted_candidate

    edit_distance, _trace = cached_distance(shifted)
    return shift_count + edit_distance, len(ref_tokens)


def _best_shift(
    hyp_tokens: list[str],
    ref_tokens: list[str],
    cached_distance: "_BeamTraceEditDistance",
    checked_candidates: int,
) -> tuple[int, list[str], int]:
    baseline_distance, reverse_trace = cached_distance(hyp_tokens)
    alignment, ref_err, hyp_err = _alignment_from_trace(_flip_trace(reverse_trace))
    best_candidate: tuple[int, int, int, int, list[str]] | None = None

    for hyp_start, ref_start, span_len in _matching_spans(hyp_tokens, ref_tokens):
        if sum(hyp_err[hyp_start : hyp_start + span_len]) == 0:
            continue
        if sum(ref_err[ref_start : ref_start + span_len]) == 0:
            continue
        aligned_target = alignment.get(ref_start)
        if aligned_target is not None and hyp_start <= aligned_target < hyp_start + span_len:
            continue

        last_target = None
        for offset in range(-1, span_len):
            ref_pos = ref_start + offset
            if ref_pos < 0:
                target_index = 0
            elif ref_pos in alignment:
                target_index = alignment[ref_pos] + 1
            else:
                break

            if target_index == last_target:
                continue
            last_target = target_index

            moved = _move_span(hyp_tokens, hyp_start, span_len, target_index)
            new_distance, _ = cached_distance(moved)
            gain = baseline_distance - new_distance
            candidate = (gain, span_len, -hyp_start, -target_index, moved)
            if best_candidate is None or candidate > best_candidate:
                best_candidate = candidate

            checked_candidates += 1
            if checked_candidates >= _SHIFT_CANDIDATE_BUDGET:
                break

        if checked_candidates >= _SHIFT_CANDIDATE_BUDGET:
            break

    if best_candidate is None:
        return 0, hyp_tokens, checked_candidates
    return best_candidate[0], best_candidate[-1], checked_candidates


def _move_span(tokens: list[str], start: int, length: int, target: int) -> list[str]:
    span = tokens[start : start + length]
    if target < start:
        return tokens[:target] + span + tokens[target:start] + tokens[start + length :]
    if target > start + length:
        return tokens[:start] + tokens[start + length : target] + span + tokens[target:]
    inner_stop = length + target
    return tokens[:start] + tokens[start + length : inner_stop] + span + tokens[inner_stop:]


def _matching_spans(hyp_tokens: list[str], ref_tokens: list[str]):
    hyp_len = len(hyp_tokens)
    ref_len = len(ref_tokens)
    for hyp_start in range(hyp_len):
        for ref_start in range(ref_len):
            if hyp_start == ref_start:
                continue
            if abs(ref_start - hyp_start) > _SHIFT_MAX_DIST:
                continue
            if hyp_tokens[hyp_start] != ref_tokens[ref_start]:
                continue

            span_len = 0
            max_len = min(_SHIFT_MAX_LEN, hyp_len - hyp_start, ref_len - ref_start)
            while span_len < max_len and hyp_tokens[hyp_start + span_len] == ref_tokens[ref_start + span_len]:
                span_len += 1

            if span_len > 0:
                yield hyp_start, ref_start, span_len


def _flip_trace(trace: str) -> str:
    return trace.translate(_FLIP_INS_DEL)


def _alignment_from_trace(trace: str) -> tuple[dict[int, int], list[int], list[int]]:
    hyp_pos = -1
    ref_pos = -1
    ref_errors: list[int] = []
    hyp_errors: list[int] = []
    alignment: dict[int, int] = {}

    for op in trace:
        if op == _OP_NOP:
            hyp_pos += 1
            ref_pos += 1
            alignment[ref_pos] = hyp_pos
            hyp_errors.append(0)
            ref_errors.append(0)
        elif op == _OP_SUB:
            hyp_pos += 1
            ref_pos += 1
            alignment[ref_pos] = hyp_pos
            hyp_errors.append(1)
            ref_errors.append(1)
        elif op == _OP_INS:
            hyp_pos += 1
            hyp_errors.append(1)
        elif op == _OP_DEL:
            ref_pos += 1
            alignment[ref_pos] = hyp_pos
            ref_errors.append(1)
        else:
            raise ValueError(f"Unknown edit operation: {op!r}")

    return alignment, ref_errors, hyp_errors


class _BeamTraceEditDistance:
    def __init__(self, ref_tokens: list[str]) -> None:
        self._ref_tokens = ref_tokens
        self._ref_len = len(ref_tokens)
        self._initial_row = [(idx, _OP_INS) for idx in range(self._ref_len + 1)]
        self._empty_row = [(_INF, _OP_UNKNOWN)] * (self._ref_len + 1)
        self._cache: dict[str, tuple[dict, tuple[tuple[int, str], ...]]] = {}
        self._cache_size = 0

    def __call__(self, hyp_tokens: list[str]) -> tuple[int, str]:
        start_idx, cached_rows = self._cache_prefix(hyp_tokens)
        distance, fresh_rows, trace = self._compute_matrix(hyp_tokens, start_idx, cached_rows)
        self._store_rows(hyp_tokens, fresh_rows)
        return distance, trace

    def _compute_matrix(
        self,
        hyp_tokens: list[str],
        start_idx: int,
        cached_rows: list[list[tuple[int, str]]],
    ) -> tuple[int, list[list[tuple[int, str]]], str]:
        hyp_len = len(hyp_tokens)
        trailing_rows = [list(self._empty_row) for _ in range(hyp_len - start_idx)]
        matrix = cached_rows + trailing_rows
        length_ratio = self._ref_len / hyp_len if hyp_tokens else 1.0
        if _BEAM_RADIUS < length_ratio / 2.0:
            beam_radius = math.ceil(length_ratio / 2.0 + _BEAM_RADIUS)
        else:
            beam_radius = _BEAM_RADIUS

        for hyp_idx in range(start_idx + 1, hyp_len + 1):
            pseudo_diag = math.floor(hyp_idx * length_ratio)
            min_ref = max(0, pseudo_diag - beam_radius)
            max_ref = min(self._ref_len + 1, pseudo_diag + beam_radius)
            if hyp_idx == hyp_len:
                max_ref = self._ref_len + 1

            for ref_idx in range(min_ref, max_ref):
                if ref_idx == 0:
                    prev_cost = matrix[hyp_idx - 1][ref_idx][0]
                    matrix[hyp_idx][ref_idx] = (prev_cost + 1, _OP_DEL)
                    continue

                if hyp_tokens[hyp_idx - 1] == self._ref_tokens[ref_idx - 1]:
                    sub_cost = 0
                    sub_op = _OP_NOP
                else:
                    sub_cost = 1
                    sub_op = _OP_SUB

                candidates = (
                    (matrix[hyp_idx - 1][ref_idx - 1][0] + sub_cost, sub_op),
                    (matrix[hyp_idx - 1][ref_idx][0] + 1, _OP_DEL),
                    (matrix[hyp_idx][ref_idx - 1][0] + 1, _OP_INS),
                )
                best_cost, best_op = matrix[hyp_idx][ref_idx]
                for cand_cost, cand_op in candidates:
                    if cand_cost < best_cost:
                        best_cost, best_op = cand_cost, cand_op
                matrix[hyp_idx][ref_idx] = (best_cost, best_op)

        trace_ops: list[str] = []
        hyp_idx = hyp_len
        ref_idx = self._ref_len
        while hyp_idx > 0 or ref_idx > 0:
            op = matrix[hyp_idx][ref_idx][1]
            trace_ops.append(op)
            if op in (_OP_NOP, _OP_SUB):
                hyp_idx -= 1
                ref_idx -= 1
            elif op == _OP_INS:
                ref_idx -= 1
            elif op == _OP_DEL:
                hyp_idx -= 1
            else:
                raise ValueError(f"Unknown edit operation: {op!r}")

        return matrix[-1][-1][0], matrix[len(cached_rows) :], "".join(reversed(trace_ops))

    def _store_rows(self, hyp_tokens: list[str], rows: list[list[tuple[int, str]]]) -> None:
        if self._cache_size >= _CACHE_LIMIT:
            return

        node = self._cache
        skip = len(hyp_tokens) - len(rows)
        for token in hyp_tokens[:skip]:
            node = node[token][0]

        for token, row in zip(hyp_tokens[skip:], rows):
            if token not in node:
                node[token] = ({}, tuple(row))
                self._cache_size += 1
            node = node[token][0]

    def _cache_prefix(self, hyp_tokens: list[str]) -> tuple[int, list[list[tuple[int, str]]]]:
        node = self._cache
        matched = 0
        rows = [self._initial_row]
        for token in hyp_tokens:
            cached = node.get(token)
            if cached is None:
                break
            matched += 1
            node, row = cached
            rows.append(list(row))
        return matched, rows
