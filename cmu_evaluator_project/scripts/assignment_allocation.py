#!/usr/bin/env python3
"""Deterministic, balanced two-annotator passage allocation."""

from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

from campaign_common import CampaignError, require_nonempty_string


def _simple_balanced_remainder(
    annotators: list[str], edge_count: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Build a simple near-regular graph with ``edge_count`` edges."""
    if edge_count == 0:
        return []
    node_count = len(annotators)
    low, extra = divmod(2 * edge_count, node_count)
    tie_order = list(annotators)
    rng.shuffle(tie_order)
    priority = {annotator: index for index, annotator in enumerate(tie_order)}
    degrees = {
        annotator: low + (priority[annotator] < extra) for annotator in annotators
    }
    edges: list[tuple[str, str]] = []
    while any(degrees.values()):
        ordered = sorted(
            degrees, key=lambda value: (-degrees[value], priority[value], value)
        )
        left = ordered[0]
        degree = degrees[left]
        candidates = [value for value in ordered[1:] if degrees[value] > 0]
        if degree > len(candidates):
            raise CampaignError(
                "could not construct a balanced unique-pair remainder"
            )
        degrees[left] = 0
        for right in candidates[:degree]:
            degrees[right] -= 1
            if degrees[right] < 0:
                raise CampaignError("invalid residual degree during allocation")
            edges.append(tuple(sorted((left, right))))
    if len(edges) != edge_count or len(edges) != len(set(edges)):
        raise CampaignError("balanced unique-pair construction failed")
    return edges


def allocate_two_annotators(
    passage_ids: list[str], annotator_ids: list[str], seed: int
) -> list[dict[str, object]]:
    """Assign each passage to two distinct annotators.

    Loads differ by at most one. Every possible annotator pair is used the
    same number of times before any pair receives one additional use.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CampaignError("assignment seed must be an integer")
    if not passage_ids or not all(isinstance(value, str) and value for value in passage_ids):
        raise CampaignError("passage_ids must be a non-empty string list")
    if len(passage_ids) != len(set(passage_ids)):
        raise CampaignError("passage_ids contain duplicates")
    if len(annotator_ids) < 2:
        raise CampaignError("at least two annotators are required")
    normalized = [
        require_nonempty_string(value, "annotator_id") for value in annotator_ids
    ]
    if len(normalized) != len(set(normalized)):
        raise CampaignError("annotator_ids contain duplicates")

    rng = random.Random(seed)
    all_pairs = [tuple(pair) for pair in combinations(normalized, 2)]
    complete_rounds, remainder = divmod(len(passage_ids), len(all_pairs))
    pairs = all_pairs * complete_rounds
    pairs.extend(_simple_balanced_remainder(normalized, remainder, rng))
    rng.shuffle(pairs)

    allocations = [
        {"passage_id": passage_id, "annotator_ids": list(pair)}
        for passage_id, pair in zip(passage_ids, pairs, strict=True)
    ]
    loads = Counter(
        annotator
        for allocation in allocations
        for annotator in allocation["annotator_ids"]
    )
    if max(loads.values()) - min(loads.values()) > 1:
        raise CampaignError("internal error: annotator loads are not balanced")
    return allocations

