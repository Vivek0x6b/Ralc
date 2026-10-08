"""Retrieval-level metrics for the benchmark.

All functions operate on sets of message ids, so duplicate ids never inflate a
score. Every number a benchmark reports is computed from these functions over
an actual run.
"""

from __future__ import annotations

from collections.abc import Iterable


def recall(selected: Iterable[str], gold: Iterable[str]) -> float:
    """Share of gold supporting messages that were selected."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(set(selected) & gold_set) / len(gold_set)


def precision(selected: Iterable[str], gold: Iterable[str]) -> float:
    """Share of selected messages that are gold supporting messages."""
    selected_set = set(selected)
    if not selected_set:
        return 0.0
    return len(selected_set & set(gold)) / len(selected_set)


def tokens_used(selected: Iterable[str], tokens_by_id: dict[str, int]) -> int:
    """Total tokens of the selected messages."""
    return sum(tokens_by_id[node_id] for node_id in set(selected))
