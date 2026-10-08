"""Unit tests for the benchmark metric functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import precision, recall, tokens_used


def test_recall_full_partial_none():
    assert recall(["a", "b"], ["a", "b"]) == 1.0
    assert recall(["a"], ["a", "b"]) == 0.5
    assert recall(["x"], ["a", "b"]) == 0.0


def test_recall_selected_superset_of_gold_is_full():
    assert recall(["a", "b", "c"], ["a", "b"]) == 1.0


def test_recall_empty_gold_is_zero():
    assert recall(["a"], []) == 0.0


def test_precision_exact_superset_empty():
    assert precision(["a", "b"], ["a", "b"]) == 1.0
    assert precision(["a", "b", "c", "d"], ["a", "b"]) == 0.5   # 2 of 4 selected are gold
    assert precision([], ["a"]) == 0.0


def test_precision_no_overlap():
    assert precision(["x", "y"], ["a"]) == 0.0


def test_tokens_used_sums_selected():
    tokens = {"a": 3, "b": 5, "c": 10}
    assert tokens_used(["a", "b"], tokens) == 8
    assert tokens_used([], tokens) == 0


def test_metrics_ignore_duplicate_ids():
    assert recall(["a", "a", "b"], ["a", "b"]) == 1.0
    assert precision(["a", "a"], ["a", "b"]) == 1.0
    assert tokens_used(["a", "a"], {"a": 4}) == 4
