"""Tests for the harder v2 benchmark dataset (built deterministically)."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from benchmarks.data.v2.build_dataset import LATE_SIGNALS, NEAR_MISS, build
from benchmarks.run import BUDGETS


@pytest.fixture(scope="module")
def data():
    return build()


def _gold_ids(questions):
    ids = set()
    for q in questions:
        ids.update(q["gold_message_ids"])
    return ids


def test_ids_sequential_and_all_unique(data):
    messages = data["messages"]
    assert [m["id"] for m in messages] == [f"m{i}" for i in range(len(messages))]
    assert len({m["content"] for m in messages}) == len(messages)


def test_at_least_20k_tokens(data):
    assert data["tokens"] >= 20000


def test_near_miss_present_and_never_gold(data):
    assert len(data["near_miss_ids"]) >= 10
    assert len(data["near_miss_ids"]) == len(NEAR_MISS)
    gold = _gold_ids(data["questions"])
    assert set(data["near_miss_ids"]).isdisjoint(gold)


def test_some_signals_placed_late(data):
    messages = data["messages"]
    n = len(messages)
    name_to_id = data["name_to_id"]
    assert len(LATE_SIGNALS) >= 3
    for name in LATE_SIGNALS:
        index = int(name_to_id[name][1:])
        assert index >= 0.85 * n, f"{name} at {index} is not in the last 15% of {n}"


def test_multi_hop_questions_have_three_to_four_supports(data):
    questions = data["questions"]
    multi = [q for q in questions if 3 <= len(q["gold_message_ids"]) <= 4]
    assert len(multi) >= 3
    ids = {m["id"] for m in data["messages"]}
    for q in questions:
        assert q["gold_message_ids"], "every question needs gold support"
        for g in q["gold_message_ids"]:
            assert g in ids


def test_budgets_add_250_and_500_for_v2():
    assert 250 in BUDGETS["v2"] and 500 in BUDGETS["v2"]
    assert BUDGETS["v1"] == [1000, 2000, 4000]


def test_v1_dataset_is_untouched():
    v1 = json.loads(
        (pathlib.Path(__file__).resolve().parents[1]
         / "benchmarks" / "data" / "conversation.json").read_text(encoding="utf-8")
    )
    assert len(v1) == 234   # the committed v1 size, unchanged by v2
