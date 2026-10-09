"""Tests for Phase 5 token budget allocation."""

import numpy as np
import pytest

from ralc.allocation import (
    ApproximateTokenCounter,
    ContextResult,
    ContextSelector,
    DefaultTokenCounter,
    SelectionConfig,
)
from ralc.graph import ContextGraph, Edge, Node
from ralc.retrieval import RankedCandidate, SemanticRetriever


class MatrixEmbedder:
    def __init__(self, mapping: dict[str, list[float]], dim: int):
        self.mapping = mapping
        self.dim = dim

    def embed(self, texts):
        rows = [np.array(self.mapping.get(t, [0.0] * self.dim), dtype=float) for t in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.dim))


class FakeCounter:
    def __init__(self, mapping: dict[str, int], approximate: bool = False):
        self.mapping = mapping
        self.approximate = approximate

    def count(self, text: str) -> int:
        return self.mapping[text]


def _graph(nodes, edges=()):
    """nodes: (id, content, seq); edges: (source, target, type)."""
    g = ContextGraph()
    for nid, content, seq in nodes:
        g.add_node(Node(nid, content, "Message", 1.0, metadata={"seq": seq}))
    for source, target, etype in edges:
        g.add_edge(Edge(source, target, etype, weight=0.5))
    return g


def _rc(node_id, score):
    return RankedCandidate(node_id, score, {}, [node_id], [])


def _no_redundancy(**kw):
    return SelectionConfig(redundancy_penalty=False, **kw)


# ==========================================================================
# Token counters
# ==========================================================================

def test_approximate_counter():
    c = ApproximateTokenCounter()
    assert c.approximate is True
    assert c.count("abcd") == 1       # ceil(4/4)
    assert c.count("abcde") == 2      # ceil(5/4)
    assert c.count("") == 0
    assert c.count("abcd") == 1       # cached, same answer


def test_default_token_counter_basic():
    c = DefaultTokenCounter()
    assert isinstance(c.approximate, bool)
    n = c.count("hello world")
    assert isinstance(n, int) and n >= 1


def test_semantic_embedding_accessor_normalized_and_cached():
    g = _graph([("A", "x", 0)])
    r = SemanticRetriever(g, MatrixEmbedder({"x": [3.0, 4.0]}, 2))
    v = r.embedding("A")
    assert float(np.linalg.norm(v)) == pytest.approx(1.0)
    assert "A" in r._vectors


# ==========================================================================
# Budget guarantee
# ==========================================================================

@pytest.mark.parametrize("budget", [0, 1, 3, 5, 7, 100])
def test_token_count_never_exceeds_budget(budget):
    g = _graph([("A", "a", 0), ("B", "b", 1), ("C", "c", 2)])
    counter = FakeCounter({"a": 3, "b": 4, "c": 2})
    sel = ContextSelector(g, counter, _no_redundancy())
    res = sel.select("q", [_rc("A", 1.0), _rc("B", 0.8), _rc("C", 0.6)], budget)
    assert res.token_count <= budget


def test_budget_zero_selects_nothing():
    g = _graph([("A", "a", 0), ("B", "b", 1)])
    counter = FakeCounter({"a": 3, "b": 4})
    res = ContextSelector(g, counter, _no_redundancy()).select("q", [_rc("A", 1.0), _rc("B", 0.8)], 0)
    assert res.nodes == []
    assert res.token_count == 0
    assert res.metadata["nodes_selected"] == 0
    assert res.metadata["nodes_skipped_for_size"] == 2
    assert res.metadata["nodes_skipped_low_score"] == 0


def test_budget_larger_than_everything_selects_all():
    g = _graph([("A", "a", 0), ("B", "b", 1)])
    counter = FakeCounter({"a": 3, "b": 4})
    res = ContextSelector(g, counter, _no_redundancy()).select("q", [_rc("A", 1.0), _rc("B", 0.8)], 1000)
    assert {n.id for n in res.nodes} == {"A", "B"}
    assert res.token_count == 7
    assert res.metadata["nodes_skipped_for_size"] == 0
    assert res.metadata["nodes_skipped_low_score"] == 0


def test_oversized_node_skipped_while_smaller_fits():
    g = _graph([("Big", "big", 0), ("Small", "small", 1)])
    counter = FakeCounter({"big": 100, "small": 2})
    res = ContextSelector(g, counter, _no_redundancy()).select(
        "q", [_rc("Big", 1.0), _rc("Small", 0.9)], 10
    )
    assert {n.id for n in res.nodes} == {"Small"}
    assert res.token_count == 2
    assert res.metadata["nodes_skipped_for_size"] == 1
    assert res.metadata["nodes_skipped_low_score"] == 0


# ==========================================================================
# Strategies
# ==========================================================================

def test_greedy_score_vs_density_choose_differently():
    g = _graph([("A", "a", 0), ("B", "b", 1)])
    counter = FakeCounter({"a": 10, "b": 2})
    cands = [_rc("A", 1.0), _rc("B", 0.6)]

    score_sel = ContextSelector(g, counter, _no_redundancy(strategy="greedy_score"))
    density_sel = ContextSelector(g, counter, _no_redundancy(strategy="greedy_density"))
    assert {n.id for n in score_sel.select("q", cands, 10).nodes} == {"A"}
    assert {n.id for n in density_sel.select("q", cands, 10).nodes} == {"B"}


def test_invalid_strategy_raises():
    g = _graph([("A", "a", 0)])
    with pytest.raises(ValueError):
        ContextSelector(g, FakeCounter({"a": 1}), _no_redundancy(strategy="bad"))


# ==========================================================================
# Redundancy (MMR) and min_score
# ==========================================================================

def _dup_graph():
    g = _graph([("D1", "d1", 0), ("D2", "d2", 1), ("X", "x", 2)])
    retriever = SemanticRetriever(g, MatrixEmbedder({"d1": [1, 0], "d2": [1, 0], "x": [0, 1]}, 2))
    counter = FakeCounter({"d1": 5, "d2": 5, "x": 5})
    return g, retriever, counter


def test_redundancy_requires_retriever():
    g = _graph([("A", "a", 0)])
    with pytest.raises(ValueError):
        ContextSelector(g, FakeCounter({"a": 1}), SelectionConfig(redundancy_penalty=True))


def test_redundancy_drops_penalized_duplicate_even_with_budget_left():
    g, retriever, counter = _dup_graph()
    cfg = SelectionConfig(strategy="greedy_score", redundancy_penalty=True,
                          redundancy_lambda=1.0, min_score=0.0)
    sel = ContextSelector(g, counter, cfg, semantic_retriever=retriever)
    res = sel.select("q", [_rc("D1", 1.0), _rc("D2", 0.95), _rc("X", 0.9)], token_budget=100)

    ids = {n.id for n in res.nodes}
    assert ids == {"D1", "X"}            # duplicate D2 dropped for the distinct X
    assert res.metadata["nodes_skipped_low_score"] == 1
    assert res.metadata["nodes_skipped_for_size"] == 0
    assert res.token_count == 10 and res.token_count < 100   # fewer tokens than budget


def test_without_redundancy_keeps_duplicate():
    g, _retriever, counter = _dup_graph()
    sel = ContextSelector(g, counter, _no_redundancy(strategy="greedy_score"))
    res = sel.select("q", [_rc("D1", 1.0), _rc("D2", 0.95), _rc("X", 0.9)], token_budget=100)
    assert "D2" in {n.id for n in res.nodes}


# ==========================================================================
# Relation-aware redundancy: waive the MMR penalty between linked nodes.
# ==========================================================================

def _dup_graph_linked(etype, source="D1", target="D2"):
    g, retriever, counter = _dup_graph()
    g.add_edge(Edge(source, target, etype, weight=0.7))
    return g, retriever, counter


@pytest.mark.parametrize("etype", ["UPDATES", "ANSWERS"])
def test_relation_aware_redundancy_keeps_linked_duplicate(etype):
    g, retriever, counter = _dup_graph_linked(etype)
    cfg = SelectionConfig(strategy="greedy_score", redundancy_penalty=True,
                          redundancy_lambda=1.0, min_score=0.0,
                          redundancy_exempt_edge_types=(etype,))
    sel = ContextSelector(g, counter, cfg, semantic_retriever=retriever)
    res = sel.select("q", [_rc("D1", 1.0), _rc("D2", 0.95), _rc("X", 0.9)], token_budget=100)
    # D2 is a near-duplicate of D1 but linked by etype, so it is kept.
    assert {n.id for n in res.nodes} == {"D1", "D2", "X"}


def test_relation_aware_redundancy_direction_insensitive():
    g, retriever, counter = _dup_graph_linked("UPDATES", source="D2", target="D1")
    cfg = SelectionConfig(strategy="greedy_score", redundancy_penalty=True,
                          redundancy_lambda=1.0,
                          redundancy_exempt_edge_types=("UPDATES",))
    sel = ContextSelector(g, counter, cfg, semantic_retriever=retriever)
    res = sel.select("q", [_rc("D1", 1.0), _rc("D2", 0.95), _rc("X", 0.9)], token_budget=100)
    assert "D2" in {n.id for n in res.nodes}


def test_relation_aware_redundancy_only_exempts_listed_types():
    g, retriever, counter = _dup_graph_linked("TEMPORALLY_FOLLOWS")
    cfg = SelectionConfig(strategy="greedy_score", redundancy_penalty=True,
                          redundancy_lambda=1.0,
                          redundancy_exempt_edge_types=("UPDATES", "ANSWERS"))
    sel = ContextSelector(g, counter, cfg, semantic_retriever=retriever)
    res = sel.select("q", [_rc("D1", 1.0), _rc("D2", 0.95), _rc("X", 0.9)], token_budget=100)
    # The edge is not one of the exempt types, so the duplicate is still dropped.
    assert {n.id for n in res.nodes} == {"D1", "X"}


def test_relation_aware_default_empty_changes_nothing():
    # An UPDATES edge exists, but the default empty exempt set means the
    # duplicate is dropped exactly as it is without the feature.
    g, retriever, counter = _dup_graph_linked("UPDATES")
    cfg = SelectionConfig(strategy="greedy_score", redundancy_penalty=True,
                          redundancy_lambda=1.0)
    assert cfg.redundancy_exempt_edge_types == ()
    sel = ContextSelector(g, counter, cfg, semantic_retriever=retriever)
    res = sel.select("q", [_rc("D1", 1.0), _rc("D2", 0.95), _rc("X", 0.9)], token_budget=100)
    assert {n.id for n in res.nodes} == {"D1", "X"}


def test_low_score_candidate_excluded_with_budget_left():
    g = _graph([("A", "a", 0), ("B", "b", 1)])
    counter = FakeCounter({"a": 2, "b": 2})
    cfg = _no_redundancy(min_score=0.5)
    res = ContextSelector(g, counter, cfg).select("q", [_rc("A", 1.0), _rc("B", 0.2)], 100)
    assert {n.id for n in res.nodes} == {"A"}
    assert res.metadata["nodes_skipped_low_score"] == 1
    assert res.metadata["nodes_skipped_for_size"] == 0
    assert res.token_count == 2


# ==========================================================================
# Output shape: chronological order, relationships, metadata
# ==========================================================================

def test_selected_nodes_in_chronological_order():
    g = _graph([("A", "a", 2), ("B", "b", 0), ("C", "c", 1)])
    counter = FakeCounter({"a": 1, "b": 1, "c": 1})
    res = ContextSelector(g, counter, _no_redundancy()).select(
        "q", [_rc("A", 1.0), _rc("B", 0.5), _rc("C", 0.7)], 100
    )
    assert [n.metadata["seq"] for n in res.nodes] == [0, 1, 2]


def test_relationships_only_between_selected_nodes():
    g = _graph(
        [("A", "a", 0), ("B", "b", 1), ("C", "c", 2)],
        edges=[("A", "B", "TEMPORALLY_FOLLOWS"), ("B", "C", "TEMPORALLY_FOLLOWS")],
    )
    counter = FakeCounter({"a": 2, "b": 2, "c": 100})
    res = ContextSelector(g, counter, _no_redundancy()).select(
        "q", [_rc("A", 1.0), _rc("B", 0.9), _rc("C", 0.8)], 5
    )
    assert {n.id for n in res.nodes} == {"A", "B"}
    assert [(e.source, e.target) for e in res.relationships] == [("A", "B")]


def test_metadata_numbers_match_reality():
    g = _graph([("A", "a", 0), ("B", "b", 1), ("C", "c", 2)])
    counter = FakeCounter({"a": 2, "b": 3, "c": 100})
    res = ContextSelector(g, counter, _no_redundancy(strategy="greedy_score")).select(
        "q", [_rc("A", 1.0), _rc("B", 0.9), _rc("C", 0.8)], 6
    )
    md = res.metadata
    assert md["budget"] == 6
    assert md["total_available_tokens"] == 105
    assert md["candidates_considered"] == 3
    assert md["nodes_selected"] == 2
    assert md["nodes_skipped_for_size"] == 1
    assert md["nodes_skipped_low_score"] == 0
    assert md["approximate"] is False
    assert md["strategy"] == "greedy_score"
    assert md["nodes_skipped_for_size"] + md["nodes_skipped_low_score"] == (
        md["candidates_considered"] - md["nodes_selected"]
    )
    assert res.token_count == 5


def test_metadata_approximate_flag_from_counter():
    g = _graph([("A", "aaaa", 0)])
    res = ContextSelector(g, ApproximateTokenCounter(), _no_redundancy()).select("q", [_rc("A", 1.0)], 100)
    assert res.metadata["approximate"] is True


def test_empty_candidates():
    g = _graph([("A", "a", 0)])
    res = ContextSelector(g, FakeCounter({"a": 1}), _no_redundancy()).select("q", [], 100)
    assert res.nodes == [] and res.relationships == [] and res.token_count == 0
    assert res.metadata["candidates_considered"] == 0


# ==========================================================================
# Query immutability
# ==========================================================================

def test_query_not_mutated_or_stored():
    g = _graph([("A", "a", 0)])
    counter = FakeCounter({"a": 2})
    query = "Why did we choose the tenant_cache layout?"
    before = str(query)
    res = ContextSelector(g, counter, _no_redundancy()).select(query, [_rc("A", 1.0)], 100)

    assert query == before
    assert not hasattr(res, "query")
    assert set(res.metadata) == {
        "budget", "total_available_tokens", "candidates_considered",
        "nodes_selected", "nodes_skipped_for_size", "nodes_skipped_low_score",
        "approximate", "strategy",
    }
    assert isinstance(res, ContextResult)
