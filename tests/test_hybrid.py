"""Tests for Phase 4 hybrid ranking."""

import numpy as np
import pytest

from ralc.extraction import HeuristicExtractor, Message
from ralc.graph import ContextGraph, Edge, Node
from ralc.retrieval import (
    Candidate,
    HybridRanker,
    RankedCandidate,
    RankingConfig,
    SemanticRetriever,
)


class MatrixEmbedder:
    """Returns the raw vector mapped to each text (zero if unknown).

    Raw (possibly unnormalized or negative) so the retriever's internal
    normalization and the cosine clipping can be exercised.
    """

    def __init__(self, mapping: dict[str, list[float]], dim: int):
        self.mapping = mapping
        self.dim = dim
        self.seen: list[list[str]] = []

    def embed(self, texts):
        self.seen.append(list(texts))
        rows = [np.array(self.mapping.get(t, [0.0] * self.dim), dtype=float) for t in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.dim))


class RecordingExtractor:
    def __init__(self):
        self._inner = HeuristicExtractor()
        self.seen_contents: list[str] = []

    def extract(self, message):
        self.seen_contents.append(message.content)
        return self._inner.extract(message)


def _graph(nodes):
    """nodes: list of (id, content, seq)."""
    g = ContextGraph()
    for nid, content, seq in nodes:
        g.add_node(Node(nid, content, "Message", 1.0, metadata={"seq": seq}))
    return g


def _cand(node_id, score=0.0, is_seed=False):
    return Candidate(
        node_id=node_id, score=score, path=[node_id], path_edge_types=[],
        is_seed=is_seed, seed_score=(score if is_seed else None),
    )


def _only(**weights):
    base = dict(semantic=0.0, relational=0.0, entity_overlap=0.0, recency=0.0, seed=0.0)
    base.update(weights)
    return RankingConfig(**base)


def _ranker(graph, embedder=None, extractor=None, config=None):
    embedder = embedder or MatrixEmbedder({}, dim=2)
    return HybridRanker(
        SemanticRetriever(graph, embedder),
        extractor or HeuristicExtractor(),
        config or RankingConfig(),
    )


# ==========================================================================
# SemanticRetriever.score
# ==========================================================================

def test_score_covers_unindexed_nodes_and_caches():
    g = _graph([("A", "a", 0), ("B", "b", 1)])
    embedder = MatrixEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0], "q": [1.0, 0.0]}, dim=2)
    r = SemanticRetriever(g, embedder)
    # no index() call; scoring must embed on demand
    scores = r.score("q", ["A", "B"])
    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(0.0)
    assert "A" in r._vectors and "B" in r._vectors   # cached


def test_score_missing_node_raises():
    g = _graph([("A", "a", 0)])
    r = SemanticRetriever(g, MatrixEmbedder({}, dim=2))
    with pytest.raises(KeyError):
        r.score("q", ["nope"])


# ==========================================================================
# Signals in isolation
# ==========================================================================

def test_semantic_signal_clipped_to_unit_interval():
    g = _graph([("P", "pos", 0), ("N", "neg", 1)])
    embedder = MatrixEmbedder({"pos": [1.0, 0.0], "neg": [-1.0, 0.0], "q": [1.0, 0.0]}, dim=2)
    ranker = _ranker(g, embedder=embedder, config=_only(semantic=1.0))
    ranked = {r.node_id: r for r in ranker.rank("q", [_cand("P"), _cand("N")])}
    assert ranked["P"].signals["semantic"] == pytest.approx(1.0)
    assert ranked["N"].signals["semantic"] == pytest.approx(0.0)   # -1 clipped to 0


def test_relational_signal_divided_by_max():
    g = _graph([("A", "a", 0), ("B", "b", 1)])
    ranker = _ranker(g, config=_only(relational=1.0))
    ranked = {r.node_id: r for r in ranker.rank("q", [_cand("A", 2.0), _cand("B", 1.0)])}
    assert ranked["A"].signals["relational"] == pytest.approx(1.0)
    assert ranked["B"].signals["relational"] == pytest.approx(0.5)


def test_recency_signal_divided_by_max():
    g = _graph([("A", "a", 4), ("B", "b", 2)])
    ranker = _ranker(g, config=_only(recency=1.0))
    ranked = {r.node_id: r for r in ranker.rank("q", [_cand("A"), _cand("B")])}
    assert ranked["A"].signals["recency"] == pytest.approx(1.0)
    assert ranked["B"].signals["recency"] == pytest.approx(0.5)


def test_seed_signal_is_raw():
    g = _graph([("A", "a", 0), ("B", "b", 1)])
    ranker = _ranker(g, config=_only(seed=1.0))
    ranked = {r.node_id: r for r in ranker.rank("q", [_cand("A", is_seed=True), _cand("B")])}
    assert ranked["A"].signals["seed"] == 1.0
    assert ranked["B"].signals["seed"] == 0.0
    assert [r.node_id for r in ranker.rank("q", [_cand("A", is_seed=True), _cand("B")])][0] == "A"


def test_entity_overlap_signal():
    g = _graph([("M", "mentions things", 0), ("N", "nothing", 1)])
    g.add_node(Node("entity:cache_key", "cache_key", "Entity", 1.0))
    g.add_edge(Edge("M", "entity:cache_key", "MENTIONS", weight=0.5))
    ranker = _ranker(g, config=_only(entity_overlap=1.0))
    ranked = {r.node_id: r for r in ranker.rank("look at the cache_key design", [_cand("M"), _cand("N")])}
    assert ranked["M"].signals["entity_overlap"] == pytest.approx(1.0)
    assert ranked["N"].signals["entity_overlap"] == pytest.approx(0.0)


def test_entity_overlap_zero_when_query_has_no_entities():
    g = _graph([("M", "m", 0), ("N", "n", 1)])
    g.add_node(Node("entity:cache_key", "cache_key", "Entity", 1.0))
    g.add_edge(Edge("M", "entity:cache_key", "MENTIONS", weight=0.5))
    ranker = _ranker(g, config=_only(entity_overlap=1.0))
    ranked = ranker.rank("nothing notable happened here", [_cand("M"), _cand("N")])
    assert all(r.signals["entity_overlap"] == 0.0 for r in ranked)


# ==========================================================================
# Normalization edge cases
# ==========================================================================

def test_relational_all_equal_not_zeroed():
    g = _graph([("A", "a", 0), ("B", "b", 0)])
    ranker = _ranker(g, config=_only(relational=1.0))
    ranked = {r.node_id: r for r in ranker.rank("q", [_cand("A", 1.0), _cand("B", 1.0)])}
    assert ranked["A"].signals["relational"] == pytest.approx(1.0)
    assert ranked["B"].signals["relational"] == pytest.approx(1.0)


def test_relational_all_zero_stays_zero():
    g = _graph([("A", "a", 0), ("B", "b", 0)])
    ranker = _ranker(g, config=_only(relational=1.0))
    ranked = ranker.rank("q", [_cand("A", 0.0), _cand("B", 0.0)])
    assert all(r.signals["relational"] == 0.0 for r in ranked)


def test_single_candidate_is_not_zeroed():
    g = _graph([("A", "a", 3)])
    ranker = _ranker(g, config=_only(relational=1.0, recency=1.0))
    ranked = ranker.rank("q", [_cand("A", 0.5)])
    assert len(ranked) == 1
    assert ranked[0].signals["relational"] == pytest.approx(1.0)   # 0.5 / max(0.5)
    assert ranked[0].signals["recency"] == pytest.approx(1.0)      # 3 / max(3)
    assert ranked[0].score == pytest.approx(2.0)


# ==========================================================================
# Weights change the order
# ==========================================================================

def test_weights_change_order():
    g = _graph([("A", "a", 0), ("B", "b", 0)])
    embedder = MatrixEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0], "q": [1.0, 0.0]}, dim=2)
    cands = [_cand("A", 1.0), _cand("B", 10.0)]   # A wins semantic, B wins relational

    sem = _ranker(g, embedder=embedder, config=_only(semantic=1.0))
    rel = _ranker(g, embedder=embedder, config=_only(relational=1.0))
    assert [r.node_id for r in sem.rank("q", cands)][0] == "A"
    assert [r.node_id for r in rel.rank("q", cands)][0] == "B"


# ==========================================================================
# Ordering, defaults, immutability
# ==========================================================================

def test_deterministic_ordering_on_ties():
    g = _graph([("A", "a", 0), ("B", "b", 0)])
    ranker = _ranker(g, config=_only(seed=1.0))   # both non-seed -> both 0
    first = [r.node_id for r in ranker.rank("q", [_cand("B"), _cand("A")])]
    second = [r.node_id for r in ranker.rank("q", [_cand("B"), _cand("A")])]
    assert first == ["A", "B"]
    assert first == second


def test_default_weights():
    cfg = RankingConfig()
    assert (cfg.semantic, cfg.relational, cfg.entity_overlap, cfg.recency, cfg.seed) == (
        1.0, 1.0, 0.5, 0.2, 0.0
    )


def test_empty_candidates():
    g = _graph([("A", "a", 0)])
    assert _ranker(g).rank("q", []) == []


def test_ranked_candidate_keeps_path_and_signals():
    g = _graph([("A", "a", 0)])
    c = Candidate("A", 1.0, ["seed", "A"], ["MENTIONS"], False, None)
    ranked = _ranker(g).rank("q", [c])
    assert isinstance(ranked[0], RankedCandidate)
    assert ranked[0].path == ["seed", "A"]
    assert ranked[0].path_edge_types == ["MENTIONS"]
    assert set(ranked[0].signals) == {"semantic", "relational", "entity_overlap", "recency", "seed"}


def test_query_is_not_mutated_and_extractor_receives_it_unchanged():
    g = _graph([("A", "a", 0)])
    embedder = MatrixEmbedder({"a": [1.0, 0.0]}, dim=2)
    recorder = RecordingExtractor()
    ranker = HybridRanker(SemanticRetriever(g, embedder), recorder, RankingConfig())

    query = "Why did we choose the tenant_cache layout?"
    before = str(query)
    ranker.rank(query, [_cand("A")])

    assert query == before
    assert recorder.seen_contents == [before]
    assert embedder.seen[0] == [before]   # query embedded unchanged (scored first)
