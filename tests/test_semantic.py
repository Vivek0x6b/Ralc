"""Tests for Phase 2 semantic seed retrieval.

A deterministic fake embedder is used throughout so the suite never downloads
a model. One integration test exercises the real SentenceTransformerEmbedder
and is skipped automatically when sentence-transformers is not installed.
"""

import importlib.util

import numpy as np
import pytest

from ralc.embeddings.local import SentenceTransformerEmbedder  # noqa: F401 (import smoke)
from ralc.graph import ContextGraph, Node
from ralc.retrieval.semantic import SemanticRetriever


# --------------------------------------------------------------------------
# Deterministic fake embedder: token counts over a fixed 4-token vocabulary
# --------------------------------------------------------------------------

class FakeEmbedder:
    VOCAB = ["a", "b", "c", "d"]

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.embed_calls: list[list[str]] = []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        rows = []
        for text in texts:
            toks = text.split()
            vec = np.array([float(toks.count(v)) for v in self.VOCAB], dtype=np.float64)
            if self.normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            rows.append(vec)
        if not rows:
            return np.zeros((0, len(self.VOCAB)))
        return np.vstack(rows)


def _node(nid, content, type="Message"):
    return Node(nid, content, type, 1.0)


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

def test_ranking_order_and_descending_scores():
    g = ContextGraph()
    g.add_node(_node("na", "a a"))   # -> (1,0,0,0)
    g.add_node(_node("nb", "b"))     # -> (0,1,0,0)
    g.add_node(_node("nc", "a b"))   # -> (.707,.707,0,0)
    r = SemanticRetriever(g, FakeEmbedder())
    r.index()

    res = r.retrieve("a", k=3)
    assert [nid for nid, _ in res] == ["na", "nc", "nb"]
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)
    assert res[0][1] == pytest.approx(1.0)
    assert res[1][1] == pytest.approx(1 / np.sqrt(2))
    assert res[2][1] == pytest.approx(0.0)


def test_k_larger_than_node_count():
    g = ContextGraph()
    g.add_node(_node("na", "a"))
    g.add_node(_node("nb", "b"))
    r = SemanticRetriever(g, FakeEmbedder())
    r.index()
    res = r.retrieve("a", k=100)
    assert len(res) == 2


def test_k_zero_or_negative_returns_empty():
    g = ContextGraph()
    g.add_node(_node("na", "a"))
    r = SemanticRetriever(g, FakeEmbedder())
    r.index()
    assert r.retrieve("a", k=0) == []
    assert r.retrieve("a", k=-1) == []


# --------------------------------------------------------------------------
# Empty states
# --------------------------------------------------------------------------

def test_empty_graph_returns_empty():
    g = ContextGraph()
    r = SemanticRetriever(g, FakeEmbedder())
    r.index()
    assert r.retrieve("a", k=5) == []


def test_retrieve_before_index_returns_empty():
    g = ContextGraph()
    g.add_node(_node("na", "a"))
    r = SemanticRetriever(g, FakeEmbedder())
    assert r.retrieve("a", k=5) == []


# --------------------------------------------------------------------------
# Node type filter
# --------------------------------------------------------------------------

def test_node_type_filter():
    g = ContextGraph()
    g.add_node(_node("msg", "a", type="Message"))
    g.add_node(_node("fact", "a", type="Fact"))
    r = SemanticRetriever(g, FakeEmbedder())
    r.index(type="Message")
    res = r.retrieve("a", k=5)
    assert [nid for nid, _ in res] == ["msg"]


# --------------------------------------------------------------------------
# Incremental indexing
# --------------------------------------------------------------------------

def test_incremental_indexing_without_full_reembed():
    g = ContextGraph()
    g.add_node(_node("m1", "a"))
    g.add_node(_node("m2", "b"))
    embedder = FakeEmbedder()
    r = SemanticRetriever(g, embedder)
    r.index()
    assert "m3" not in {nid for nid, _ in r.retrieve("a", k=5)}

    g.add_node(_node("m3", "a a a"))
    r.index_node("m3")

    # the incremental call embedded exactly one text, not the whole graph
    assert embedder.embed_calls[-1] == ["a a a"]
    assert "m3" in {nid for nid, _ in r.retrieve("a", k=5)}


def test_index_node_missing_raises():
    g = ContextGraph()
    r = SemanticRetriever(g, FakeEmbedder())
    with pytest.raises(KeyError):
        r.index_node("nope")


# --------------------------------------------------------------------------
# Query immutability (required from the start)
# --------------------------------------------------------------------------

def test_query_is_not_mutated():
    g = ContextGraph()
    g.add_node(_node("na", "a"))
    embedder = FakeEmbedder()
    r = SemanticRetriever(g, embedder)
    r.index()

    query = "Why did we choose PostgreSQL?"
    before = str(query)
    r.retrieve(query, k=2)

    assert query == before
    # the embedder received the query unchanged
    assert embedder.embed_calls[-1] == [before]


# --------------------------------------------------------------------------
# Internal normalization (does not trust the embedder's magnitudes)
# --------------------------------------------------------------------------

def test_unnormalized_embedder_still_ranks_by_cosine():
    g = ContextGraph()
    g.add_node(_node("aligned", "a"))        # raw (1,0,0,0)
    g.add_node(_node("big", "a a b b"))      # raw (2,2,0,0)
    r = SemanticRetriever(g, FakeEmbedder(normalize=False))
    r.index()

    res = r.retrieve("a", k=2)
    # unnormalized dot would put "big" (dot=2) above "aligned" (dot=1);
    # internal normalization gives the correct cosine ranking instead.
    assert res[0][0] == "aligned"
    assert res[0][1] == pytest.approx(1.0)
    assert res[1][1] == pytest.approx(2 / np.sqrt(8))
    assert res[0][1] > res[1][1]


def test_empty_content_node_scores_zero_without_crashing():
    g = ContextGraph()
    g.add_node(_node("good", "a"))
    g.add_node(_node("empty", ""))   # zero vector
    r = SemanticRetriever(g, FakeEmbedder())
    r.index()

    res = dict(r.retrieve("a", k=5))
    assert res["empty"] == pytest.approx(0.0)
    # an empty query must not crash either
    res2 = r.retrieve("", k=5)
    assert len(res2) == 2
    assert all(s == pytest.approx(0.0) for _, s in res2)


# --------------------------------------------------------------------------
# Integration with the real embedder (skipped if not installed)
# --------------------------------------------------------------------------

_ST_INSTALLED = importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(not _ST_INSTALLED, reason="sentence-transformers not installed")
def test_integration_real_sentence_transformer():
    g = ContextGraph()
    g.add_node(_node("cat", "The cat sat quietly on the warm mat."))
    g.add_node(_node("finance", "Quarterly revenue exceeded every analyst expectation."))
    r = SemanticRetriever(g, SentenceTransformerEmbedder())
    r.index()

    res = r.retrieve("How much money did the company make last quarter?", k=2)
    assert res[0][0] == "finance"
    assert res[0][1] >= res[1][1]
    assert all(-1.0001 <= s <= 1.0001 for _, s in res)
