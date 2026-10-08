"""Tests for the ContextManager facade (Phases 1 to 5 wired together)."""

import numpy as np

from ralc import ContextManager
from ralc.allocation import SelectionConfig


class DictEmbedder:
    """One-hot embedder keyed by exact text; unknown text -> zero vector."""

    def __init__(self, mapping: dict[str, int], dim: int):
        self.mapping = mapping
        self.dim = dim

    def embed(self, texts):
        rows = []
        for text in texts:
            vec = np.zeros(self.dim)
            if text in self.mapping:
                vec[self.mapping[text]] = 1.0
            rows.append(vec)
        return np.vstack(rows) if rows else np.zeros((0, self.dim))


class WordCounter:
    approximate = False

    def count(self, text: str) -> int:
        return len(text.split())


def _manager(embedder, **kwargs):
    return ContextManager(embedder=embedder, token_counter=WordCounter(), **kwargs)


# ==========================================================================
# End to end
# ==========================================================================

def test_add_message_then_retrieve_returns_relevant_within_budget():
    m = {
        "Why did we choose PostgreSQL?": 0,
        "We decided to use PostgreSQL for storage.": 0,
        "PostgreSQL gives transactional consistency.": 1,
        "The weather is nice today.": 2,
    }
    cm = _manager(DictEmbedder(m, dim=3))
    ids = [
        cm.add_message("user", "We decided to use PostgreSQL for storage."),
        cm.add_message("assistant", "PostgreSQL gives transactional consistency."),
        cm.add_message("user", "The weather is nice today."),
    ]
    m0, m1, m2 = ids

    result = cm.retrieve("Why did we choose PostgreSQL?", token_budget=11, seed_k=1)
    selected = {n.id for n in result.nodes}
    assert m0 in selected and m1 in selected   # the PostgreSQL-relevant turns
    assert m2 not in selected                   # the weather turn dropped for budget
    assert result.token_count <= 11


def test_empty_history_retrieve_returns_empty_result():
    cm = _manager(DictEmbedder({}, dim=2))
    result = cm.retrieve("anything", token_budget=100)
    assert result.nodes == []
    assert result.token_count == 0
    assert result.to_text() == ""
    assert result.metadata["semantic_seeds"] == 0
    assert result.metadata["nodes_explored"] == 0
    assert result.metadata["candidates_ranked"] == 0
    assert result.metadata["final_nodes"] == 0
    assert result.metadata["relationships_used"] == 0


# ==========================================================================
# Persistence
# ==========================================================================

def test_save_and_reload_continues_conversation(tmp_path):
    path = str(tmp_path / "graph.db")
    embedder = DictEmbedder({"q": 0, "alpha one": 0, "beta two": 1, "gamma three": 2}, dim=3)

    cm1 = _manager(embedder, storage=path)
    assert cm1.add_message("user", "alpha one") == "m0"
    assert cm1.add_message("assistant", "beta two") == "m1"
    cm1.save()

    cm2 = _manager(embedder, storage=path)
    assert cm2.add_message("user", "gamma three") == "m2"   # seq continues from the reloaded graph
    assert len(cm2.graph.nodes(type="Message")) == 3

    result = cm2.retrieve("q", token_budget=100, seed_k=1)
    assert len(result.nodes) >= 1


# ==========================================================================
# Section 34 invariant: query immutability and query absence from output
# ==========================================================================

def test_query_unchanged_and_absent_from_to_text():
    embedder = DictEmbedder({"why was gamma chosen": 0, "alpha": 0, "beta": 1}, dim=2)
    cm = _manager(embedder)
    cm.add_message("user", "alpha")
    cm.add_message("user", "beta")

    query = "why was gamma chosen"
    before = str(query)
    result = cm.retrieve(query, token_budget=100, seed_k=1)

    assert query == before                       # original_query == query_after_ralc
    text = result.to_text()
    assert query not in text
    assert "gamma" not in text                   # no query token leaked into the context
    assert "alpha" in text                        # real message content is present


# ==========================================================================
# Pipeline stats
# ==========================================================================

def test_metadata_pipeline_stats_match_reality():
    embedder = DictEmbedder({"q": 0, "alpha": 0, "beta": 1}, dim=2)
    cm = _manager(embedder)
    cm.add_message("user", "alpha")      # m0
    cm.add_message("user", "beta")       # m1 (TEMPORALLY_FOLLOWS m0)

    result = cm.retrieve("q", token_budget=100, seed_k=1)
    md = result.metadata
    assert md["semantic_seeds"] == 1
    assert md["nodes_explored"] == 2             # m0 and m1 touched during expansion
    assert md["candidates_ranked"] == 2
    assert md["final_nodes"] == 2
    assert md["relationships_used"] == 1         # the single temporal edge between them
    assert md["final_nodes"] == len(result.nodes)
    assert md["relationships_used"] == len(result.relationships)


# ==========================================================================
# PPR strategy through the facade
# ==========================================================================

def test_ppr_strategy_through_facade():
    embedder = DictEmbedder({"q": 0, "alpha": 0, "beta": 1}, dim=2)
    cm = _manager(embedder, expansion_strategy="ppr")
    cm.add_message("user", "alpha")
    cm.add_message("user", "beta")

    result = cm.retrieve("q", token_budget=100, seed_k=1)
    assert len(result.nodes) >= 1
    assert result.metadata["nodes_explored"] >= 1
    assert result.token_count <= 100
