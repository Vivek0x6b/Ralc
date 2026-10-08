"""Tests for Phase 3 relational expansion (hop_decay and ppr)."""

import numpy as np
import pytest

from ralc.extraction import HeuristicExtractor, Message, MessageLinker
from ralc.graph import ContextGraph, Edge, Node
from ralc.retrieval.relational import (
    Candidate,
    ExpansionConfig,
    RelationalExpander,
)
from ralc.retrieval.semantic import SemanticRetriever


def _msg(nid):
    return Node(nid, nid, "Message", 1.0)


def _entity(nid):
    return Node(nid, nid, "Entity", 1.0)


def _by_id(candidates):
    return {c.node_id: c for c in candidates}


# ==========================================================================
# hop_decay: decay, hop limit, best path
# ==========================================================================

def test_hop_decay_scores_and_hop_limit():
    g = ContextGraph()
    for nid in ("s", "a", "b"):
        g.add_node(_msg(nid))
    g.add_edge(Edge("s", "a", "REL", weight=1.0))
    g.add_edge(Edge("a", "b", "REL", weight=1.0))

    cfg = ExpansionConfig(max_hops=2, decay=0.5, hub_penalty=False)
    out = _by_id(RelationalExpander(g, "hop_decay", cfg).expand([("s", 1.0)]))
    assert out["s"].score == pytest.approx(1.0)
    assert out["a"].score == pytest.approx(0.5)    # seed * weight * decay
    assert out["b"].score == pytest.approx(0.25)   # * weight * decay again

    cfg1 = ExpansionConfig(max_hops=1, decay=0.5, hub_penalty=False)
    out1 = _by_id(RelationalExpander(g, "hop_decay", cfg1).expand([("s", 1.0)]))
    assert "a" in out1
    assert "b" not in out1


def test_hop_decay_best_path_recording():
    g = ContextGraph()
    for nid in ("s", "a", "b"):
        g.add_node(_msg(nid))
    g.add_edge(Edge("s", "a", "REL", weight=1.0))
    g.add_edge(Edge("a", "b", "REL", weight=1.0))

    cfg = ExpansionConfig(max_hops=2, decay=0.5, hub_penalty=False)
    out = _by_id(RelationalExpander(g, "hop_decay", cfg).expand([("s", 1.0)]))
    assert out["s"].path == ["s"] and out["s"].path_edge_types == []
    assert out["a"].path == ["s", "a"] and out["a"].path_edge_types == ["REL"]
    assert out["b"].path == ["s", "a", "b"]
    assert out["b"].path_edge_types == ["REL", "REL"]


def test_hop_decay_edge_type_filter():
    g = ContextGraph()
    for nid in ("s", "x", "y"):
        g.add_node(_msg(nid))
    g.add_edge(Edge("s", "x", "A", weight=1.0))
    g.add_edge(Edge("s", "y", "B", weight=1.0))

    cfg = ExpansionConfig(edge_types=["A"], hub_penalty=False)
    ids = {c.node_id for c in RelationalExpander(g, "hop_decay", cfg).expand([("s", 1.0)])}
    assert "x" in ids
    assert "y" not in ids


# ==========================================================================
# hop_decay: entities are bridges (one hop, no decay on arrival)
# ==========================================================================

def test_entity_bridge_is_one_hop():
    g = ContextGraph()
    g.add_node(_msg("s"))
    g.add_node(_msg("c"))
    g.add_node(_entity("e"))
    g.add_edge(Edge("s", "e", "MENTIONS", weight=0.5))
    g.add_edge(Edge("c", "e", "MENTIONS", weight=0.5))

    cfg = ExpansionConfig(max_hops=1, decay=0.5, hub_penalty=False, edge_types=["MENTIONS"])
    out = _by_id(RelationalExpander(g, "hop_decay", cfg).expand([("s", 1.0)]))
    # c is reachable within a single hop because the entity bridge is free of a hop
    assert "c" in out
    assert out["c"].path == ["s", "e", "c"]
    assert out["c"].path_edge_types == ["MENTIONS", "MENTIONS"]


def test_entity_connection_at_least_as_high_as_single_temporal():
    # Hub penalty disabled to isolate the hop/decay structure (penalty is
    # covered separately). With equal edge weights, an entity-bridged message
    # must not score below a message one TEMPORALLY_FOLLOWS edge away.
    g = ContextGraph()
    for nid in ("s", "b", "c"):
        g.add_node(_msg(nid))
    g.add_node(_entity("e"))
    g.add_edge(Edge("s", "b", "TEMPORALLY_FOLLOWS", weight=0.5))
    g.add_edge(Edge("s", "e", "MENTIONS", weight=0.5))
    g.add_edge(Edge("c", "e", "MENTIONS", weight=0.5))

    cfg = ExpansionConfig(max_hops=2, decay=0.5, hub_penalty=False)  # all edge types
    out = _by_id(RelationalExpander(g, "hop_decay", cfg).expand([("s", 1.0)]))
    assert out["c"].score >= out["b"].score
    assert out["c"].score == pytest.approx(out["b"].score)


# ==========================================================================
# hop_decay: hub penalty
# ==========================================================================

def test_hub_penalty_lowers_score_through_high_degree_entity():
    g = ContextGraph()
    for nid in ("s", "t1", "t2"):
        g.add_node(_msg(nid))
    g.add_node(_entity("e_low"))
    g.add_node(_entity("e_high"))
    g.add_edge(Edge("s", "e_low", "MENTIONS", weight=0.5))
    g.add_edge(Edge("t1", "e_low", "MENTIONS", weight=0.5))     # e_low degree 2
    g.add_edge(Edge("s", "e_high", "MENTIONS", weight=0.5))
    g.add_edge(Edge("t2", "e_high", "MENTIONS", weight=0.5))
    for i in range(8):                                          # inflate e_high degree to 10
        fid = f"f{i}"
        g.add_node(_msg(fid))
        g.add_edge(Edge(fid, "e_high", "MENTIONS", weight=0.5))

    cfg_on = ExpansionConfig(max_hops=1, decay=0.5, hub_penalty=True, edge_types=["MENTIONS"])
    on = _by_id(RelationalExpander(g, "hop_decay", cfg_on).expand([("s", 1.0)]))
    assert on["t1"].score > on["t2"].score

    cfg_off = ExpansionConfig(max_hops=1, decay=0.5, hub_penalty=False, edge_types=["MENTIONS"])
    off = _by_id(RelationalExpander(g, "hop_decay", cfg_off).expand([("s", 1.0)]))
    assert off["t1"].score == pytest.approx(off["t2"].score)


# ==========================================================================
# return type filter, ordering, cap, seed flags
# ==========================================================================

def test_return_type_filter_excludes_entities_by_default():
    g = ContextGraph()
    g.add_node(_msg("s"))
    g.add_node(_msg("c"))
    g.add_node(_entity("e"))
    g.add_edge(Edge("s", "e", "MENTIONS", weight=0.5))
    g.add_edge(Edge("c", "e", "MENTIONS", weight=0.5))

    cfg = ExpansionConfig(max_hops=2, hub_penalty=False, edge_types=["MENTIONS"])
    default_ids = {c.node_id for c in RelationalExpander(g, "hop_decay", cfg).expand([("s", 1.0)])}
    assert "e" not in default_ids
    assert {"s", "c"} <= default_ids

    cfg2 = ExpansionConfig(max_hops=2, hub_penalty=False, edge_types=["MENTIONS"],
                           return_types=["Message", "Entity"])
    with_entities = {c.node_id for c in RelationalExpander(g, "hop_decay", cfg2).expand([("s", 1.0)])}
    assert "e" in with_entities


def test_deterministic_ordering():
    g = ContextGraph()
    for nid in ("s", "a", "b"):
        g.add_node(_msg(nid))
    g.add_edge(Edge("s", "a", "REL", weight=1.0))
    g.add_edge(Edge("s", "b", "REL", weight=1.0))   # a and b tie on score

    cfg = ExpansionConfig(max_hops=1, decay=0.5, hub_penalty=False)
    exp = RelationalExpander(g, "hop_decay", cfg)
    first = [c.node_id for c in exp.expand([("s", 1.0)])]
    second = [c.node_id for c in exp.expand([("s", 1.0)])]
    assert first == ["s", "a", "b"]   # score desc, then id asc
    assert first == second


def test_max_candidates_cap_keeps_seed():
    g = ContextGraph()
    for nid in ("s", "a", "b", "c"):
        g.add_node(_msg(nid))
    g.add_edge(Edge("s", "a", "REL", weight=1.0))
    g.add_edge(Edge("a", "b", "REL", weight=1.0))
    g.add_edge(Edge("b", "c", "REL", weight=1.0))

    cfg = ExpansionConfig(max_hops=3, decay=0.5, hub_penalty=False, max_candidates=2)
    out = RelationalExpander(g, "hop_decay", cfg).expand([("s", 1.0)])
    ids = [c.node_id for c in out]
    assert ids == ["s", "a"]            # seed plus the single highest expansion


def test_seed_flags_set():
    g = ContextGraph()
    g.add_node(_msg("s"))
    g.add_node(_msg("a"))
    g.add_edge(Edge("s", "a", "REL", weight=1.0))
    cfg = ExpansionConfig(hub_penalty=False)
    out = _by_id(RelationalExpander(g, "hop_decay", cfg).expand([("s", 0.8)]))
    assert out["s"].is_seed is True and out["s"].seed_score == 0.8
    assert out["a"].is_seed is False and out["a"].seed_score is None


def test_empty_seeds_and_missing_seed():
    g = ContextGraph()
    g.add_node(_msg("s"))
    exp = RelationalExpander(g, "hop_decay", ExpansionConfig())
    assert exp.expand([]) == []
    with pytest.raises(KeyError):
        exp.expand([("nope", 1.0)])


# ==========================================================================
# ppr
# ==========================================================================

def test_ppr_returns_connected_and_excludes_disconnected():
    g = ContextGraph()
    for nid in ("a", "b", "c", "z"):
        g.add_node(_msg(nid))
    g.add_edge(Edge("a", "b", "REL", weight=1.0))
    g.add_edge(Edge("b", "c", "REL", weight=1.0))
    # z is isolated

    out = RelationalExpander(g, "ppr", ExpansionConfig()).expand([("a", 1.0)])
    ids = {c.node_id for c in out}
    assert {"a", "b", "c"} <= ids
    assert "z" not in ids
    assert max(c.score for c in out) == pytest.approx(1.0)   # normalized
    seed = _by_id(out)["a"]
    assert seed.is_seed is True and seed.seed_score == 1.0


def test_ppr_is_bidirectional():
    g = ContextGraph()
    g.add_node(_msg("a"))
    g.add_node(_msg("b"))
    g.add_edge(Edge("a", "b", "REL", weight=1.0))   # directed a -> b only
    out = {c.node_id for c in RelationalExpander(g, "ppr", ExpansionConfig()).expand([("a", 1.0)])}
    assert "b" in out   # reached despite edge direction


def test_invalid_strategy_raises():
    g = ContextGraph()
    with pytest.raises(ValueError):
        RelationalExpander(g, "nonsense", ExpansionConfig())


# ==========================================================================
# End to end: linker -> semantic seeds -> relational find through an entity
# ==========================================================================

class DictEmbedder:
    """Maps exact strings to one-hot basis vectors; unknown text -> zero."""

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


def test_end_to_end_relational_find_through_entity():
    convo = [
        Message("user", "Isolation uses the tenant_cache layer."),
        Message("assistant", "Understood and noted."),
        Message("user", "The tenant_cache was added last week."),
    ]
    g = ContextGraph()
    ids = MessageLinker(HeuristicExtractor()).ingest(g, convo)
    m0, m1, m2 = ids

    query = "How does multi-tenant isolation work?"
    embedder = DictEmbedder(
        {
            query: 0,
            "Isolation uses the tenant_cache layer.": 0,   # same basis -> cosine 1 with query
            "The tenant_cache was added last week.": 1,    # orthogonal: zero overlap with query
            "Understood and noted.": 2,
        },
        dim=3,
    )
    retriever = SemanticRetriever(g, embedder)
    retriever.index(type="Message")
    seeds = retriever.retrieve(query, k=1)
    assert [s[0] for s in seeds] == [m0]

    cfg = ExpansionConfig(max_hops=2, edge_types=["MENTIONS"])   # only entity bridges
    out = _by_id(RelationalExpander(g, "hop_decay", cfg).expand(seeds))

    # m2 shares no words with the query but is reached through the shared entity
    assert m2 in out
    assert out[m2].is_seed is False
    assert out[m2].path == [m0, "entity:tenant_cache", m2]
    assert out[m2].path_edge_types == ["MENTIONS", "MENTIONS"]
    assert out[m2].score > 0.0
    # the filler assistant message has no entity, so it is not pulled in
    assert m1 not in out


def test_hop_decay_records_nodes_visited_including_bridges():
    g = ContextGraph()
    g.add_node(_msg("s"))
    g.add_node(_msg("c"))
    g.add_node(_entity("e"))
    g.add_edge(Edge("s", "e", "MENTIONS", weight=0.5))
    g.add_edge(Edge("c", "e", "MENTIONS", weight=0.5))
    cfg = ExpansionConfig(max_hops=1, hub_penalty=False, edge_types=["MENTIONS"])
    exp = RelationalExpander(g, "hop_decay", cfg)
    exp.expand([("s", 1.0)])
    assert exp.last_stats["nodes_visited"] == 3   # s, e (bridge), c


def test_ppr_records_nodes_visited():
    g = ContextGraph()
    for nid in ("a", "b", "z"):
        g.add_node(_msg(nid))
    g.add_edge(Edge("a", "b", "REL", weight=1.0))   # z disconnected
    exp = RelationalExpander(g, "ppr", ExpansionConfig())
    exp.expand([("a", 1.0)])
    assert exp.last_stats["nodes_visited"] == 2     # a, b; z excluded


def test_empty_seeds_record_zero_visited():
    g = ContextGraph()
    g.add_node(_msg("s"))
    exp = RelationalExpander(g, "hop_decay", ExpansionConfig())
    exp.expand([])
    assert exp.last_stats["nodes_visited"] == 0


def test_candidate_is_dataclass_shape():
    c = Candidate(node_id="x", score=1.0, path=["x"], path_edge_types=[], is_seed=True, seed_score=1.0)
    assert (c.node_id, c.score, c.is_seed, c.seed_score) == ("x", 1.0, True, 1.0)
