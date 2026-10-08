"""Tests for the Phase 1 graph layer: Node, Edge, ContextGraph."""

import pytest

from ralc.graph import ContextGraph, Edge, Node


# --------------------------------------------------------------------------
# Node / Edge dataclasses
# --------------------------------------------------------------------------

def test_node_creation():
    n = Node(id="n1", content="hello", type="Message", timestamp=1000.0)
    assert n.id == "n1"
    assert n.content == "hello"
    assert n.type == "Message"
    assert n.timestamp == 1000.0
    assert n.metadata == {}
    assert n.embedding_ref is None


def test_node_with_metadata_and_embedding_ref():
    n = Node("n2", "c", "Fact", 1.0, {"k": "v"}, "emb-7")
    assert n.metadata == {"k": "v"}
    assert n.embedding_ref == "emb-7"


def test_edge_creation_defaults():
    e = Edge(source="a", target="b", type="RELATED_TO")
    assert e.source == "a"
    assert e.target == "b"
    assert e.type == "RELATED_TO"
    assert e.weight == 1.0
    assert e.metadata == {}


def test_edge_creation_explicit():
    e = Edge("a", "b", "DEPENDS_ON", weight=0.91, metadata={"x": 1}, created_at=5.0)
    assert e.weight == 0.91
    assert e.metadata == {"x": 1}
    assert e.created_at == 5.0


# --------------------------------------------------------------------------
# add_node / add_edge
# --------------------------------------------------------------------------

def test_add_and_get_node():
    g = ContextGraph()
    n = Node("n1", "hello", "Message", 1.0)
    g.add_node(n)
    assert g.get("n1") == n


def test_get_missing_node_raises():
    g = ContextGraph()
    with pytest.raises(KeyError):
        g.get("nope")


def test_add_duplicate_node_id_raises():
    g = ContextGraph()
    g.add_node(Node("n1", "a", "Message", 1.0))
    with pytest.raises(ValueError):
        g.add_node(Node("n1", "b", "Fact", 2.0))


def test_add_edge_missing_endpoint_raises():
    g = ContextGraph()
    g.add_node(Node("a", "a", "Message", 1.0))
    # target "b" does not exist
    with pytest.raises(ValueError):
        g.add_edge(Edge("a", "b", "RELATED_TO"))
    # source "c" does not exist
    with pytest.raises(ValueError):
        g.add_edge(Edge("c", "a", "RELATED_TO"))


def test_duplicate_edge_type_same_pair_raises():
    g = ContextGraph()
    g.add_node(Node("a", "a", "Message", 1.0))
    g.add_node(Node("b", "b", "Message", 1.0))
    g.add_edge(Edge("a", "b", "RELATED_TO"))
    with pytest.raises(ValueError):
        g.add_edge(Edge("a", "b", "RELATED_TO"))


def test_different_type_same_pair_allowed():
    g = ContextGraph()
    g.add_node(Node("a", "a", "Message", 1.0))
    g.add_node(Node("b", "b", "Message", 1.0))
    g.add_edge(Edge("a", "b", "RELATED_TO"))
    g.add_edge(Edge("a", "b", "MENTIONS"))
    assert len(g.edges()) == 2


def test_same_type_reverse_direction_allowed():
    g = ContextGraph()
    g.add_node(Node("a", "a", "Message", 1.0))
    g.add_node(Node("b", "b", "Message", 1.0))
    g.add_edge(Edge("a", "b", "RELATED_TO"))
    g.add_edge(Edge("b", "a", "RELATED_TO"))
    assert len(g.edges()) == 2


# --------------------------------------------------------------------------
# nodes() / edges() filtering
# --------------------------------------------------------------------------

def test_nodes_filter_by_type():
    g = ContextGraph()
    g.add_node(Node("a", "a", "Message", 1.0))
    g.add_node(Node("b", "b", "Fact", 1.0))
    g.add_node(Node("c", "c", "Message", 1.0))
    assert {n.id for n in g.nodes()} == {"a", "b", "c"}
    assert {n.id for n in g.nodes(type="Message")} == {"a", "c"}


def test_edges_filter_by_type():
    g = ContextGraph()
    g.add_node(Node("a", "a", "Message", 1.0))
    g.add_node(Node("b", "b", "Message", 1.0))
    g.add_edge(Edge("a", "b", "RELATED_TO"))
    g.add_edge(Edge("a", "b", "MENTIONS"))
    assert len(g.edges()) == 2
    assert len(g.edges(type="MENTIONS")) == 1


# --------------------------------------------------------------------------
# neighbors (both directions, one (id, Edge) pair per matching edge)
# --------------------------------------------------------------------------

def _diamond():
    g = ContextGraph()
    for nid in ("a", "b", "c"):
        g.add_node(Node(nid, nid, "Message", 1.0))
    e_ab1 = Edge("a", "b", "RELATED_TO")
    e_ab2 = Edge("a", "b", "MENTIONS")
    e_ca = Edge("c", "a", "DEPENDS_ON")
    g.add_edge(e_ab1)
    g.add_edge(e_ab2)
    g.add_edge(e_ca)
    return g, e_ab1, e_ab2, e_ca


def test_neighbors_both_directions_no_dedup():
    g, e_ab1, e_ab2, e_ca = _diamond()
    pairs = g.neighbors("a")
    # two outgoing edges to b (not deduped) + one incoming edge from c
    assert len(pairs) == 3
    assert ("b", e_ab1) in pairs
    assert ("b", e_ab2) in pairs
    assert ("c", e_ca) in pairs


def test_neighbors_type_filter():
    g, e_ab1, e_ab2, e_ca = _diamond()
    pairs = g.neighbors("a", types=["MENTIONS"])
    assert pairs == [("b", e_ab2)]


def test_neighbors_missing_node_raises():
    g = ContextGraph()
    with pytest.raises(KeyError):
        g.neighbors("nope")


# --------------------------------------------------------------------------
# bfs (bidirectional, hop distances)
# --------------------------------------------------------------------------

def _chain():
    g = ContextGraph()
    for nid in ("a", "b", "c", "d"):
        g.add_node(Node(nid, nid, "Message", 1.0))
    g.add_edge(Edge("a", "b", "NEXT"))
    g.add_edge(Edge("b", "c", "NEXT"))
    g.add_edge(Edge("c", "d", "OTHER"))
    return g


def test_bfs_zero_hops_returns_seeds():
    g = _chain()
    assert g.bfs(["a"], max_hops=0) == {"a": 0}


def test_bfs_hop_limit():
    g = _chain()
    assert g.bfs(["a"], max_hops=1) == {"a": 0, "b": 1}
    assert g.bfs(["a"], max_hops=2) == {"a": 0, "b": 1, "c": 2}


def test_bfs_is_bidirectional():
    g = _chain()
    # b is reached from a (incoming edge a->b) as well as forward
    assert g.bfs(["b"], max_hops=1) == {"b": 0, "a": 1, "c": 1}


def test_bfs_edge_type_filter():
    g = _chain()
    # only NEXT edges traversable: a-b-c reachable, d (behind OTHER) is not
    assert g.bfs(["a"], max_hops=5, edge_types=["NEXT"]) == {"a": 0, "b": 1, "c": 2}


def test_bfs_unknown_seed_raises():
    g = _chain()
    with pytest.raises(KeyError):
        g.bfs(["nope"], max_hops=1)


# --------------------------------------------------------------------------
# SQLite persistence round-trip
# --------------------------------------------------------------------------

def test_save_load_round_trip_lossless(tmp_path):
    g = ContextGraph()
    g.add_node(Node("a", "alpha", "Message", 1000.5, {"tags": ["x", "y"], "n": 3}, "emb-1"))
    g.add_node(Node("b", "beta", "Fact", 2000.25, {}, None))
    g.add_edge(Edge("a", "b", "DEPENDS_ON", weight=0.91, metadata={"why": "test"}, created_at=42.0))
    g.add_edge(Edge("a", "b", "MENTIONS"))

    path = tmp_path / "graph.db"
    g.save(path)
    g2 = ContextGraph.load(path)

    assert sorted(g2.nodes(), key=lambda n: n.id) == sorted(g.nodes(), key=lambda n: n.id)
    assert sorted(g2.edges(), key=lambda e: (e.source, e.target, e.type)) == sorted(
        g.edges(), key=lambda e: (e.source, e.target, e.type)
    )
