"""Tests for the heuristic extractor and the message linker."""

import time

import pytest

from ralc.extraction import (
    DEFAULT_EDGE_WEIGHTS,
    Extractor,
    HeuristicExtractor,
    Message,
    MessageLinker,
    Signals,
)
from ralc.graph import ContextGraph


# ==========================================================================
# HeuristicExtractor: entity patterns
# ==========================================================================

def _entities(content, vocabulary=None):
    ex = HeuristicExtractor(vocabulary=vocabulary)
    return ex.extract(Message("user", content)).entities


def test_extractor_satisfies_protocol():
    assert isinstance(HeuristicExtractor(), Extractor)


def test_backticked_term():
    assert "redis" in _entities("please use `redis` as the store")


def test_file_path():
    ents = _entities("see ralc/graph/nodes.py for the detail")
    assert "ralc/graph/nodes.py" in ents
    # the nested filename is not emitted as a separate entity
    assert "nodes.py" not in ents


def test_filename():
    assert "app.py" in _entities("edit app.py now")


def test_filename_ignores_numeric_tokens():
    assert _entities("upgrade to 3.10 and torch 2.14.1") == []


def test_snake_and_upper_snake():
    ents = _entities("set user_id when TEMPORALLY_FOLLOWS fires")
    assert "user_id" in ents
    assert "TEMPORALLY_FOLLOWS" in ents


def test_camel_and_pascal():
    ents = _entities("call getUser then build a ContextGraph")
    assert "getUser" in ents
    assert "ContextGraph" in ents


def test_plain_capitalized_words_and_acronyms_are_not_entities():
    assert _entities("Hello World this is the API layer") == []


def test_vocabulary_matching_is_case_insensitive_and_canonical():
    ents = _entities("we use postgresql and REDIS daily", vocabulary=["PostgreSQL", "Redis"])
    assert "PostgreSQL" in ents
    assert "Redis" in ents


def test_within_message_dedup_is_case_insensitive_first_spelling():
    ents = _entities("`getUser` then getUser and GetUser again")
    assert ents == ["getUser"]


# ==========================================================================
# HeuristicExtractor: decision detection
# ==========================================================================

@pytest.mark.parametrize(
    "content",
    [
        "we decided to use postgres",
        "let's go with redis",
        "we should switch to a new scheme",
        "use B instead of A",
    ],
)
def test_decision_cue_phrases(content):
    assert HeuristicExtractor().extract(Message("user", content)).is_decision is True


def test_non_decision():
    assert HeuristicExtractor().extract(Message("user", "the weather is nice")).is_decision is False


# ==========================================================================
# MessageLinker: nodes, seq, temporal chain
# ==========================================================================

def _linker():
    return MessageLinker(HeuristicExtractor())


def test_message_nodes_and_seq():
    g = ContextGraph()
    ids = _linker().ingest(g, [
        Message("user", "first"),
        Message("assistant", "second"),
        Message("user", "third"),
    ])
    assert ids == ["m0", "m1", "m2"]
    assert g.get("m0").metadata == {"role": "user", "seq": 0, "is_decision": False}
    assert g.get("m1").metadata == {"role": "assistant", "seq": 1, "is_decision": False}
    assert g.get("m2").metadata == {"role": "user", "seq": 2, "is_decision": False}


def test_message_timestamp_used_when_provided_else_wallclock():
    g = ContextGraph()
    before = time.time()
    _linker().ingest(g, [
        Message("user", "pinned", timestamp=123.5),
        Message("user", "auto"),
    ])
    assert g.get("m0").timestamp == 123.5
    assert g.get("m1").timestamp >= before


def test_temporal_chain():
    g = ContextGraph()
    _linker().ingest(g, [Message("user", "a"), Message("user", "b"), Message("user", "c")])
    pairs = {(e.source, e.target) for e in g.edges(type="TEMPORALLY_FOLLOWS")}
    assert pairs == {("m1", "m0"), ("m2", "m1")}


# ==========================================================================
# MessageLinker: mentions and entity dedup across messages
# ==========================================================================

def test_mentions_edges():
    g = ContextGraph()
    _linker().ingest(g, [Message("user", "use `redis` here")])
    mentions = g.edges(type="MENTIONS")
    assert len(mentions) == 1
    assert mentions[0].source == "m0"
    assert mentions[0].target == "entity:redis"
    assert g.get("entity:redis").content == "redis"


def test_entity_dedup_across_messages_case_insensitive():
    g = ContextGraph()
    MessageLinker(HeuristicExtractor(vocabulary=["PostgreSQL"])).ingest(g, [
        Message("user", "We use PostgreSQL."),
        Message("user", "postgresql is fine"),
        Message("user", "`Postgresql` again"),
    ])
    entities = g.nodes(type="Entity")
    assert len(entities) == 1
    assert entities[0].id == "entity:postgresql"
    assert entities[0].content == "PostgreSQL"   # first-seen spelling
    assert len(g.edges(type="MENTIONS")) == 3


# ==========================================================================
# MessageLinker: ANSWERS
# ==========================================================================

def test_answers_links_assistant_to_preceding_user():
    g = ContextGraph()
    _linker().ingest(g, [
        Message("user", "Why Postgres?"),
        Message("assistant", "Because transactions."),
        Message("assistant", "Also durability."),
    ])
    pairs = {(e.source, e.target) for e in g.edges(type="ANSWERS")}
    # only m1 (assistant after user) answers; m2 follows an assistant, so no edge
    assert pairs == {("m1", "m0")}


# ==========================================================================
# MessageLinker: UPDATES
# ==========================================================================

def test_updates_links_to_most_recent_sharing_decision():
    g = ContextGraph()
    _linker().ingest(g, [
        Message("user", "we decided the cache_key format"),
        Message("user", "let's go with a new cache_key scheme"),
        Message("user", "switch to another cache_key design"),
    ])
    pairs = {(e.source, e.target) for e in g.edges(type="UPDATES")}
    assert pairs == {("m1", "m0"), ("m2", "m1")}
    assert ("m2", "m0") not in pairs


def test_updates_requires_shared_entity():
    g = ContextGraph()
    _linker().ingest(g, [
        Message("user", "we decided on cache_key"),
        Message("user", "let's go with tenant_id"),
    ])
    assert g.edges(type="UPDATES") == []


# ==========================================================================
# MessageLinker: second ingest continues the conversation
# ==========================================================================

def test_second_ingest_continues_chain_and_seq():
    g = ContextGraph()
    linker = _linker()
    first = linker.ingest(g, [Message("user", "Why?")])
    second = linker.ingest(g, [Message("assistant", "Because"), Message("user", "ok")])

    assert first == ["m0"]
    assert second == ["m1", "m2"]
    assert g.get("m1").metadata["seq"] == 1
    assert g.get("m2").metadata["seq"] == 2

    temporal = {(e.source, e.target) for e in g.edges(type="TEMPORALLY_FOLLOWS")}
    assert ("m1", "m0") in temporal   # crosses the ingest boundary
    assert ("m2", "m1") in temporal

    answers = {(e.source, e.target) for e in g.edges(type="ANSWERS")}
    assert answers == {("m1", "m0")}  # assistant (call 2) answers user (call 1)


# ==========================================================================
# MessageLinker: state survives save/load and a fresh linker
# ==========================================================================

def test_cross_call_state_survives_save_load(tmp_path):
    g = ContextGraph()
    MessageLinker(HeuristicExtractor()).ingest(g, [
        Message("user", "Why?"),                          # m0
        Message("assistant", "we decided on cache_key"),  # m1, decision, entity cache_key
    ])

    path = tmp_path / "graph.db"
    g.save(path)
    g2 = ContextGraph.load(path)

    # a brand-new linker, continuing from the reloaded graph
    ids = MessageLinker(HeuristicExtractor()).ingest(g2, [
        Message("user", "let's go with a new cache_key"),  # m2, decision, entity cache_key
    ])

    assert ids == ["m2"]                                   # seq continued from the graph
    assert g2.get("m2").metadata["seq"] == 2

    temporal = {(e.source, e.target) for e in g2.edges(type="TEMPORALLY_FOLLOWS")}
    assert ("m2", "m1") in temporal                        # chain continued across reload

    updates = {(e.source, e.target) for e in g2.edges(type="UPDATES")}
    assert ("m2", "m1") in updates                         # earlier decision recovered from graph


# ==========================================================================
# MessageLinker: weights and immutability
# ==========================================================================

def test_edge_weights_from_config_and_override():
    g = ContextGraph()
    _linker().ingest(g, [Message("user", "a"), Message("user", "b")])
    tf = g.edges(type="TEMPORALLY_FOLLOWS")[0]
    assert tf.weight == DEFAULT_EDGE_WEIGHTS["TEMPORALLY_FOLLOWS"]

    g2 = ContextGraph()
    MessageLinker(HeuristicExtractor(), weights={"TEMPORALLY_FOLLOWS": 0.99}).ingest(
        g2, [Message("user", "a"), Message("user", "b")]
    )
    assert g2.edges(type="TEMPORALLY_FOLLOWS")[0].weight == 0.99


def test_linker_never_mutates_message_content():
    contents = ["We use `redis` for cache_key", "switch to postgres"]
    messages = [Message("user", contents[0]), Message("assistant", contents[1])]
    g = ContextGraph()
    _linker().ingest(g, messages)

    assert [m.content for m in messages] == contents
    assert g.get("m0").content == contents[0]
    assert g.get("m1").content == contents[1]


# ==========================================================================
# MessageLinker: topic-based UPDATES linking (Gemma topics)
# ==========================================================================

class _FixedExtractor:
    """Returns a preset Signals per message content."""

    def __init__(self, by_content):
        self.by_content = by_content

    def extract(self, message):
        return self.by_content[message.content]


class _OneHotEmbedder:
    def __init__(self, mapping, dim):
        self.mapping = mapping
        self.dim = dim

    def embed(self, texts):
        import numpy as np
        rows = []
        for t in texts:
            v = np.zeros(self.dim)
            if t in self.mapping:
                v[self.mapping[t]] = 1.0
            rows.append(v)
        return np.vstack(rows) if rows else np.zeros((0, self.dim))


def _topic_graph(threshold, topic_vec):
    sig = {
        "mongo": Signals(entities=["mongodb"], is_decision=True, topic="data store choice"),
        "postgres": Signals(entities=["postgres"], is_decision=True, topic="the main datastore"),
    }
    g = ContextGraph()
    MessageLinker(_FixedExtractor(sig), embedder=_OneHotEmbedder(topic_vec, dim=2),
                  topic_threshold=threshold).ingest(
        g, [Message("user", "mongo"), Message("user", "postgres")])
    return g


def test_topic_updates_links_decisions_without_shared_entity():
    g = _topic_graph(0.6, {"data store choice": 0, "the main datastore": 0})
    edges = g.edges(type="UPDATES")
    assert {(e.source, e.target) for e in edges} == {("m1", "m0")}
    assert edges[0].metadata["reason"] == "topic"
    assert edges[0].metadata["similarity"] == pytest.approx(1.0)


def test_topic_below_threshold_does_not_link():
    g = _topic_graph(0.6, {"data store choice": 0, "the main datastore": 1})
    assert g.edges(type="UPDATES") == []


def test_topic_linking_off_without_embedder():
    sig = {
        "mongo": Signals(entities=["mongodb"], is_decision=True, topic="data store"),
        "postgres": Signals(entities=["postgres"], is_decision=True, topic="data store"),
    }
    g = ContextGraph()
    MessageLinker(_FixedExtractor(sig)).ingest(
        g, [Message("user", "mongo"), Message("user", "postgres")])
    assert g.edges(type="UPDATES") == []   # entity-only linker ignores topics


def test_shared_entity_links_with_entity_reason():
    sig = {
        "a": Signals(entities=["cache_key"], is_decision=True, topic="caching"),
        "b": Signals(entities=["cache_key"], is_decision=True, topic="something else"),
    }
    g = ContextGraph()
    MessageLinker(_FixedExtractor(sig), embedder=_OneHotEmbedder({"caching": 0, "something else": 1}, 2),
                  topic_threshold=0.6).ingest(g, [Message("user", "a"), Message("user", "b")])
    edges = g.edges(type="UPDATES")
    assert {(e.source, e.target) for e in edges} == {("m1", "m0")}
    assert edges[0].metadata["reason"] == "entity"
    assert edges[0].metadata["shared_entity"] == "cache_key"
