"""Ingest messages into a ContextGraph, building heuristic relationships."""

from __future__ import annotations

import time

import numpy as np

from ralc.extraction.base import Extractor, Message
from ralc.graph.edges import Edge
from ralc.graph.graph import ContextGraph
from ralc.graph.nodes import Node

_ENTITY_PREFIX = "entity:"

# Experimental starting weights. These are rough guesses, NOT validated;
# expect them to change once ranking is benchmarked.
DEFAULT_EDGE_WEIGHTS = {
    "TEMPORALLY_FOLLOWS": 0.3,
    "MENTIONS": 0.5,
    "ANSWERS": 0.8,
    "UPDATES": 0.7,
}


class MessageLinker:
    """Turns an ordered message stream into a relational context graph.

    The linker holds no cross-call state of its own: each :meth:`ingest`
    derives where the conversation left off from the graph itself (sequence
    number, previous message, and earlier decisions). That state therefore
    survives save/load and a fresh linker continues the conversation.
    """

    def __init__(self, extractor: Extractor, weights: dict | None = None,
                 embedder=None, topic_threshold: float = 0.6):
        self.extractor = extractor
        self.weights = {**DEFAULT_EDGE_WEIGHTS, **(weights or {})}
        # When an embedder is supplied, two decisions also link by UPDATES if
        # their topic strings are embedding-similar at or above the threshold,
        # not only when they share an entity. With no embedder, linking is
        # entity-only exactly as before.
        self.embedder = embedder
        self.topic_threshold = topic_threshold
        self._topic_vectors: dict[str, np.ndarray] = {}

    def ingest(self, graph: ContextGraph, messages: list[Message]) -> list[str]:
        """Add messages to ``graph`` and return their node ids in order."""
        seq, prev_id, prev_role, decisions = self._state_from_graph(graph)
        created: list[str] = []
        for message in messages:
            signals = self.extractor.extract(message)
            msg_id = f"m{seq}"
            timestamp = message.timestamp if message.timestamp is not None else time.time()
            metadata = {
                "role": message.role,
                "seq": seq,
                "is_decision": signals.is_decision,
            }
            # Only store a topic when there is one, so messages without a topic
            # keep the exact metadata shape they had before.
            if signals.topic is not None:
                metadata["topic"] = signals.topic
            graph.add_node(
                Node(
                    id=msg_id,
                    content=message.content,
                    type="Message",
                    timestamp=timestamp,
                    metadata=metadata,
                )
            )
            created.append(msg_id)

            if prev_id is not None:
                graph.add_edge(
                    Edge(msg_id, prev_id, "TEMPORALLY_FOLLOWS",
                         weight=self.weights["TEMPORALLY_FOLLOWS"])
                )
                if message.role == "assistant" and prev_role == "user":
                    graph.add_edge(
                        Edge(msg_id, prev_id, "ANSWERS",
                             weight=self.weights["ANSWERS"])
                    )

            entity_keys = self._link_entities(graph, msg_id, signals.entities, timestamp)

            if signals.is_decision:
                self._link_update(graph, msg_id, entity_keys, signals.topic, decisions)
                decisions.append((msg_id, entity_keys, signals.topic))

            prev_id = msg_id
            prev_role = message.role
            seq += 1
        return created

    # -- state recovery ----------------------------------------------------

    def _state_from_graph(
        self, graph: ContextGraph
    ) -> tuple[int, str | None, str | None, list[tuple[str, set[str], str | None]]]:
        messages = graph.nodes(type="Message")
        if not messages:
            return 0, None, None, []
        messages.sort(key=lambda n: n.metadata["seq"])
        last = messages[-1]
        decisions = [
            (node.id, self._entity_keys_of(graph, node.id), node.metadata.get("topic"))
            for node in messages
            if node.metadata.get("is_decision")
        ]
        return last.metadata["seq"] + 1, last.id, last.metadata.get("role"), decisions

    def _entity_keys_of(self, graph: ContextGraph, msg_id: str) -> set[str]:
        keys = set()
        for neighbor_id, edge in graph.neighbors(msg_id, types=["MENTIONS"]):
            if edge.source == msg_id and neighbor_id.startswith(_ENTITY_PREFIX):
                keys.add(neighbor_id[len(_ENTITY_PREFIX):])
        return keys

    # -- edge building -----------------------------------------------------

    def _link_entities(
        self, graph: ContextGraph, msg_id: str, entities: list[str], timestamp: float
    ) -> set[str]:
        keys: set[str] = set()
        for spelling in entities:
            key = spelling.lower()
            keys.add(key)
            entity_id = f"{_ENTITY_PREFIX}{key}"
            if not self._has_node(graph, entity_id):
                graph.add_node(
                    Node(id=entity_id, content=spelling, type="Entity", timestamp=timestamp)
                )
            graph.add_edge(
                Edge(msg_id, entity_id, "MENTIONS", weight=self.weights["MENTIONS"])
            )
        return keys

    def _link_update(
        self,
        graph: ContextGraph,
        msg_id: str,
        entity_keys: set[str],
        topic: str | None,
        decisions: list[tuple[str, set[str], str | None]],
    ) -> None:
        for prev_id, prev_keys, prev_topic in reversed(decisions):
            shared = entity_keys & prev_keys
            if shared:
                graph.add_edge(Edge(msg_id, prev_id, "UPDATES", weight=self.weights["UPDATES"],
                                    metadata={"reason": "entity",
                                              "shared_entity": sorted(shared)[0]}))
                return
            similarity = self._topic_similarity(topic, prev_topic)
            if similarity is not None and similarity >= self.topic_threshold:
                graph.add_edge(Edge(msg_id, prev_id, "UPDATES", weight=self.weights["UPDATES"],
                                    metadata={"reason": "topic",
                                              "similarity": similarity,
                                              "topics": [topic, prev_topic]}))
                return

    def _topic_similarity(self, a: str | None, b: str | None) -> float | None:
        """Cosine between two topic strings, or None when it cannot apply."""
        if self.embedder is None or not a or not b:
            return None
        return float(self._topic_vector(a) @ self._topic_vector(b))

    def _topic_vector(self, topic: str) -> np.ndarray:
        if topic not in self._topic_vectors:
            vector = np.asarray(self.embedder.embed([topic])[0], dtype=float)
            norm = float(np.linalg.norm(vector))
            self._topic_vectors[topic] = vector / norm if norm > 0.0 else vector
        return self._topic_vectors[topic]

    @staticmethod
    def _has_node(graph: ContextGraph, node_id: str) -> bool:
        try:
            graph.get(node_id)
            return True
        except KeyError:
            return False
