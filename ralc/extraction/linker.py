"""Ingest messages into a ContextGraph, building heuristic relationships."""

from __future__ import annotations

import time

from ralc.extraction.base import Extractor, Message
from ralc.graph.edges import Edge
from ralc.graph.graph import ContextGraph
from ralc.graph.nodes import Node

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

    The linker keeps conversation state across :meth:`ingest` calls, so a later
    call continues the temporal chain and sequence numbering from where the
    previous one left off.
    """

    def __init__(self, extractor: Extractor, weights: dict | None = None):
        self.extractor = extractor
        self.weights = {**DEFAULT_EDGE_WEIGHTS, **(weights or {})}
        self._seq = 0
        self._prev_id: str | None = None
        self._prev_role: str | None = None
        # (message id, set of lowercased entity keys) for each decision so far.
        self._decisions: list[tuple[str, set[str]]] = []

    def ingest(self, graph: ContextGraph, messages: list[Message]) -> list[str]:
        """Add messages to ``graph`` and return their node ids in order."""
        created: list[str] = []
        for message in messages:
            signals = self.extractor.extract(message)
            seq = self._seq
            msg_id = f"m{seq}"
            timestamp = message.timestamp if message.timestamp is not None else time.time()
            graph.add_node(
                Node(
                    id=msg_id,
                    content=message.content,
                    type="Message",
                    timestamp=timestamp,
                    metadata={"role": message.role, "seq": seq},
                )
            )
            created.append(msg_id)

            if self._prev_id is not None:
                graph.add_edge(
                    Edge(msg_id, self._prev_id, "TEMPORALLY_FOLLOWS",
                         weight=self.weights["TEMPORALLY_FOLLOWS"])
                )
                if message.role == "assistant" and self._prev_role == "user":
                    graph.add_edge(
                        Edge(msg_id, self._prev_id, "ANSWERS",
                             weight=self.weights["ANSWERS"])
                    )

            entity_keys = self._link_entities(graph, msg_id, signals.entities, timestamp)

            if signals.is_decision:
                self._link_update(graph, msg_id, entity_keys)
                self._decisions.append((msg_id, entity_keys))

            self._prev_id = msg_id
            self._prev_role = message.role
            self._seq += 1
        return created

    def _link_entities(
        self, graph: ContextGraph, msg_id: str, entities: list[str], timestamp: float
    ) -> set[str]:
        keys: set[str] = set()
        for spelling in entities:
            key = spelling.lower()
            keys.add(key)
            entity_id = f"entity:{key}"
            if not self._has_node(graph, entity_id):
                graph.add_node(
                    Node(id=entity_id, content=spelling, type="Entity", timestamp=timestamp)
                )
            graph.add_edge(
                Edge(msg_id, entity_id, "MENTIONS", weight=self.weights["MENTIONS"])
            )
        return keys

    def _link_update(self, graph: ContextGraph, msg_id: str, entity_keys: set[str]) -> None:
        for prev_id, prev_keys in reversed(self._decisions):
            if entity_keys & prev_keys:
                graph.add_edge(
                    Edge(msg_id, prev_id, "UPDATES", weight=self.weights["UPDATES"])
                )
                return

    @staticmethod
    def _has_node(graph: ContextGraph, node_id: str) -> bool:
        try:
            graph.get(node_id)
            return True
        except KeyError:
            return False
