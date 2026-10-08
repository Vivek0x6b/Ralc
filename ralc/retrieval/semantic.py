"""Semantic seed retrieval: embed nodes, score a query by cosine similarity."""

from __future__ import annotations

import numpy as np

from ralc.embeddings.local import Embedder
from ralc.graph.graph import ContextGraph


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit length; a zero vector is left unchanged."""
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        return vec / norm
    return vec


class SemanticRetriever:
    """Embeds graph nodes and retrieves seeds by cosine similarity to a query.

    Vectors are L2-normalized inside the retriever at index and query time, so
    rankings are correct even if the embedder returns unnormalized vectors.
    Zero vectors (for example from empty content) score 0 rather than dividing
    by zero.
    """

    def __init__(self, graph: ContextGraph, embedder: Embedder):
        self.graph = graph
        self.embedder = embedder
        self._vectors: dict[str, np.ndarray] = {}

    def index(self, type: str | None = None) -> None:
        """Embed all nodes (optionally filtered by node type), replacing any
        existing index."""
        self._vectors = {}
        nodes = self.graph.nodes(type=type)
        if not nodes:
            return
        matrix = self.embedder.embed([node.content for node in nodes])
        for node, vector in zip(nodes, matrix):
            self._vectors[node.id] = _normalize(vector)

    def index_node(self, node_id: str) -> None:
        """Embed and add a single node without re-embedding the rest.

        Raises KeyError if the node is not in the graph.
        """
        node = self.graph.get(node_id)
        vector = self.embedder.embed([node.content])[0]
        self._vectors[node_id] = _normalize(vector)

    def score(self, query: str, node_ids: list[str]) -> dict[str, float]:
        """Return the cosine between ``query`` and each node in ``node_ids``.

        Nodes not already indexed are embedded on demand (from their graph
        content) and cached, so non-seed candidates can be scored too. Raises
        KeyError if a node id is not in the graph. The query is only read.
        """
        if not node_ids:
            return {}
        query_vector = _normalize(self.embedder.embed([query])[0])
        missing = [nid for nid in node_ids if nid not in self._vectors]
        if missing:
            matrix = self.embedder.embed([self.graph.get(nid).content for nid in missing])
            for node_id, vector in zip(missing, matrix):
                self._vectors[node_id] = _normalize(vector)
        return {node_id: float(self._vectors[node_id] @ query_vector) for node_id in node_ids}

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the top ``k`` (node_id, cosine score) pairs, highest first.

        Returns an empty list when nothing is indexed or ``k <= 0``. The query
        is only read: it is passed to the embedder unchanged.
        """
        if k <= 0 or not self._vectors:
            return []
        query_vector = _normalize(self.embedder.embed([query])[0])
        ids = list(self._vectors)
        matrix = np.vstack([self._vectors[node_id] for node_id in ids])
        scores = matrix @ query_vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [(ids[i], float(scores[i])) for i in order]
