"""ContextGraph: a relational context store over a networkx MultiDiGraph."""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from pathlib import Path

import networkx as nx

from ralc.graph.edges import Edge
from ralc.graph.nodes import Node


class ContextGraph:
    """A directed multigraph of context nodes and typed relationships.

    Nodes are keyed by ``id``; the ``Node``/``Edge`` object is carried on the
    networkx element under the ``data`` attribute. Edge keys are the edge type,
    so there is at most one edge of a given type per ordered (source, target).
    """

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    # -- construction ------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add a node. Raises ValueError if the id already exists."""
        if node.id in self._g:
            raise ValueError(f"node id already exists: {node.id!r}")
        self._g.add_node(node.id, data=node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge.

        Raises ValueError if either endpoint is missing, or if an edge of the
        same type already connects the same ordered pair.
        """
        if edge.source not in self._g:
            raise ValueError(f"source node missing: {edge.source!r}")
        if edge.target not in self._g:
            raise ValueError(f"target node missing: {edge.target!r}")
        if self._g.has_edge(edge.source, edge.target, key=edge.type):
            raise ValueError(
                f"duplicate edge type {edge.type!r} for "
                f"({edge.source!r} -> {edge.target!r})"
            )
        self._g.add_edge(edge.source, edge.target, key=edge.type, data=edge)

    # -- access ------------------------------------------------------------

    def get(self, node_id: str) -> Node:
        """Return the node with ``node_id``. Raises KeyError if absent."""
        if node_id not in self._g:
            raise KeyError(node_id)
        return self._g.nodes[node_id]["data"]

    def nodes(self, type: str | None = None) -> list[Node]:
        """Return all nodes, optionally filtered by node type."""
        return [
            data
            for _, data in self._g.nodes(data="data")
            if type is None or data.type == type
        ]

    def edges(self, type: str | None = None) -> list[Edge]:
        """Return all edges, optionally filtered by edge type."""
        return [
            data
            for _, _, key, data in self._g.edges(keys=True, data="data")
            if type is None or key == type
        ]

    def neighbors(
        self, node_id: str, types: list[str] | None = None
    ) -> list[tuple[str, Edge]]:
        """Return (neighbor_id, edge) pairs in both directions.

        One pair per matching edge, with no deduplication: two edges of
        different types to the same neighbor yield two pairs. ``types``, when
        given, restricts the result to edges whose type is in the collection.
        Raises KeyError if the node is absent.
        """
        if node_id not in self._g:
            raise KeyError(node_id)
        pairs: list[tuple[str, Edge]] = []
        for _, target, key, data in self._g.out_edges(node_id, keys=True, data="data"):
            if types is None or key in types:
                pairs.append((target, data))
        for source, _, key, data in self._g.in_edges(node_id, keys=True, data="data"):
            if types is None or key in types:
                pairs.append((source, data))
        return pairs

    # -- traversal ---------------------------------------------------------

    def bfs(
        self,
        seed_ids: list[str],
        max_hops: int,
        edge_types: list[str] | None = None,
    ) -> dict[str, int]:
        """Breadth-first reachability from ``seed_ids``.

        Returns a mapping of reachable node id to its minimum hop distance.
        Seeds are at hop 0. Traversal is bidirectional (edges are followed
        regardless of direction) and restricted to ``edge_types`` when given.
        Raises KeyError if any seed is absent.
        """
        for seed in seed_ids:
            if seed not in self._g:
                raise KeyError(seed)

        dist: dict[str, int] = {}
        queue: deque[str] = deque()
        for seed in seed_ids:
            if seed not in dist:
                dist[seed] = 0
                queue.append(seed)

        while queue:
            current = queue.popleft()
            if dist[current] >= max_hops:
                continue
            for neighbor, _edge in self.neighbors(current, types=edge_types):
                if neighbor not in dist:
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
        return dist

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the graph to a SQLite database at ``path`` (overwriting)."""
        conn = sqlite3.connect(str(path))
        try:
            cur = conn.cursor()
            cur.execute("DROP TABLE IF EXISTS nodes")
            cur.execute("DROP TABLE IF EXISTS edges")
            cur.execute(
                "CREATE TABLE nodes "
                "(id TEXT PRIMARY KEY, content TEXT, type TEXT, "
                "timestamp REAL, metadata TEXT, embedding_ref TEXT)"
            )
            cur.execute(
                "CREATE TABLE edges "
                "(source TEXT, target TEXT, type TEXT, weight REAL, "
                "metadata TEXT, created_at REAL)"
            )
            cur.executemany(
                "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (n.id, n.content, n.type, n.timestamp,
                     json.dumps(n.metadata), n.embedding_ref)
                    for n in self.nodes()
                ],
            )
            cur.executemany(
                "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (e.source, e.target, e.type, e.weight,
                     json.dumps(e.metadata), e.created_at)
                    for e in self.edges()
                ],
            )
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def load(cls, path: str | Path) -> ContextGraph:
        """Load a graph previously written by :meth:`save`."""
        graph = cls()
        conn = sqlite3.connect(str(path))
        try:
            cur = conn.cursor()
            for id_, content, type_, timestamp, metadata, embedding_ref in cur.execute(
                "SELECT id, content, type, timestamp, metadata, embedding_ref FROM nodes"
            ):
                graph.add_node(
                    Node(id_, content, type_, timestamp,
                         json.loads(metadata), embedding_ref)
                )
            for source, target, type_, weight, metadata, created_at in cur.execute(
                "SELECT source, target, type, weight, metadata, created_at FROM edges"
            ):
                graph.add_edge(
                    Edge(source, target, type_, weight,
                         json.loads(metadata), created_at)
                )
        finally:
            conn.close()
        return graph
