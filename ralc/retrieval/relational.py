"""Relational expansion: grow a scored seed set through the context graph.

Two strategies are provided so they can be compared later:

* ``hop_decay`` walks outward from the seeds, multiplying seed score by edge
  weights and a per-hop decay, and records the best path to each node.
* ``ppr`` runs personalized PageRank from the seeds over an undirected,
  weighted view of the graph.

Entity nodes are treated as bridges in ``hop_decay``: arriving at an Entity
costs the edge weight but no decay and does not consume a hop, so
Message -> Entity -> Message is a single hop. The hub penalty still applies
when leaving an Entity. This keeps entity-connected messages from being
penalized roughly twice as hard as a direct temporal neighbor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx

from ralc.graph.graph import ContextGraph

_ENTITY_TYPE = "Entity"
_STRATEGIES = ("hop_decay", "ppr")


@dataclass
class ExpansionConfig:
    max_hops: int = 2
    decay: float = 0.5
    edge_types: list[str] | None = None            # None means every type
    max_candidates: int | None = None              # None means uncapped
    return_types: list[str] = field(default_factory=lambda: ["Message"])
    # Hub penalty is experimental: an entity linked to many messages should not
    # wire everything together, so traversing out of it is scaled down by a
    # factor that shrinks with its degree. Not validated.
    hub_penalty: bool = True
    hub_log_base: float = 2.0


@dataclass
class Candidate:
    node_id: str
    score: float
    path: list[str]
    path_edge_types: list[str]
    is_seed: bool
    seed_score: float | None


class RelationalExpander:
    def __init__(self, graph: ContextGraph, strategy: str = "hop_decay",
                 config: ExpansionConfig | None = None):
        if strategy not in _STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {_STRATEGIES}")
        self.graph = graph
        self.strategy = strategy
        self.config = config or ExpansionConfig()
        self._degree_cache: dict[str, int] = {}
        # Real stats from the most recent expand() call, for the demo dashboard.
        self.last_stats: dict[str, int] = {"nodes_visited": 0}

    def expand(self, seeds: list[tuple[str, float]]) -> list[Candidate]:
        if not seeds:
            self.last_stats = {"nodes_visited": 0}
            return []
        for node_id, _ in seeds:
            self.graph.get(node_id)   # raises KeyError if a seed is missing
        seed_scores = {node_id: score for node_id, score in seeds}
        if self.strategy == "hop_decay":
            scored, visited = self._hop_decay(seeds)
        else:
            scored, visited = self._ppr(seed_scores)
        self.last_stats = {"nodes_visited": len(visited)}
        return self._finalize(scored, seed_scores)

    # -- hop_decay ---------------------------------------------------------

    def _hop_decay(self, seeds: list[tuple[str, float]]):
        cfg = self.config
        # best[node] = (value, path, edge_types)
        best: dict[str, tuple[float, list[str], list[str]]] = {}
        touched: set[str] = set()

        def consider(node_id, value, path, etypes):
            current = best.get(node_id)
            if current is None or value > current[0]:
                best[node_id] = (value, path, etypes)
                return True
            return False

        def dfs(node_id, value, hops, path, etypes, visited):
            touched.add(node_id)
            consider(node_id, value, path, etypes)
            source_is_entity = self.graph.get(node_id).type == _ENTITY_TYPE
            neighbors = sorted(
                self.graph.neighbors(node_id, types=cfg.edge_types),
                key=lambda pair: (pair[0], pair[1].type),
            )
            for neighbor_id, edge in neighbors:
                if neighbor_id in visited:
                    continue
                dest_is_entity = self.graph.get(neighbor_id).type == _ENTITY_TYPE
                if dest_is_entity:
                    # Arrive at a bridge: edge weight only, no decay, no hop.
                    mult = edge.weight
                    if source_is_entity and cfg.hub_penalty:
                        mult *= self._penalty(node_id)
                    new_hops = hops
                elif source_is_entity:
                    # Leave a bridge: decay and hub penalty, no second weight.
                    mult = cfg.decay
                    if cfg.hub_penalty:
                        mult *= self._penalty(node_id)
                    new_hops = hops + 1
                else:
                    mult = edge.weight * cfg.decay
                    new_hops = hops + 1
                if new_hops > cfg.max_hops:
                    continue
                dfs(
                    neighbor_id,
                    value * mult,
                    new_hops,
                    path + [neighbor_id],
                    etypes + [edge.type],
                    visited | {neighbor_id},
                )

        for node_id, score in sorted(seeds, key=lambda s: (-s[1], s[0])):
            dfs(node_id, score, 0, [node_id], [], {node_id})

        scored = {nid: (val, path, etypes) for nid, (val, path, etypes) in best.items()}
        return scored, touched

    def _penalty(self, node_id: str) -> float:
        return 1.0 / math.log(self.config.hub_log_base + self._degree(node_id))

    def _degree(self, node_id: str) -> int:
        if node_id not in self._degree_cache:
            self._degree_cache[node_id] = len(self.graph.neighbors(node_id))
        return self._degree_cache[node_id]

    # -- ppr ---------------------------------------------------------------

    def _ppr(self, seed_scores: dict[str, float]):
        cfg = self.config
        undirected = nx.Graph()
        for node in self.graph.nodes():
            undirected.add_node(node.id)
        for edge in self.graph.edges():
            if cfg.edge_types is not None and edge.type not in cfg.edge_types:
                continue
            if undirected.has_edge(edge.source, edge.target):
                undirected[edge.source][edge.target]["weight"] += edge.weight
            else:
                undirected.add_edge(edge.source, edge.target, weight=edge.weight)

        reachable: set[str] = set()
        for seed in seed_scores:
            reachable |= nx.node_connected_component(undirected, seed)

        total = sum(seed_scores.values())
        if total > 0:
            personalization = {node: seed_scores.get(node, 0.0) for node in undirected}
        else:
            personalization = {node: (1.0 if node in seed_scores else 0.0) for node in undirected}
        ranks = nx.pagerank(undirected, personalization=personalization, weight="weight")

        scored = {
            node_id: (ranks[node_id], [node_id], [])
            for node_id in reachable
        }
        return scored, reachable

    # -- shared finalization ----------------------------------------------

    def _finalize(self, scored, seed_scores):
        cfg = self.config
        return_types = set(cfg.return_types)

        if self.strategy == "ppr":
            kept = {
                nid: data for nid, data in scored.items()
                if nid in seed_scores or self.graph.get(nid).type in return_types
            }
            max_score = max((data[0] for data in kept.values()), default=0.0)
            if max_score > 0:
                scored = {nid: (val / max_score, path, et) for nid, (val, path, et) in kept.items()}
            else:
                scored = kept
        else:
            scored = {
                nid: data for nid, data in scored.items()
                if nid in seed_scores or self.graph.get(nid).type in return_types
            }

        candidates = [
            Candidate(
                node_id=nid,
                score=value,
                path=path,
                path_edge_types=etypes,
                is_seed=nid in seed_scores,
                seed_score=seed_scores.get(nid) if nid in seed_scores else None,
            )
            for nid, (value, path, etypes) in scored.items()
        ]
        candidates.sort(key=lambda c: (-c.score, c.node_id))
        return self._cap(candidates)

    def _cap(self, candidates: list[Candidate]) -> list[Candidate]:
        cap = self.config.max_candidates
        if cap is None or len(candidates) <= cap:
            return candidates
        chosen = [c for c in candidates if c.is_seed]   # seeds are never dropped
        for candidate in candidates:
            if len(chosen) >= cap:
                break
            if not candidate.is_seed:
                chosen.append(candidate)
        chosen.sort(key=lambda c: (-c.score, c.node_id))
        return chosen
