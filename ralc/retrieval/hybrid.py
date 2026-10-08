"""Hybrid ranking: combine semantic and relational signals into one score."""

from __future__ import annotations

from dataclasses import dataclass, field

from ralc.extraction.base import Extractor, Message
from ralc.retrieval.relational import Candidate
from ralc.retrieval.semantic import SemanticRetriever

_ENTITY_PREFIX = "entity:"
_SIGNAL_NAMES = ("semantic", "relational", "entity_overlap", "recency", "seed")


@dataclass
class RankingConfig:
    # Experimental weights, NOT validated. Seed is 0 by default because it
    # largely double-counts semantic similarity; recency is low so relevance
    # outweighs recency.
    semantic: float = 1.0
    relational: float = 1.0
    entity_overlap: float = 0.5
    recency: float = 0.2
    seed: float = 0.0


@dataclass
class RankedCandidate:
    node_id: str
    score: float
    signals: dict[str, float]
    path: list[str]
    path_edge_types: list[str] = field(default_factory=list)


class HybridRanker:
    def __init__(self, semantic_retriever: SemanticRetriever, extractor: Extractor,
                 config: RankingConfig | None = None):
        self.semantic = semantic_retriever
        self.extractor = extractor
        self.config = config or RankingConfig()

    def rank(self, query: str, candidates: list[Candidate]) -> list[RankedCandidate]:
        if not candidates:
            return []
        graph = self.semantic.graph
        ids = [c.node_id for c in candidates]

        cosines = self.semantic.score(query, ids)
        query_entities = {e.lower() for e in self.extractor.extract(Message("user", query)).entities}

        semantic = {nid: _clip01(cosines[nid]) for nid in ids}
        relational = _divide_by_max({c.node_id: c.score for c in candidates})
        recency = _divide_by_max({
            c.node_id: float(graph.get(c.node_id).metadata.get("seq", 0)) for c in candidates
        })
        entity_overlap = {
            c.node_id: self._entity_overlap(graph, c.node_id, query_entities) for c in candidates
        }
        seed = {c.node_id: (1.0 if c.is_seed else 0.0) for c in candidates}

        weights = self.config
        ranked = []
        for candidate in candidates:
            nid = candidate.node_id
            signals = {
                "semantic": semantic[nid],
                "relational": relational[nid],
                "entity_overlap": entity_overlap[nid],
                "recency": recency[nid],
                "seed": seed[nid],
            }
            score = (
                weights.semantic * signals["semantic"]
                + weights.relational * signals["relational"]
                + weights.entity_overlap * signals["entity_overlap"]
                + weights.recency * signals["recency"]
                + weights.seed * signals["seed"]
            )
            ranked.append(
                RankedCandidate(nid, score, signals, candidate.path, candidate.path_edge_types)
            )
        ranked.sort(key=lambda r: (-r.score, r.node_id))
        return ranked

    def _entity_overlap(self, graph, node_id: str, query_entities: set[str]) -> float:
        if not query_entities:
            return 0.0
        mentioned = set()
        for neighbor_id, edge in graph.neighbors(node_id, types=["MENTIONS"]):
            if edge.source == node_id and neighbor_id.startswith(_ENTITY_PREFIX):
                mentioned.add(neighbor_id[len(_ENTITY_PREFIX):])
        return len(query_entities & mentioned) / len(query_entities)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _divide_by_max(raw: dict[str, float]) -> dict[str, float]:
    hi = max(raw.values(), default=0.0)
    if hi <= 0.0:
        return {key: 0.0 for key in raw}
    return {key: value / hi for key, value in raw.items()}
