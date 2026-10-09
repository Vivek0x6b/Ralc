"""Select the highest-value context that fits a token budget."""

from __future__ import annotations

from dataclasses import dataclass

from ralc.allocation.budget import TokenCounter
from ralc.allocation.result import ContextResult
from ralc.graph.graph import ContextGraph
from ralc.retrieval.hybrid import RankedCandidate
from ralc.retrieval.semantic import SemanticRetriever

_STRATEGIES = ("greedy_score", "greedy_density")


@dataclass
class SelectionConfig:
    strategy: str = "greedy_score"
    # Redundancy penalty (MMR style) is experimental and not validated: a
    # candidate's score is reduced by its peak similarity to already-selected
    # nodes, so near-duplicates are less likely to be chosen.
    redundancy_penalty: bool = True
    redundancy_lambda: float = 0.5
    # Relation-aware redundancy: skip the redundancy penalty between a candidate
    # and an already-selected node that the graph links by a direct edge (either
    # direction) of one of these types, so a complementary linked pair (a
    # decision and its update, a question and its answer) is not treated as a
    # near-duplicate. Default empty, which reproduces the plain MMR behavior.
    redundancy_exempt_edge_types: tuple[str, ...] = ()
    # A candidate whose adjusted score is <= min_score is never selected, even
    # if it fits; RALC may return fewer tokens than the budget allows.
    min_score: float = 0.0


class ContextSelector:
    def __init__(self, graph: ContextGraph, token_counter: TokenCounter,
                 config: SelectionConfig | None = None,
                 semantic_retriever: SemanticRetriever | None = None):
        config = config or SelectionConfig()
        if config.strategy not in _STRATEGIES:
            raise ValueError(
                f"unknown strategy {config.strategy!r}; expected one of {_STRATEGIES}"
            )
        if config.redundancy_penalty and semantic_retriever is None:
            raise ValueError("redundancy_penalty requires a semantic_retriever")
        self.graph = graph
        self.token_counter = token_counter
        self.config = config
        self.semantic_retriever = semantic_retriever

    def select(self, query: str, ranked: list[RankedCandidate], token_budget: int) -> ContextResult:
        cfg = self.config
        tokens = {r.node_id: self.token_counter.count(self.graph.get(r.node_id).content)
                  for r in ranked}
        score_of = {r.node_id: r.score for r in ranked}

        selected: list[str] = []
        remaining = token_budget
        while True:
            best_id = None
            best_key = None
            for candidate in ranked:
                nid = candidate.node_id
                if nid in selected:
                    continue
                adjusted = self._adjusted(nid, score_of[nid], selected)
                if adjusted <= cfg.min_score:
                    continue
                if tokens[nid] > remaining:
                    continue
                metric = adjusted / max(tokens[nid], 1) if cfg.strategy == "greedy_density" else adjusted
                key = (-metric, nid)
                if best_key is None or key < best_key:
                    best_key = key
                    best_id = nid
            if best_id is None:
                break
            selected.append(best_id)
            remaining -= tokens[best_id]

        return self._build_result(ranked, tokens, score_of, selected, token_budget)

    def _adjusted(self, node_id: str, score: float, selected: list[str]) -> float:
        cfg = self.config
        if not cfg.redundancy_penalty or not selected:
            return score
        exempt = self._exempt_partners(node_id) if cfg.redundancy_exempt_edge_types else frozenset()
        peak = max(
            (self._similarity(node_id, other) for other in selected if other not in exempt),
            default=0.0,
        )
        return score - cfg.redundancy_lambda * peak

    def _exempt_partners(self, node_id: str) -> set[str]:
        """Node ids linked to ``node_id`` by an exempt edge type (either direction)."""
        types = list(self.config.redundancy_exempt_edge_types)
        return {neighbor_id for neighbor_id, _edge in self.graph.neighbors(node_id, types=types)}

    def _similarity(self, a: str, b: str) -> float:
        vector_a = self.semantic_retriever.embedding(a)
        vector_b = self.semantic_retriever.embedding(b)
        return max(0.0, float(vector_a @ vector_b))

    def _build_result(self, ranked, tokens, score_of, selected, token_budget) -> ContextResult:
        selected_set = set(selected)
        nodes = [self.graph.get(nid) for nid in selected_set]
        nodes.sort(key=lambda n: (n.metadata.get("seq", 0), n.id))
        token_count = sum(tokens[nid] for nid in selected_set)

        relationships = [
            edge for edge in self.graph.edges()
            if edge.source in selected_set and edge.target in selected_set
        ]
        relationships.sort(key=lambda e: (e.source, e.target, e.type))

        skipped_low_score = 0
        skipped_for_size = 0
        for candidate in ranked:
            nid = candidate.node_id
            if nid in selected_set:
                continue
            if self._adjusted(nid, score_of[nid], selected) <= self.config.min_score:
                skipped_low_score += 1
            else:
                skipped_for_size += 1

        metadata = {
            "budget": token_budget,
            "total_available_tokens": sum(tokens.values()),
            "candidates_considered": len(ranked),
            "nodes_selected": len(selected_set),
            "nodes_skipped_for_size": skipped_for_size,
            "nodes_skipped_low_score": skipped_low_score,
            "approximate": bool(self.token_counter.approximate),
            "strategy": self.config.strategy,
        }
        return ContextResult(
            nodes=nodes, token_count=token_count,
            relationships=relationships, metadata=metadata,
        )
