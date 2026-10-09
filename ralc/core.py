"""ContextManager: the simple facade wiring the RALC pipeline together."""

from __future__ import annotations

from pathlib import Path

from ralc.allocation.budget import DefaultTokenCounter, TokenCounter
from ralc.allocation.result import ContextResult
from ralc.allocation.selector import ContextSelector, SelectionConfig
from ralc.embeddings.local import Embedder, SentenceTransformerEmbedder
from ralc.extraction.base import Extractor, Message
from ralc.extraction.heuristic import HeuristicExtractor
from ralc.extraction.linker import MessageLinker
from ralc.graph.graph import ContextGraph
from ralc.retrieval.hybrid import HybridRanker, RankingConfig
from ralc.retrieval.relational import ExpansionConfig, RelationalExpander
from ralc.retrieval.semantic import SemanticRetriever


class ContextManager:
    """Store conversation context and retrieve a budget-bounded, relationally
    selected slice of it for an LLM.

    The query is passed through the pipeline for ranking only; it is never
    stored in or altered by the returned ContextResult.
    """

    def __init__(
        self,
        storage: str | Path | None = None,
        embedder: Embedder | None = None,
        extractor: Extractor | None = None,
        token_counter: TokenCounter | None = None,
        expansion_strategy: str = "hop_decay",
        expansion_config: ExpansionConfig | None = None,
        ranking_config: RankingConfig | None = None,
        selection_config: SelectionConfig | None = None,
        link_topics: bool = False,
        topic_threshold: float = 0.6,
    ):
        self.storage = storage
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.extractor = extractor or HeuristicExtractor()
        self.token_counter = token_counter or DefaultTokenCounter()

        if storage is not None and Path(storage).exists():
            self.graph = ContextGraph.load(storage)
        else:
            self.graph = ContextGraph()

        self.retriever = SemanticRetriever(self.graph, self.embedder)
        # Topic linking is opt-in: with link_topics off the linker is entity-only,
        # exactly as before, so default behavior and existing results are unchanged.
        self.linker = MessageLinker(
            self.extractor,
            embedder=self.embedder if link_topics else None,
            topic_threshold=topic_threshold,
        )
        self.expander = RelationalExpander(
            self.graph, strategy=expansion_strategy, config=expansion_config
        )
        self.ranker = HybridRanker(self.retriever, self.extractor, ranking_config)
        self.selector = ContextSelector(
            self.graph, self.token_counter, selection_config,
            semantic_retriever=self.retriever,
        )

        # Rebuild embedding vectors for a reloaded graph (they are not persisted).
        if self.graph.nodes(type="Message"):
            self.retriever.index(type="Message")

    def add_message(self, role: str, content: str, timestamp: float | None = None) -> str:
        """Ingest one message and index it; return its node id."""
        node_id = self.linker.ingest(self.graph, [Message(role, content, timestamp)])[0]
        self.retriever.index_node(node_id)
        return node_id

    def retrieve(self, query: str, token_budget: int, seed_k: int = 8) -> ContextResult:
        """Build a budget-bounded context for ``query`` through the pipeline."""
        seeds = self.retriever.retrieve(query, k=seed_k)
        candidates = self.expander.expand(seeds)
        ranked = self.ranker.rank(query, candidates)
        result = self.selector.select(query, ranked, token_budget)
        result.metadata.update(
            {
                "semantic_seeds": len(seeds),
                "nodes_explored": self.expander.last_stats.get("nodes_visited", 0),
                "candidates_ranked": len(ranked),
                "final_nodes": len(result.nodes),
                "relationships_used": len(result.relationships),
            }
        )
        return result

    def save(self, path: str | Path | None = None) -> None:
        """Write the graph to ``path`` (or the configured storage)."""
        target = path or self.storage
        if target is None:
            raise ValueError("no storage path configured; pass path to save()")
        self.graph.save(target)
