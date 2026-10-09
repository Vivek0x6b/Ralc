"""Hackathon demo: one question, one budget, vector search vs RALC with Gemma.

Cached data only, no API calls. On the v1 dataset it answers
"What did we use for hosting before we changed it?" at a 1000 token budget and
shows that vector search drops the AWS message while RALC, using the Gemma graph
and relation-aware selection, keeps both the Heroku and the AWS message.

Every number is computed here from the actual run. Run from the repo root:

    python -m examples.demo
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ralc import ContextManager
from ralc.allocation.budget import DefaultTokenCounter
from ralc.allocation.selector import SelectionConfig
from ralc.embeddings.local import SentenceTransformerEmbedder
from ralc.extraction import HeuristicExtractor
from ralc.retrieval.hybrid import RankingConfig
from ralc.retrieval.relational import ExpansionConfig

from benchmarks.experiments.relational_ablation import ralc_ids
from benchmarks.gemma_cache import GEMMA_MODEL, load_gemma_cache
from benchmarks.run import fill_to_budget

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "data"
QUESTION = "What did we use for hosting before we changed it?"
BUDGET = 1000
REQUIRED = ("m33", "m137")   # Heroku decision, AWS switch
SEED_K = 10


class CachedGemmaV1:
    """Reads the cached Gemma signals (original prompt version) with a heuristic
    fallback. No API calls. This rebuilds the exact Gemma graph the benchmark
    used, where the AWS message links to the Heroku one by an UPDATES edge."""

    def __init__(self, cache, model):
        self.cache = cache
        self.model = model
        self.heuristic = HeuristicExtractor()

    def _key(self, content):
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"{self.model}|0|1|{digest}"

    def extract(self, message):
        return self.cache.get(self._key(message.content)) or self.heuristic.extract(message)


def show(selected_ids, by_id, order):
    for nid in sorted(set(selected_ids), key=lambda i: order[i]):
        m = by_id[nid]
        mark = f"   <= {nid} is a required message" if nid in REQUIRED else ""
        print(f"    {nid:>5}  {m['role']:<9}  {m['content'][:70]}{mark}")


def found(selected_ids):
    s = set(selected_ids)
    return sum(1 for r in REQUIRED if r in s)


def tokens(selected_ids, tokens_by_id):
    return sum(tokens_by_id[i] for i in set(selected_ids))


def main():
    messages = json.loads((DATA / "conversation.json").read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in messages}
    order = {m["id"]: i for i, m in enumerate(messages)}
    counter = DefaultTokenCounter()
    tokens_by_id = {m["id"]: counter.count(m["content"]) for m in messages}
    all_ids = [m["id"] for m in messages]

    embedder = SentenceTransformerEmbedder()

    print("RALC demo: one question, one budget, two ways to choose context")
    print("=" * 64)
    print(f"Question: {QUESTION}")
    print(f"Token budget: {BUDGET}")
    print(f"Required messages for the full answer: {REQUIRED[0]} (Heroku), {REQUIRED[1]} (AWS)")
    print()

    # Heuristic graph backs vector search and shows the missing links.
    heuristic_mgr = ContextManager(embedder=embedder, extractor=HeuristicExtractor(),
                                   token_counter=counter)
    for m in messages:
        heuristic_mgr.add_message(m["role"], m["content"])

    scored = heuristic_mgr.retriever.retrieve(QUESTION, k=len(all_ids))
    vector_ids = fill_to_budget([nid for nid, _ in scored], tokens_by_id, BUDGET)
    print("-" * 64)
    print("1) VECTOR SEARCH (pick the most similar messages)")
    print("-" * 64)
    show(vector_ids, by_id, order)
    print(f"  Heroku m33 found: {'m33' in vector_ids} | AWS m137 found: {'m137' in vector_ids}")
    print()

    # Gemma graph from cache (original prompt), entity linking.
    cache = load_gemma_cache()
    gemma_mgr = ContextManager(embedder=embedder, extractor=CachedGemmaV1(cache, GEMMA_MODEL),
                               token_counter=counter)
    for m in messages:
        gemma_mgr.add_message(m["role"], m["content"])

    heuristic_updates = len(heuristic_mgr.graph.edges(type="UPDATES"))
    gemma_updates = len(gemma_mgr.graph.edges(type="UPDATES"))
    hosting_edge = [e for e in gemma_mgr.graph.edges(type="UPDATES")
                    if (e.source, e.target) == ("m137", "m33")]
    print("-" * 64)
    print("2) THE LINK GEMMA FOUND")
    print("-" * 64)
    for e in hosting_edge:
        print(f"  {e.source} UPDATES {e.target}: the AWS message replaces the Heroku one")
    print(f"  Heuristic graph UPDATES edges: {heuristic_updates}")
    print(f"  Gemma graph UPDATES edges:     {gemma_updates}")
    print()

    # RALC: Gemma graph + relation-aware selection (keep linked pairs).
    ralc_ids_selected = ralc_ids(
        gemma_mgr, QUESTION, BUDGET,
        exp_cfg=ExpansionConfig(), strategy="hop_decay", rank_cfg=RankingConfig(),
        sel_cfg=SelectionConfig(redundancy_exempt_edge_types=("UPDATES", "ANSWERS")),
        seed_k=SEED_K,
    )
    print("-" * 64)
    print("3) RALC (Gemma graph, keep linked pairs)")
    print("-" * 64)
    show(ralc_ids_selected, by_id, order)
    print(f"  Heroku m33 found: {'m33' in ralc_ids_selected} | AWS m137 found: {'m137' in ralc_ids_selected}")
    print()

    print("=" * 64)
    print("SUMMARY")
    print(f"  vector search  required found {found(vector_ids)}/2   tokens {tokens(vector_ids, tokens_by_id)}")
    print(f"  RALC + Gemma   required found {found(ralc_ids_selected)}/2   tokens {tokens(ralc_ids_selected, tokens_by_id)}")


if __name__ == "__main__":
    main()
