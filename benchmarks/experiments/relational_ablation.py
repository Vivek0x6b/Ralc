"""Relation-aware selection ablation (see benchmarks/PREREGISTRATION_relational.md).

Runs the non-Gemma baselines (recent, vector, graph, ralc_heuristic, full) plus
three config variants (relations off, redundancy off, relation-aware redundancy)
on the heuristic graph, for v1 and v2. For v1 it also runs ralc_gemma and
ralc_gemma + relation-aware, built from benchmarks/results/gemma_cache.json with
no API calls (heuristic fallback for any uncached string, matching run.py).

Reports, per question type (relationship vs lookup) and per budget: complete-
story rate, mean recall, mean tokens, and per-question wins and losses against
vector. Run from the repo root:

    python -m benchmarks.experiments.relational_ablation
"""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

from ralc import ContextManager
from ralc.allocation.budget import DefaultTokenCounter
from ralc.allocation.selector import ContextSelector, SelectionConfig
from ralc.embeddings.local import SentenceTransformerEmbedder
from ralc.extraction import HeuristicExtractor, Message
from ralc.extraction.gemma import GemmaExtractor
from ralc.retrieval.hybrid import HybridRanker, RankingConfig
from ralc.retrieval.relational import ExpansionConfig, RelationalExpander

from benchmarks.gemma_cache import GEMMA_MODEL, load_gemma_cache
from benchmarks.metrics import recall as recall_metric
from benchmarks.run import fill_to_budget

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS_PATH = HERE / "relational_ablation_results.json"
MD_PATH = HERE / "relational_ablation_results.md"

BUDGETS = {"v1": [1000, 2000, 4000], "v2": [250, 500, 1000, 2000, 4000]}
SEED_K = 10
EXEMPT = ("UPDATES", "ANSWERS")


class CachedGemmaExtractor:
    """The cached plain-Gemma signals, with a heuristic fallback for any string
    not in the cache. Makes no API calls, matching how run.py evaluates
    ralc_gemma over warm-failure strings."""

    def __init__(self, cache, keyer, heuristic):
        self.cache = cache
        self.keyer = keyer
        self.heuristic = heuristic
        self.misses = 0

    def extract(self, message: Message):
        key = self.keyer.cache_key(message.content)
        if key in self.cache:
            return self.cache[key]
        self.misses += 1
        return self.heuristic.extract(message)


def qtype(question: dict) -> str:
    """relationship if 2 or more gold messages, else lookup."""
    return "relationship" if len(question["gold_message_ids"]) >= 2 else "lookup"


def ralc_ids(manager, query, budget, *, exp_cfg, strategy, rank_cfg, sel_cfg):
    """Run the RALC pipeline over an already-built manager with given configs."""
    seeds = manager.retriever.retrieve(query, k=SEED_K)
    candidates = RelationalExpander(manager.graph, strategy=strategy, config=exp_cfg).expand(seeds)
    ranked = HybridRanker(manager.retriever, manager.extractor, rank_cfg).rank(query, candidates)
    selector = ContextSelector(manager.graph, manager.token_counter, sel_cfg,
                               semantic_retriever=manager.retriever)
    return [n.id for n in selector.select(query, ranked, budget).nodes]


def build_methods(manager, all_ids, tokens_by_id, gemma_manager=None):
    """Map method name to a function (query, budget) -> selected ids."""
    retriever = manager.retriever
    recent_order = list(reversed(all_ids))

    def method_recent(query, budget):
        return fill_to_budget(recent_order, tokens_by_id, budget)

    def method_vector(query, budget):
        scored = retriever.retrieve(query, k=len(all_ids))
        return fill_to_budget([nid for nid, _ in scored], tokens_by_id, budget)

    def method_graph(query, budget):
        seeds = retriever.retrieve(query, k=SEED_K)
        candidates = RelationalExpander(manager.graph, strategy="hop_decay").expand(seeds)
        ordered = [c.node_id for c in sorted(candidates, key=lambda c: (-c.score, c.node_id))]
        return fill_to_budget(ordered, tokens_by_id, budget)

    def method_full(query, budget):
        return list(all_ids)

    defaults = dict(exp_cfg=ExpansionConfig(), strategy="hop_decay", rank_cfg=RankingConfig())

    def method_ralc_heuristic(query, budget):
        return ralc_ids(manager, query, budget, sel_cfg=SelectionConfig(), **defaults)

    def method_relations_off(query, budget):
        return ralc_ids(manager, query, budget,
                        exp_cfg=ExpansionConfig(max_hops=0), strategy="hop_decay",
                        rank_cfg=RankingConfig(relational=0.0, entity_overlap=0.0),
                        sel_cfg=SelectionConfig())

    def method_redundancy_off(query, budget):
        return ralc_ids(manager, query, budget,
                        sel_cfg=SelectionConfig(redundancy_penalty=False), **defaults)

    def method_relation_aware(query, budget):
        return ralc_ids(manager, query, budget,
                        sel_cfg=SelectionConfig(redundancy_exempt_edge_types=EXEMPT), **defaults)

    methods = {
        "recent": method_recent,
        "vector": method_vector,
        "graph": method_graph,
        "ralc_heuristic": method_ralc_heuristic,
        "relations_off": method_relations_off,
        "redundancy_off": method_redundancy_off,
        "relation_aware": method_relation_aware,
        "full": method_full,
    }

    if gemma_manager is not None:
        def method_ralc_gemma(query, budget):
            return ralc_ids(gemma_manager, query, budget, sel_cfg=SelectionConfig(), **defaults)

        def method_ralc_gemma_relation_aware(query, budget):
            return ralc_ids(gemma_manager, query, budget,
                            sel_cfg=SelectionConfig(redundancy_exempt_edge_types=EXEMPT), **defaults)

        methods["ralc_gemma"] = method_ralc_gemma
        methods["ralc_gemma_relation_aware"] = method_ralc_gemma_relation_aware

    return methods


def run_dataset(name, embedder):
    data_dir = DATA if name == "v1" else DATA / "v2"
    messages = json.loads((data_dir / "conversation.json").read_text(encoding="utf-8"))
    questions = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
    counter = DefaultTokenCounter()
    tokens_by_id = {m["id"]: counter.count(m["content"]) for m in messages}
    all_ids = [m["id"] for m in messages]

    print(f"[{name}] building heuristic graph ...", flush=True)
    manager = ContextManager(embedder=embedder, extractor=HeuristicExtractor(), token_counter=counter)
    for m in messages:
        manager.add_message(m["role"], m["content"])

    gemma_manager = None
    gemma_misses = None
    if name == "v1":
        cache = load_gemma_cache()
        keyer = GemmaExtractor(model=GEMMA_MODEL, hybrid=False)
        gemma_extractor = CachedGemmaExtractor(cache, keyer, HeuristicExtractor())
        print(f"[{name}] building gemma graph from cache ({len(cache)} entries) ...", flush=True)
        gemma_manager = ContextManager(embedder=embedder, extractor=gemma_extractor, token_counter=counter)
        for m in messages:
            gemma_manager.add_message(m["role"], m["content"])
        gemma_misses = gemma_extractor.misses
        print(f"[{name}] gemma cache misses (heuristic fallback): {gemma_misses}", flush=True)

    methods = build_methods(manager, all_ids, tokens_by_id, gemma_manager)
    budgets = BUDGETS[name]

    # per_question[i][budget][method] = {recall, complete, tokens, selected}
    per_question = []
    for q in questions:
        gold = q["gold_message_ids"]
        entry = {"question": q["question"], "type": qtype(q), "gold": gold,
                 "results": {str(b): {} for b in budgets}}
        for b in budgets:
            for mname, fn in methods.items():
                snapshot = str(q["question"])
                ids = fn(q["question"], b)
                assert q["question"] == snapshot, f"{mname} mutated the query"
                rec = recall_metric(ids, gold)
                entry["results"][str(b)][mname] = {
                    "recall": rec,
                    "complete": 1 if rec == 1.0 else 0,
                    "tokens": sum(tokens_by_id[i] for i in set(ids)),
                    "selected": sorted(set(ids)),
                }
        per_question.append(entry)

    del manager, gemma_manager
    gc.collect()
    return {
        "method_order": list(methods.keys()),
        "budgets": budgets,
        "types": ["relationship", "lookup"],
        "counts": {t: sum(1 for q in questions if qtype(q) == t) for t in ("relationship", "lookup")},
        "gemma_cache_misses": gemma_misses,
        "per_question": per_question,
    }


def aggregate(dataset_result):
    """Per type and budget: complete-story rate, mean recall, mean tokens, and
    wins/losses vs vector on complete-story."""
    methods = dataset_result["method_order"]
    budgets = dataset_result["budgets"]
    pq = dataset_result["per_question"]
    agg = {}
    for b in budgets:
        agg[str(b)] = {}
        for t in ("relationship", "lookup", "all"):
            rows = [q for q in pq if t == "all" or q["type"] == t]
            n = len(rows)
            agg[str(b)][t] = {}
            for m in methods:
                cs = [q["results"][str(b)][m]["complete"] for q in rows]
                rc = [q["results"][str(b)][m]["recall"] for q in rows]
                tk = [q["results"][str(b)][m]["tokens"] for q in rows]
                wins = sum(1 for q in rows
                           if q["results"][str(b)][m]["complete"] == 1
                           and q["results"][str(b)]["vector"]["complete"] == 0)
                losses = sum(1 for q in rows
                             if q["results"][str(b)][m]["complete"] == 0
                             and q["results"][str(b)]["vector"]["complete"] == 1)
                agg[str(b)][t][m] = {
                    "complete_story_rate": sum(cs) / n if n else 0.0,
                    "mean_recall": sum(rc) / n if n else 0.0,
                    "mean_tokens": sum(tk) / n if n else 0.0,
                    "wins_vs_vector": wins,
                    "losses_vs_vector": losses,
                }
    return agg


def render_md(result):
    lines = ["# Relation-aware selection ablation", ""]
    lines.append(f"- generated: {result['generated_at']}")
    lines.append("")
    for name in result["datasets"]:
        d = result["datasets"][name]
        agg = d["aggregate"]
        lines.append(f"## {name} (relationship {d['counts']['relationship']}, "
                     f"lookup {d['counts']['lookup']}; gemma cache misses {d['gemma_cache_misses']})")
        lines.append("")
        for b in d["budgets"]:
            lines.append(f"### Budget {b}")
            lines.append("")
            lines.append("| method | type | complete-story | mean recall | mean tokens | win/loss vs vector |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for m in d["method_order"]:
                for t in ("relationship", "lookup"):
                    a = agg[str(b)][t][m]
                    lines.append(f"| {m} | {t} | {a['complete_story_rate']:.2f} | "
                                 f"{a['mean_recall']:.2f} | {a['mean_tokens']:.0f} | "
                                 f"{a['wins_vs_vector']}/{a['losses_vs_vector']} |")
            lines.append("")
    return "\n".join(lines)


def main():
    embedder = SentenceTransformerEmbedder()
    datasets = {}
    for name in ("v1", "v2"):
        d = run_dataset(name, embedder)
        d["aggregate"] = aggregate(d)
        datasets[name] = d

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed_k": SEED_K,
        "exempt_edge_types": list(EXEMPT),
        "datasets": datasets,
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_md(result)
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.name} and {MD_PATH.name}\n")
    print(md)


if __name__ == "__main__":
    main()
