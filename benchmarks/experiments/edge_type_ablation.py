"""Edge-type ablation: which graph relationship drives full RALC's gain.

Full RALC on the heuristic graph, then the same pipeline with one edge type
removed from the graph at a time (TEMPORALLY_FOLLOWS, ANSWERS, MENTIONS), and a
variant keeping only TEMPORALLY_FOLLOWS. Each variant's seed_k is raised so its
average candidate pool matches full RALC's, so a change cannot be blamed on a
smaller pool. Heuristic graph only, v1 and v2, no API calls.

Run from the repo root:

    python -m benchmarks.experiments.edge_type_ablation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ralc import ContextManager
from ralc.allocation.budget import DefaultTokenCounter
from ralc.allocation.selector import ContextSelector, SelectionConfig
from ralc.embeddings.local import SentenceTransformerEmbedder
from ralc.extraction import HeuristicExtractor
from ralc.graph.graph import ContextGraph
from ralc.retrieval.hybrid import HybridRanker, RankingConfig
from ralc.retrieval.relational import ExpansionConfig, RelationalExpander
from ralc.retrieval.semantic import SemanticRetriever

from benchmarks.experiments.relational_ablation import qtype
from benchmarks.metrics import recall as recall_metric

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS_PATH = HERE / "edge_type_ablation_results.json"
MD_PATH = HERE / "edge_type_ablation_results.md"

BUDGETS = {"v1": [1000, 2000, 4000], "v2": [250, 500, 1000, 2000, 4000]}
SEED_K = 10
ALL_TYPES = ["TEMPORALLY_FOLLOWS", "ANSWERS", "MENTIONS", "UPDATES"]

# (method name, kept edge types). full_ralc keeps everything at the natural
# seed_k; the rest are pool-matched.
VARIANTS = [
    ("full_ralc", ALL_TYPES),
    ("remove_TEMPORALLY_FOLLOWS", ["ANSWERS", "MENTIONS", "UPDATES"]),
    ("remove_ANSWERS", ["TEMPORALLY_FOLLOWS", "MENTIONS", "UPDATES"]),
    ("remove_MENTIONS", ["TEMPORALLY_FOLLOWS", "ANSWERS", "UPDATES"]),
    ("only_TEMPORALLY_FOLLOWS", ["TEMPORALLY_FOLLOWS"]),
]


def filtered_graph(full: ContextGraph, keep: list[str]) -> ContextGraph:
    fg = ContextGraph()
    for node in full.nodes():
        fg.add_node(node)
    for edge in full.edges():
        if edge.type in keep:
            fg.add_edge(edge)
    return fg


class Pipeline:
    """Full RALC pipeline bound to one (possibly edge-filtered) graph."""

    def __init__(self, graph, embedder, vectors, counter):
        self.graph = graph
        self.retriever = SemanticRetriever(graph, embedder)
        self.retriever._vectors = dict(vectors)   # reuse the full-graph embeddings
        self.extractor = HeuristicExtractor()
        self.counter = counter

    def pool(self, query, seed_k):
        seeds = self.retriever.retrieve(query, k=seed_k)
        return len(RelationalExpander(self.graph, config=ExpansionConfig()).expand(seeds))

    def select(self, query, budget, seed_k):
        seeds = self.retriever.retrieve(query, k=seed_k)
        candidates = RelationalExpander(self.graph, config=ExpansionConfig()).expand(seeds)
        ranked = HybridRanker(self.retriever, self.extractor, RankingConfig()).rank(query, candidates)
        selector = ContextSelector(self.graph, self.counter, SelectionConfig(),
                                   semantic_retriever=self.retriever)
        return [n.id for n in selector.select(query, ranked, budget).nodes]


def matched_seed_k(pipeline, questions, target, max_k):
    """Smallest seed_k whose average pool is nearest the target (pool grows
    monotonically with seed_k, so stop at the first value that reaches it)."""
    best_k, best_gap, prev = max_k, None, None
    for k in range(SEED_K, max_k + 1):
        avg = sum(pipeline.pool(q["question"], k) for q in questions) / len(questions)
        gap = abs(avg - target)
        if best_gap is None or gap < best_gap:
            best_k, best_gap = k, gap
        if avg >= target:
            # One step past the crossover only grows the gap; prefer the nearer
            # of this step and the previous one.
            if prev is not None and abs(prev[1] - target) <= gap:
                best_k = prev[0]
            return best_k, round(sum(pipeline.pool(q["question"], best_k) for q in questions) / len(questions), 1)
        prev = (k, avg)
    avg = sum(pipeline.pool(q["question"], best_k) for q in questions) / len(questions)
    return best_k, round(avg, 1)


def run_dataset(name, embedder):
    data_dir = DATA if name == "v1" else DATA / "v2"
    messages = json.loads((data_dir / "conversation.json").read_text(encoding="utf-8"))
    questions = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
    counter = DefaultTokenCounter()
    tokens_by_id = {m["id"]: counter.count(m["content"]) for m in messages}
    n_messages = len(messages)
    budgets = BUDGETS[name]

    print(f"[{name}] building heuristic graph ...", flush=True)
    manager = ContextManager(embedder=embedder, extractor=HeuristicExtractor(), token_counter=counter)
    for m in messages:
        manager.add_message(m["role"], m["content"])
    vectors = dict(manager.retriever._vectors)

    full_pipe = Pipeline(manager.graph, embedder, vectors, counter)
    target_pool = sum(full_pipe.pool(q["question"], SEED_K) for q in questions) / len(questions)
    print(f"[{name}] full RALC avg pool (seed_k {SEED_K}): {target_pool:.1f}", flush=True)

    method_seed_k = {}
    pipelines = {}
    for mname, keep in VARIANTS:
        pipe = full_pipe if mname == "full_ralc" else Pipeline(
            filtered_graph(manager.graph, keep), embedder, vectors, counter)
        pipelines[mname] = pipe
        if mname == "full_ralc":
            method_seed_k[mname] = (SEED_K, round(target_pool, 1))
        else:
            k, achieved = matched_seed_k(pipe, questions, target_pool, n_messages)
            method_seed_k[mname] = (k, achieved)
            print(f"[{name}] {mname}: matched seed_k {k} (avg pool {achieved})", flush=True)

    per_question = []
    for q in questions:
        gold = q["gold_message_ids"]
        entry = {"question": q["question"], "type": qtype(q), "results": {str(b): {} for b in budgets}}
        for b in budgets:
            for mname, _ in VARIANTS:
                k = method_seed_k[mname][0]
                snapshot = str(q["question"])
                ids = pipelines[mname].select(q["question"], b, k)
                assert q["question"] == snapshot, f"{mname} mutated the query"
                rec = recall_metric(ids, gold)
                entry["results"][str(b)][mname] = {
                    "recall": rec, "complete": 1 if rec == 1.0 else 0,
                    "tokens": sum(tokens_by_id[i] for i in set(ids)),
                }
        per_question.append(entry)

    return {
        "budgets": budgets,
        "method_order": [m for m, _ in VARIANTS],
        "method_seed_k": method_seed_k,
        "target_pool": round(target_pool, 1),
        "counts": {t: sum(1 for q in questions if qtype(q) == t) for t in ("relationship", "lookup")},
        "per_question": per_question,
    }


def agg(dataset, metric):
    """mean of `metric` per method, type, budget."""
    out = {}
    for b in dataset["budgets"]:
        out[str(b)] = {}
        for t in ("relationship", "lookup"):
            rows = [q for q in dataset["per_question"] if q["type"] == t]
            out[str(b)][t] = {
                m: sum(q["results"][str(b)][m][metric] for q in rows) / len(rows)
                for m, _ in VARIANTS
            }
    return out


def render_md(result):
    lines = ["# Edge-type ablation", "",
             f"- generated: {result['generated_at']}",
             "- metric columns: relationship complete-story / recall, lookup complete-story / recall",
             ""]
    for name, d in result["datasets"].items():
        cs = agg(d, "complete")
        rc = agg(d, "recall")
        sk = d["method_seed_k"]
        lines.append(f"## {name} (relationship {d['counts']['relationship']}, lookup "
                     f"{d['counts']['lookup']}; full RALC avg pool {d['target_pool']})")
        lines.append("")
        lines.append("Matched seed_k (avg pool): " + ", ".join(
            f"{m} {sk[m][0]} ({sk[m][1]})" for m, _ in VARIANTS))
        lines.append("")
        for b in d["budgets"]:
            lines.append(f"### Budget {b}")
            lines.append("")
            lines.append("| method | rel complete | rel recall | lookup complete | lookup recall |")
            lines.append("| --- | --- | --- | --- | --- |")
            for m, _ in VARIANTS:
                lines.append(f"| {m} | {cs[str(b)]['relationship'][m]:.2f} | "
                             f"{rc[str(b)]['relationship'][m]:.2f} | "
                             f"{cs[str(b)]['lookup'][m]:.2f} | {rc[str(b)]['lookup'][m]:.2f} |")
            lines.append("")
        # Which removal hurts relationship complete-story most, averaged over budgets.
        budgets = d["budgets"]
        mean_rel_cs = {m: sum(cs[str(b)]['relationship'][m] for b in budgets) / len(budgets)
                       for m, _ in VARIANTS}
        base = mean_rel_cs["full_ralc"]
        drops = {m: base - mean_rel_cs[m] for m, _ in VARIANTS if m.startswith("remove_")}
        worst = max(drops, key=drops.get)
        lines.append(f"Mean relationship complete-story over budgets: " + ", ".join(
            f"{m} {mean_rel_cs[m]:.2f}" for m, _ in VARIANTS))
        lines.append("")
        lines.append(f"Biggest drop from removing one edge type: {worst} "
                     f"(-{drops[worst]:.2f} vs full RALC {base:.2f}).")
        lines.append("")
    return "\n".join(lines)


def main():
    embedder = SentenceTransformerEmbedder()
    datasets = {name: run_dataset(name, embedder) for name in ("v1", "v2")}
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "datasets": datasets}
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_md(result)
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.name} and {MD_PATH.name}\n")
    print(md)


if __name__ == "__main__":
    main()
