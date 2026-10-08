"""Run the RALC retrieval benchmark and write JSON plus a markdown table.

Every number written comes from this actual run. Nothing is hand-edited.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from ralc import ContextManager
from ralc.allocation.budget import DefaultTokenCounter
from ralc.embeddings.local import SentenceTransformerEmbedder
from ralc.extraction import GemmaExtractor, HeuristicExtractor, Signals
from ralc.retrieval.relational import RelationalExpander

from benchmarks.metrics import precision, recall, tokens_used

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
GEMMA_CACHE = RESULTS / "gemma_cache.json"
BUDGETS = [1000, 2000, 4000]
SEED_K = 10
GEMMA_MODEL = "gemma-4-26b-a4b-it"


# -- Gemma disk cache (kept in the harness; GemmaExtractor is unchanged) -----

def load_gemma_cache() -> dict:
    if not GEMMA_CACHE.exists():
        return {}
    raw = json.loads(GEMMA_CACHE.read_text(encoding="utf-8"))
    return {k: Signals(entities=v["entities"], is_decision=v["is_decision"]) for k, v in raw.items()}


def save_gemma_cache(cache: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    raw = {k: {"entities": list(s.entities), "is_decision": bool(s.is_decision)}
           for k, s in cache.items()}
    GEMMA_CACHE.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def fill_to_budget(ordered_ids, tokens_by_id, budget):
    selected, remaining = [], budget
    for node_id in ordered_ids:
        cost = tokens_by_id[node_id]
        if cost <= remaining:
            selected.append(node_id)
            remaining -= cost
    return selected


def ingest_all(manager, messages, label):
    print(f"  ingesting {len(messages)} messages into {label} ...", flush=True)
    for message in messages:
        for attempt in range(5):
            try:
                manager.add_message(message["role"], message["content"])
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                # Transient API errors (for example a 500) should not abort the
                # whole run; strictness is preserved, we just retry the message.
                print(f"    transient {type(exc).__name__} on {label}; "
                      f"retry {attempt + 1}/5", flush=True)
                time.sleep(2 * (attempt + 1))


def main():
    messages = json.loads((DATA / "conversation.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))

    counter = DefaultTokenCounter()
    tokens_by_id = {m["id"]: counter.count(m["content"]) for m in messages}
    all_ids = [m["id"] for m in messages]
    recent_order = list(reversed(all_ids))
    total_tokens = sum(tokens_by_id.values())

    embedder = SentenceTransformerEmbedder()
    has_key = bool(os.environ.get("GEMINI_API_KEY"))

    print("Building managers (this performs Gemma API calls on the first run):", flush=True)
    heuristic_mgr = ContextManager(embedder=embedder, extractor=HeuristicExtractor(),
                                   token_counter=counter)
    ingest_all(heuristic_mgr, messages, "ralc_heuristic")

    gemma_cache = load_gemma_cache()
    gemma_extractors = {}
    managers = {"ralc_heuristic": heuristic_mgr}
    if has_key:
        gemma_ex = GemmaExtractor(model=GEMMA_MODEL, strict=True, cache=gemma_cache,
                                  max_retries=6, retry_base_delay=2.0)
        gemma_mgr = ContextManager(embedder=embedder, extractor=gemma_ex, token_counter=counter)
        ingest_all(gemma_mgr, messages, "ralc_gemma")
        save_gemma_cache(gemma_cache)   # persist progress before the next manager
        managers["ralc_gemma"] = gemma_mgr
        gemma_extractors["ralc_gemma"] = gemma_ex

        hybrid_ex = GemmaExtractor(model=GEMMA_MODEL, hybrid=True, strict=True, cache=gemma_cache,
                                   max_retries=6, retry_base_delay=2.0)
        hybrid_mgr = ContextManager(embedder=embedder, extractor=hybrid_ex, token_counter=counter)
        ingest_all(hybrid_mgr, messages, "ralc_hybrid")
        managers["ralc_hybrid"] = hybrid_mgr
        gemma_extractors["ralc_hybrid"] = hybrid_ex
        save_gemma_cache(gemma_cache)
    else:
        print("  GEMINI_API_KEY not set: skipping ralc_gemma and ralc_hybrid.", flush=True)

    # Baselines reuse the heuristic manager's graph, retriever, and expander.
    retriever = heuristic_mgr.retriever
    graph_expander = RelationalExpander(heuristic_mgr.graph, strategy="hop_decay")

    def method_recent(query, budget):
        return fill_to_budget(recent_order, tokens_by_id, budget)

    def method_vector(query, budget):
        scored = retriever.retrieve(query, k=len(all_ids))
        return fill_to_budget([nid for nid, _ in scored], tokens_by_id, budget)

    def method_graph(query, budget):
        seeds = retriever.retrieve(query, k=SEED_K)
        candidates = graph_expander.expand(seeds)
        ordered = [c.node_id for c in sorted(candidates, key=lambda c: (-c.score, c.node_id))]
        return fill_to_budget(ordered, tokens_by_id, budget)

    def make_ralc(manager):
        def run(query, budget):
            return [n.id for n in manager.retrieve(query, token_budget=budget, seed_k=SEED_K).nodes]
        return run

    methods = {
        "recent": method_recent,
        "vector": method_vector,
        "graph": method_graph,
        "ralc_heuristic": make_ralc(managers["ralc_heuristic"]),
    }
    if has_key:
        methods["ralc_gemma"] = make_ralc(managers["ralc_gemma"])
        methods["ralc_hybrid"] = make_ralc(managers["ralc_hybrid"])
    methods["full"] = lambda query, budget: list(all_ids)

    per_question = []
    aggregate = {b: {name: {"recall": [], "precision": [], "tokens": [], "latency": []}
                     for name in methods} for b in BUDGETS}

    print("Running retrieval ...", flush=True)
    for q in questions:
        gold = q["gold_message_ids"]
        entry = {"question": q["question"], "gold_message_ids": gold, "results": {}}
        for budget in BUDGETS:
            entry["results"][str(budget)] = {}
            for name, fn in methods.items():
                query = q["question"]
                snapshot = str(query)
                start = time.perf_counter()
                selected = fn(query, budget)
                latency = time.perf_counter() - start
                assert query == snapshot, "the query string was mutated by a method"

                r = recall(selected, gold)
                p = precision(selected, gold)
                used = tokens_used(selected, tokens_by_id)
                entry["results"][str(budget)][name] = {
                    "recall": r, "precision": p, "tokens": used,
                    "latency": latency, "selected": sorted(selected),
                }
                aggregate[budget][name]["recall"].append(r)
                aggregate[budget][name]["precision"].append(p)
                aggregate[budget][name]["tokens"].append(used)
                aggregate[budget][name]["latency"].append(latency)
        per_question.append(entry)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    aggregate_means = {
        str(b): {name: {k: mean(v[k]) for k in ("recall", "precision", "tokens", "latency")}
                 for name, v in aggregate[b].items()}
        for b in BUDGETS
    }

    versions = {}
    for pkg in ("ralc", "networkx", "numpy", "sentence-transformers", "google-genai", "tiktoken"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {"embedder": embedder.model_name, "gemma": GEMMA_MODEL if has_key else None},
        "versions": versions,
        "dataset": {"messages": len(messages), "tokens": total_tokens, "questions": len(questions)},
        "budgets": BUDGETS,
        "seed_k": SEED_K,
        "token_counter_approximate": counter.approximate,
        "gemma_stats": {name: dict(ex.stats) for name, ex in gemma_extractors.items()},
        "methods": list(methods),
        "aggregate_means": aggregate_means,
        "per_question": per_question,
    }

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RESULTS / f"{stamp}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (RESULTS / f"{stamp}.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"\nWrote results/{stamp}.json and results/{stamp}.md")
    print("\n" + render_markdown(result))


def render_markdown(result) -> str:
    methods = result["methods"]
    lines = ["# RALC benchmark results", ""]
    lines.append(f"- generated: {result['generated_at']}")
    lines.append(f"- dataset: {result['dataset']['messages']} messages, "
                 f"{result['dataset']['tokens']} tokens, {result['dataset']['questions']} questions")
    lines.append(f"- embedder: {result['models']['embedder']}; gemma: {result['models']['gemma']}")
    lines.append(f"- token counter approximate: {result['token_counter_approximate']}")
    lines.append(f"- versions: {result['versions']}")
    if result["gemma_stats"]:
        lines.append(f"- gemma stats: {result['gemma_stats']}")
    lines.append("")

    for budget in result["budgets"]:
        means = result["aggregate_means"][str(budget)]
        lines.append(f"## Budget {budget} tokens (means over {result['dataset']['questions']} questions)")
        lines.append("")
        lines.append("| method | recall | precision | tokens | latency (s) |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name in methods:
            m = means[name]
            lines.append(f"| {name} | {m['recall']:.3f} | {m['precision']:.3f} | "
                         f"{m['tokens']:.0f} | {m['latency']:.4f} |")
        lines.append("")

    lines.append("## Per-question recall")
    lines.append("")
    for budget in result["budgets"]:
        lines.append(f"### Budget {budget}")
        lines.append("")
        lines.append("| question | " + " | ".join(methods) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in methods) + " |")
        for entry in result["per_question"]:
            cells = []
            for name in methods:
                cells.append(f"{entry['results'][str(budget)][name]['recall']:.2f}")
            question = entry["question"].replace("|", "/")
            lines.append(f"| {question} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
