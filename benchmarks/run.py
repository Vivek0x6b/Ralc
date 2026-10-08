"""Run the RALC retrieval benchmark and write JSON plus a markdown table.

Every number written comes from this actual run. Nothing is hand-edited.

To keep peak memory down, methods are built and evaluated one group at a time
and released before the next group. Warm the Gemma cache first with
benchmarks/warm_gemma_cache.py so this run makes no API calls.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from ralc import ContextManager
from ralc.allocation.budget import DefaultTokenCounter
from ralc.embeddings.local import SentenceTransformerEmbedder
from ralc.extraction import GemmaExtractor, HeuristicExtractor, Message
from ralc.retrieval.relational import RelationalExpander

from benchmarks.gemma_cache import (
    GEMMA_MODEL,
    RESULTS,
    extraction_texts,
    load_failures,
    load_gemma_cache,
)
from benchmarks.metrics import precision, recall, tokens_used

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BUDGETS = [1000, 2000, 4000]
SEED_K = 10
CANONICAL_METHODS = ["recent", "vector", "graph", "ralc_heuristic",
                     "ralc_gemma", "ralc_hybrid", "full"]
GEMMA_METHODS = {"ralc_gemma", "ralc_hybrid"}


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
                print(f"    transient {type(exc).__name__} on {label}; "
                      f"retry {attempt + 1}/5", flush=True)
                time.sleep(2 * (attempt + 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the RALC retrieval benchmark.")
    parser.add_argument(
        "--methods",
        help="comma separated subset to run, e.g. "
             "recent,vector,graph,ralc_heuristic,full. Default: all available. "
             "gemma_cache.json is read or written only when a gemma method is selected.",
    )
    args = parser.parse_args(argv)

    messages = json.loads((DATA / "conversation.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))

    counter = DefaultTokenCounter()
    tokens_by_id = {m["id"]: counter.count(m["content"]) for m in messages}
    all_ids = [m["id"] for m in messages]
    recent_order = list(reversed(all_ids))
    total_tokens = sum(tokens_by_id.values())

    embedder = SentenceTransformerEmbedder()
    has_key = bool(os.environ.get("GEMINI_API_KEY"))

    available = [m for m in CANONICAL_METHODS if m not in GEMMA_METHODS or has_key]
    if args.methods:
        requested = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in requested if m not in CANONICAL_METHODS]
        if unknown:
            parser.error(f"unknown methods {unknown}; choose from {CANONICAL_METHODS}")
        need_key = [m for m in requested if m in GEMMA_METHODS and not has_key]
        if need_key:
            parser.error(f"methods {need_key} require GEMINI_API_KEY to be set")
        selected = [m for m in CANONICAL_METHODS if m in requested]
    else:
        selected = list(available)
    selected_set = set(selected)
    method_order = selected
    print(f"methods: {method_order}", flush=True)

    per_question = [
        {"question": q["question"], "gold_message_ids": q["gold_message_ids"],
         "results": {str(b): {} for b in BUDGETS}}
        for q in questions
    ]
    aggregate = {b: {name: {"recall": [], "precision": [], "tokens": [], "latency": []}
                     for name in method_order} for b in BUDGETS}
    gemma_stats = {}
    heuristic_fallbacks = {}

    def evaluate(name, fn):
        print(f"  evaluating {name} ...", flush=True)
        for entry, q in zip(per_question, questions):
            gold = q["gold_message_ids"]
            for budget in BUDGETS:
                query = q["question"]
                snapshot = str(query)
                start = time.perf_counter()
                selected_ids = fn(query, budget)
                latency = time.perf_counter() - start
                assert query == snapshot, "the query string was mutated by a method"
                entry["results"][str(budget)][name] = {
                    "recall": recall(selected_ids, gold),
                    "precision": precision(selected_ids, gold),
                    "tokens": tokens_used(selected_ids, tokens_by_id),
                    "latency": latency,
                    "selected": sorted(selected_ids),
                }
                for metric in ("recall", "precision", "tokens", "latency"):
                    aggregate[budget][name][metric].append(entry["results"][str(budget)][name][metric])

    def make_ralc(manager):
        def run(query, budget):
            return [n.id for n in manager.retrieve(query, token_budget=budget, seed_k=SEED_K).nodes]
        return run

    if "recent" in selected_set:
        def method_recent(query, budget):
            return fill_to_budget(recent_order, tokens_by_id, budget)
        evaluate("recent", method_recent)

    # The heuristic graph backs vector, graph, and ralc_heuristic.
    if selected_set & {"vector", "graph", "ralc_heuristic"}:
        print("Group: heuristic graph", flush=True)
        heuristic_mgr = ContextManager(embedder=embedder, extractor=HeuristicExtractor(),
                                       token_counter=counter)
        ingest_all(heuristic_mgr, messages, "ralc_heuristic")
        retriever = heuristic_mgr.retriever
        graph_expander = RelationalExpander(heuristic_mgr.graph, strategy="hop_decay")

        def method_vector(query, budget):
            scored = retriever.retrieve(query, k=len(all_ids))
            return fill_to_budget([nid for nid, _ in scored], tokens_by_id, budget)

        def method_graph(query, budget):
            seeds = retriever.retrieve(query, k=SEED_K)
            candidates = graph_expander.expand(seeds)
            ordered = [c.node_id for c in sorted(candidates, key=lambda c: (-c.score, c.node_id))]
            return fill_to_budget(ordered, tokens_by_id, budget)

        if "vector" in selected_set:
            evaluate("vector", method_vector)
        if "graph" in selected_set:
            evaluate("graph", method_graph)
        if "ralc_heuristic" in selected_set:
            evaluate("ralc_heuristic", make_ralc(heuristic_mgr))
        del heuristic_mgr, retriever, graph_expander
        gc.collect()

    # Only read or write gemma_cache.json when a Gemma method is selected, so a
    # concurrent warm-up that owns that file is never disturbed.
    if selected_set & GEMMA_METHODS:
        cache = load_gemma_cache()          # read-only here; the warm-up owns the file
        failures = load_failures()
        texts = extraction_texts(messages, questions)
        heuristic = HeuristicExtractor()
        for name, hybrid in (("ralc_gemma", False), ("ralc_hybrid", True)):
            if name not in selected_set:
                continue
            print(f"Group: {name}", flush=True)
            extractor = GemmaExtractor(model=GEMMA_MODEL, hybrid=hybrid, strict=True,
                                       cache=cache, max_retries=5, retry_base_delay=2.0)
            # Use the heuristic for exactly the strings the warm-up could not get
            # from Gemma, by seeding their cache entries in memory so the strict
            # extractor serves them without an API call.
            mode = "hybrid" if hybrid else "gemma"
            substituted = 0
            for failure in failures:
                if failure["mode"] != mode:
                    continue
                content = texts[failure["index"]]
                key = extractor.cache_key(content)
                if key not in cache:
                    cache[key] = heuristic.extract(Message("user", content))
                    substituted += 1
            heuristic_fallbacks[name] = substituted
            print(f"  heuristic substituted for {substituted} failed strings", flush=True)
            manager = ContextManager(embedder=embedder, extractor=extractor, token_counter=counter)
            ingest_all(manager, messages, name)
            evaluate(name, make_ralc(manager))
            gemma_stats[name] = dict(extractor.stats)
            del manager, extractor
            gc.collect()

    if "full" in selected_set:
        evaluate("full", lambda query, budget: list(all_ids))

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
        "gemma_stats": gemma_stats,
        "heuristic_fallbacks": heuristic_fallbacks,
        "methods": method_order,
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
    lines.append(f"- methods: {', '.join(result['methods'])}")
    lines.append(f"- dataset: {result['dataset']['messages']} messages, "
                 f"{result['dataset']['tokens']} tokens, {result['dataset']['questions']} questions")
    lines.append(f"- embedder: {result['models']['embedder']}; gemma: {result['models']['gemma']}")
    lines.append(f"- token counter approximate: {result['token_counter_approximate']}")
    lines.append(f"- versions: {result['versions']}")
    if result["gemma_stats"]:
        lines.append(f"- gemma stats: {result['gemma_stats']}")
    if result.get("heuristic_fallbacks"):
        lines.append(f"- heuristic fallbacks (failed warm strings): {result['heuristic_fallbacks']}")
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
            cells = [f"{entry['results'][str(budget)][name]['recall']:.2f}" for name in methods]
            question = entry["question"].replace("|", "/")
            lines.append(f"| {question} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
