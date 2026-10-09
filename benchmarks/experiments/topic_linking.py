"""Topic-linking report (see benchmarks/PREREGISTRATION_topic_linking.md).

From the warmed gemma_cache.json (no API calls): build the Gemma graph for v1
and v2 two ways, entity-only linking and entity-or-topic linking, on the same
extracted signals. Reports total UPDATES edges, which of the four decision pairs
are linked and why, the is_decision/topic of the eight decision-pair messages
with per-pair topic similarity at thresholds 0.50/0.60/0.70, a scan of every
topic-created UPDATES edge for false links, and complete-story by question type
with relation-aware selection for entity-only vs entity-or-topic (threshold 0.60,
the primary). Writes a human-review file of every UPDATES edge.

Run from the repo root after re-extraction:

    python -m benchmarks.experiments.topic_linking
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ralc import ContextManager
from ralc.allocation.budget import DefaultTokenCounter
from ralc.allocation.selector import SelectionConfig
from ralc.embeddings.local import SentenceTransformerEmbedder
from ralc.extraction import HeuristicExtractor
from ralc.extraction.gemma import GemmaExtractor
from ralc.retrieval.hybrid import RankingConfig
from ralc.retrieval.relational import ExpansionConfig

from benchmarks.experiments.relational_ablation import CachedGemmaExtractor, qtype, ralc_ids
from benchmarks.gemma_cache import GEMMA_MODEL, load_gemma_cache
from benchmarks.metrics import recall as recall_metric

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS_PATH = HERE / "topic_linking_results.json"
MD_PATH = HERE / "topic_linking_results.md"

BUDGETS = {"v1": [1000, 2000, 4000], "v2": [250, 500, 1000, 2000, 4000]}
SEED_K = 10
THRESHOLDS = [0.50, 0.60, 0.70]
PRIMARY = 0.60
EXEMPT = ("UPDATES", "ANSWERS")

# The decision pairs the questions depend on (fixed by the datasets).
PAIRS = {
    "v1": {"database": ("m10", "m100"), "hosting": ("m33", "m137")},
    "v2": {"database": ("m8", "m90"), "hosting": ("m29", "m238")},
}


def build_manager(messages, embedder, counter, cache, keyer, link_topics, threshold):
    extractor = CachedGemmaExtractor(cache, keyer, HeuristicExtractor())
    manager = ContextManager(embedder=embedder, extractor=extractor, token_counter=counter,
                             link_topics=link_topics, topic_threshold=threshold)
    for m in messages:
        manager.add_message(m["role"], m["content"])
    return manager, extractor


def updates_between(graph, a, b):
    for e in graph.edges(type="UPDATES"):
        if {e.source, e.target} == {a, b}:
            return e
    return None


def topic_similarity(embedder, ta, tb):
    if not ta or not tb:
        return None
    va = np.asarray(embedder.embed([ta])[0], dtype=float)
    vb = np.asarray(embedder.embed([tb])[0], dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return None
    return float((va / na) @ (vb / nb))


def complete_story_by_type(manager, questions, budgets):
    out = {}
    sel = SelectionConfig(redundancy_exempt_edge_types=EXEMPT)
    for b in budgets:
        out[str(b)] = {}
        for t in ("relationship", "lookup"):
            rows = [q for q in questions if qtype(q) == t]
            complete = 0
            for q in rows:
                ids = ralc_ids(manager, q["question"], b, exp_cfg=ExpansionConfig(),
                               strategy="hop_decay", rank_cfg=RankingConfig(), sel_cfg=sel)
                if recall_metric(ids, q["gold_message_ids"]) == 1.0:
                    complete += 1
            out[str(b)][t] = complete / len(rows) if rows else 0.0
    return out


def write_review_file(path, graph, by_id):
    lines = []
    for e in sorted(graph.edges(type="UPDATES"), key=lambda e: (e.source, e.target)):
        reason = e.metadata.get("reason", "?")
        if reason == "entity":
            why = f"shared entity: {e.metadata.get('shared_entity')}"
        elif reason == "topic":
            why = (f"topic match (similarity {e.metadata.get('similarity'):.3f}): "
                   f"{e.metadata.get('topics')}")
        else:
            why = reason
        lines.append(f"UPDATES {e.source} -> {e.target} | {why}")
        lines.append(f"  [{e.source}] {by_id[e.source]['content']}")
        lines.append(f"  [{e.target}] {by_id[e.target]['content']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_dataset(name, embedder, cache, keyer):
    data_dir = DATA / "v2" if name == "v2" else DATA
    messages = json.loads((data_dir / "conversation.json").read_text(encoding="utf-8"))
    questions = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in messages}
    counter = DefaultTokenCounter()
    budgets = BUDGETS[name]

    print(f"[{name}] building entity-only and topic graphs ...", flush=True)
    entity_mgr, entity_ex = build_manager(messages, embedder, counter, cache, keyer, False, PRIMARY)
    topic_mgrs = {}
    for thr in THRESHOLDS:
        topic_mgrs[thr], _ = build_manager(messages, embedder, counter, cache, keyer, True, thr)

    heuristic_fallbacks = entity_ex.misses

    # UPDATES counts.
    updates_counts = {"entity_only": len(entity_mgr.graph.edges(type="UPDATES"))}
    for thr in THRESHOLDS:
        updates_counts[f"topic_{thr:.2f}"] = len(topic_mgrs[thr].graph.edges(type="UPDATES"))

    # The four pairs: linked or not, and why, under entity-only and topic@primary.
    pair_report = {}
    for label, (a, b) in PAIRS[name].items():
        ta = topic_mgrs[PRIMARY].graph.get(a).metadata.get("topic")
        tb = topic_mgrs[PRIMARY].graph.get(b).metadata.get("topic")
        sim = topic_similarity(embedder, ta, tb)
        entry = {
            "pair": [a, b],
            "topics": [ta, tb],
            "topic_similarity": None if sim is None else round(sim, 3),
            "clears_threshold": {f"{thr:.2f}": (sim is not None and sim >= thr) for thr in THRESHOLDS},
        }
        for mode, graph in (("entity_only", entity_mgr.graph), (f"topic_{PRIMARY:.2f}", topic_mgrs[PRIMARY].graph)):
            e = updates_between(graph, a, b)
            entry[mode] = None if e is None else {
                "reason": e.metadata.get("reason"),
                "shared_entity": e.metadata.get("shared_entity"),
                "similarity": (round(e.metadata["similarity"], 3) if "similarity" in e.metadata else None),
            }
        pair_report[label] = entry

    # is_decision and topic for the eight decision-pair messages.
    decision_messages = {}
    for label, (a, b) in PAIRS[name].items():
        for nid in (a, b):
            node = entity_mgr.graph.get(nid)
            decision_messages[nid] = {"is_decision": node.metadata.get("is_decision"),
                                      "topic": node.metadata.get("topic")}

    # False-link scan: every topic-created UPDATES edge at the primary threshold.
    topic_edges = []
    for e in topic_mgrs[PRIMARY].graph.edges(type="UPDATES"):
        if e.metadata.get("reason") == "topic":
            topic_edges.append({
                "edge": [e.source, e.target],
                "similarity": round(e.metadata["similarity"], 3),
                "topics": e.metadata.get("topics"),
                "source_preview": by_id[e.source]["content"][:70],
                "target_preview": by_id[e.target]["content"][:70],
            })

    write_review_file(HERE / f"topic_updates_review_{name}.txt", topic_mgrs[PRIMARY].graph, by_id)

    # Complete-story by type with relation-aware selection.
    cs_entity = complete_story_by_type(entity_mgr, questions, budgets)
    cs_topic = complete_story_by_type(topic_mgrs[PRIMARY], questions, budgets)

    return {
        "budgets": budgets,
        "counts": {t: sum(1 for q in questions if qtype(q) == t) for t in ("relationship", "lookup")},
        "heuristic_fallbacks": heuristic_fallbacks,
        "updates_counts": updates_counts,
        "pairs": pair_report,
        "decision_messages": decision_messages,
        "topic_edges": topic_edges,
        "complete_story": {"entity_only": cs_entity, f"topic_{PRIMARY:.2f}": cs_topic},
    }


def render_md(result):
    lines = ["# Topic-linking report", "",
             f"- generated: {result['generated_at']}",
             f"- primary threshold: {PRIMARY}; sensitivity: {THRESHOLDS}",
             ""]
    for name, d in result["datasets"].items():
        lines.append(f"## {name} (relationship {d['counts']['relationship']}, lookup "
                     f"{d['counts']['lookup']}; heuristic fallbacks {d['heuristic_fallbacks']})")
        lines.append("")
        lines.append("UPDATES edge counts: " + ", ".join(
            f"{k} {v}" for k, v in d["updates_counts"].items()))
        lines.append("")
        lines.append("Decision pairs:")
        for label, e in d["pairs"].items():
            el = e["entity_only"]
            tl = e[f"topic_{PRIMARY:.2f}"]
            lines.append(f"- {label} {e['pair']}: topic_similarity={e['topic_similarity']} "
                         f"topics={e['topics']}")
            lines.append(f"    entity_only link: {el}")
            lines.append(f"    topic@{PRIMARY:.2f} link: {tl}")
            lines.append(f"    clears threshold: {e['clears_threshold']}")
        lines.append("")
        lines.append("Eight decision-pair messages (is_decision, topic):")
        for nid, v in d["decision_messages"].items():
            lines.append(f"- {nid}: is_decision={v['is_decision']} topic={v['topic']!r}")
        lines.append("")
        lines.append(f"Topic-created UPDATES edges at {PRIMARY:.2f} (for false-link review): "
                     f"{len(d['topic_edges'])}")
        for te in d["topic_edges"]:
            lines.append(f"- {te['edge']} sim={te['similarity']} topics={te['topics']}")
            lines.append(f"    [{te['edge'][0]}] {te['source_preview']}")
            lines.append(f"    [{te['edge'][1]}] {te['target_preview']}")
        lines.append("")
        lines.append("Complete-story by type with relation-aware selection:")
        lines.append("")
        lines.append("| budget | entity rel | topic rel | entity lookup | topic lookup |")
        lines.append("| --- | --- | --- | --- | --- |")
        eo = d["complete_story"]["entity_only"]
        to = d["complete_story"][f"topic_{PRIMARY:.2f}"]
        for b in d["budgets"]:
            lines.append(f"| {b} | {eo[str(b)]['relationship']:.2f} | {to[str(b)]['relationship']:.2f} | "
                         f"{eo[str(b)]['lookup']:.2f} | {to[str(b)]['lookup']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    embedder = SentenceTransformerEmbedder()
    cache = load_gemma_cache()
    keyer = GemmaExtractor(model=GEMMA_MODEL, hybrid=False)
    datasets = {name: run_dataset(name, embedder, cache, keyer) for name in ("v1", "v2")}
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "datasets": datasets}
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_md(result)
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.name} and {MD_PATH.name}\n")
    print(md)


if __name__ == "__main__":
    main()
