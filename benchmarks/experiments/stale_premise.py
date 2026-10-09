"""Stale-context premise experiment (see PREREGISTRATION_stale.md).

Does a superseded decision in the context make Gemma answer a current-state
question with the old decision instead of the current one? No RALC code is
changed: this only reads the v2 conversation and reuses the existing retrieval
and rendering code paths.

Run from the repo root with GEMINI_API_KEY set:

    python -m benchmarks.experiments.stale_premise

Answers are cached to disk so the run is resumable and re-runs make no new API
calls. Every answer is kept verbatim; the automatic labels are advisory only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from ralc import ContextManager
from ralc.allocation.budget import DefaultTokenCounter
from ralc.embeddings.local import SentenceTransformerEmbedder
from ralc.extraction import HeuristicExtractor
from ralc.extraction.gemma import GemmaExtractor

from benchmarks.run import fill_to_budget

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "v2"
RESULTS_PATH = HERE / "stale_premise_results.json"
CACHE_PATH = HERE / "stale_premise_cache.json"

GEMMA_MODEL = "gemma-4-26b-a4b-it"
TEMPERATURE = 0
TIMEOUT_SECONDS = 45.0
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
REPS = 3
SEED_K = 10
D_BUDGETS = [250, 500]

PROMPT_TEMPLATE = (
    "You are answering a question using only the chat messages below.\n"
    "Read the messages, then answer the question in one or two sentences.\n"
    "If the messages do not contain the answer, say you do not know.\n\n"
    "Messages:\n{context}\n\n"
    "Question: {question}"
)

QUESTIONS = [
    {"id": "Q1", "topic": "db", "type": "current",
     "text": "Which database are we using for the main data store now?",
     "new_marker": "postgres", "old_marker": "mongo"},
    {"id": "Q2", "topic": "db", "type": "history",
     "text": "What database did we originally plan to use before we switched?",
     "new_marker": "postgres", "old_marker": "mongo"},
    {"id": "Q3", "topic": "hosting", "type": "current",
     "text": "Where are we deploying the app now?",
     "new_marker": "aws", "old_marker": "heroku"},
    {"id": "Q4", "topic": "hosting", "type": "history",
     "text": "What did we use for hosting before we changed it?",
     "new_marker": "aws", "old_marker": "heroku"},
]

# Hand-built contexts, original message ids only, in conversation order.
HANDBUILT = {
    "db": {"A": ["m90"], "B": ["m8", "m90"], "C": ["m8"]},
    "hosting": {"A": ["m238"], "B": ["m29", "m238"], "C": ["m29"]},
}

# Condition D bookkeeping: which ids count as the current decision, the old
# decision, and the topic near-miss distractors.
NEW_DECISION_IDS = {"db": {"m90", "m91"}, "hosting": {"m238"}}
OLD_DECISION_IDS = {"db": {"m8"}, "hosting": {"m29"}}
NEAR_MISS_IDS = {"db": {"m10", "m81", "m115"}, "hosting": {"m20", "m30", "m124"}}


def render(ids, by_id):
    """Render messages in the RALC to_text style: '{role}: {content}', blank
    line between. ``ids`` must already be in conversation order."""
    return "\n\n".join(f"{by_id[i]['role']}: {by_id[i]['content']}" for i in ids)


def context_entry(topic, ids, by_id, order_index):
    ordered = sorted(set(ids), key=lambda i: order_index[i])
    id_set = set(ids)
    return {
        "ids": ordered,
        "contains_new_decision": bool(id_set & NEW_DECISION_IDS[topic]),
        "contains_old_decision": bool(id_set & OLD_DECISION_IDS[topic]),
        "topic_near_miss_ids": sorted(id_set & NEAR_MISS_IDS[topic], key=lambda i: order_index[i]),
        "text": render(ordered, by_id),
    }


def label(answer, question):
    """Advisory bucket for one answer. Hand review is the real result."""
    low = answer.lower()
    new_hit = question["new_marker"] in low
    old_hit = question["old_marker"] in low
    if new_hit and old_hit:
        bucket = "both"
    elif new_hit:
        bucket = "new_only"
    elif old_hit:
        bucket = "old_only"
    else:
        bucket = "neither"

    if question["type"] == "current":
        verdict = {"new_only": "correct(new)", "both": "correct(both)",
                   "old_only": "STALE(old)", "neither": "unclear"}[bucket]
    else:  # history: the old decision is the correct answer
        verdict = {"old_only": "correct(old)", "both": "correct(both)",
                   "new_only": "ANACHRONISM(new)", "neither": "unclear"}[bucket]
    return {"new_marker_hit": new_hit, "old_marker_hit": old_hit,
            "bucket": bucket, "verdict": verdict}


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _get_client():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key,
                        http_options={"timeout": int(TIMEOUT_SECONDS * 1000)})


def call_gemma(client, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMMA_MODEL, contents=prompt, config={"temperature": TEMPERATURE},
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("empty or blocked response from the model")
            return text.strip()
        except Exception as exc:
            if GemmaExtractor._is_retryable(exc) and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            raise


def cached_answer(cache, client, prompt, rep):
    raw = f"{GEMMA_MODEL}|{TEMPERATURE}|{rep}|{prompt}"
    key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key], True
    answer = call_gemma(client, prompt)
    cache[key] = answer
    save_cache(cache)
    return answer, False


def build_d_contexts(messages, by_id, order_index, tokens_by_id):
    """The real vector and ralc_heuristic selections for Q1 and Q3, using the
    identical code path as benchmarks/run.py."""
    print("Building condition D contexts (loading embedder, ingesting v2) ...", flush=True)
    embedder = SentenceTransformerEmbedder()
    manager = ContextManager(embedder=embedder, extractor=HeuristicExtractor(),
                             token_counter=DefaultTokenCounter())
    for message in messages:
        manager.add_message(message["role"], message["content"])
    retriever = manager.retriever
    all_ids = [m["id"] for m in messages]

    out = {}
    for question in QUESTIONS:
        if question["type"] != "current":
            continue
        query = question["text"]
        topic = question["topic"]
        scored = retriever.retrieve(query, k=len(all_ids))
        vector_order = [nid for nid, _ in scored]
        contexts = {}
        for budget in D_BUDGETS:
            vector_ids = fill_to_budget(vector_order, tokens_by_id, budget)
            ralc_ids = [n.id for n in
                        manager.retrieve(query, token_budget=budget, seed_k=SEED_K).nodes]
            contexts[f"D:vector@{budget}"] = context_entry(topic, vector_ids, by_id, order_index)
            contexts[f"D:ralc_heuristic@{budget}"] = context_entry(topic, ralc_ids, by_id, order_index)
        out[question["id"]] = contexts
    return out


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set; this experiment needs it to call Gemma.")

    messages = json.loads((DATA / "conversation.json").read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in messages}
    order_index = {m["id"]: i for i, m in enumerate(messages)}
    counter = DefaultTokenCounter()
    tokens_by_id = {m["id"]: counter.count(m["content"]) for m in messages}

    d_contexts = build_d_contexts(messages, by_id, order_index, tokens_by_id)

    # Assemble every context per question: hand-built A/B/C, plus D for the
    # current-state questions.
    contexts_by_q = {}
    for question in QUESTIONS:
        qid, topic = question["id"], question["topic"]
        entries = {}
        for cond in ("A", "B", "C"):
            entries[cond] = context_entry(topic, HANDBUILT[topic][cond], by_id, order_index)
        if qid in d_contexts:
            entries.update(d_contexts[qid])
        contexts_by_q[qid] = entries

    cache = load_cache()
    client = _get_client()
    runs = []
    for question in QUESTIONS:
        qid = question["id"]
        for cond, entry in contexts_by_q[qid].items():
            prompt = PROMPT_TEMPLATE.format(context=entry["text"], question=question["text"])
            for rep in range(REPS):
                answer, hit = cached_answer(cache, client, prompt, rep)
                marks = label(answer, question)
                runs.append({
                    "question_id": qid, "type": question["type"], "topic": topic,
                    "condition": cond, "rep": rep, "answer": answer,
                    **marks,
                })
                print(f"  {qid} {cond} rep{rep} [{'cache' if hit else 'api'}] "
                      f"{marks['verdict']}: {answer[:70]}", flush=True)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": GEMMA_MODEL,
        "settings": {"temperature": TEMPERATURE, "timeout_seconds": TIMEOUT_SECONDS,
                     "max_retries": MAX_RETRIES, "reps": REPS, "seed_k": SEED_K,
                     "d_budgets": D_BUDGETS},
        "prompt_template": PROMPT_TEMPLATE,
        "questions": QUESTIONS,
        "contexts": contexts_by_q,
        "runs": runs,
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.relative_to(HERE.parent.parent)}")
    print("\n" + render_table(result))


def render_table(result):
    runs = result["runs"]
    contexts = result["contexts"]
    lines = ["# Stale-context premise: answers", ""]
    lines.append(f"- model: {result['model']}, temperature {result['settings']['temperature']}, "
                 f"{result['settings']['reps']} reps")
    lines.append("")
    for question in result["questions"]:
        qid = question["id"]
        lines.append(f"## {qid} ({question['type']}, {question['topic']}): {question['text']}")
        correct = question["new_marker"] if question["type"] == "current" else question["old_marker"]
        lines.append(f"correct answer marker: {correct}")
        lines.append("")
        conditions = list(contexts[qid].keys())
        for cond in conditions:
            entry = contexts[qid][cond]
            header = f"- {cond}: ids={entry['ids']}"
            if cond.startswith("D:"):
                header += (f" | new={entry['contains_new_decision']} "
                           f"old={entry['contains_old_decision']} "
                           f"near_miss={entry['topic_near_miss_ids']}")
            lines.append(header)
            for rep in range(result["settings"]["reps"]):
                r = next(x for x in runs if x["question_id"] == qid
                         and x["condition"] == cond and x["rep"] == rep)
                lines.append(f"    rep{rep} [{r['verdict']}] {r['answer']}")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
