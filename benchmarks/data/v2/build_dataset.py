"""Deterministically build the harder v2 benchmark conversation and questions.

v2 reuses v1's signal facts and generic distractor generator, and adds:
- near-miss distractors that discuss the same topics (Postgres, hosting, cache,
  tokens) without being the answer, so semantic retrieval gets real false
  positives,
- some signal messages placed late in the conversation,
- multi-hop questions that need three to four supporting messages.

v1 is left untouched; v2 writes its own conversation.json and questions.json
here. Every message is unique and the conversation clears 20k tokens. Run with
`python -m benchmarks.data.v2.build_dataset`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ralc.allocation.budget import DefaultTokenCounter

from benchmarks.data.build_dataset import QUESTIONS as V1_QUESTIONS
from benchmarks.data.build_dataset import SIGNALS, _distractor

HERE = Path(__file__).resolve().parent
TOKEN_TARGET = 21000

# Signals placed at the very end of the conversation (after the token top-up),
# so they land in the last part of the transcript rather than clustering early.
LATE_SIGNALS = ["deploy_aws_switch", "db_replicas", "auth_refresh_rotate"]

# Near-miss distractors: same topics as the answers, but not the answer. They
# are never gold, so they test precision and semantic false positives.
NEAR_MISS: list[tuple[str, str]] = [
    ("user", "Honestly I still wonder if Postgres is overkill for our size. I lost that "
             "debate and I have mostly made my peace with it, but do not be shocked if I "
             "bring it up again at the next offsite when I am feeling a little braver."),
    ("assistant", "Another Heroku pricing email just landed and the increases are genuinely "
                  "rough. I am not proposing anything, just venting, because every time I open "
                  "that billing page I feel a bit older and a bit poorer than I did before."),
    ("user", "Someone in the hallway asked whether we had looked at Render or Fly for hosting. "
             "We glanced at them very early on but never seriously, and I did not want to "
             "relitigate the whole hosting question standing by the coffee machine."),
    ("assistant", "Is the cache actually doing anything? The invoice list still feels sluggish "
                  "to me some mornings, although that might just be my laptop having opinions. "
                  "This is not a bug report, just a vague vibe I wanted to put on the record."),
    ("user", "We should really write down how the cache key gets built one of these days, "
             "because right now it is tribal knowledge living in two people's heads, which is "
             "exactly the kind of thing that comes back to bite you about six months later."),
    ("assistant", "My login token expired right in the middle of a customer demo again, which "
                  "was mildly humiliating. I am not asking anyone to change anything, I just "
                  "need to complain about it to people who understand the very specific pain."),
    ("user", "Support got a question about whether refresh tokens work on the mobile app yet. "
             "I think they do, but I am honestly not certain, so if someone actually knows the "
             "answer please drop it in the thread and save them the guesswork and the wait."),
    ("assistant", "A friend of mine benchmarked Mongo against Postgres on a weekend side "
                  "project and wrote it up. It is a fun read and completely irrelevant to any "
                  "decision we have ever made, but I enjoyed it and will share the link later."),
    ("user", "Finance asked whether the nightly billing job could also email them a small "
             "summary of what ran. That is a separate ask from the charging behavior and I did "
             "not want it tangled up with anything else, so I am flagging it on its own here."),
    ("assistant", "A customer wanted to know if search covers the text inside attachments. It "
                  "does not today, which is a different question from which search engine we "
                  "run, so I told them it was on the someday pile and gently left it there."),
    ("user", "I keep seeing the word Postgres in my dreams at this point, which I am choosing "
             "to read as a sign that I am overdue for a vacation rather than any kind of deep "
             "technical insight about the data layer or the choices we have made around it."),
    ("assistant", "Reminder that AWS has a free tier for a lot of services if you are tinkering "
                  "at home. Nothing to do with our setup, I just watched three people get "
                  "surprise bills this quarter and I have become a one person crusade about it."),
]

# ("sig", name) emits a signal, ("near", None) the next near-miss, ("fill", k) k
# generic distractors. The three LATE_SIGNALS are emitted separately, at the end.
TIMELINE: list[tuple[str, object]] = [
    ("fill", 8), ("sig", "db_mongo_plan"), ("sig", "db_mongo_ack"), ("near", None),
    ("fill", 8), ("sig", "auth_decision"), ("near", None),
    ("fill", 8), ("sig", "deploy_heroku"), ("near", None),
    ("fill", 8), ("sig", "cache_add"), ("near", None),
    ("fill", 8), ("sig", "email_decision"), ("near", None),
    ("fill", 8), ("sig", "billing_job"), ("near", None),
    ("fill", 8), ("sig", "search_consider"), ("sig", "search_decision"), ("near", None),
    ("fill", 8), ("sig", "ratelimit_decision"), ("near", None),
    ("fill", 8), ("sig", "db_postgres_switch"), ("sig", "db_postgres_reason"), ("near", None),
    ("fill", 8), ("sig", "cache_bug"), ("sig", "cache_cause"), ("sig", "cache_fix"), ("near", None),
    ("fill", 8), ("sig", "billing_bug"), ("sig", "billing_fix"), ("near", None),
    ("fill", 8), ("near", None),
]

MULTIHOP: list[dict] = [
    {"question": "Walk me through the whole database decision from start to finish.",
     "gold_facts": ["started on MongoDB", "switched to Postgres for transactions",
                    "added read replicas for the heavy reports"],
     "gold": ["db_mongo_plan", "db_postgres_switch", "db_postgres_reason", "db_replicas"],
     "vector_friendly": False},
    {"question": "Tell me the full story of the cache incident, from the cause to the fix.",
     "gold_facts": ["added a Redis cache", "tenants briefly saw each other's invoices",
                    "cache key was missing the account", "account folded into the key"],
     "gold": ["cache_add", "cache_bug", "cache_cause", "cache_fix"],
     "vector_friendly": False},
    {"question": "Summarize the billing double charge incident end to end.",
     "gold_facts": ["nightly billing job", "double charged after a retry", "idempotency key fix"],
     "gold": ["billing_job", "billing_bug", "billing_fix"],
     "vector_friendly": False},
]

QUESTIONS = V1_QUESTIONS + MULTIHOP


def build():
    messages: list[dict] = []
    name_to_id: dict[str, str] = {}
    near_miss_ids: list[str] = []
    seen: set[str] = set()
    gen_index = 0
    near_index = 0

    def emit(role: str, content: str) -> str:
        node_id = f"m{len(messages)}"
        messages.append({"id": node_id, "role": role, "content": content})
        seen.add(content)
        return node_id

    def emit_generic() -> None:
        nonlocal gen_index
        while True:
            role, content = _distractor(gen_index)
            gen_index += 1
            if content not in seen:
                emit(role, content)
                return

    def emit_near() -> None:
        nonlocal near_index
        role, content = NEAR_MISS[near_index]
        near_index += 1
        near_miss_ids.append(emit(role, content))

    for kind, value in TIMELINE:
        if kind == "sig":
            name_to_id[value] = emit(*SIGNALS[value])
        elif kind == "near":
            emit_near()
        else:
            for _ in range(int(value)):
                emit_generic()

    counter = DefaultTokenCounter()
    total = sum(counter.count(m["content"]) for m in messages)
    while total < TOKEN_TARGET:
        emit_generic()
        total += counter.count(messages[-1]["content"])

    # Late signals: emit at the end so they sit in the final part of the chat.
    for name in LATE_SIGNALS:
        name_to_id[name] = emit(*SIGNALS[name])
        total += counter.count(messages[-1]["content"])
        emit_generic()
        total += counter.count(messages[-1]["content"])

    assert near_index == len(NEAR_MISS), "every near-miss must be placed exactly once"
    assert len({m["content"] for m in messages}) == len(messages), "messages not all unique"
    assert total >= 20000, f"conversation is only {total} tokens"

    questions = [
        {"question": q["question"],
         "gold_facts": q["gold_facts"],
         "gold_message_ids": [name_to_id[name] for name in q["gold"]],
         "gold_signal_names": q["gold"],
         "vector_friendly": q["vector_friendly"]}
        for q in QUESTIONS
    ]

    HERE.mkdir(exist_ok=True)
    (HERE / "conversation.json").write_text(json.dumps(messages, indent=2), encoding="utf-8")
    (HERE / "questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")

    return {
        "messages": messages,
        "questions": questions,
        "name_to_id": name_to_id,
        "near_miss_ids": near_miss_ids,
        "tokens": total,
    }


if __name__ == "__main__":
    result = build()
    print(f"messages: {len(result['messages'])}")
    print(f"unique messages: {len({m['content'] for m in result['messages']})}")
    print(f"tokens: {result['tokens']}")
    print(f"near-miss distractors: {len(result['near_miss_ids'])}")
    print(f"questions: {len(result['questions'])} "
          f"(multi-hop with 3 to 4 supports: "
          f"{sum(1 for q in result['questions'] if 3 <= len(q['gold_message_ids']) <= 4)})")
    late = [result["name_to_id"][n] for n in LATE_SIGNALS]
    print(f"late signals at: {late} of {len(result['messages'])} messages")
