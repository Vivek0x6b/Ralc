# Pre-registration: does stale (superseded) context make Gemma answer a current-state question wrongly?

Date: 2026-10-09
Dataset: benchmarks/data/v2 (the harder set, 244 messages).
No RALC code is changed by this experiment. It only reads the v2 conversation
and reuses the existing retrieval and rendering code paths.

## Premise under test

Including a message that states a decision which was later replaced (a stale or
superseded decision) can make an LLM answer a current-state question with the
old decision instead of the current one.

RALC is not evaluated here as a retriever. This experiment asks a narrower
question: given a fixed context, does the presence of stale content change the
answer the model gives to an unchanged question.

## The two superseded decisions in v2

These are the only two decisions in v2 that a later message genuinely replaces
(not merely refines, adds to, or fixes). This set is final for this experiment.

| Topic | Old decision message | New (current) decision message |
| --- | --- | --- |
| Main data store | m8 (user): keep everything in MongoDB for the first cut | m90 (user): change of mind, move the main store over to Postgres (m91 is the assistant reason, excluded from the hand-built contexts) |
| Hosting / deploy | m29 (user): push it to Heroku to begin with | m238 (assistant): Heroku got expensive, moving where the app runs over to AWS as containers |

## Questions

The question text is passed to the model unchanged. Current-state questions are
deliberately phrased so they do not name the answer, otherwise a context cannot
mislead the model.

| Q | Topic | Type | Question text | Correct answer |
| --- | --- | --- | --- | --- |
| Q1 | DB | current-state | Which database are we using for the main data store now? | Postgres |
| Q2 | DB | history | What database did we originally plan to use before we switched? | MongoDB |
| Q3 | hosting | current-state | Where are we deploying the app now? | AWS |
| Q4 | hosting | history | What did we use for hosting before we changed it? | Heroku |

Q2, Q3, Q4 are taken verbatim from benchmarks/data/v2/questions.json. Q1 is a
neutral rephrase of the dataset's "Why did we choose Postgres for the main
database?" so that the question does not contain the word Postgres.

## Conditions

All contexts are rendered in conversation order in the RALC `to_text` style:
each node as `"{role}: {content}"`, nodes joined by a blank line. The same
single prompt template (below) wraps every context; only the message set
differs between conditions.

Hand-built conditions (all four questions), original messages only:

| Condition | DB questions (Q1, Q2) | Hosting questions (Q3, Q4) |
| --- | --- | --- |
| A, newer only | m90 | m238 |
| B, older then newer | m8, m90 | m29, m238 |
| C, older only | m8 | m29 |

Condition D (current-state questions Q1 and Q3 only): the exact context that the
benchmark's `vector` and `ralc_heuristic` methods actually select, at token
budgets 250 and 500. Four contexts per question (two methods by two budgets).

- Retrieval is run on the experiment's own question text (the Q1 rephrase for
  the DB context, the Q3 dataset text for the hosting context), so the same
  query drives both retrieval and the final answer.
- The selection code path is identical to benchmarks/run.py:
  - `vector`: `retriever.retrieve(query, k=all_ids)` then `fill_to_budget`.
  - `ralc_heuristic`: `manager.retrieve(query, token_budget=b, seed_k=10).nodes`.
  - Embedder: SentenceTransformerEmbedder (default model). Extractor:
    HeuristicExtractor. Token counter: DefaultTokenCounter.
- These selections are deterministic, so each D context is fixed; only the
  model's answer can vary across repetitions.

For every D context the results file records:
- the sorted list of message ids it contains,
- `contains_new_decision`: whether it includes the current-decision message
  (m90 or m91 for DB, m238 for hosting),
- `contains_old_decision`: whether it includes the old-decision message
  (m8 for DB, m29 for hosting),
- `topic_near_miss_ids`: which topic near-miss distractors it contains
  (DB: m10, m81, m115; hosting: m20, m30, m124).

## Model and call settings

- Model: gemma-4-26b-a4b-it, via the existing google-genai client (the same path
  as ralc.extraction.gemma.GemmaExtractor). API key from GEMINI_API_KEY.
- temperature: 0.
- timeout: 45 seconds.
- retries: up to 3, exponential backoff, only on genuine server or rate-limit
  errors (429, 500, 503, 504 and the matching markers), reusing the existing
  retry policy. A blocked or empty response is a real failure, not retried.
- repetitions: 3 per (question, condition) to check consistency.
- Every response is cached to disk, so the run is resumable and re-runs make no
  new API calls.

Total calls: hand-built 4 questions x 3 conditions x 3 reps = 36; condition D
2 questions x 4 contexts x 3 reps = 24; 60 in all.

## Prompt template (identical for every call)

```
You are answering a question using only the chat messages below.
Read the messages, then answer the question in one or two sentences.
If the messages do not contain the answer, say you do not know.

Messages:
{context}

Question: {question}
```

`{context}` is the to_text-style rendering of the condition's messages.
`{question}` is the unchanged question text.

## Automatic first-pass label (advisory only)

Every answer is reviewed by hand. The automatic label is a convenience, not the
result. For each question two case-insensitive substring markers are defined:

| Q | new marker | old marker |
| --- | --- | --- |
| Q1 (DB current) | postgres | mongo |
| Q2 (DB history) | postgres | mongo |
| Q3 (hosting current) | aws | heroku |
| Q4 (hosting history) | heroku (new in time, but the correct answer) | (see note) |

The labeler reports which markers appear, in four buckets: `new_only`,
`old_only`, `both`, `neither`. Mapping to correctness:

- Current-state questions (Q1, Q3): correct = `new_only` or `both` (an answer
  that correctly narrates the change, for example "moved from Heroku to AWS", is
  not a stale failure); stale failure = `old_only`; `neither` = unclear.
- History questions (Q2, Q4): correct = the old decision is named, so correct =
  `old_only` or `both`; anachronism failure = `new_only`; `neither` = unclear.

For Q4 the correct answer (Heroku) is the old decision, so its markers are
new=aws (anachronism), old=heroku (correct); the same four-bucket logic applies.

## What result supports or rejects the premise

- Supports the premise: on a current-state question, condition A reliably gives
  the current answer, but condition B (old then new) or condition D (real
  retrieval output) gives the old answer (`old_only`) in a meaningful share of
  the repetitions. Adding stale content degraded a current-state answer.
- Rejects the premise: condition B and condition D match condition A (the
  current answer) across all repetitions. Stale content was present but did not
  mislead.
- Controls: condition C (old only) on a current-state question should give the
  old answer or "I do not know", confirming the model reads the supplied context
  rather than answering from prior knowledge. On history questions, condition B
  should name the old decision as "before"; naming the new decision there is a
  distinct anachronism failure worth flagging.

## Stated limitation

Both newer messages (m90, m238) announce the change explicitly ("I have changed
my mind", "we are moving ... over to AWS"). So a reject result on the hand-built
conditions A to C only shows that the model is not misled when the current
message spells out that a change happened. It does not cover changes that are
implied rather than announced. Condition D is the decision-relevant test,
because it uses the context that real retrieval actually produces, where the
current message may sit among old and near-miss messages without any explicit
"this replaces the old choice" framing in the selected set.
