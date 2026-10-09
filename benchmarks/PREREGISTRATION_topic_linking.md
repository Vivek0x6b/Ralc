# Pre-registration: Gemma topic linking for UPDATES edges

Date: 2026-10-09
Datasets: benchmarks/data (v1) and benchmarks/data/v2.

## Problem

On the Gemma graph the hosting decision pair is linked by an UPDATES edge
(the AWS message names "Heroku", a shared entity), but the database pair
(MongoDB vs Postgres) is not, because the two messages share no entity. Both
decisions are about the same subject (the main data store), so UPDATES should be
able to link decisions on the same topic, not only on a shared entity.

## Change under test

1. Gemma extraction also returns a short `topic` for decision messages (a few
   words naming what the decision is about), added to Signals as an optional
   field. HeuristicExtractor still returns no topic. PROMPT_VERSION is bumped so
   old cache entries are not reused. Topics come only from Gemma: no hand-written
   vocabulary or topic list is added.
2. Two decision messages get an UPDATES edge if they share an entity OR their
   topics match. Topics are phrased inconsistently, so a match is decided by
   embedding cosine similarity of the two topic strings (all-MiniLM-L6-v2, the
   project embedder) at or above a fixed threshold, not by exact text. The
   linker keeps its nearest-only, single-edge behavior; only the match condition
   widens.
3. Extraction is batched: several messages per Gemma call (each with an id),
   parsing a JSON array back, with a fallback to single-message calls for any
   batch that fails to parse. Real token usage is logged from usage_metadata.
   Batch size 15, 45 second timeout, 3 retries on server/rate-limit errors.

## Fixed parameters (set now, before any results)

- Primary topic-match threshold: cosine >= 0.60.
- Secondary sensitivity check: the same run is also evaluated at 0.50 and 0.70
  and reported alongside. These are reported only; they are not used to pick the
  threshold. The primary result is 0.60.
- Batch size: 15. Estimated calls for a full re-extraction of both datasets:
  about 18 (263 unique strings across v1 and v2 at batch 15), up to about 25 if
  some batches fall back to single calls.

## Expected results (directions, not numbers)

- UPDATES count: v1 currently has 31 UPDATES edges (entity-only linking). Topic
  OR entity linking should raise the count, because same-subject decisions that
  share no entity can now connect. Exact counts reported after the run.
- The four decision pairs: the hosting pairs (v1 m33/m137, v2 m29/m238) already
  link via the shared "Heroku" entity. The database pairs (v1 m10/m100, v2
  m8/m90) are expected to link via topic (both are about the main data store) if
  their topic similarity clears 0.60.
- Effect on selection: with relation-aware redundancy (exempt edge types
  UPDATES and ANSWERS), the database decision-change question and the multi-hop
  database story should gain complete-story at tight budgets, because the
  Mongo/Postgres pair becomes UPDATES-linked and the MMR exemption keeps both
  halves instead of dropping the near-duplicate second half. Lookup questions
  should be unchanged. Stated as a direction; measured after the run.

## What will be reported (step 6)

After re-extracting v1 and v2 in strict mode (batched), the relational ablation
is rerun with Gemma graphs for both datasets (the ablation is extended to build
the v2 Gemma graph now that v2 is warmed). The report includes:

1. Total UPDATES edges per dataset, for entity-only linking and for
   entity-or-topic linking, so the topic effect is isolated. The two graphs use
   the same extracted signals; only the linking rule differs.
2. Which of the four pairs are linked, and by what (entity, or topic with the
   similarity value).
3. is_decision and topic for all eight decision-pair messages (v1 m10, m100,
   m33, m137; v2 m8, m90, m29, m238), plus the topic similarity for each of the
   four pairs, at thresholds 0.50, 0.60, 0.70.
4. Any visible false links between genuinely unrelated decisions (a scan of all
   topic-created UPDATES edges, flagged by hand review).
5. Complete-story rate by question type (relationship vs lookup), per budget,
   for entity-only vs entity-or-topic Gemma graphs with relation-aware selection,
   next to the existing methods.

## Integrity checks

- Batch leakage: a sample of about 20 messages is extracted both batched and
  singly, and any differences in entities or is_decision are reported. Batching
  must not change the per-message signals.
- Every UPDATES edge is written to a human-review file with both message texts
  and the matching reason (shared entity, naming it; or topic match with the two
  topic strings and the cosine similarity).
- The new behavior is implemented tests-first and the full existing test suite
  stays green. Signals(entities=..., is_decision=...) stays equal to one with
  topic=None, so existing equality assertions hold.
