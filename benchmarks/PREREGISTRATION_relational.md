# Pre-registration: relation-aware selection, measured by ablation

Date: 2026-10-09
Datasets: benchmarks/data (v1) and benchmarks/data/v2.
This experiment adds one config switch to RALC (relation-aware redundancy,
default off) and measures it with an ablation. No weights are tuned and no
domain vocabulary is added (choosing entity names we know the answers depend on
would be tuning to the test).

## Premise

RALC's MMR redundancy penalty reduces a candidate's score by its peak embedding
similarity to already-selected nodes. On a decision-change question, the old and
new decisions are near-duplicates, so the penalty can drop the second half of
the answer. Relation-aware redundancy waives that penalty between two nodes that
the graph directly links by a meaningful edge (UPDATES or ANSWERS), on the view
that a linked pair is complementary, not redundant.

Hypothesis: relation-aware redundancy raises the complete-story rate on
relationship questions without hurting lookups.

## Main caveat (heuristic-graph diagnosis, read-only, no code changed)

In the heuristic graph the feature is expected to be inert, because the pairs it
targets are not linked:

- The heuristic graph has 0 UPDATES edges in both v1 and v2. The heuristic
  `is_decision` fires only on the literal cues ("we decided", "let's go with",
  "switch to", "instead of"), and the natural-language signal messages contain
  none of them, so nothing is flagged as a decision and no UPDATES edge is ever
  built.
- Postgres, Heroku, and AWS are not extracted as entities by the heuristic
  (its entity regex only catches internal-uppercase tokens like MongoDB), so the
  two sides of a change share no entity either.
- Direct check of the pairs (heuristic graph): v1 m10/m100 (Mongo/Postgres) and
  m33/m137 (Heroku/AWS) have no edge of any type between them and no shared
  entity; the same holds for the v2 pairs m8/m90 and m29/m238.

Consequence: under the heuristic extractor, relation-aware redundancy keyed on
UPDATES or ANSWERS cannot change the result on the decision-change questions.
It will be reported as inert if that is what happens. The ablation still isolates
the mechanism: redundancy-off is expected to recover the decision-change
questions, confirming that MMR is what drops the second half.

## Secondary diagnosis (Gemma graph, v1, from gemma_cache.json, no API calls)

Built with the cached Gemma signals only (232/234 strings cached, 2 distractor
misses):

- The v1 Gemma graph has 31 UPDATES edges.
- Hosting pair m33/m137: linked by an UPDATES edge (m137 UPDATES m33). Gemma
  extracts "Heroku" as an entity in both messages, because the AWS switch
  message names Heroku explicitly.
- DB pair m10/m100: not linked, no shared entity. Gemma extracts "mongodb" for
  m10 and "postgres"/"main store"/"rows" for m100, with no overlap. All five
  decision strings are flagged as decisions by Gemma.

Consequence: on the Gemma graph, relation-aware redundancy can plausibly affect
the hosting decision-change question (the pair is UPDATES-linked) but still not
the database one (the pair is unlinked). So the v1 ablation includes ralc_gemma
and ralc_gemma + relation-aware (cached data only, no API calls). v2 Gemma
coverage is not warmed, so Gemma variants are v1 only.

## Question tags (relationship vs lookup)

Rule: relationship if the question has 2 or more gold messages (decision changes,
cause and fix, multi-step stories); lookup if exactly 1 gold message (one fact in
one place).

v1 (14 questions): relationship (8):
- Why did we choose Postgres for the main database? (2)
- What database did we originally plan to use before we switched? (2)
- Why does each company only see its own cached invoice list now? (2)
- What caused a customer to see invoices that belonged to another company? (2)
- What did we use for hosting before we changed it? (2)
- How did we stop the nightly job from charging people twice? (2)
- Why did the team move away from its first choice of data store? (3)
- What did we do about the slow invoice list page? (2)

v1 lookup (6):
- Where are we deploying the app now? (1)
- Did we decide to use Elasticsearch? (1)
- Which service do we use to send invoice emails? (1)
- How long are login tokens valid and how do we keep people signed in? (1)
- What change makes a stolen refresh token only usable once? (1)
- What is the API rate limit? (1)

v2 (17 questions): the same 14, plus three multi-hop relationship questions
(Walk me through the whole database decision; Tell me the full story of the cache
incident; Summarize the billing double charge incident). v2: relationship 11,
lookup 6.

## Complete-story metric

complete_story(question) = 1 if every gold message id for that question was
selected, else 0. This is per-question recall == 1.0. Reported as a rate (share
of questions) per question type and per budget.

## Variants (config switches; only relation-aware is new)

- full RALC: default configs.
- relations off: ExpansionConfig(max_hops=0) so expansion returns seeds only,
  plus RankingConfig(relational=0.0, entity_overlap=0.0). Existing switches.
- redundancy off: SelectionConfig(redundancy_penalty=False). Existing switch.
- relation-aware redundancy (new): SelectionConfig(redundancy_exempt_edge_types
  = ("UPDATES", "ANSWERS")). New field, default () so existing behavior and all
  existing results are unchanged. When non-empty, the selector skips the MMR
  penalty between a candidate and an already-selected node that the graph links
  by a direct edge (either direction) of one of those types. Edge types are
  configurable.

## Ablation plan

- Methods: non-Gemma baselines (recent, vector, graph, ralc_heuristic, full) plus
  the three variant configs on the heuristic graph. For v1 only, also ralc_gemma
  and ralc_gemma + relation-aware, built from gemma_cache.json with no API calls
  (heuristic fallback for any uncached string, matching the benchmark runner).
- Budgets: v1 [1000, 2000, 4000]; v2 [250, 500, 1000, 2000, 4000].
- Report, per question type (relationship vs lookup) and per budget: complete-
  story rate, mean recall, mean tokens, and per-question wins and losses against
  vector (a win is a question where the method gets the complete story and vector
  does not; a loss is the reverse).
- The query string is asserted unchanged by every method, as in the main runner.

## Implementation and integrity

- The new switch is implemented tests-first. Tests assert: default empty set
  reproduces current selection exactly; with the switch on, a node that MMR would
  drop as redundant is kept when it is UPDATES- or ANSWERS-linked to a selected
  node; with the switch off it is dropped.
- The full existing test suite must stay green.
- No weights are changed. No vocabulary is added.

## Amendment (2026-10-09): corrected gold labels and complete-story metric

This amendment comes from a by-hand review of the gold labels, done before the
topic-linking results were seen. questions.json is unchanged; the labels are
added alongside in question_labels.json (one per dataset, built by
benchmarks/data/build_labels.py).

- required_message_ids: the minimal set of gold messages that together state
  every gold fact. gold_message_ids stays the supporting set.
- Complete-story is measured on required_message_ids (every required message
  selected). Recall stays on gold_message_ids.
- Question tags come from the required set: two or more required messages is a
  relationship question, exactly one is a lookup.

required != gold for: v1 Q4 [m113], Q7 [m126], Q12 [m10, m101]; v2 Q4 [m102],
Q7 [m114], Q12 [m8, m91], Q15 [m8, m91, m240]. In each case the fix or reason
message already states every fact of the dropped message. New tag counts: v1
relationship 6 / lookup 8 (was 8 / 6); v2 relationship 9 / lookup 8 (was 11 / 6),
as Q4 and Q7 move from relationship to lookup.

Review note on Q8 ("Did we decide to use Elasticsearch?"): the word Elasticsearch
appears in no message in either dataset; the gold message says "a separate search
cluster ... Postgres full text search". The gold set is left unchanged; this is
recorded so the label is read with that in mind.

The relational ablation is recomputed with this metric and these tags in
benchmarks/experiments/relational_ablation_corrected.md, old and new side by
side. No API calls and no re-extraction were involved.
