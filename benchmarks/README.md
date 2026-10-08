# RALC benchmarks

This directory holds a small, reproducible, retrieval-level benchmark that
compares RALC against simpler baselines under a fixed token budget.

## Dataset

The dataset in `data/` is **synthetic and LLM-generated**. `data/build_dataset.py`
deterministically assembles:

- `conversation.json`: one long software-project conversation (around 250
  messages, over 20k tokens). It contains architecture decisions, decisions that
  are later reversed (so `UPDATES` edges matter), bugs and their fixes, and a lot
  of unrelated chatter as distractors. The distractors deliberately reuse the
  same entity names as the real discussion, so the graph has realistic noisy
  hubs that do not actually answer any question.
- `questions.json`: 14 questions. Each has the question text, the gold answer's
  key facts, and the gold supporting message ids. Some questions need
  information spread across distant messages; some have low word overlap with
  their supporting messages; at least three are phrased so plain vector search
  should do well. For questions about a reversed decision, the gold set includes
  the latest decision message.

The signal messages are written in natural conversational language, not shaped
around the heuristic extractor's patterns, so the test is fair rather than built
for RALC to win. **The gold labels are to be reviewed by hand**; regenerate the
files with `python benchmarks/data/build_dataset.py`.

## Methods

All methods select messages under the same token budget per run:

- `recent`: the most recent messages that fit.
- `vector`: semantic top-k, filled to the budget in score order.
- `graph`: relational expansion only, filled in expansion-score order (no hybrid
  ranking, no allocation strategy), over the heuristic-built graph.
- `ralc_heuristic`: the full ContextManager with HeuristicExtractor.
- `ralc_gemma`: the full ContextManager with GemmaExtractor(strict=True).
- `ralc_hybrid`: the full ContextManager with GemmaExtractor(hybrid=True, strict=True).
- `full`: all messages, reported for token count only.

## Metrics

Retrieval-level only, no LLM judge: recall of the gold supporting messages,
precision, tokens used, and retrieval latency. Runs at budgets of 1000, 2000,
and 4000 tokens.

## Running

```bash
pip install -e ".[embed,tokens,gemma]"
export GEMINI_API_KEY=...        # required for the gemma and hybrid methods
python benchmarks/run.py
```

Results are written to `results/` as JSON plus a markdown table, recording the
date, model names, package versions, and the Gemma extractor stats. Gemma
extraction results are cached to disk (`results/gemma_cache.json`) so re-runs do
not re-call the API.

Every number in the results comes from an actual run. Nothing is hand-edited. If
RALC does not beat a baseline, the table shows it.
