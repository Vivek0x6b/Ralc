# RALC: Relational Context Allocation

Give an LLM less context, but richer context.

## The problem

An LLM only sees the text you send it. A long chat history does not fit in the
token budget, so something has to choose which pieces to send. RALC is that
chooser: it sits between your app and the model and selects a small, high-value
slice of the history, leaving the question and the final answer to the model.

## How it works

1. Store each message as a node in a graph.
2. Connect the nodes: a pattern extractor (regex, no model) or a Gemma extractor
   adds links such as "answers", "mentions the same thing", and "this decision
   replaced that one".
3. For a query, find the most similar messages by embedding (the seeds).
4. Follow the links out from the seeds to pull in related messages.
5. Rank the candidates by similarity, relationship strength, and recency.
6. Fill the token budget with the highest-value messages.
7. Hand back the selected messages in conversation order, with the original
   query passed through unchanged.

The model you answer with is yours (GPT, Claude, Gemini, a local model, anything).
RALC never rewrites the query or the answer.

## Install

```bash
pip install -e ".[embed,tokens]"
```

Extras: `embed` (local embeddings via sentence-transformers), `tokens` (exact
token counts via tiktoken), `gemma` (Gemma extraction via google-genai). The
core install needs only networkx and numpy; heavy packages load lazily.

## Usage

```python
from ralc import ContextManager

ralc = ContextManager(storage="./memory")

ralc.add_message("user", "We decided to use PostgreSQL.")
ralc.add_message("assistant", "The primary reason was transactional consistency.")

result = ralc.retrieve(query="Why did we choose PostgreSQL?", token_budget=4000)

print(result.to_text())   # the selected context, in conversation order
```

`result.to_text()` is ready to drop into your prompt alongside the unchanged
question. To use Gemma for the linking step instead of the pattern extractor:

```python
from ralc.extraction import GemmaExtractor   # needs the gemma extra and GEMINI_API_KEY

ralc = ContextManager(storage="./memory", extractor=GemmaExtractor())
```

See `examples/quickstart.py` for a longer runnable version.

## Results

Measured on one synthetic benchmark (details below). Every number comes from an
actual run; the result tables are linked.

- On multi-message questions (answers that need two or more connected messages),
  turning relationships on beats turning them off at every token budget, even
  after matching the candidate-pool size so the win is not just from seeing more
  messages. See
  [relational_ablation_results.md](benchmarks/experiments/relational_ablation_results.md)
  and the corrected-metric version
  [relational_ablation_corrected.md](benchmarks/experiments/relational_ablation_corrected.md).
- The pattern extractor finds 0 "this decision replaced that one" links in the
  test conversation; swapping in a Gemma extractor finds 31. See the diagnosis in
  [PREREGISTRATION_relational.md](benchmarks/PREREGISTRATION_relational.md) and
  [topic_linking_results.md](benchmarks/experiments/topic_linking_results.md).
- Overall, RALC ties plain vector search on recall while using relationships to
  do better on the multi-message questions.

## Limitations

- One synthetic conversation, 14 to 17 questions. Small and not independently
  validated.
- One Gemma model, one embedding model.
- The benchmark measures which context is retrieved, not the quality of the
  final answer the LLM writes from it.

## Future work (not built yet)

- Importing real chat export files.
- An MCP server so any MCP-aware client can use RALC.
- Drop-in integrations for common agent frameworks.

## Disclosure

This project was started the day before the hackathon.

## License

[Apache-2.0](LICENSE)
