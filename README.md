# RALC: Relational Context Allocation

> Give an LLM less context, but richer context.

RALC is an open-source, model-agnostic context optimization engine for LLM
applications. It represents accumulated context as a relational graph and
allocates a limited token budget based on both relevance and the relationships
between pieces of information. It is the context layer between an application and its
LLM, not the model that answers.

## Install

```bash
pip install -e .            # core: networkx, numpy
pip install -e ".[embed]"   # local embeddings via sentence-transformers
pip install -e ".[tokens]"  # exact token counts via tiktoken
```

The core package has no heavy dependencies. Local embeddings load lazily, so
`sentence-transformers` is only imported the first time you embed.

## Usage

```python
from ralc import ContextManager

ralc = ContextManager(storage="./memory")

ralc.add_message("user", "We decided to use PostgreSQL.")
ralc.add_message("assistant", "The primary reason was transactional consistency.")

result = ralc.retrieve(
    query="Why did we choose PostgreSQL?",
    token_budget=4000,
)

print(result.to_text())   # the selected context, ready for your prompt
print(result.metadata)    # real numbers: seeds, nodes explored, tokens, and more
```

`result` is a `ContextResult` with the selected nodes (in conversation order),
the token count, the relationships between selected nodes, and metadata. The
original query is passed through unchanged and never stored in the result. See
`examples/quickstart.py` for a longer runnable version.

## Status

Early development. The pipeline is built out in phases: graph, semantic
retrieval, relational expansion, hybrid ranking, and token allocation, wired
together by `ContextManager`.

## License

[Apache-2.0](LICENSE)
