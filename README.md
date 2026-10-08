# RALC: Relational Context Allocation

> Give an LLM less context, but richer context.

RALC is an open-source, model-agnostic context optimization engine for LLM
applications. It represents accumulated context as a relational graph and
allocates a limited token budget based on both relevance and the relationships
between pieces of information. It is the context layer between an application and its
LLM, not the model that answers.

## Status

Early development. The repository scaffold is in place; core functionality is
being built out in phases (graph → semantic retrieval → relational retrieval →
hybrid ranking → token allocation).

## License

[Apache-2.0](LICENSE)
