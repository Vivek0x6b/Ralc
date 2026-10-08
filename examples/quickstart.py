"""Quickstart: store a short conversation, then retrieve within a token budget.

Run with the embedding extra installed:

    pip install -e ".[embed]"
    python examples/quickstart.py
"""

from ralc import ContextManager


def main() -> None:
    ralc = ContextManager(storage="./memory")

    ralc.add_message("user", "We decided to use PostgreSQL.")
    ralc.add_message("assistant", "The primary reason was transactional consistency.")
    ralc.add_message("user", "We also added a Redis cache in front of it.")
    ralc.add_message("assistant", "The cache key includes the tenant id for isolation.")
    ralc.add_message("user", "Lunch options near the office are pretty limited.")

    result = ralc.retrieve(
        query="Why did we choose PostgreSQL?",
        token_budget=4000,
    )

    print("=== selected context ===")
    print(result.to_text())
    print("\n=== metadata ===")
    for key, value in result.metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
