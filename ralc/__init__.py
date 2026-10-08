"""RALC: Relational Context Allocation.

A model-agnostic, local-first context optimization engine for LLM applications.
"""

__version__ = "0.0.1"

from ralc.core import ContextManager

__all__ = ["ContextManager", "__version__"]
