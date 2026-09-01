"""SAGE-KG: Sequential Uncertainty Resolution for knowledge-graph construction."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.2"
__all__ = ["SAGEKG", "Graph", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "SAGEKG":
        from sage_kg.pipeline import SAGEKG

        return SAGEKG
    if name == "Graph":
        from sage_kg.models import Graph

        return Graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
