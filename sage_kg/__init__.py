"""SAGE-KG: Sequential Uncertainty Resolution for knowledge-graph construction."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__ = ["SAGEKG", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "SAGEKG":
        from sage_kg.pipeline import SAGEKG

        return SAGEKG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
