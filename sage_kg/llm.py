"""Ollama helpers used by extraction and querying.

CrewAI 1.x talks to Ollama through its own LLM class (LiteLLM). Answering
uses the local Ollama HTTP API so we do not depend on langchain-community.
"""

from __future__ import annotations

import json
import urllib.request
from types import SimpleNamespace


def ollama_tag(model: str) -> str:
    """Strip an ``ollama/`` prefix if the caller already added one."""
    return model.split("ollama/", 1)[-1]


def chat_llm(model: str, temperature: float = 0, base_url: str = "http://127.0.0.1:11434"):
    """LLM object for CrewAI agents."""
    from crewai import LLM

    return LLM(
        model=f"ollama/{ollama_tag(model)}",
        base_url=base_url,
        temperature=temperature,
    )


class OllamaCompleter:
    """Minimal stand-in for the old LangChain ``Ollama.complete()`` API."""

    def __init__(self, model: str, temperature: float = 0, base_url: str = "http://127.0.0.1:11434"):
        self.model = ollama_tag(model)
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str):
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            text = json.loads(resp.read())["response"]
        return SimpleNamespace(text=text)
