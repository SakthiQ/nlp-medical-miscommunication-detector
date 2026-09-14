"""A thin client for a local Ollama server.

Kept separate from explainer.py so the LLM provider can be swapped (a smaller model for
development, a hosted API for a demo) without touching the prompt or parsing logic.
"""

from __future__ import annotations

import os

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.environ.get("LLM_MODEL", "llama3")

# Measured: 2 KB entries ~11-20s, 6 entries ~21s on an 8B CPU model. A typical report
# explains a handful of terms, so 120s leaves headroom without hanging indefinitely.
REQUEST_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "120"))


class LLMUnavailableError(Exception):
    """The LLM could not be reached, or did not respond in time."""


def generate_json(prompt: str, num_predict: int = 800) -> str:
    """Call Ollama in JSON mode and return the raw response text (still a JSON string).

    Raises LLMUnavailableError on any connection, timeout, or non-2xx failure. Does not
    parse or validate the JSON — that is the caller's job, since the caller knows the
    expected schema.
    """
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": num_predict},
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:
        # requests.exceptions.JSONDecodeError (Ollama returned a non-JSON body despite a
        # 2xx status) is itself a RequestException, so a malformed response is caught
        # here too, not just connection/timeout/HTTP-status failures.
        raise LLMUnavailableError(f"{MODEL_NAME} via Ollama: {exc}") from exc

    text = body.get("response")
    if not text:
        raise LLMUnavailableError(f"{MODEL_NAME} via Ollama: empty response")
    return text
