"""Embedding backends behind one small interface.

`HashingEmbedder` is deterministic and needs no network or model download, so
the index, the tests and CI all work offline. Swap in `OpenAIEmbedder` (or any
object with the same two methods) for semantic recall in production.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import httpx
import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype("float32")


_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it its many much my of on or "
    "that the their this to was what when where which who why will with you your long after before".split()
)


def _stem(word: str) -> str:
    """Tiny suffix stripper: enough to match limit/limits and refund/refunded/refunding."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix) and not word.endswith("ss"):
            return word[: -len(suffix)]
    return word


class HashingEmbedder:
    """Signed feature hashing over stemmed unigrams and bigrams (the 'hashing trick').

    Lexical, not semantic: it matches shared vocabulary, not paraphrases. That is
    enough for a deterministic test corpus; use a real embedding model for recall
    on natural questions.
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        words = [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, bucket] += sign
        # Sublinear term frequency: a word repeated five times is not five times as relevant.
        out = np.sign(out) * np.log1p(np.abs(out))
        return _normalise(out)


class OpenAIEmbedder:
    """Any OpenAI-compatible /embeddings endpoint."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", dim: int = 1536):
        self.name, self.dim = model, dim
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 96):
            body = {"model": self.name, "input": texts[start : start + 96]}
            response = httpx.post(self._url, json=body, headers=self._headers, timeout=60)
            response.raise_for_status()
            vectors.extend(item["embedding"] for item in response.json()["data"])
        return _normalise(np.asarray(vectors, dtype="float32"))


def from_env() -> Embedder:
    model = os.getenv("EMBEDDING_MODEL")
    if model and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbedder(
            model,
            os.environ["OPENAI_API_KEY"],
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            int(os.getenv("EMBEDDING_DIM", "1536")),
        )
    return HashingEmbedder()
