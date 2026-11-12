"""Document chunking and the FAISS vector index."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss

from .embeddings import Embedder


@dataclass(frozen=True)
class Chunk:
    id: str  # "<file stem>#<n>" - this is what the agent cites
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def chunk_markdown(source: str, text: str, max_chars: int = 900) -> list[Chunk]:
    """Split on headings, then pack paragraphs up to `max_chars` so a chunk
    never cuts a paragraph in half. Each chunk keeps its heading for context."""
    stem = Path(source).stem
    chunks: list[Chunk] = []
    sections = re.split(r"^(#{1,3} .+)$", text, flags=re.M)
    heading = stem
    for part in sections:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{1,3} ", part):
            heading = part.lstrip("# ").strip()
            continue
        buffer = ""
        for paragraph in re.split(r"\n\s*\n", part):
            paragraph = " ".join(paragraph.split())
            if buffer and len(buffer) + len(paragraph) + 1 > max_chars:
                chunks.append(Chunk(f"{stem}#{len(chunks) + 1}", source, heading, buffer))
                buffer = ""
            buffer = f"{buffer} {paragraph}".strip()
        if buffer:
            chunks.append(Chunk(f"{stem}#{len(chunks) + 1}", source, heading, buffer))
    return chunks


class VectorIndex:
    """Exact inner-product search over normalised vectors (= cosine similarity).

    IndexFlatIP is the right choice up to ~100k chunks: no training step, exact
    results, and it keeps evaluation free of ANN recall noise.
    """

    def __init__(self, embedder: Embedder, chunks: list[Chunk], index: faiss.Index):
        self.embedder, self.chunks, self._index = embedder, chunks, index

    @classmethod
    def build(cls, embedder: Embedder, chunks: list[Chunk]) -> VectorIndex:
        if not chunks:
            raise ValueError("cannot build an index from zero chunks")
        vectors = embedder.embed([f"{c.heading}\n{c.text}" for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(embedder, chunks, index)

    @classmethod
    def from_directory(cls, embedder: Embedder, folder: str | Path) -> VectorIndex:
        chunks: list[Chunk] = []
        for path in sorted(Path(folder).glob("**/*.md")):
            chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8")))
        return cls.build(embedder, chunks)

    def search(self, query: str, k: int = 4, min_score: float = 0.1) -> list[Hit]:
        vector = self.embedder.embed([query])
        scores, ids = self._index.search(vector, min(k, len(self.chunks)))
        return [
            Hit(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0 and s >= min_score
        ]

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(folder / "index.faiss"))
        meta = {"embedder": self.embedder.name, "chunks": [asdict(c) for c in self.chunks]}
        (folder / "chunks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, embedder: Embedder, folder: str | Path) -> VectorIndex:
        folder = Path(folder)
        meta = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            # Vectors from one embedding model are meaningless to another.
            raise ValueError(f"index was built with {meta['embedder']!r}, not {embedder.name!r}; re-run ingest")
        index = faiss.read_index(str(folder / "index.faiss"))
        return cls(embedder, [Chunk(**c) for c in meta["chunks"]], index)

    def __len__(self) -> int:
        return len(self.chunks)
