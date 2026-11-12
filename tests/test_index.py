import numpy as np
import pytest

from agent.embeddings import HashingEmbedder
from agent.index import VectorIndex, chunk_markdown


def test_chunks_follow_headings_and_get_stable_ids():
    chunks = chunk_markdown(
        "guide.md", "# Guide\n\nIntro.\n\n## Setup\n\nStep one.\n\nStep two.\n\n## Usage\n\nRun it."
    )
    assert [(c.id, c.heading) for c in chunks] == [("guide#1", "Guide"), ("guide#2", "Setup"), ("guide#3", "Usage")]
    assert chunks[1].text == "Step one. Step two."


def test_long_sections_split_on_paragraph_boundaries():
    paragraphs = [f"Paragraph {i} " + "word " * 40 for i in range(6)]
    chunks = chunk_markdown("long.md", "# Long\n\n" + "\n\n".join(paragraphs), max_chars=500)
    assert len(chunks) > 1
    assert all(c.text.startswith("Paragraph") for c in chunks)  # no paragraph cut in half
    assert " ".join(c.text for c in chunks).count("Paragraph") == 6  # nothing lost


def test_embeddings_are_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=256)
    a, b = embedder.embed(["refund window"]), embedder.embed(["refund window"])
    assert np.array_equal(a, b) and a.shape == (1, 256)
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_stemming_matches_inflected_forms():
    embedder = HashingEmbedder()
    same = (embedder.embed(["refunded payments"]) @ embedder.embed(["refund payment"]).T).item()
    different = (embedder.embed(["refunded payments"]) @ embedder.embed(["webhook signature"]).T).item()
    assert same > 0.9 and different < 0.1


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How long after capture can a payment be refunded?", "refunds#1"),
        ("After rolling an API key, how long does the old key keep working?", "authentication#2"),
        ("How do I verify a webhook signature?", "webhooks#2"),
        ("production keys requests per minute", "rate-limits#3"),
    ],
)
def test_search_ranks_the_right_passage_first(index, query, expected):
    assert index.search(query, k=3)[0].chunk.id == expected


def test_search_drops_irrelevant_results(index):
    assert index.search("quantum chromodynamics lattice gauge") == []


def test_save_and_load_round_trip(index, tmp_path):
    index.save(tmp_path)
    loaded = VectorIndex.load(HashingEmbedder(), tmp_path)
    query = "webhook retries"
    assert [h.chunk.id for h in loaded.search(query)] == [h.chunk.id for h in index.search(query)]


def test_loading_with_a_different_embedder_is_refused(index, tmp_path):
    index.save(tmp_path)
    with pytest.raises(ValueError, match="re-run ingest"):
        VectorIndex.load(HashingEmbedder(dim=512), tmp_path)


def test_empty_corpus_is_an_error():
    with pytest.raises(ValueError):
        VectorIndex.build(HashingEmbedder(), [])
