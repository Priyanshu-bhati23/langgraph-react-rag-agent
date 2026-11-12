from pathlib import Path

import pytest

from agent.embeddings import HashingEmbedder
from agent.index import VectorIndex
from agent.tools import make_tools

DOCS = Path(__file__).parent.parent / "data" / "docs"


@pytest.fixture(scope="session")
def index() -> VectorIndex:
    return VectorIndex.from_directory(HashingEmbedder(), DOCS)


@pytest.fixture(scope="session")
def tools(index):
    return make_tools(index)
