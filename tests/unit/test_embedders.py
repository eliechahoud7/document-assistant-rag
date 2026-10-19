from pathlib import Path
from typing import Any

import numpy as np
import pytest

from doc_assistant.config import Settings
from doc_assistant.domain.errors import EmbeddingModelNotFoundError
from doc_assistant.domain.ports import Embedder
from doc_assistant.indexing.embedders import E5Embedder


class RecordingEncoder:
    """Faux SentenceTransformer : enregistre les appels, renvoie des vecteurs constants."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def encode(self, inputs: list[str], **kwargs: Any) -> Any:
        self.calls.append((inputs, kwargs))
        return np.full((len(inputs), self.dimension), 0.5, dtype=np.float32)

    def get_embedding_dimension(self) -> int | None:
        return self.dimension


def make_embedder(encoder: RecordingEncoder, batch_size: int = 8) -> E5Embedder:
    return E5Embedder(
        Path("absent"), "intfloat/multilingual-e5-small", batch_size=batch_size, encoder=encoder
    )


def test_documents_get_passage_prefix() -> None:
    encoder = RecordingEncoder()

    vectors = make_embedder(encoder).embed_documents(["Premier.", "Second."])

    assert encoder.calls[0][0] == ["passage: Premier.", "passage: Second."]
    assert vectors == [[0.5] * 4, [0.5] * 4]
    assert all(isinstance(x, float) for x in vectors[0])


def test_query_gets_query_prefix() -> None:
    encoder = RecordingEncoder()

    vector = make_embedder(encoder).embed_query("Qu'est-ce que le RAG ?")

    assert encoder.calls[0][0] == ["query: Qu'est-ce que le RAG ?"]
    assert vector == [0.5] * 4


def test_encoding_options() -> None:
    encoder = RecordingEncoder()

    make_embedder(encoder, batch_size=16).embed_documents(["x"])

    options = encoder.calls[0][1]
    assert options["normalize_embeddings"] is True
    assert options["batch_size"] == 16
    assert options["show_progress_bar"] is False


def test_empty_input_does_not_call_the_model() -> None:
    encoder = RecordingEncoder()

    assert make_embedder(encoder).embed_documents([]) == []
    assert encoder.calls == []


def test_metadata() -> None:
    embedder: Embedder = make_embedder(RecordingEncoder(dimension=384))

    assert embedder.dimension == 384
    assert embedder.model_name == "intfloat/multilingual-e5-small"


def test_unknown_dimension_is_an_error() -> None:
    encoder = RecordingEncoder()
    encoder.get_embedding_dimension = lambda: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="dimension inconnue"):
        _ = make_embedder(encoder).dimension


def test_missing_model_gives_actionable_error(tmp_path: Path) -> None:
    embedder = E5Embedder(tmp_path / "multilingual-e5-small", "intfloat/multilingual-e5-small")

    with pytest.raises(EmbeddingModelNotFoundError, match="download-model"):
        embedder.embed_query("question")


def test_from_settings() -> None:
    settings = Settings(models_dir=Path("m"), embedding_batch_size=4, embedding_device="cpu")

    embedder = E5Embedder.from_settings(settings)

    assert embedder.model_dir == Path("m") / "multilingual-e5-small"
    assert (embedder.batch_size, embedder.device) == (4, "cpu")
