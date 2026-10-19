from collections.abc import Callable
from pathlib import Path

import pytest

from doc_assistant.domain.errors import EmbeddingModelMismatchError, InvalidPDFError
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.ingestion.pdf_loader import PypdfLoader
from doc_assistant.services.indexing_service import IndexingService
from fakes import FakeEmbedder, InMemoryVectorStore

MakePdf = Callable[..., Path]


def make_service(
    chunk_size: int = 200,
) -> tuple[IndexingService, FakeEmbedder, InMemoryVectorStore]:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    service = IndexingService(
        PypdfLoader(), RecursiveChunker(chunk_size, chunk_size // 10), embedder, store
    )
    return service, embedder, store


def test_indexes_all_chunks_with_passage_embeddings(make_pdf: MakePdf) -> None:
    service, embedder, store = make_service()
    path = make_pdf([["Introduction au RAG."], [], ["Conclusion."]], name="rag.pdf")

    report = service.index_pdf(path)

    assert report.filename == "rag.pdf"
    assert (report.page_count, report.empty_pages, report.chunk_count) == (3, (2,), 2)
    assert report.skipped is False
    assert store.count() == 2
    assert embedder.document_calls == [["Introduction au RAG.", "Conclusion."]]


def test_same_content_is_not_indexed_twice(make_pdf: MakePdf) -> None:
    service, embedder, store = make_service()
    service.index_pdf(make_pdf([["Contenu."]], name="a.pdf"))

    report = service.index_pdf(make_pdf([["Contenu."]], name="copie.pdf"))

    assert report.skipped is True
    assert report.chunk_count == 0
    assert store.count() == 1
    assert len(embedder.document_calls) == 1  # aucun calcul inutile


def test_force_replaces_previous_chunks(make_pdf: MakePdf) -> None:
    path = make_pdf([["mot " * 200]])
    first, _, store = make_service(chunk_size=100)
    first.index_pdf(path)
    many = store.count()
    # Même store, chunks plus gros : moins de chunks, les anciens doivent disparaître.
    second = IndexingService(PypdfLoader(), RecursiveChunker(400, 40), FakeEmbedder(), store)

    report = second.index_pdf(path, force=True)

    assert report.skipped is False
    assert store.count() == report.chunk_count < many


def test_invalid_pdf_leaves_store_untouched(tmp_path: Path) -> None:
    service, _, store = make_service()
    path = tmp_path / "faux.pdf"
    path.write_bytes(b"pas un pdf")

    with pytest.raises(InvalidPDFError):
        service.index_pdf(path)
    assert store.count() == 0


def test_failed_embedding_keeps_previous_index(
    make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, embedder, store = make_service()
    path = make_pdf([["Version indexée."]])
    service.index_pdf(path)

    def boom(_texts: object) -> list[list[float]]:
        raise RuntimeError("panne du modèle")

    monkeypatch.setattr(embedder, "embed_documents", boom)

    with pytest.raises(RuntimeError):
        service.index_pdf(path, force=True)
    assert store.count() == 1


def test_embedder_and_store_must_use_the_same_model() -> None:
    with pytest.raises(EmbeddingModelMismatchError):
        IndexingService(
            PypdfLoader(),
            RecursiveChunker(100, 10),
            FakeEmbedder(model_name="modele-b"),
            InMemoryVectorStore("modele-a"),
        )
