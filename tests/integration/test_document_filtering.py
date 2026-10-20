"""Recherche limitée à un document, sur un vrai index ChromaDB (dossier temporaire).

Le filtre porte sur le ``doc_id`` enregistré dans les métadonnées de chaque chunk (début du
SHA-256 du PDF) : jamais sur le nom de fichier. Il est appliqué par l'index pendant la
recherche, donc avant que le contexte ne soit envoyé au LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from conftest import MakePdf
from doc_assistant import bootstrap
from doc_assistant.config import Settings
from doc_assistant.domain.errors import DocumentScopeError
from doc_assistant.domain.models import Answer, Chunk, SearchResult
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.library_service import DocumentLibrary
from doc_assistant.services.qa_service import QAService
from fakes import FakeEmbedder, InMemoryVectorStore, ScriptedAnswerGenerator

OTTERS = [["Les loutres de mer dorment en se tenant la patte."], ["Les loutres flottent."]]
VOLCANOES = [["Un volcan rejette de la lave lors d'une éruption."]]


class RecordingGenerator(ScriptedAnswerGenerator):
    def __init__(self) -> None:
        super().__init__({}, model_name="fake-gemini")
        self.passages: list[list[SearchResult]] = []

    def generate(self, question: str, passages: Sequence[SearchResult]) -> Answer:
        self.passages.append(list(passages))
        return super().generate(question, passages)


@pytest.fixture
def setup(tmp_path: Path, make_pdf: MakePdf) -> tuple[Settings, DocumentLibrary, dict[str, str]]:
    settings = Settings(data_dir=tmp_path / "data", context_neighbors=1)
    library = bootstrap.build_document_library(settings, FakeEmbedder())
    ids = {}
    for name, pages, src in (
        ("Loutres.pdf", OTTERS, "a.pdf"),
        ("Volcans.pdf", VOLCANOES, "b.pdf"),
        ("notes.pdf", [["Les loutres de mer dorment aussi ici."]], "c.pdf"),
        ("notes.pdf", [["Le volcan de lave gronde encore."]], "d.pdf"),  # même nom
    ):
        outcome = library.add(name, make_pdf(pages, name=src).read_bytes())
        ids[f"{name}:{src}"] = outcome.entry.doc_id
    return settings, library, ids


def qa(settings: Settings) -> tuple[QAService, RecordingGenerator]:
    generator = RecordingGenerator()
    return bootstrap.build_qa_service(settings, FakeEmbedder(), generator), generator


def test_filter_excludes_more_similar_chunks_of_other_documents(setup: Any) -> None:
    settings, _, ids = setup
    service, generator = qa(settings)
    volcanoes = ids["Volcans.pdf:b.pdf"]

    service.ask("Comment dorment les loutres de mer ?", top_k=5, doc_id=volcanoes)

    sent = generator.passages[-1]
    assert sent
    assert {p.chunk.doc_id for p in sent} == {volcanoes}  # rien d'autre envoyé au LLM


def test_neighbours_stay_inside_the_selected_document(setup: Any) -> None:
    settings, _, ids = setup
    service, _ = qa(settings)
    otters = ids["Loutres.pdf:a.pdf"]

    passages = service.retrieve("Les loutres flottent", top_k=1, doc_id=otters)

    assert len(passages) == 2  # le chunk retrouvé et son voisin (context_neighbors=1)
    assert {p.chunk.doc_id for p in passages} == {otters}


def test_no_filter_keeps_cross_document_retrieval(setup: Any) -> None:
    settings, _, ids = setup
    service, _ = qa(settings)

    passages = service.retrieve("loutres de mer et volcan de lave", top_k=10)

    assert {ids["Loutres.pdf:a.pdf"], ids["Volcans.pdf:b.pdf"]} <= {
        p.chunk.doc_id for p in passages
    }


def test_duplicate_filenames_are_filtered_by_document_id(setup: Any) -> None:
    settings, library, ids = setup
    service, _ = qa(settings)
    first, second = ids["notes.pdf:c.pdf"], ids["notes.pdf:d.pdf"]

    assert first != second
    assert [e.filename for e in library.documents()].count("notes.pdf") == 2
    for doc_id in (first, second):
        passages = service.retrieve("loutres de mer et volcan de lave", top_k=10, doc_id=doc_id)
        assert {p.chunk.doc_id for p in passages} == {doc_id}


def test_unknown_document_returns_nothing_rather_than_other_chunks(setup: Any) -> None:
    settings, _, _ = setup
    service, generator = qa(settings)

    result = service.ask("loutres", doc_id="ffffffffffffffff")

    assert result.passages == ()
    assert result.answer.abstained
    assert generator.passages == []  # le LLM n'est même pas appelé


class FilterIgnoringStore(InMemoryVectorStore):
    """Index défectueux qui ignorerait le filtre par document."""

    def search(
        self, query_embedding: Sequence[float], top_k: int, *, doc_id: str | None = None
    ) -> list[SearchResult]:
        return super().search(query_embedding, top_k)


def test_foreign_passages_are_refused_before_reaching_the_llm() -> None:
    """Garde-fou : même si un index ignorait le filtre, rien ne part au LLM."""
    embedder, store = FakeEmbedder(), FilterIgnoringStore()
    texts = {"aaaaaaaaaaaaaaaa": "Les loutres dorment.", "bbbbbbbbbbbbbbbb": "Le volcan gronde."}
    for doc_id, text in texts.items():
        chunk = Chunk(f"{doc_id}:0", doc_id, f"{doc_id}.pdf", 1, 0, 0, text)
        store.add([chunk], embedder.embed_documents([text]))
    generator = RecordingGenerator()
    service = QAService(Retriever(embedder, store, default_top_k=5), generator)

    with pytest.raises(DocumentScopeError, match="rien n'est envoyé au LLM"):
        service.ask("loutres volcan", doc_id="bbbbbbbbbbbbbbbb")

    assert generator.passages == []
