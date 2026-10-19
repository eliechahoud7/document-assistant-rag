from itertools import pairwise

import pytest

from doc_assistant.config import Settings
from doc_assistant.domain.models import Document, PageContent
from doc_assistant.domain.ports import Chunker
from doc_assistant.ingestion.chunking import RecursiveChunker

SHA = "b" * 64
SENTENCE = "Le modèle retrouve les passages pertinents avant de répondre."


def make_document(*texts: str) -> Document:
    pages = tuple(PageContent(number, text) for number, text in enumerate(texts, start=1))
    return Document(doc_id="doc42", filename="rapport.pdf", sha256=SHA, pages=pages)


def long_text(sentences: int) -> str:
    return " ".join(f"{SENTENCE} ({i})" for i in range(sentences))


def test_short_page_gives_a_single_chunk() -> None:
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=20).split(make_document(SENTENCE))

    assert len(chunks) == 1
    assert chunks[0].text == SENTENCE
    assert chunks[0].start_char == 0


def test_long_page_is_split_within_size_limit() -> None:
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=40).split(make_document(long_text(20)))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=60).split(make_document(long_text(20)))

    for previous, current in pairwise(chunks):
        assert current.start_char < previous.start_char + len(previous.text)
        shared = previous.text[current.start_char - previous.start_char :]
        assert current.text.startswith(shared)


def test_no_overlap_when_overlap_is_zero() -> None:
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=0).split(make_document(long_text(20)))

    for previous, current in pairwise(chunks):
        assert current.start_char >= previous.start_char + len(previous.text)


def test_start_char_points_to_the_chunk_in_its_page() -> None:
    document = make_document(long_text(15), long_text(8))
    chunks = RecursiveChunker(chunk_size=150, chunk_overlap=30).split(document)

    for chunk in chunks:
        page_text = document.pages[chunk.page_number - 1].text
        assert page_text[chunk.start_char : chunk.start_char + len(chunk.text)] == chunk.text


def test_metadata_is_preserved_and_indexes_are_contiguous() -> None:
    document = make_document(long_text(10), "", long_text(10))
    chunks = RecursiveChunker(chunk_size=150, chunk_overlap=30).split(document)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page_number for chunk in chunks} == {1, 3}  # page 2 vide : ignorée
    assert all(chunk.doc_id == "doc42" for chunk in chunks)
    assert all(chunk.filename == "rapport.pdf" for chunk in chunks)
    assert [chunk.chunk_id for chunk in chunks] == [f"doc42:{i}" for i in range(len(chunks))]
    pages_in_order = [chunk.page_number for chunk in chunks]
    assert pages_in_order == sorted(pages_in_order)


def test_chunks_never_span_two_pages() -> None:
    document = make_document("Fin de la page un.", "Début de la page deux.")
    chunks = RecursiveChunker(chunk_size=500, chunk_overlap=50).split(document)

    assert [(chunk.page_number, chunk.text) for chunk in chunks] == [
        (1, "Fin de la page un."),
        (2, "Début de la page deux."),
    ]


def test_splitting_is_deterministic() -> None:
    document = make_document(long_text(12))
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)

    assert chunker.split(document) == chunker.split(document)


def test_paragraph_boundaries_are_preferred() -> None:
    first, second = "A" * 90, "B" * 90
    chunks = RecursiveChunker(chunk_size=120, chunk_overlap=10).split(
        make_document(f"{first}\n\n{second}")
    )

    assert [chunk.text for chunk in chunks] == [first, second]


def test_from_settings_uses_configuration() -> None:
    chunker = RecursiveChunker.from_settings(Settings(chunk_size=300, chunk_overlap=30))

    assert (chunker.chunk_size, chunker.chunk_overlap) == (300, 30)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=size, chunk_overlap=overlap)


def test_satisfies_chunker_port() -> None:
    chunker: Chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

    assert chunker.split(make_document("")) == []
