from typing import Any

import pytest

from doc_assistant.domain.models import Chunk, Document, PageContent

SHA = "a" * 64


def make_chunk(**overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "chunk_id": "doc:0",
        "doc_id": "doc",
        "filename": "cours.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "start_char": 0,
        "text": "Un passage.",
    }
    fields.update(overrides)
    return Chunk(**fields)


def test_page_numbers_start_at_one() -> None:
    with pytest.raises(ValueError):
        PageContent(page_number=0, text="x")


@pytest.mark.parametrize(("text", "expected"), [("", True), ("  \n\t", True), ("a", False)])
def test_page_is_empty(text: str, expected: bool) -> None:
    assert PageContent(page_number=1, text=text).is_empty is expected


def test_document_exposes_empty_pages() -> None:
    document = Document(
        doc_id="doc",
        filename="cours.pdf",
        sha256=SHA,
        pages=(PageContent(1, "texte"), PageContent(2, ""), PageContent(3, "suite")),
    )

    assert document.page_count == 3
    assert document.empty_pages == (2,)


@pytest.mark.parametrize(
    "fields",
    [
        {"doc_id": ""},
        {"filename": ""},
        {"sha256": "abc"},
        {"sha256": "A" * 64},
        {"pages": (PageContent(2, "x"),)},
        {"pages": (PageContent(1, "x"), PageContent(1, "y"))},
    ],
)
def test_document_validation(fields: dict[str, Any]) -> None:
    base: dict[str, Any] = {
        "doc_id": "doc",
        "filename": "cours.pdf",
        "sha256": SHA,
        "pages": (PageContent(1, "x"),),
    }
    with pytest.raises(ValueError):
        Document(**(base | fields))


def test_valid_chunk() -> None:
    chunk = make_chunk(page_number=3, chunk_index=7, start_char=120)

    assert (chunk.page_number, chunk.chunk_index, chunk.start_char) == (3, 7, 120)


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_id": ""},
        {"doc_id": ""},
        {"filename": ""},
        {"page_number": 0},
        {"chunk_index": -1},
        {"start_char": -1},
        {"text": "   "},
    ],
)
def test_chunk_validation(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        make_chunk(**overrides)


def test_models_are_immutable() -> None:
    chunk = make_chunk()

    with pytest.raises(AttributeError):
        chunk.text = "modifié"  # type: ignore[misc]
