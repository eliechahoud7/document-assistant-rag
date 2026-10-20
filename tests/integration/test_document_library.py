"""Bibliothèque de documents sur un vrai index ChromaDB (dossier temporaire, embedder factice).

Couvre la liste persistante, les doublons et surtout l'isolation des suppressions : supprimer
un PDF ne retire jamais les chunks d'un autre document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import MakePdf
from doc_assistant import bootstrap
from doc_assistant.config import Settings
from doc_assistant.domain.errors import DocumentNotFoundError, DocumentRegistryError
from doc_assistant.services.library_service import DocumentLibrary
from fakes import FakeEmbedder

LOUTRES = [["Les loutres de mer dorment en se tenant la patte."], ["Elles flottent sur le dos."]]
VOLCANS = [["Un volcan rejette de la lave et des cendres lors d'une éruption."]]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def library(settings: Settings) -> DocumentLibrary:
    return bootstrap.build_document_library(settings, FakeEmbedder())


def chunks_by_doc(settings: Settings) -> dict[str, int]:
    """Comptage indépendant de la bibliothèque : un nouvel accès à l'index."""
    store = bootstrap.build_store(settings, FakeEmbedder())
    return {d.filename: d.chunk_count for d in store.list_documents()}


def test_uploaded_documents_are_listed_with_details(make_pdf: MakePdf, settings: Settings) -> None:
    lib = library(settings)

    a = lib.add("Loutres.pdf", make_pdf(LOUTRES, name="a.pdf").read_bytes())
    b = lib.add("Volcans.pdf", make_pdf(VOLCANS, name="b.pdf").read_bytes())

    assert (a.kind, b.kind) == ("indexed", "indexed")
    entries = lib.documents()
    assert [(e.filename, e.page_count, e.chunk_count, e.status, e.source) for e in entries] == [
        ("Loutres.pdf", 2, 2, "indexed", "app"),
        ("Volcans.pdf", 1, 1, "indexed", "app"),
    ]
    assert all(e.added_at is not None for e in entries)


def test_list_persists_across_restarts(make_pdf: MakePdf, settings: Settings) -> None:
    library(settings).add("Loutres.pdf", make_pdf(LOUTRES).read_bytes())

    restarted = library(settings)  # nouveaux objets, mêmes fichiers

    assert [e.filename for e in restarted.documents()] == ["Loutres.pdf"]
    assert (settings.data_dir / "documents.json").is_file()


def test_duplicate_content_is_not_indexed_twice(make_pdf: MakePdf, settings: Settings) -> None:
    lib = library(settings)
    data = make_pdf(LOUTRES).read_bytes()
    lib.add("Loutres.pdf", data)

    again = lib.add("Copie de loutres.pdf", data)

    assert again.kind == "duplicate"
    assert again.duplicate_of == "Loutres.pdf"
    assert [e.filename for e in lib.documents()] == ["Loutres.pdf"]
    assert chunks_by_doc(settings) == {"Loutres.pdf": 2}


def test_deleting_one_document_keeps_every_other_chunk(
    make_pdf: MakePdf, settings: Settings
) -> None:
    lib = library(settings)
    keep = lib.add("Volcans.pdf", make_pdf(VOLCANS, name="b.pdf").read_bytes())
    gone = lib.add("Loutres.pdf", make_pdf(LOUTRES, name="a.pdf").read_bytes())
    stored_keep = settings.uploads_dir / keep.entry.doc_id / "Volcans.pdf"

    outcome = lib.delete(gone.entry.doc_id)

    assert (outcome.filename, outcome.chunks_removed, outcome.file_removed) == (
        "Loutres.pdf",
        2,
        True,
    )
    assert chunks_by_doc(settings) == {"Volcans.pdf": 1}  # l'autre document est intact
    assert [e.filename for e in library(settings).documents()] == ["Volcans.pdf"]
    assert not (settings.uploads_dir / gone.entry.doc_id).exists()
    assert stored_keep.is_file()
    hits = bootstrap.build_retriever(settings, FakeEmbedder()).search("volcan lave", top_k=5)
    assert {h.chunk.filename for h in hits} == {"Volcans.pdf"}


def test_document_indexed_with_the_cli_is_listed_and_deletable(
    make_pdf: MakePdf, settings: Settings
) -> None:
    embedder = FakeEmbedder()
    bootstrap.build_indexing_service(settings, embedder).index_pdf(
        make_pdf(VOLCANS, name="cli.pdf")
    )  # comme « python -m doc_assistant index »
    lib = library(settings)
    lib.add("Loutres.pdf", make_pdf(LOUTRES, name="a.pdf").read_bytes())

    cli_entry = next(e for e in lib.documents() if e.filename == "cli.pdf")

    assert (cli_entry.source, cli_entry.page_count, cli_entry.added_at) == ("external", None, None)
    assert lib.delete(cli_entry.doc_id).file_removed is False  # aucun PDF conservé à supprimer
    assert chunks_by_doc(settings) == {"Loutres.pdf": 2}


def test_unknown_or_malformed_identifiers_delete_nothing(
    make_pdf: MakePdf, settings: Settings
) -> None:
    lib = library(settings)
    lib.add("Loutres.pdf", make_pdf(LOUTRES).read_bytes())

    for bad in ("ffffffffffffffff", "../../etc", ""):
        with pytest.raises(DocumentNotFoundError):
            lib.delete(bad)

    assert chunks_by_doc(settings) == {"Loutres.pdf": 2}


def test_unreadable_registry_falls_back_to_the_index(make_pdf: MakePdf, settings: Settings) -> None:
    lib = library(settings)
    lib.add("Loutres.pdf", make_pdf(LOUTRES).read_bytes())
    (settings.data_dir / "documents.json").write_text("{corrompu", encoding="utf-8")

    with pytest.raises(DocumentRegistryError):
        lib.documents()
    fallback = lib.documents_from_index()

    assert [(e.filename, e.chunk_count) for e in fallback] == [("Loutres.pdf", 2)]
