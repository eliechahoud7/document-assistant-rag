"""Envoi de PDF depuis l'interface : contrôles, indexation par le pipeline existant, doublons.

Hors ligne : vrai chargeur pypdf et vrai découpeur sur des PDF générés, embedder factice et
index en mémoire.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import MakePdf
from doc_assistant.domain.errors import (
    EmptyDocumentError,
    IngestionError,
    InvalidPDFError,
    UploadRejectedError,
)
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.ingestion.pdf_loader import PypdfLoader
from doc_assistant.services.indexing_service import IndexingService
from doc_assistant.services.upload_service import (
    MAX_FILENAME_LENGTH,
    MAX_UPLOAD_BYTES,
    MEGABYTE,
    check_upload,
    index_uploaded_pdf,
    safe_filename,
)
from fakes import FakeEmbedder, InMemoryVectorStore

TEXT = [["Les loutres de mer dorment en se tenant par la patte."], ["Deuxième page du cours."]]


def make_service() -> tuple[IndexingService, InMemoryVectorStore]:
    store = InMemoryVectorStore()
    service = IndexingService(PypdfLoader(), RecursiveChunker(800, 100), FakeEmbedder(), store)
    return service, store


def pdf_bytes(make_pdf: MakePdf, pages: list[list[str]] = TEXT, name: str = "src.pdf") -> bytes:
    return make_pdf(pages, name=name).read_bytes()


def test_valid_pdf_is_indexed_and_stored(make_pdf: MakePdf, tmp_path: Path) -> None:
    service, store = make_service()
    data = pdf_bytes(make_pdf)
    uploads = tmp_path / "uploads"

    result = index_uploaded_pdf("Loutres.pdf", data, service, uploads)

    report = result.report
    assert not report.skipped
    assert (report.filename, report.page_count, report.chunk_count) == ("Loutres.pdf", 2, 2)
    assert store.count() == 2
    assert result.stored_path == uploads / report.doc_id / "Loutres.pdf"
    assert result.stored_path.read_bytes() == data  # copie fidèle du fichier envoyé


def test_duplicate_content_is_not_reindexed_even_under_another_name(
    make_pdf: MakePdf, tmp_path: Path
) -> None:
    service, store = make_service()
    data = pdf_bytes(make_pdf)
    uploads = tmp_path / "uploads"
    first = index_uploaded_pdf("cours.pdf", data, service, uploads)

    second = index_uploaded_pdf("copie du cours.pdf", data, service, uploads)

    assert second.report.skipped
    assert second.report.doc_id == first.report.doc_id  # même SHA-256
    assert second.stored_path is None
    assert store.count() == 2  # aucun chunk ajouté
    assert [p.name for p in uploads.rglob("*.pdf")] == ["cours.pdf"]


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("notes.txt", b"%PDF-1.4 ...", "seuls les fichiers PDF"),
        ("vide.pdf", b"", "le fichier est vide"),
        ("texte.pdf", b"Ceci est un fichier texte renomme en .pdf", "pas un PDF valide"),
    ],
)
def test_rejected_uploads_are_neither_indexed_nor_stored(
    tmp_path: Path, name: str, data: bytes, message: str
) -> None:
    service, store = make_service()
    uploads = tmp_path / "uploads"

    with pytest.raises(UploadRejectedError, match=message):
        index_uploaded_pdf(name, data, service, uploads)

    assert store.count() == 0
    assert not uploads.exists()


def test_too_large_upload_is_rejected(make_pdf: MakePdf, tmp_path: Path) -> None:
    service, store = make_service()
    data = pdf_bytes(make_pdf)

    with pytest.raises(UploadRejectedError, match="trop volumineux"):
        index_uploaded_pdf("gros.pdf", data, service, tmp_path / "uploads", max_bytes=len(data) - 1)

    assert store.count() == 0


def assert_loader_error(data: bytes, error: type[IngestionError], tmp_path: Path) -> None:
    service, store = make_service()
    uploads = tmp_path / "uploads"

    with pytest.raises(error):
        index_uploaded_pdf("document.pdf", data, service, uploads)

    assert store.count() == 0
    assert not uploads.exists()


def test_corrupted_pdf_reports_the_loader_error(tmp_path: Path) -> None:
    assert_loader_error(b"%PDF-1.7\n corrupted content, no objects", InvalidPDFError, tmp_path)


def test_pdf_without_extractable_text_reports_the_loader_error(
    make_pdf: MakePdf, tmp_path: Path
) -> None:
    scanned = make_pdf([[], []], name="scan.pdf").read_bytes()  # pages sans texte (scan)

    assert_loader_error(scanned, EmptyDocumentError, tmp_path)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Cours.pdf", "Cours.pdf"),
        ("rapport.PDF", "rapport.pdf"),
        ("../../secret/passwd.pdf", "passwd.pdf"),  # aucun chemin conservé
        ("C:\\Users\\moi\\Mon CV (final).pdf", "Mon CV _final.pdf"),
        ("Cours d'été.pdf", "Cours d_été.pdf"),  # lettres accentuées conservées
        (".pdf", "document.pdf"),
        ("", "document.pdf"),
    ],
)
def test_safe_filename(name: str, expected: str) -> None:
    assert safe_filename(name) == expected


def test_safe_filename_is_bounded() -> None:
    name = safe_filename("a" * 500 + ".pdf")

    assert len(name) == MAX_FILENAME_LENGTH
    assert name.endswith(".pdf")


def test_too_many_pages_is_rejected_before_any_embedding(make_pdf: MakePdf, tmp_path: Path) -> None:
    service, store = make_service()
    three_pages = pdf_bytes(make_pdf, [["Page 1."], ["Page 2."], ["Page 3."]])

    with pytest.raises(UploadRejectedError, match=r"trop de pages \(3 ; maximum 2"):
        index_uploaded_pdf("long.pdf", three_pages, service, tmp_path / "uploads", max_pages=2)

    assert store.count() == 0
    assert not (tmp_path / "uploads").exists()


def test_page_limit_accepts_a_document_at_the_limit(make_pdf: MakePdf, tmp_path: Path) -> None:
    service, store = make_service()

    result = index_uploaded_pdf(
        "cours.pdf", pdf_bytes(make_pdf), service, tmp_path / "uploads", max_pages=2
    )

    assert result.report.page_count == 2
    assert store.count() > 0


def test_unreadable_pdf_still_reports_the_loader_error_with_a_page_limit(tmp_path: Path) -> None:
    service, _ = make_service()

    with pytest.raises(InvalidPDFError):
        index_uploaded_pdf(
            "abime.pdf", b"%PDF-1.4 tronque", service, tmp_path / "uploads", max_pages=50
        )


def test_size_limit_counts_megabytes_like_the_upload_widget() -> None:
    """Régression : « 20 Mo » affiché dans l'interface, mais « maximum 21 Mo » dans l'erreur
    (division par 10^6 d'une limite de 20 * 1024 * 1024 octets)."""
    assert MAX_UPLOAD_BYTES == 20 * MEGABYTE == 20 * 1024 * 1024
    check_upload("limite.pdf", b"%PDF-" + b"0" * (MAX_UPLOAD_BYTES - 5))  # exactement la limite

    with pytest.raises(UploadRejectedError, match=r"\(20\.0 Mo ; maximum 20 Mo\)"):
        check_upload("gros.pdf", b"%PDF-" + b"0" * (MAX_UPLOAD_BYTES - 4))
