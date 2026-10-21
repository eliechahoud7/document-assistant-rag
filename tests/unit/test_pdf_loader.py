import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pypdf import PageObject, PdfWriter

from doc_assistant.domain.errors import EmptyDocumentError, InvalidPDFError
from doc_assistant.domain.ports import DocumentLoader
from doc_assistant.ingestion.pdf_loader import PypdfLoader

MakePdf = Callable[..., Path]


def test_loads_pages_in_order_with_filename(make_pdf: MakePdf) -> None:
    path = make_pdf([["Première page."], ["Deuxième page.", "Suite."]], name="cours.pdf")

    document = PypdfLoader().load(path)

    assert document.filename == "cours.pdf"
    assert document.page_count == 2
    assert [page.page_number for page in document.pages] == [1, 2]
    assert document.pages[0].text == "Première page."
    assert document.pages[1].text == "Deuxième page.\nSuite."
    assert document.empty_pages == ()


def test_french_characters_survive_extraction(make_pdf: MakePdf) -> None:
    text = "Été, œuvre, naïveté, garçon (À bientôt)"

    document = PypdfLoader().load(make_pdf([[text]]))

    assert document.pages[0].text == text


def test_text_is_cleaned(make_pdf: MakePdf) -> None:
    document = PypdfLoader().load(make_pdf([["Une infor-", "mation   utile"]]))

    assert document.pages[0].text == "Une information utile"


def test_identity_is_derived_from_content(make_pdf: MakePdf) -> None:
    path = make_pdf([["Contenu."]], name="a.pdf")
    copy = make_pdf([["Contenu."]], name="b.pdf")

    first, second = PypdfLoader().load(path), PypdfLoader().load(copy)

    assert first.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert first.doc_id == first.sha256[:16]
    assert first.doc_id == second.doc_id  # même contenu, même identifiant
    assert first.filename != second.filename


def test_blank_pages_are_reported(make_pdf: MakePdf, caplog: pytest.LogCaptureFixture) -> None:
    path = make_pdf([["Texte."], [], ["Encore du texte."], []])

    with caplog.at_level(logging.WARNING, logger="doc_assistant.ingestion.pdf_loader"):
        document = PypdfLoader().load(path)

    assert document.page_count == 4
    assert document.empty_pages == (2, 4)
    assert "2, 4" in caplog.text


def test_pdf_without_any_text_is_rejected(make_pdf: MakePdf) -> None:
    with pytest.raises(EmptyDocumentError, match="aucun texte"):
        PypdfLoader().load(make_pdf([[], []]))


def test_pdf_without_pages_is_rejected(make_pdf: MakePdf) -> None:
    with pytest.raises(EmptyDocumentError, match="aucune page"):
        PypdfLoader().load(make_pdf([]))


@pytest.mark.parametrize("content", [b"", b"ceci n'est pas un PDF", b"%PDF-1.7\n tronqu"])
def test_invalid_files_are_rejected(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "faux.pdf"
    path.write_bytes(content)

    with pytest.raises(InvalidPDFError, match=r"faux\.pdf"):
        PypdfLoader().load(path)


def test_password_protected_pdf_is_rejected(make_pdf: MakePdf) -> None:
    path = make_pdf([["Secret."]])
    writer = PdfWriter(clone_from=path)
    writer.encrypt(user_password="motdepasse", algorithm="RC4-128")
    with path.open("wb") as stream:
        writer.write(stream)

    with pytest.raises(InvalidPDFError, match="mot de passe"):
        PypdfLoader().load(path)


def test_failing_page_is_treated_as_empty(
    make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = make_pdf([["Page saine."], ["Page abîmée."]])
    original = PageObject.extract_text

    def flaky(self: PageObject, *args: Any, **kwargs: Any) -> str:
        text = original(self, *args, **kwargs)
        if "abîmée" in text:
            raise ValueError("flux corrompu")
        return text

    monkeypatch.setattr(PageObject, "extract_text", flaky)

    with caplog.at_level(logging.WARNING, logger="doc_assistant.ingestion.pdf_loader"):
        document = PypdfLoader().load(path)

    assert document.pages[0].text == "Page saine."
    assert document.empty_pages == (2,)
    assert "page 2 : extraction impossible" in caplog.text


def test_logs_never_contain_the_uploaded_filename(
    make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Un nom de fichier envoyé peut être une donnée personnelle : les journaux ne gardent que
    l'identifiant du document (et le type de l'erreur, pas son message)."""
    path = make_pdf([["Page saine."], ["Page abîmée."], []], name="Dossier-medical-Dupont.pdf")
    original = PageObject.extract_text

    def flaky(self: PageObject, *args: Any, **kwargs: Any) -> str:
        text = original(self, *args, **kwargs)
        if "abîmée" in text:
            raise ValueError("flux corrompu près de Dupont")
        return text

    monkeypatch.setattr(PageObject, "extract_text", flaky)

    with caplog.at_level(logging.WARNING, logger="doc_assistant.ingestion.pdf_loader"):
        document = PypdfLoader().load(path)

    assert f"document {document.doc_id}, page 2 : extraction impossible (ValueError)" in caplog.text
    assert f"document {document.doc_id} : 2 page(s) sans texte extractible : 2, 3" in caplog.text
    assert "Dupont" not in caplog.text and "Dossier" not in caplog.text
    assert document.filename == "Dossier-medical-Dupont.pdf"  # le nom reste dans le document


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PypdfLoader().load(tmp_path / "absent.pdf")


def test_satisfies_loader_port(make_pdf: MakePdf) -> None:
    loader: DocumentLoader = PypdfLoader()

    assert loader.load(make_pdf([["ok"]])).page_count == 1
