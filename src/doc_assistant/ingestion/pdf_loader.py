"""Extraction du texte des PDF page par page, avec pypdf."""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path

from pypdf import PdfReader

from doc_assistant.domain.errors import EmptyDocumentError, InvalidPDFError
from doc_assistant.domain.models import Document, PageContent
from doc_assistant.ingestion.cleaning import clean_text

logger = logging.getLogger(__name__)

DOC_ID_LENGTH = 16


class PypdfLoader:
    """Charge un PDF à texte sélectionnable. Ne fait pas d'OCR."""

    def load(self, path: Path) -> Document:
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()

        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise InvalidPDFError(f"{path.name} : PDF protégé par mot de passe")
            raw_pages = list(reader.pages)
        except InvalidPDFError:
            raise
        except Exception as exc:  # pypdf lève des exceptions variées sur un fichier malformé
            raise InvalidPDFError(f"{path.name} : PDF illisible ({exc})") from exc

        pages: list[PageContent] = []
        for number, raw_page in enumerate(raw_pages, start=1):
            try:
                text = clean_text(raw_page.extract_text())
            except Exception as exc:  # une page abîmée ne doit pas bloquer tout le document
                logger.warning("%s, page %d : extraction impossible (%s)", path.name, number, exc)
                text = ""
            pages.append(PageContent(page_number=number, text=text))

        document = Document(
            doc_id=sha256[:DOC_ID_LENGTH],
            filename=path.name,
            sha256=sha256,
            pages=tuple(pages),
        )

        if document.page_count == 0:
            raise EmptyDocumentError(f"{path.name} : le PDF ne contient aucune page")
        if len(document.empty_pages) == document.page_count:
            raise EmptyDocumentError(
                f"{path.name} : aucun texte extractible (PDF scanné ? l'OCR n'est pas géré)"
            )
        if document.empty_pages:
            logger.warning(
                "%s : %d page(s) sans texte extractible : %s",
                path.name,
                len(document.empty_pages),
                ", ".join(map(str, document.empty_pages)),
            )
        return document
