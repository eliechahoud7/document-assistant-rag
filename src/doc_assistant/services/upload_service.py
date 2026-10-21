"""Cas d'usage : indexer un PDF envoyé par l'utilisateur (interface web).

Couche fine au-dessus du pipeline existant : contrôles du fichier reçu, puis
``IndexingService.index_pdf`` (extraction, découpage, embeddings, ChromaDB, déduplication par
SHA-256). Seuls les PDF acceptés sont conservés, dans ``Settings.uploads_dir`` :
``<uploads>/<doc_id>/<nom>.pdf`` (``doc_id`` = début du SHA-256, comme dans l'index).
"""

from __future__ import annotations

import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from doc_assistant.domain.errors import UploadRejectedError
from doc_assistant.ingestion.pdf_loader import count_pdf_pages
from doc_assistant.services.indexing_service import IndexingService, IndexReport

# « Mo » = 1024 * 1024 octets partout dans l'application : c'est ainsi que le composant d'envoi
# de Streamlit compte dans un navigateur sous Windows (1 000 000 ailleurs, donc plus strict). La
# limite du serveur n'est ainsi jamais plus stricte que celle du navigateur.
MEGABYTE = 1024 * 1024
MAX_UPLOAD_BYTES = 20 * MEGABYTE
MAX_FILENAME_LENGTH = 120
PDF_MAGIC = b"%PDF-"
# La signature doit figurer au début du fichier (des octets parasites la précèdent parfois).
_MAGIC_SEARCH_BYTES = 1024
_UNSAFE_CHARS_RE = re.compile(r"[^\w.\- ]+")


@dataclass(frozen=True, slots=True)
class UploadResult:
    report: IndexReport
    stored_path: Path | None  # None si le document était déjà indexé


def safe_filename(name: str) -> str:
    """Nom de fichier sûr, lisible dans les citations : sans chemin ni caractère spécial."""
    base = Path(name.replace("\\", "/")).name
    base = unicodedata.normalize("NFKC", base).strip()
    stem = base[:-4] if base.lower().endswith(".pdf") else base
    stem = _UNSAFE_CHARS_RE.sub("_", stem).strip(" ._") or "document"
    return stem[: MAX_FILENAME_LENGTH - 4] + ".pdf"


def check_upload(name: str, data: bytes, *, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """Refuse un fichier vide, trop gros ou qui n'est pas un PDF (avant toute extraction)."""
    if not name.lower().endswith(".pdf"):
        raise UploadRejectedError(f"{name} : seuls les fichiers PDF (.pdf) sont acceptés")
    if not data:
        raise UploadRejectedError(f"{name} : le fichier est vide")
    if len(data) > max_bytes:
        raise UploadRejectedError(
            f"{name} : fichier trop volumineux ({len(data) / MEGABYTE:.1f} Mo ; "
            f"maximum {max_bytes / MEGABYTE:.0f} Mo)"
        )
    if PDF_MAGIC not in data[:_MAGIC_SEARCH_BYTES]:
        raise UploadRejectedError(f"{name} : ce fichier n'est pas un PDF valide")


def index_uploaded_pdf(
    name: str,
    data: bytes,
    service: IndexingService,
    uploads_dir: Path | None,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_pages: int | None = None,
) -> UploadResult:
    """Contrôle, indexe avec le pipeline existant, puis conserve le PDF s'il est nouveau
    (``uploads_dir`` ``None`` : le PDF n'est jamais conservé, seuls ses passages sont indexés).

    ``max_pages`` est vérifié avant toute extraction de texte et tout calcul d'embeddings.
    Lève ``UploadRejectedError`` (fichier refusé) ou une ``IngestionError`` du chargeur
    (PDF illisible, protégé, sans texte) : dans ces cas, rien n'est indexé ni conservé.
    """
    check_upload(name, data, max_bytes=max_bytes)
    if max_pages is not None:
        pages = count_pdf_pages(data)
        if pages is not None and pages > max_pages:
            raise UploadRejectedError(
                f"{name} : trop de pages ({pages} ; maximum {max_pages} par document)"
            )
    filename = safe_filename(name)
    # Fichier temporaire au nom définitif : c'est ce nom qui apparaît dans les citations.
    with tempfile.TemporaryDirectory(prefix="docassist-upload-") as tmp:
        path = Path(tmp) / filename
        path.write_bytes(data)
        report = service.index_pdf(path)
        if report.skipped or uploads_dir is None:
            return UploadResult(report, None)
        target = uploads_dir / report.doc_id / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return UploadResult(report, target)
