"""Cas d'usage : la bibliothèque de documents (lister, ajouter, supprimer).

Un document est identifié par son ``doc_id`` (début du SHA-256 de son contenu), enregistré
dans les métadonnées de chacun de ses chunks : la suppression ne vise que ces chunks
(``VectorStore.delete_document``) et ne peut pas toucher un autre document.

La liste fusionne trois sources : les chunks de l'index (référence de ce qui est indexé), le
registre de l'application (date, pages) et les PDF conservés dans le dossier des envois. Un
document indexé avec la CLI ou envoyé avant l'existence du registre apparaît donc aussi.

Bibliothèque d'un visiteur (envois publics) : les documents ``protected`` (démonstration) ne
peuvent pas être supprimés, et les limites (taille, pages, nombre de documents ajoutés) sont
vérifiées ici, côté serveur, quelle que soit l'interface.

Bibliothèque personnelle d'un utilisateur connecté : le registre est conservé avec l'index
(``ChunkRecordRegistry``) et ``uploads_dir`` vaut ``None`` : aucun PDF n'est gardé sur le
disque du serveur, seuls ses passages indexés.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from doc_assistant.domain.errors import (
    DocumentNotFoundError,
    DocumentProtectedError,
    LibraryClosedError,
    UploadRejectedError,
)
from doc_assistant.domain.models import IndexedDocument
from doc_assistant.domain.ports import VectorStore
from doc_assistant.indexing.document_registry import DocumentRecord
from doc_assistant.ingestion.pdf_loader import document_id
from doc_assistant.services.indexing_service import IndexingService
from doc_assistant.services.upload_service import MAX_UPLOAD_BYTES, index_uploaded_pdf

_DOC_ID_RE = re.compile(r"[0-9a-f]{16}")

DocumentStatus = Literal["indexed", "missing_chunks"]
DocumentSource = Literal["app", "external"]


class RecordStore(Protocol):
    """Registre des documents : fichier JSON local (``DocumentRegistry``) ou métadonnées des
    passages dans l'index (``ChunkRecordRegistry``). Lève ``DocumentRegistryError`` s'il est
    illisible."""

    def records(self) -> dict[str, DocumentRecord]: ...

    def put(self, record: DocumentRecord) -> None: ...

    def remove(self, doc_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class LibraryEntry:
    doc_id: str
    filename: str
    page_count: int | None  # None : inconnu (document indexé hors de l'application)
    chunk_count: int
    added_at: datetime | None
    status: DocumentStatus  # missing_chunks : enregistré mais absent de l'index
    source: DocumentSource  # app : envoyé par l'interface ; external : CLI ou autre


@dataclass(frozen=True, slots=True)
class AddOutcome:
    filename: str
    kind: Literal["indexed", "duplicate"]
    entry: LibraryEntry
    duplicate_of: str | None = None  # nom du document déjà indexé au même contenu


@dataclass(frozen=True, slots=True)
class DeleteOutcome:
    filename: str
    chunks_removed: int
    file_removed: bool


class DocumentLibrary:
    def __init__(
        self,
        indexing: IndexingService,
        store: VectorStore,
        registry: RecordStore,
        uploads_dir: Path | None,
        *,
        protected: frozenset[str] = frozenset(),
        max_documents: int | None = None,
        max_bytes: int = MAX_UPLOAD_BYTES,
        max_pages: int | None = None,
        lock: threading.Lock | None = None,
        is_open: Callable[[], bool] | None = None,
    ) -> None:
        """``lock`` : verrou partagé avec toute autre opération sur les mêmes données (par
        exemple la suppression du compte) ; ``is_open`` : vérifié sous ce verrou, ``False`` une
        fois la bibliothèque supprimée (``LibraryClosedError`` à l'ajout)."""
        self._indexing = indexing
        self._store = store
        self._registry = registry
        self._uploads_dir = uploads_dir
        self.protected = protected  # jamais supprimés, hors du quota ``max_documents``
        self._max_documents = max_documents
        self._max_bytes = max_bytes
        self._max_pages = max_pages
        self._lock = lock or threading.Lock()  # un ajout ou une suppression à la fois
        self._is_open = is_open or (lambda: True)

    def documents(self) -> list[LibraryEntry]:
        """Documents de la bibliothèque ; lève ``DocumentRegistryError`` si le registre est
        illisible (``documents_from_index`` donne alors au moins le contenu de l'index)."""
        return self._merge(self._registry.records())

    def documents_from_index(self) -> list[LibraryEntry]:
        """Liste dégradée, à partir de l'index et des fichiers conservés seulement."""
        return self._merge({})

    def add(self, name: str, data: bytes) -> AddOutcome:
        """Indexe un PDF avec le pipeline existant ; un contenu déjà indexé n'est pas réindexé.

        Lève ``UploadRejectedError`` (fichier refusé, limite atteinte) ou une autre
        ``IngestionError`` : rien n'est alors modifié.
        """
        with self._lock:
            if not self._is_open():
                raise LibraryClosedError(
                    f"{name} : votre compte vient d'être supprimé, rien n'a été ajouté"
                )
            self._check_quota(name, data)
            result = index_uploaded_pdf(
                name,
                data,
                self._indexing,
                self._uploads_dir,
                max_bytes=self._max_bytes,
                max_pages=self._max_pages,
            )
            report = result.report
            existing = {d.doc_id: d for d in self._store.list_documents()}
            if report.skipped:
                known = self._registry.records().get(report.doc_id)
                entry = self._entry(report.doc_id, existing.get(report.doc_id), known)
                return AddOutcome(report.filename, "duplicate", entry, entry.filename)
            stored = (
                result.stored_path.relative_to(self._uploads_dir)
                if result.stored_path and self._uploads_dir
                else None
            )
            record = DocumentRecord(
                doc_id=report.doc_id,
                filename=report.filename,
                sha256=hashlib.sha256(data).hexdigest(),
                page_count=report.page_count,
                chunk_count=report.chunk_count,
                empty_pages=report.empty_pages,
                added_at=datetime.now(UTC).isoformat(timespec="seconds"),
                stored_file=stored.as_posix() if stored else None,
            )
            self._registry.put(record)
            entry = self._entry(report.doc_id, existing.get(report.doc_id), record)
            return AddOutcome(report.filename, "indexed", entry)

    def delete(self, doc_id: str) -> DeleteOutcome:
        """Supprime les chunks du document, puis son PDF conservé et son enregistrement.

        Seuls les chunks portant ce ``doc_id`` sont supprimés ; seul le dossier
        ``<envois>/<doc_id>/`` est touché, et seulement ses fichiers PDF.
        """
        if not _DOC_ID_RE.fullmatch(doc_id):
            raise DocumentNotFoundError(f"identifiant de document invalide : {doc_id!r}")
        if doc_id in self.protected:
            raise DocumentProtectedError("ce document de démonstration ne peut pas être supprimé")
        with self._lock:
            if not self._is_open():
                raise DocumentNotFoundError("votre compte a été supprimé : document introuvable")
            records = self._registry.records()
            entry = next((e for e in self._merge(records) if e.doc_id == doc_id), None)
            if entry is None:
                raise DocumentNotFoundError(f"document introuvable : {doc_id}")
            chunks_removed = self._store.delete_document(doc_id)  # d'abord l'index
            file_removed = self._remove_stored_files(doc_id)
            self._registry.remove(doc_id)
            return DeleteOutcome(entry.filename, chunks_removed, file_removed)

    # ------------------------------------------------------------------ interne

    def _check_quota(self, name: str, data: bytes) -> None:
        """Refuse un nouveau document au-delà de ``max_documents`` ajoutés (les documents
        protégés ne comptent pas ; renvoyer un document déjà présent reste possible)."""
        if self._max_documents is None:
            return
        present = {d.doc_id for d in self._store.list_documents()}
        added = len(present - self.protected)
        if document_id(data) not in present and added >= self._max_documents:
            raise UploadRejectedError(
                f"{name} : limite de {self._max_documents} documents ajoutés atteinte ; "
                "supprimez-en un pour en ajouter un autre"
            )

    def _merge(self, records: dict[str, DocumentRecord]) -> list[LibraryEntry]:
        indexed = {d.doc_id: d for d in self._store.list_documents()}
        ids = set(indexed) | set(records)
        entries = [self._entry(i, indexed.get(i), records.get(i)) for i in ids]
        return sorted(entries, key=lambda e: (e.filename.lower(), e.doc_id))

    def _entry(
        self, doc_id: str, indexed: IndexedDocument | None, record: DocumentRecord | None
    ) -> LibraryEntry:
        chunks = indexed.chunk_count if indexed else 0
        stored = self._stored_pdf(doc_id)
        if record is not None:
            return LibraryEntry(
                doc_id,
                record.filename,
                record.page_count,
                chunks,
                datetime.fromisoformat(record.added_at),
                "indexed" if chunks else "missing_chunks",
                "app",
            )
        added = datetime.fromtimestamp(stored.stat().st_mtime, UTC) if stored else None
        return LibraryEntry(
            doc_id,
            indexed.filename if indexed else (stored.name if stored else doc_id),
            None,
            chunks,
            added,
            "indexed" if chunks else "missing_chunks",
            "app" if stored else "external",
        )

    def _stored_pdf(self, doc_id: str) -> Path | None:
        if self._uploads_dir is None:
            return None
        folder = self._uploads_dir / doc_id
        pdfs = sorted(folder.glob("*.pdf")) if folder.is_dir() else []
        return pdfs[0] if pdfs else None

    def _remove_stored_files(self, doc_id: str) -> bool:
        if self._uploads_dir is None:
            return False
        folder = self._uploads_dir / doc_id
        if not folder.is_dir():
            return False
        removed = False
        for pdf in folder.glob("*.pdf"):
            pdf.unlink()
            removed = True
        with contextlib.suppress(OSError):  # rmdir échoue si un fichier inattendu reste
            folder.rmdir()
        return removed
