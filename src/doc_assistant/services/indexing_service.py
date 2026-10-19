"""Cas d'usage : indexer un PDF (chargement -> découpage -> embeddings -> stockage)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path

from doc_assistant.domain.models import Document
from doc_assistant.domain.ports import (
    Chunker,
    DocumentLoader,
    Embedder,
    VectorStore,
    ensure_same_embedding_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexReport:
    doc_id: str
    filename: str
    page_count: int
    empty_pages: tuple[int, ...]
    chunk_count: int
    skipped: bool
    seconds: float


class IndexingService:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        ensure_same_embedding_model(embedder, store)
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def index_pdf(self, path: Path, *, force: bool = False) -> IndexReport:
        """Indexe un PDF. Un document déjà présent (même contenu) est ignoré sauf si ``force``."""
        start = time.perf_counter()
        report = self.index_document(self._loader.load(path), force=force)
        return replace(report, seconds=time.perf_counter() - start)

    def index_document(self, document: Document, *, force: bool = False) -> IndexReport:
        """Indexe un document déjà chargé (même règle de déduplication que ``index_pdf``)."""
        start = time.perf_counter()
        if self._store.has_document(document.doc_id) and not force:
            logger.debug("%s déjà indexé (doc_id=%s) : ignoré", document.filename, document.doc_id)
            return IndexReport(
                doc_id=document.doc_id,
                filename=document.filename,
                page_count=document.page_count,
                empty_pages=document.empty_pages,
                chunk_count=0,
                skipped=True,
                seconds=time.perf_counter() - start,
            )

        chunks = self._chunker.split(document)
        # Embeddings calculés avant toute suppression : en cas d'échec, l'ancien index reste intact.
        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])
        removed = self._store.delete_document(document.doc_id)
        self._store.add(chunks, embeddings)

        logger.debug(
            "%s : %d chunks indexés (%d anciens remplacés)",
            document.filename,
            len(chunks),
            removed,
        )
        return IndexReport(
            doc_id=document.doc_id,
            filename=document.filename,
            page_count=document.page_count,
            empty_pages=document.empty_pages,
            chunk_count=len(chunks),
            skipped=False,
            seconds=time.perf_counter() - start,
        )
