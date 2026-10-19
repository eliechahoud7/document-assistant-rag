"""Découpage des documents en chunks, page par page."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_assistant.config import Settings
from doc_assistant.domain.models import Chunk, Document


class RecursiveChunker:
    """Découpe chaque page séparément pour que chaque chunk corresponde à une seule page.

    Les tailles sont en caractères. Le splitter essaie de couper d'abord entre paragraphes,
    puis entre lignes, puis entre mots, et en dernier recours au milieu d'un mot.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size doit être strictement positif")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap doit vérifier 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> RecursiveChunker:
        return cls(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    def split(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in document.pages:
            if page.is_empty:
                continue
            for piece in self._splitter.create_documents([page.text]):
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}:{index}",
                        doc_id=document.doc_id,
                        filename=document.filename,
                        page_number=page.page_number,
                        chunk_index=index,
                        start_char=int(piece.metadata["start_index"]),
                        text=piece.page_content,
                    )
                )
        return chunks
