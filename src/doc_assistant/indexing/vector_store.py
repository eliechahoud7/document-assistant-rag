"""Adaptateur ChromaDB : stockage vectoriel local et persistant."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import chromadb
from chromadb.api.types import Where
from chromadb.config import Settings as ChromaSettings

from doc_assistant.config import Settings
from doc_assistant.domain.errors import EmbeddingModelMismatchError
from doc_assistant.domain.models import Chunk, SearchResult
from doc_assistant.domain.ports import Embedder


class ChromaVectorStore:
    """Collection Chroma liée à un unique modèle d'embeddings.

    Le nom et la dimension du modèle sont enregistrés dans les métadonnées de la
    collection à sa création ; rouvrir la collection avec un autre modèle lève
    ``EmbeddingModelMismatchError``. Les vecteurs sont toujours fournis par l'appelant :
    Chroma n'utilise jamais sa fonction d'embedding par défaut (qui téléchargerait un modèle).
    """

    def __init__(
        self, path: Path, collection_name: str, embedding_model: str, dimension: int
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._embedding_model = embedding_model
        self._dimension = dimension
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Sur une collection existante, get_or_create conserve ses métadonnées d'origine.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": embedding_model, "dimension": dimension},
            embedding_function=None,
        )
        self._check_compatibility(self._collection.metadata or {})

    @classmethod
    def from_settings(cls, settings: Settings, embedder: Embedder) -> ChromaVectorStore:
        """Collection liée au modèle qui produit réellement les vecteurs : celui de l'embedder."""
        return cls(
            settings.chroma_dir,
            settings.chroma_collection,
            embedder.model_name,
            embedder.dimension,
        )

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks mais {len(embeddings)} vecteurs")
        if any(len(vector) != self._dimension for vector in embeddings):
            raise ValueError(f"tous les vecteurs doivent avoir la dimension {self._dimension}")
        batch = self._client.get_max_batch_size()
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            vectors: list[Sequence[float]] = list(embeddings[start : start + batch])
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in part],
                embeddings=vectors,
                documents=[chunk.text for chunk in part],
                metadatas=[self._to_metadata(chunk) for chunk in part],
            )

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k doit être strictement positif")
        available = self._collection.count()
        if available == 0:
            return []
        queries: list[Sequence[float]] = [query_embedding]
        result = self._collection.query(
            query_embeddings=queries,
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            # Distance cosinus de Chroma = 1 - similarité cosinus.
            SearchResult(chunk=self._to_chunk(chunk_id, text, meta), score=1.0 - distance)
            for chunk_id, text, meta, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def get_chunks(self, doc_id: str, chunk_indices: Sequence[int]) -> list[Chunk]:
        if not chunk_indices:
            return []
        membership: dict[Literal["$in", "$nin"], list[str | int | float | bool]] = {
            "$in": list(chunk_indices)
        }
        where: Where = {"$and": [{"doc_id": doc_id}, {"chunk_index": membership}]}
        result = self._collection.get(
            where=where,
            include=["documents", "metadatas"],
        )
        chunks = [
            self._to_chunk(chunk_id, text, meta)
            for chunk_id, text, meta in zip(
                result["ids"], result["documents"] or [], result["metadatas"] or [], strict=True
            )
        ]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def has_document(self, doc_id: str) -> bool:
        return bool(self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])["ids"])

    def delete_document(self, doc_id: str) -> int:
        ids = self._collection.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self._collection.count()

    def _check_compatibility(self, stored: Mapping[str, Any]) -> None:
        stored_model = stored.get("embedding_model")
        stored_dimension = stored.get("dimension")
        if stored_model != self._embedding_model or stored_dimension != self._dimension:
            raise EmbeddingModelMismatchError(
                f"la collection {self._collection.name!r} a été construite avec "
                f"{stored_model!r} (dimension {stored_dimension}) ; impossible de l'utiliser avec "
                f"{self._embedding_model!r} (dimension {self._dimension}). Utilisez une autre "
                "collection (DOCASSIST_CHROMA_COLLECTION) ou réindexez dans une collection vide."
            )

    def _to_metadata(self, chunk: Chunk) -> dict[str, str | int]:
        return {
            "doc_id": chunk.doc_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "embedding_model": self._embedding_model,
        }

    @staticmethod
    def _to_chunk(chunk_id: str, text: str, meta: Mapping[str, Any]) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id=str(meta["doc_id"]),
            filename=str(meta["filename"]),
            page_number=int(meta["page_number"]),
            chunk_index=int(meta["chunk_index"]),
            start_char=int(meta["start_char"]),
            text=text,
        )
