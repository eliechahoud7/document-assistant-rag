"""Interfaces (ports) dont dépendent les services. Les implémentations sont des adaptateurs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from doc_assistant.domain.errors import EmbeddingModelMismatchError
from doc_assistant.domain.models import Answer, Chunk, Document, SearchResult


class DocumentLoader(Protocol):
    def load(self, path: Path) -> Document:
        """Lit un fichier et renvoie son contenu nettoyé, page par page."""
        ...


class Chunker(Protocol):
    def split(self, document: Document) -> list[Chunk]:
        """Découpe un document en chunks ordonnés, sans franchir les limites de page."""
        ...


class Embedder(Protocol):
    """Transforme des textes en vecteurs.

    Questions et passages ont des méthodes distinctes : les modèles asymétriques
    (comme E5) les encodent différemment.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """Stocke des chunks et leurs vecteurs, tous produits par un même modèle."""

    @property
    def embedding_model(self) -> str: ...

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        """Renvoie au plus ``top_k`` résultats, du plus similaire au moins similaire."""
        ...

    def get_chunks(self, doc_id: str, chunk_indices: Sequence[int]) -> list[Chunk]:
        """Chunks du document aux positions demandées (celles qui existent), triés par position."""
        ...

    def has_document(self, doc_id: str) -> bool: ...

    def delete_document(self, doc_id: str) -> int:
        """Supprime tous les chunks du document et renvoie leur nombre."""
        ...

    def count(self) -> int: ...


class AnswerGenerator(Protocol):
    """Rédige une réponse fondée uniquement sur les passages fournis, ou s'abstient."""

    @property
    def model_name(self) -> str: ...

    def generate(self, question: str, passages: Sequence[SearchResult]) -> Answer: ...


def ensure_same_embedding_model(embedder: Embedder, store: VectorStore) -> None:
    if embedder.model_name != store.embedding_model:
        raise EmbeddingModelMismatchError(
            f"l'index utilise le modèle {store.embedding_model!r} "
            f"mais l'embedder est {embedder.model_name!r}"
        )
