"""Adaptateur d'embeddings E5 (sentence-transformers, CPU, fichiers locaux uniquement)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from doc_assistant.config import Settings
from doc_assistant.domain.errors import EmbeddingModelNotFoundError


class SentenceEncoder(Protocol):
    """Sous-ensemble de ``SentenceTransformer`` utilisé ici (permet d'injecter un faux)."""

    def encode(self, inputs: list[str], **kwargs: Any) -> Any: ...

    def get_embedding_dimension(self) -> int | None: ...


class E5Embedder:
    """Embeddings ``intfloat/multilingual-e5-*``.

    E5 a été entraîné avec des préfixes : ``"query: "`` pour les questions et
    ``"passage: "`` pour les textes indexés. Les omettre dégrade nettement la recherche.
    Les vecteurs sont normalisés (norme 1) : la similarité cosinus est alors un produit
    scalaire.
    """

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        model_dir: Path,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        encoder: SentenceEncoder | None = None,
    ) -> None:
        self.model_dir = model_dir
        self._model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._encoder = encoder

    @classmethod
    def from_settings(cls, settings: Settings) -> E5Embedder:
        return cls(
            settings.embedding_model_dir,
            settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        dimension = self._load().get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"dimension inconnue pour le modèle {self._model_name}")
        return dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self.PASSAGE_PREFIX + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.QUERY_PREFIX + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result: list[list[float]] = vectors.tolist()
        return result

    def _load(self) -> SentenceEncoder:
        """Charge le modèle au premier usage : l'import de torch prend plusieurs secondes."""
        if self._encoder is None:
            if not (self.model_dir / "model.safetensors").is_file():
                raise EmbeddingModelNotFoundError(
                    f"modèle introuvable dans {self.model_dir}. "
                    "Téléchargez-le avec : python -m doc_assistant download-model"
                )
            # Garantie supplémentaire : la bibliothèque Hugging Face ne contacte pas le Hub.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                str(self.model_dir), device=self.device, local_files_only=True
            )
        return self._encoder
