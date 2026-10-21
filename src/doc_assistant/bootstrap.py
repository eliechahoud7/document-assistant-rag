"""Assemblage des adaptateurs concrets (le seul endroit qui les connaît tous)."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chromadb.api import ClientAPI

from doc_assistant.config import Settings
from doc_assistant.domain.errors import LLMConfigurationError
from doc_assistant.domain.ports import AnswerGenerator, Embedder, VectorStore
from doc_assistant.indexing.document_registry import DocumentRegistry
from doc_assistant.indexing.embedders import E5Embedder
from doc_assistant.indexing.vector_store import ChromaVectorStore, open_persistent_client
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.ingestion.pdf_loader import PypdfLoader
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.indexing_service import IndexingService
from doc_assistant.services.library_service import DocumentLibrary
from doc_assistant.services.qa_service import QAService
from doc_assistant.services.upload_service import MAX_UPLOAD_BYTES
from doc_assistant.services.user_libraries import UserIdentity, UserLibraries, UserLimits

UserStoreKind = Literal["chroma-cloud", "local"]
# Identifiants Chroma Cloud : lus uniquement dans l'environnement (Secrets de Streamlit
# Community Cloud), jamais dans un fichier versionné, jamais journalisés.
CHROMA_CLOUD_ENV = ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE")


def build_embedder(settings: Settings) -> Embedder:
    """Embedder E5 ; enveloppé d'une traduction des requêtes si elle est activée."""
    embedder = E5Embedder.from_settings(settings)
    if settings.query_translation == "none":
        return embedder
    from doc_assistant.retrieval.marian_translator import MarianTranslator
    from doc_assistant.retrieval.query_translation import TranslatingEmbedder

    # Modèle absent : TranslationModelNotFoundError dès la construction, avant toute requête.
    return TranslatingEmbedder(embedder, MarianTranslator.from_settings(settings))


def build_store(settings: Settings, embedder: Embedder) -> ChromaVectorStore:
    return ChromaVectorStore.from_settings(settings, embedder)


def build_indexing_service(
    settings: Settings, embedder: Embedder, store: VectorStore | None = None
) -> IndexingService:
    return IndexingService(
        loader=PypdfLoader(),
        chunker=RecursiveChunker.from_settings(settings),
        embedder=embedder,
        store=store if store is not None else build_store(settings, embedder),
    )


def build_document_library(
    settings: Settings,
    embedder: Embedder,
    *,
    protected: frozenset[str] = frozenset(),
    max_documents: int | None = None,
    max_bytes: int | None = None,
    max_pages: int | None = None,
    store: VectorStore | None = None,
) -> DocumentLibrary:
    """Bibliothèque de l'index ``settings.data_dir`` : un seul accès à l'index, partagé par
    l'indexation, la liste et la suppression. Les options (documents protégés, limites, index
    déjà ouvert) servent aux bibliothèques des visiteurs ; par défaut, aucune restriction
    supplémentaire."""
    store = store if store is not None else build_store(settings, embedder)
    return DocumentLibrary(
        build_indexing_service(settings, embedder, store),
        store,
        DocumentRegistry.for_data_dir(settings.data_dir),
        settings.uploads_dir,
        protected=protected,
        max_documents=max_documents,
        max_bytes=MAX_UPLOAD_BYTES if max_bytes is None else max_bytes,
        max_pages=max_pages,
    )


@dataclass(frozen=True, slots=True)
class UserStore:
    """Base des bibliothèques personnelles : la base de chaque utilisateur, et le nombre de
    bibliothèques qu'elle contient (plafond de comptes)."""

    client_for: Callable[[UserIdentity], ClientAPI]
    count_accounts: Callable[[], int]


def build_user_store(kind: UserStoreKind, local_dir: Path) -> UserStore:
    """Chroma Cloud : une base partagée (persistante, commune à tous les appareils), qui doit
    être réservée à l'application (toutes ses collections comptent comme des comptes). En
    local : un dossier ``local_dir/<clé>`` par utilisateur (développement et tests ; voir
    ``services/user_libraries.py`` pour la raison)."""
    if kind == "local":

        def count_local() -> int:
            folders = sorted(local_dir.iterdir()) if local_dir.is_dir() else []
            return sum(
                1 for f in folders if f.is_dir() and open_persistent_client(f).count_collections()
            )

        return UserStore(
            lambda identity: open_persistent_client(local_dir / identity.key), count_local
        )
    import chromadb

    api_key, tenant, database = (os.environ[name].strip() for name in CHROMA_CLOUD_ENV)
    client = chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
    return UserStore(lambda _identity: client, client.count_collections)


def build_user_libraries(
    settings: Settings, embedder: Embedder, store: UserStore, limits: UserLimits
) -> UserLibraries:
    return UserLibraries(
        store.client_for,
        embedder,
        limits,
        lambda vectors: build_indexing_service(settings, embedder, vectors),
        store.count_accounts,
    )


def build_retriever(
    settings: Settings, embedder: Embedder, store: VectorStore | None = None
) -> Retriever:
    return Retriever(
        embedder,
        store if store is not None else build_store(settings, embedder),
        default_top_k=settings.retrieval_top_k,
    )


def build_generator(settings: Settings) -> AnswerGenerator:
    """Générateur LLM ; erreur explicite si aucun fournisseur n'est configuré."""
    if settings.llm_provider == "none":
        raise LLMConfigurationError(
            "aucun fournisseur LLM configuré : définissez DOCASSIST_LLM_PROVIDER=gemini "
            "(et GEMINI_API_KEY) pour obtenir des réponses (commande ask, interface web)"
        )
    if settings.llm_provider == "gemini":
        from doc_assistant.generation.gemini_generator import GeminiGenerator

        return GeminiGenerator.from_settings(settings)
    raise LLMConfigurationError(f"fournisseur LLM non implémenté : {settings.llm_provider!r}")


def build_qa_service(
    settings: Settings,
    embedder: Embedder,
    generator: AnswerGenerator,
    *,
    store: VectorStore | None = None,
) -> QAService:
    """``store`` : index déjà ouvert à réutiliser (sinon, un nouvel accès à l'index)."""
    return QAService(
        build_retriever(settings, embedder, store),
        generator,
        context_neighbors=settings.context_neighbors,
    )
