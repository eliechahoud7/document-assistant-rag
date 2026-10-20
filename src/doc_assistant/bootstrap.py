"""Assemblage des adaptateurs concrets (le seul endroit qui les connaît tous)."""

from __future__ import annotations

from doc_assistant.config import Settings
from doc_assistant.domain.errors import LLMConfigurationError
from doc_assistant.domain.ports import AnswerGenerator, Embedder, VectorStore
from doc_assistant.indexing.document_registry import DocumentRegistry
from doc_assistant.indexing.embedders import E5Embedder
from doc_assistant.indexing.vector_store import ChromaVectorStore
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.ingestion.pdf_loader import PypdfLoader
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.indexing_service import IndexingService
from doc_assistant.services.library_service import DocumentLibrary
from doc_assistant.services.qa_service import QAService


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


def build_document_library(settings: Settings, embedder: Embedder) -> DocumentLibrary:
    """Bibliothèque de l'index ``settings.data_dir`` : un seul accès à l'index, partagé par
    l'indexation, la liste et la suppression."""
    store = build_store(settings, embedder)
    return DocumentLibrary(
        build_indexing_service(settings, embedder, store),
        store,
        DocumentRegistry.for_data_dir(settings.data_dir),
        settings.uploads_dir,
    )


def build_retriever(settings: Settings, embedder: Embedder) -> Retriever:
    return Retriever(
        embedder, build_store(settings, embedder), default_top_k=settings.retrieval_top_k
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
    settings: Settings, embedder: Embedder, generator: AnswerGenerator
) -> QAService:
    return QAService(
        build_retriever(settings, embedder),
        generator,
        context_neighbors=settings.context_neighbors,
    )
