"""Assemblage des adaptateurs concrets (le seul endroit qui les connaît tous)."""

from __future__ import annotations

from doc_assistant.config import Settings
from doc_assistant.domain.errors import LLMConfigurationError
from doc_assistant.domain.ports import AnswerGenerator, Embedder
from doc_assistant.indexing.embedders import E5Embedder
from doc_assistant.indexing.vector_store import ChromaVectorStore
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.ingestion.pdf_loader import PypdfLoader
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.indexing_service import IndexingService
from doc_assistant.services.qa_service import QAService


def build_embedder(settings: Settings) -> Embedder:
    return E5Embedder.from_settings(settings)


def build_store(settings: Settings, embedder: Embedder) -> ChromaVectorStore:
    return ChromaVectorStore.from_settings(settings, embedder)


def build_indexing_service(settings: Settings, embedder: Embedder) -> IndexingService:
    return IndexingService(
        loader=PypdfLoader(),
        chunker=RecursiveChunker.from_settings(settings),
        embedder=embedder,
        store=build_store(settings, embedder),
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
            "(et GEMINI_API_KEY) pour utiliser la commande ask"
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
