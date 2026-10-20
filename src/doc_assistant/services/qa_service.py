"""Cas d'usage : répondre à une question à partir des passages retrouvés."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from doc_assistant.domain.errors import DocumentScopeError
from doc_assistant.domain.models import Answer, SearchResult
from doc_assistant.domain.ports import AnswerGenerator
from doc_assistant.generation.answers import no_passage_answer
from doc_assistant.retrieval.retriever import Retriever


@dataclass(frozen=True, slots=True)
class QAResult:
    answer: Answer
    passages: tuple[SearchResult, ...]


class QAService:
    """Recherche (identique à la commande ``search``) puis génération.

    Aucun seuil de similarité n'est appliqué : l'évaluation a montré que les scores ne
    séparent pas les questions répondables des autres. Sans passage, on s'abstient sans
    appeler le LLM. ``context_neighbors`` (0 par défaut) ajoute à chaque passage retrouvé
    ses chunks voisins, pour les réponses coupées à une frontière de chunk ou de page.
    """

    def __init__(
        self, retriever: Retriever, generator: AnswerGenerator, *, context_neighbors: int = 0
    ) -> None:
        if context_neighbors < 0:
            raise ValueError("context_neighbors doit être positif ou nul")
        self._retriever = retriever
        self._generator = generator
        self.context_neighbors = context_neighbors

    def retrieve(
        self, question: str, top_k: int | None = None, *, doc_id: str | None = None
    ) -> list[SearchResult]:
        """Passages envoyés au LLM. Avec ``doc_id``, recherche et voisins (même document)
        proviennent uniquement de ce document."""
        results = self._retriever.search(question, top_k=top_k, doc_id=doc_id)
        passages = self._retriever.expand_with_neighbors(results, self.context_neighbors)
        foreign = sorted({p.chunk.doc_id for p in passages} - {doc_id}) if doc_id else []
        if foreign:
            raise DocumentScopeError(
                f"passages d'autres documents ({', '.join(foreign)}) retrouvés pour le document "
                f"{doc_id} : rien n'est envoyé au LLM"
            )
        return passages

    def answer(self, question: str, passages: Sequence[SearchResult]) -> QAResult:
        if not passages:
            return QAResult(no_passage_answer(self._generator.model_name), ())
        return QAResult(self._generator.generate(question.strip(), passages), tuple(passages))

    def ask(
        self, question: str, top_k: int | None = None, *, doc_id: str | None = None
    ) -> QAResult:
        return self.answer(question, self.retrieve(question, top_k=top_k, doc_id=doc_id))
