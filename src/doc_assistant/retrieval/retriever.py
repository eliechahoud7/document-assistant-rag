"""Recherche sémantique : question -> passages les plus proches."""

from __future__ import annotations

from collections.abc import Sequence

from doc_assistant.domain.models import SearchResult
from doc_assistant.domain.ports import Embedder, VectorStore, ensure_same_embedding_model


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, *, default_top_k: int = 5) -> None:
        ensure_same_embedding_model(embedder, store)
        if default_top_k <= 0:
            raise ValueError("default_top_k doit être strictement positif")
        self._embedder = embedder
        self._store = store
        self.default_top_k = default_top_k

    def search(
        self, question: str, top_k: int | None = None, *, doc_id: str | None = None
    ) -> list[SearchResult]:
        """Passages les plus proches ; ``doc_id`` limite la recherche à un seul document."""
        question = question.strip()
        if not question:
            raise ValueError("la question ne peut pas être vide")
        k = self.default_top_k if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k doit être strictement positif")
        return self._store.search(self._embedder.embed_query(question), k, doc_id=doc_id)

    def expand_with_neighbors(
        self, results: Sequence[SearchResult], window: int
    ) -> list[SearchResult]:
        """Ajoute à chaque résultat ses ``window`` chunks voisins dans le même document.

        Un passage coupé en fin de chunk (ou de page) retrouve ainsi sa suite. L'ordre suit
        le classement : pour chaque résultat, ses voisins précédents, lui-même, ses suivants ;
        un chunk déjà présent n'est pas répété. Un voisin reprend le score du résultat qui l'a
        amené (il n'a pas été classé lui-même). Le classement de ``search`` est inchangé.
        """
        if window < 0:
            raise ValueError("window doit être positif ou nul")
        if window == 0:
            return list(results)
        expanded: list[SearchResult] = []
        seen: set[str] = set()
        for result in results:
            anchor = result.chunk
            indices = range(anchor.chunk_index - window, anchor.chunk_index + window + 1)
            wanted = [i for i in indices if i >= 0]
            for chunk in self._store.get_chunks(anchor.doc_id, wanted):
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    expanded.append(SearchResult(chunk, result.score))
        return expanded
