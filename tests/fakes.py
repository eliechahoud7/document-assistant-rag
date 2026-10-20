"""Implémentations factices et déterministes des ports, pour des tests rapides et hors ligne."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from doc_assistant.domain.models import Answer, Chunk, Citation, IndexedDocument, SearchResult

_WORD_RE = re.compile(r"\w+")


class FakeEmbedder:
    """Sac de mots haché : deux textes partageant des mots ont des vecteurs proches.

    C'est une recherche purement lexicale, sans aucune notion de sens ni de langue :
    elle sert aussi de point de comparaison pour mesurer l'apport d'un vrai modèle.
    """

    def __init__(self, dimension: int = 256, model_name: str = "fake-hash") -> None:
        self._dimension = dimension
        self._model_name = model_name
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for word in _WORD_RE.findall(text.lower()):
            bucket = int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "little")
            vector[bucket % self._dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


class InMemoryVectorStore:
    """Recherche exhaustive par similarité cosinus, en mémoire."""

    def __init__(self, embedding_model: str = "fake-hash") -> None:
        self._embedding_model = embedding_model
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("autant de vecteurs que de chunks attendus")
        for chunk, vector in zip(chunks, embeddings, strict=True):
            self._items[chunk.chunk_id] = (chunk, list(vector))

    def search(
        self, query_embedding: Sequence[float], top_k: int, *, doc_id: str | None = None
    ) -> list[SearchResult]:
        scored = [
            SearchResult(chunk, sum(a * b for a, b in zip(query_embedding, vector, strict=True)))
            for chunk, vector in self._items.values()
            if doc_id is None or chunk.doc_id == doc_id
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def get_chunks(self, doc_id: str, chunk_indices: Sequence[int]) -> list[Chunk]:
        wanted = set(chunk_indices)
        found = [
            chunk
            for chunk, _ in self._items.values()
            if chunk.doc_id == doc_id and chunk.chunk_index in wanted
        ]
        return sorted(found, key=lambda chunk: chunk.chunk_index)

    def has_document(self, doc_id: str) -> bool:
        return any(chunk.doc_id == doc_id for chunk, _ in self._items.values())

    def delete_document(self, doc_id: str) -> int:
        ids = [cid for cid, (chunk, _) in self._items.items() if chunk.doc_id == doc_id]
        for chunk_id in ids:
            del self._items[chunk_id]
        return len(ids)

    def count(self) -> int:
        return len(self._items)

    def list_documents(self) -> list[IndexedDocument]:
        stats: dict[str, IndexedDocument] = {}
        for chunk, _ in self._items.values():
            known = stats.get(chunk.doc_id)
            stats[chunk.doc_id] = IndexedDocument(
                chunk.doc_id,
                known.filename if known else chunk.filename,
                (known.chunk_count if known else 0) + 1,
                max(known.last_page if known else 0, chunk.page_number),
            )
        return sorted(stats.values(), key=lambda d: (d.filename.lower(), d.doc_id))


class FakeAnswerGenerator:
    """Générateur factice : cite le premier passage, ou s'abstient si ``abstain``."""

    def __init__(self, model_name: str = "fake-llm", *, abstain: bool = False) -> None:
        self._model_name = model_name
        self.abstain = abstain
        self.calls: list[tuple[str, list[SearchResult]]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, question: str, passages: Sequence[SearchResult]) -> Answer:
        self.calls.append((question, list(passages)))
        if self.abstain or not passages:
            return Answer("", (), True, self._model_name, abstain_reason="fake abstention")
        first = passages[0].chunk
        return Answer(
            f"Réponse fondée sur {first.filename} [1].",
            (Citation(1, first),),
            False,
            self._model_name,
        )


class ScriptedAnswerGenerator:
    """Générateur scripté par question : texte de réponse (cite le passage 1), "abstain",
    ou exception à lever. Les questions absentes du script citent le passage 1.
    """

    def __init__(self, script: dict[str, str | Exception], model_name: str = "scripted") -> None:
        self.script = script
        self._model_name = model_name
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, question: str, passages: Sequence[SearchResult]) -> Answer:
        self.calls.append(question)
        action = self.script.get(question, "Réponse [1].")
        if isinstance(action, Exception):
            raise action
        if action == "abstain" or not passages:
            return Answer("", (), True, self._model_name, abstain_reason="scripted abstention")
        return Answer(action, (Citation(1, passages[0].chunk),), False, self._model_name)


class KeywordAnswerGenerator:
    """Générateur sensible aux preuves : répond seulement si ``keyword`` figure dans un passage
    fourni (et cite ce passage), sinon s'abstient. Modélise « répondre si et seulement si
    l'information est dans le contexte ».
    """

    def __init__(self, keyword: str, model_name: str = "keyword") -> None:
        self.keyword = keyword.lower()
        self._model_name = model_name
        self.calls: list[list[SearchResult]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, question: str, passages: Sequence[SearchResult]) -> Answer:
        self.calls.append(list(passages))
        for number, result in enumerate(passages, start=1):
            if self.keyword in result.chunk.text.lower():
                return Answer(
                    f"{self.keyword} [{number}]",
                    (Citation(number, result.chunk),),
                    False,
                    self._model_name,
                )
        return Answer("", (), True, self._model_name, abstain_reason="information absente")
