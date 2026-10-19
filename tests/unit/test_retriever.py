import pytest

from doc_assistant.domain.errors import EmbeddingModelMismatchError
from doc_assistant.domain.models import Chunk
from doc_assistant.retrieval.retriever import Retriever
from fakes import FakeEmbedder, InMemoryVectorStore

TEXTS = [
    "Les chats dorment beaucoup pendant la journée.",
    "La bourse de Paris a clôturé en hausse.",
    "Le football est un sport collectif.",
]


def make_retriever(default_top_k: int = 2) -> tuple[Retriever, FakeEmbedder]:
    embedder = FakeEmbedder()
    store = InMemoryVectorStore()
    chunks = [Chunk(f"d:{i}", "d", "notes.pdf", i + 1, i, 0, text) for i, text in enumerate(TEXTS)]
    store.add(chunks, embedder.embed_documents(TEXTS))
    return Retriever(embedder, store, default_top_k=default_top_k), embedder


def test_returns_most_similar_chunks_first() -> None:
    retriever, _ = make_retriever()

    results = retriever.search("Combien de temps dorment les chats ?")

    assert results[0].chunk.page_number == 1
    assert results[0].score >= results[1].score


def test_top_k() -> None:
    retriever, _ = make_retriever(default_top_k=2)

    assert len(retriever.search("chats")) == 2
    assert len(retriever.search("chats", top_k=1)) == 1
    assert len(retriever.search("chats", top_k=10)) == 3


def test_question_is_stripped_and_embedded_as_query() -> None:
    retriever, embedder = make_retriever()

    retriever.search("  les chats  ")

    assert embedder.query_calls == ["les chats"]


@pytest.mark.parametrize("question", ["", "   \n"])
def test_empty_question_is_rejected(question: str) -> None:
    retriever, _ = make_retriever()

    with pytest.raises(ValueError, match="vide"):
        retriever.search(question)


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    retriever, _ = make_retriever()

    with pytest.raises(ValueError):
        retriever.search("chats", top_k=top_k)


def test_invalid_default_top_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        Retriever(FakeEmbedder(), InMemoryVectorStore(), default_top_k=0)


def test_embedder_and_store_must_use_the_same_model() -> None:
    with pytest.raises(EmbeddingModelMismatchError):
        Retriever(FakeEmbedder(model_name="autre-modele"), InMemoryVectorStore("fake-hash"))


def make_neighbor_retriever() -> Retriever:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    texts = ["alpha one", "beta two", "gamma three", "delta four"]
    chunks = [Chunk(f"d:{i}", "d", "doc.pdf", 1 + i // 2, i, 0, t) for i, t in enumerate(texts)]
    chunks.append(Chunk("e:1", "e", "other.pdf", 1, 1, 0, "beta two other"))
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return Retriever(embedder, store)


def test_expand_with_neighbors_order_dedup_and_boundaries() -> None:
    retriever = make_neighbor_retriever()
    hits = [r for r in retriever.search("beta two", top_k=5) if r.chunk.doc_id == "d"]
    first, last = (
        next(r for r in hits if r.chunk.chunk_index == 1),
        next(r for r in retriever.search("delta four", top_k=5) if r.chunk.chunk_id == "d:3"),
    )

    expanded = retriever.expand_with_neighbors([first, last], window=1)

    assert [r.chunk.chunk_id for r in expanded] == ["d:0", "d:1", "d:2", "d:3"]  # d:2 non répété
    assert [r.score for r in expanded[:3]] == [first.score] * 3  # score hérité du résultat
    assert all(r.chunk.doc_id == "d" for r in expanded)  # jamais un autre document


def test_expand_with_window_zero_is_identity() -> None:
    retriever = make_neighbor_retriever()
    results = retriever.search("beta two", top_k=2)

    assert retriever.expand_with_neighbors(results, window=0) == results


def test_expand_rejects_negative_window() -> None:
    retriever = make_neighbor_retriever()

    with pytest.raises(ValueError):
        retriever.expand_with_neighbors([], window=-1)
