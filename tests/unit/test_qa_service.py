import pytest

from doc_assistant.domain.models import Chunk
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.qa_service import QAService
from fakes import FakeAnswerGenerator, FakeEmbedder, InMemoryVectorStore

TEXTS = ["Les chats dorment beaucoup.", "La bourse de Paris monte.", "Le football se joue à onze."]


def make_service(*, empty: bool = False) -> tuple[QAService, FakeAnswerGenerator]:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    if not empty:
        chunks = [Chunk(f"d:{i}", "d", "notes.pdf", i + 1, i, 0, t) for i, t in enumerate(TEXTS)]
        store.add(chunks, embedder.embed_documents(TEXTS))
    generator = FakeAnswerGenerator()
    return QAService(Retriever(embedder, store, default_top_k=2), generator), generator


def test_ask_sends_retrieved_passages_to_generator() -> None:
    service, generator = make_service()

    result = service.ask("Combien dorment les chats ?")

    question, passages = generator.calls[0]
    assert question == "Combien dorment les chats ?"
    assert len(passages) == 2  # top_k par défaut du retriever
    assert result.passages == tuple(passages)
    assert result.answer.citations[0].chunk == passages[0].chunk


def test_top_k_is_forwarded() -> None:
    service, generator = make_service()

    service.ask("chats", top_k=3)

    assert len(generator.calls[0][1]) == 3


def test_no_passages_abstains_without_calling_the_generator() -> None:
    service, generator = make_service(empty=True)

    result = service.ask("Combien dorment les chats ?")

    assert result.answer.abstained
    assert result.answer.model == "fake-llm"
    assert generator.calls == []


def test_empty_question_is_rejected_before_generation() -> None:
    service, generator = make_service()

    with pytest.raises(ValueError):
        service.ask("   ")
    assert generator.calls == []
