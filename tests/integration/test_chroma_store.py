"""ChromaDB réel, dans un dossier temporaire, avec des embeddings factices."""

import shutil
from pathlib import Path

import pytest

from doc_assistant.domain.errors import EmbeddingModelMismatchError
from doc_assistant.domain.models import Chunk
from doc_assistant.domain.ports import VectorStore
from doc_assistant.indexing.vector_store import ChromaVectorStore
from fakes import FakeEmbedder

EMBEDDER = FakeEmbedder(dimension=64)


def open_store(path: Path, model: str = "fake-hash", dimension: int = 64) -> ChromaVectorStore:
    return ChromaVectorStore(path, "chunks", embedding_model=model, dimension=dimension)


def make_chunks(doc_id: str, texts: list[str], filename: str = "doc.pdf") -> list[Chunk]:
    return [
        Chunk(
            f"{doc_id}:{i}",
            doc_id,
            filename,
            page_number=i + 1,
            chunk_index=i,
            start_char=10 * i,
            text=text,
        )
        for i, text in enumerate(texts)
    ]


def fill(store: ChromaVectorStore, chunks: list[Chunk]) -> None:
    store.add(chunks, EMBEDDER.embed_documents([chunk.text for chunk in chunks]))


def test_search_orders_by_similarity_and_restores_metadata(tmp_path: Path) -> None:
    store: VectorStore = open_store(tmp_path)
    chunks = make_chunks("d1", ["les chats dorment", "la bourse monte", "les chats mangent"])
    store.add(chunks, EMBEDDER.embed_documents([chunk.text for chunk in chunks]))

    results = store.search(EMBEDDER.embed_query("les chats dorment"), top_k=2)

    assert [result.chunk for result in results] == [chunks[0], chunks[2]]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[0].score > results[1].score


def test_embedding_model_is_stored_in_chunk_metadata(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    fill(store, make_chunks("d1", ["texte"]))

    assert store._collection is not None
    stored = store._collection.get(include=["metadatas"])["metadatas"]

    assert stored is not None
    assert stored[0] == {
        "doc_id": "d1",
        "filename": "doc.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "start_char": 0,
        "embedding_model": "fake-hash",
    }


def test_data_persists_after_reopening(tmp_path: Path) -> None:
    fill(open_store(tmp_path), make_chunks("d1", ["un", "deux"]))

    reopened = open_store(tmp_path)

    assert reopened.count() == 2
    assert reopened.has_document("d1")


def test_closing_releases_the_files_so_the_folder_can_be_deleted(tmp_path: Path) -> None:
    """Régression (Windows) : un index ouvert empêchait la suppression de son dossier."""
    folder = tmp_path / "session" / "chroma"
    store = open_store(folder)
    fill(store, make_chunks("d1", ["un", "deux"]))

    store.close()
    store.close()  # idempotent
    shutil.rmtree(folder.parent)

    assert not folder.parent.exists()


def test_reopening_with_another_model_is_refused(tmp_path: Path) -> None:
    fill(open_store(tmp_path, model="modele-a"), make_chunks("d1", ["texte"]))

    with pytest.raises(EmbeddingModelMismatchError, match="modele-a"):
        open_store(tmp_path, model="modele-b")
    with pytest.raises(EmbeddingModelMismatchError):
        open_store(tmp_path, model="modele-a", dimension=128)
    assert open_store(tmp_path, model="modele-a").count() == 1  # rien n'a été modifié


def test_delete_document_only_removes_its_chunks(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    fill(store, make_chunks("d1", ["a", "b"]) + make_chunks("d2", ["c"]))

    assert store.delete_document("d1") == 2
    assert store.delete_document("d1") == 0
    assert not store.has_document("d1")
    assert store.has_document("d2")
    assert store.count() == 1


def test_adding_same_ids_twice_does_not_duplicate(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    chunks = make_chunks("d1", ["a", "b"])

    fill(store, chunks)
    fill(store, chunks)

    assert store.count() == 2


def test_empty_store_returns_no_results(tmp_path: Path) -> None:
    assert open_store(tmp_path).search(EMBEDDER.embed_query("question"), top_k=5) == []


def test_top_k_larger_than_collection(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    fill(store, make_chunks("d1", ["a", "b"]))

    assert len(store.search(EMBEDDER.embed_query("a"), top_k=50)) == 2


def test_large_inputs_are_split_into_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = open_store(tmp_path)
    monkeypatch.setattr(store._client, "get_max_batch_size", lambda: 3)

    fill(store, make_chunks("d1", [f"texte {i}" for i in range(10)]))

    assert store.count() == 10


@pytest.mark.parametrize(
    ("chunk_count", "vectors"),
    [(2, [[0.0] * 64]), (1, [[0.0] * 10])],
)
def test_invalid_embeddings_are_rejected(
    tmp_path: Path, chunk_count: int, vectors: list[list[float]]
) -> None:
    store = open_store(tmp_path)

    with pytest.raises(ValueError):
        store.add(make_chunks("d1", ["a", "b"][:chunk_count]), vectors)


def test_invalid_top_k_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        open_store(tmp_path).search([0.0] * 64, top_k=0)


def test_get_chunks_by_position(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    fill(store, make_chunks("d1", ["a", "b", "c"]) + make_chunks("d2", ["x", "y"]))

    found = store.get_chunks("d1", [2, 0, 7])

    assert [(c.doc_id, c.chunk_index, c.text) for c in found] == [("d1", 0, "a"), ("d1", 2, "c")]
    assert found[0] == make_chunks("d1", ["a"])[0]  # métadonnées restituées à l'identique
    assert store.get_chunks("d1", []) == []
    assert store.get_chunks("absent", [0]) == []
