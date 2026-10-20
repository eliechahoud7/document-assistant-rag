"""Régression : afficher la bibliothèque ne charge pas le modèle d'embeddings.

Avant correctif, ouvrir l'index demandait ``embedder.dimension``, ce qui chargeait le modèle
E5 (≈ 9 s mesurées) à chaque ouverture de la page, simplement pour lister les documents.
L'embedder ci-dessous échoue s'il est chargé : listes, suppressions et service de réponses
doivent fonctionner sans lui ; seuls l'indexation et la recherche l'utilisent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import MakePdf
from doc_assistant import bootstrap
from doc_assistant.config import Settings
from doc_assistant.domain.errors import EmbeddingModelMismatchError
from doc_assistant.indexing.vector_store import ChromaVectorStore
from fakes import FakeEmbedder, ScriptedAnswerGenerator


class ModelLoadError(AssertionError):
    pass


class UnloadedEmbedder(FakeEmbedder):
    """Même nom de modèle que l'index ; tout usage qui chargerait le modèle échoue."""

    @property
    def dimension(self) -> int:
        raise ModelLoadError("le modèle d'embeddings a été chargé")


@pytest.fixture
def settings(tmp_path: Path, make_pdf: MakePdf) -> Settings:
    settings = Settings(data_dir=tmp_path / "data")
    library = bootstrap.build_document_library(settings, FakeEmbedder())
    for name, line in (("a.pdf", "Les loutres dorment."), ("b.pdf", "Les volcans grondent.")):
        library.add(name, make_pdf([[line]], name=name).read_bytes())
    return settings


def test_listing_and_deleting_do_not_load_the_model(settings: Settings) -> None:
    library = bootstrap.build_document_library(settings, UnloadedEmbedder())

    names = [e.filename for e in library.documents()]
    removed = library.delete(next(e.doc_id for e in library.documents() if e.filename == "a.pdf"))

    assert names == ["a.pdf", "b.pdf"]
    assert removed.chunks_removed == 1
    assert [e.filename for e in library.documents()] == ["b.pdf"]


def test_building_the_answer_service_does_not_load_the_model(settings: Settings) -> None:
    bootstrap.build_qa_service(settings, UnloadedEmbedder(), ScriptedAnswerGenerator({}))


def test_existing_collection_never_asks_for_the_dimension(settings: Settings) -> None:
    def never() -> int:
        raise ModelLoadError("dimension demandée")

    store = ChromaVectorStore(settings.chroma_dir, "chunks", "fake-hash", never)

    assert store.count() == 2


def test_new_collection_still_asks_for_the_dimension(tmp_path: Path) -> None:
    calls: list[int] = []

    def provider() -> int:
        calls.append(1)
        return 8

    ChromaVectorStore(tmp_path / "chroma", "chunks", "fake-hash", provider)

    assert calls == [1]  # créer une collection exige la dimension du modèle


def test_other_model_is_still_rejected_without_loading_it(settings: Settings) -> None:
    def never() -> int:
        raise ModelLoadError("dimension demandée")

    with pytest.raises(EmbeddingModelMismatchError):
        ChromaVectorStore(settings.chroma_dir, "chunks", "autre/modele", never)
