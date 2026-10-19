"""Extension du contexte aux chunks voisins : cas inspirés des échecs Q05, Q08, Q14, Q15.

Index en mémoire, embeddings lexicaux factices, générateur « sensible aux preuves » qui ne
répond que si l'information est réellement dans les passages reçus. Aucun appel réseau.
"""

from __future__ import annotations

import pytest

from doc_assistant.domain.models import Chunk
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.qa_service import QAService
from fakes import FakeEmbedder, InMemoryVectorStore, KeywordAnswerGenerator

LECTURE = [
    # (index, page, texte) : un même document, chunks consécutifs
    (12, 5, "Labeling errors are common in many practical problems."),
    (
        13,
        5,
        "Allowing misclassified examples. Perhaps the simplest way to permit errors in the "
        "maximum margin classifier is to",
    ),
    (14, 5, "introduce slack variables for the margin constraints in the optimization problem."),
    (30, 2, "Estimating the coefficients requires inverting an n by n matrix. This is the cost of"),
    (
        31,
        2,
        "dealing with inner products as opposed to feature vectors directly; the benefit is "
        "substantial.",
    ),
    (
        133,
        38,
        "For example, we fit our model on 60% of the training set and use the remaining 40%.",
    ),
    (
        134,
        39,
        "More, less? How much of our data can we spare for a validation set? With k-fold we "
        "train on 80 instances and test on 20.",
    ),
    (200, 80, "Clustering assigns each instance to the closest centroid; compute the centroids."),
    (201, 81, "How to choose k? Look for an elbow in the error curve, or use silhouette analysis."),
]
OTHER_DOC = [(14, 5, "Unrelated document chunk with the same index: slack 40% elbow inverting.")]


def build(*, neighbors: int, keyword: str) -> tuple[QAService, KeywordAnswerGenerator]:
    embedder, store = FakeEmbedder(), InMemoryVectorStore()
    chunks = [Chunk(f"lec:{i}", "lec", "lecture.pdf", p, i, 0, t) for i, p, t in LECTURE]
    chunks += [Chunk(f"oth:{i}", "oth", "other.pdf", p, i, 0, t) for i, p, t in OTHER_DOC]
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    generator = KeywordAnswerGenerator(keyword)
    retriever = Retriever(embedder, store, default_top_k=1)
    return QAService(retriever, generator, context_neighbors=neighbors), generator


@pytest.mark.parametrize(
    ("label", "question", "keyword", "expected_page"),
    [
        (
            "Q05 : la suite est dans le chunk suivant",
            "simplest way to permit errors in the maximum margin classifier",
            "slack",
            5,
        ),
        (
            "Q08 : la réponse termine le chunk précédent",
            "dealing with inner products as opposed to feature vectors",
            "inverting",
            2,
        ),
        (
            "Q14 : la réponse est sur la page précédente",
            "how much of our data can we spare for a validation set",
            "40%",
            38,
        ),
        (
            "Q15 : la réponse est dans le chunk suivant",
            "clustering assigns each instance to the closest centroid",
            "silhouette",
            81,
        ),
    ],
)
def test_neighbors_bring_the_missing_evidence(
    label: str, question: str, keyword: str, expected_page: int
) -> None:
    without, _ = build(neighbors=0, keyword=keyword)
    with_neighbors, generator = build(neighbors=1, keyword=keyword)

    before = without.ask(question).answer
    after = with_neighbors.ask(question).answer

    assert before.abstained, label  # comportement d'origine : l'information n'est pas fournie
    assert not after.abstained, label
    cited = after.citations[0]
    # La citation est le chunk voisin réel : page et fichier d'origine, jamais inventés.
    assert (cited.filename, cited.page_number) == ("lecture.pdf", expected_page)
    assert keyword in cited.chunk.text.lower()
    assert all(p.chunk.doc_id == "lec" for p in generator.calls[0])  # pas d'autre document


def test_genuinely_unanswerable_question_still_abstains() -> None:
    service, generator = build(neighbors=1, keyword="batch normalization")

    result = service.ask("permit errors in the maximum margin classifier")

    assert result.answer.abstained
    assert len(generator.calls[0]) == 3  # contexte étendu fourni, mais sans l'information


def test_no_passages_still_means_no_llm_call() -> None:
    generator = KeywordAnswerGenerator("slack")
    service = QAService(
        Retriever(FakeEmbedder(), InMemoryVectorStore()), generator, context_neighbors=1
    )

    assert service.ask("anything").answer.abstained
    assert generator.calls == []


def test_default_keeps_original_context() -> None:
    service, generator = build(neighbors=0, keyword="slack")

    service.ask("simplest way to permit errors in the maximum margin classifier")

    assert [p.chunk.chunk_index for p in generator.calls[0]] == [13]


def test_negative_neighbors_rejected() -> None:
    with pytest.raises(ValueError):
        build(neighbors=-1, keyword="x")
