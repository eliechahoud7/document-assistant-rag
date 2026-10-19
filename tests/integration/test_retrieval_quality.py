"""Test de fumée de la qualité de recherche, avec le vrai modèle E5 (marqueur ``slow``).

Lancer avec : ``python -m pytest -m slow -s`` (``-s`` affiche le tableau des résultats).
Ce n'est pas une évaluation statistiquement solide (12 questions) : voir la phase 3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from doc_assistant.config import Settings
from doc_assistant.domain.ports import Embedder
from doc_assistant.indexing.embedders import E5Embedder
from doc_assistant.indexing.vector_store import ChromaVectorStore
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.ingestion.pdf_loader import PypdfLoader
from doc_assistant.retrieval.metrics import hit_rate_at_k, mean_reciprocal_rank
from doc_assistant.retrieval.retriever import Retriever
from doc_assistant.services.indexing_service import IndexingService
from fakes import FakeEmbedder
from pdf_factory import write_pdf

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
CORPUS = json.loads((ROOT / "tests" / "fixtures" / "retrieval_corpus.json").read_text("utf-8"))
SETTINGS = Settings(models_dir=ROOT / "models")
TOP_K = 5


@dataclass(frozen=True)
class Evaluation:
    ranks: list[int | None]

    @property
    def hit1(self) -> float:
        return hit_rate_at_k(self.ranks, 1)

    @property
    def hit3(self) -> float:
        return hit_rate_at_k(self.ranks, 3)

    @property
    def mrr(self) -> float:
        return mean_reciprocal_rank(self.ranks)


def evaluate(embedder: Embedder, workdir: Path) -> Evaluation:
    pdf_dir = workdir / "pdfs"
    pdf_dir.mkdir(parents=True)
    store = ChromaVectorStore(workdir / "chroma", "eval", embedder.model_name, embedder.dimension)
    service = IndexingService(
        PypdfLoader(), RecursiveChunker.from_settings(SETTINGS), embedder, store
    )
    for document in CORPUS["documents"]:
        pages = [[text] for text in document["pages"]]
        service.index_pdf(write_pdf(pdf_dir / document["filename"], pages))

    retriever = Retriever(embedder, store)
    ranks: list[int | None] = []
    for query in CORPUS["queries"]:
        found = [
            (r.chunk.filename, r.chunk.page_number)
            for r in retriever.search(query["question"], top_k=TOP_K)
        ]
        target = (query["filename"], query["page"])
        ranks.append(found.index(target) + 1 if target in found else None)
    return Evaluation(ranks)


@pytest.fixture(scope="module")
def e5() -> E5Embedder:
    if not (SETTINGS.embedding_model_dir / "model.safetensors").is_file():
        pytest.skip("modèle E5 absent : python -m doc_assistant download-model")
    return E5Embedder.from_settings(SETTINGS)


def test_semantic_retrieval_quality(e5: E5Embedder, tmp_path: Path) -> None:
    semantic = evaluate(e5, tmp_path / "e5")
    lexical = evaluate(FakeEmbedder(), tmp_path / "lexical")

    print(f"\n{'question':<75} {'langues':<7} {'E5':>4} {'lexical':>8}")
    for query, rank_e5, rank_lex in zip(
        CORPUS["queries"], semantic.ranks, lexical.ranks, strict=True
    ):
        print(
            f"{query['question'][:74]:<75} {query['langs']:<7} {rank_e5 or '-':>4} "
            f"{rank_lex or '-':>8}"
        )
    for name, ev in (("E5", semantic), ("lexical", lexical)):
        print(f"{name:<8} Hit@1={ev.hit1:.2f}  Hit@3={ev.hit3:.2f}  MRR={ev.mrr:.3f}")

    # Mesuré le 2026-10-19 (e5-small, commit 614241f6) : E5 Hit@1 = Hit@3 = MRR = 1.00 ;
    # lexical Hit@1 = 0.08, Hit@3 = 0.42, MRR = 0.22. Score parfait = corpus facile (sujets
    # très éloignés) : ce test détecte une régression grossière, il ne compare pas des modèles.
    # Les seuils tolèrent une erreur sur 12 questions.
    assert semantic.hit3 >= 0.9
    assert semantic.mrr >= 0.8
    assert semantic.mrr > lexical.mrr
