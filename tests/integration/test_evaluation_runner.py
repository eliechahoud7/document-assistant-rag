"""Évaluation de bout en bout avec ChromaDB temporaire et embeddings factices (lexicaux)."""

import json
from pathlib import Path
from typing import Any

import pytest

from doc_assistant.domain.models import Chunk
from doc_assistant.evaluation.dataset import load_dataset
from doc_assistant.evaluation.runner import (
    EvaluationRun,
    build_report,
    format_summary,
    run_evaluation,
)
from doc_assistant.indexing.vector_store import ChromaVectorStore
from doc_assistant.ingestion.chunking import RecursiveChunker
from fakes import FakeEmbedder

CONFIG = {"embedding_model": "fake-hash", "chunk_size": 200, "chunk_overlap": 20}
DATASET: dict[str, Any] = {
    "name": "mini",
    "kind": "synthetic",
    "documents": [
        {
            "filename": "animaux.pdf",
            "lang": "fr",
            "pages": ["Le chat dort sur le canapé.", "Le chien court dans le jardin."],
        },
        {"filename": "finance.pdf", "lang": "fr", "pages": ["La bourse monte fortement."]},
    ],
    "questions": [
        {
            "id": "q1",
            "lang": "fr",
            "category": "simple",
            "answerable": True,
            "question": "Où dort le chat ?",
            "references": [{"filename": "animaux.pdf", "page": 1, "quote": "chat dort"}],
        },
        {
            "id": "q2",
            "lang": "fr",
            "category": "simple",
            "answerable": True,
            "question": "Que fait le chien ?",
            "references": [{"filename": "animaux.pdf", "page": 2, "quote": "chien court"}],
        },
        {
            "id": "q3",
            "lang": "fr",
            "category": "piege",
            "answerable": True,
            "question": "Le jardin ?",
            "references": [{"filename": "finance.pdf", "page": 1, "quote": "bourse monte"}],
        },
        {"id": "u1", "lang": "fr", "answerable": False, "question": "Quelle heure est-il ?"},
    ],
}


def evaluate(tmp_path: Path, name: str = "run", **kwargs: Any) -> EvaluationRun:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(DATASET, ensure_ascii=False), encoding="utf-8")
    embedder = FakeEmbedder(dimension=64)
    store = ChromaVectorStore(tmp_path / name, "evaluation", embedder.model_name, 64)
    return run_evaluation(
        load_dataset(path), embedder, RecursiveChunker(200, 20), store, top_k=3, **kwargs
    )


def test_ranks_are_computed_from_quotes(tmp_path: Path) -> None:
    run = evaluate(tmp_path)
    by_id = {r.question.id: r for r in run.results}

    assert by_id["q1"].reference_ranks == (1,)
    assert by_id["q2"].first_relevant_rank == 1
    assert by_id["q3"].first_relevant_rank != 1  # "Le jardin ?" attire la page du chien
    assert by_id["u1"].reference_ranks == ()
    assert run.chunk_count == 3


def test_report_separates_retrieval_and_absence_detection(tmp_path: Path) -> None:
    report = build_report(evaluate(tmp_path), CONFIG)

    overall = report["retrieval"]["overall"]
    assert overall["n"] == 3  # seules les questions répondables
    assert set(overall) == {"n", "recall@1", "recall@3", "hit@1", "hit@3", "mrr"}  # top_k=3
    assert set(report["retrieval"]["by_category"]) == {"simple", "piege"}
    assert report["retrieval"]["by_category"]["simple"]["recall@1"] == 1.0
    assert set(report["retrieval"]["by_language_pair"]) == {"fr->fr"}
    absence = report["absence_detection"]
    assert absence["answerable_top1"]["n"] == 3
    assert absence["unanswerable_top1"]["n"] == 1
    assert 0.0 <= absence["auroc"] <= 1.0
    assert report["dataset"]["kind"] == "synthetic"
    assert report["config"]["chunk_count"] == 3
    assert len(report["questions"]) == 4


def test_report_is_reproducible(tmp_path: Path) -> None:
    first = build_report(evaluate(tmp_path, "a"), CONFIG)
    second = build_report(evaluate(tmp_path, "b"), CONFIG)
    for report in (first, second):
        report.pop("generated_at")

    assert first == second


def test_summary_flags_synthetic_results(tmp_path: Path) -> None:
    summary = format_summary(build_report(evaluate(tmp_path), CONFIG))

    assert "SYNTHÉTIQUE" in summary
    assert "AUROC" in summary
    assert "recall@1" in summary


def test_no_absence_analysis_without_unanswerable_questions(tmp_path: Path) -> None:
    original = DATASET["questions"]
    DATASET["questions"] = original[:3]
    try:
        report = build_report(evaluate(tmp_path), CONFIG)
    finally:
        DATASET["questions"] = original

    assert report["absence_detection"] is None
    assert "AUROC" not in format_summary(report)


def test_refuses_a_non_empty_index(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "existant", "evaluation", "fake-hash", 64)
    store.add([Chunk("x:0", "x", "x.pdf", 1, 0, 0, "donnée existante")], [[0.1] * 64])
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(DATASET, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="index vide"):
        run_evaluation(load_dataset(path), FakeEmbedder(64), RecursiveChunker(200, 20), store)
    assert store.count() == 1  # rien n'a été écrit


def test_refuses_quotes_longer_than_chunks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        evaluate(tmp_path, chunk_size=5)


def test_invalid_top_k(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(DATASET, ensure_ascii=False), encoding="utf-8")
    store = ChromaVectorStore(tmp_path / "s", "evaluation", "fake-hash", 64)

    with pytest.raises(ValueError, match="top_k"):
        run_evaluation(
            load_dataset(path), FakeEmbedder(64), RecursiveChunker(200, 20), store, top_k=0
        )
