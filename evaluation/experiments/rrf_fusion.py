"""Expérience isolée : fusion de rangs (RRF) entre la question d'origine et sa traduction.

Hors application : ce script n'est importé par aucun module de ``src/`` et ne change aucune
valeur par défaut. Il compare, sur un même index temporaire, trois conditions de recherche :

- ``baseline`` : question d'origine (comportement par défaut de l'application) ;
- ``translation`` : question française traduite en anglais (DOCASSIST_QUERY_TRANSLATION=fr-en) ;
- ``rrf`` : Reciprocal Rank Fusion des classements d'origine et traduit (Cormack, Clarke et
  Büttcher, SIGIR 2009) : score(d) = somme sur les requêtes de 1 / (k + rang(d)), k = 60
  (valeur standard, non réglée sur ce jeu). Une question anglaise n'a qu'un classement : sa
  fusion est son classement d'origine.

Pour la détection d'absence, le score d'un passage fusionné est le maximum de ses deux
similarités cosinus (un score RRF n'est pas comparable à un cosinus). Les conditions baseline et
traduction doivent reproduire exactement les rapports existants, sinon rien n'est écrit.

Usage (racine du projet) :
    python evaluation/experiments/rrf_fusion.py evaluation/datasets/real_v2.json \
        --baseline evaluation/results/real_v2__e5-small__cs800-co100__corrected.json \
        --translation evaluation/results/real_v2__e5-small__cs800-co100__fr2en-opusmt.json \
        -o evaluation/results/real_v2__e5-small__cs800-co100__fr2en-rrf60.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from doc_assistant.config import Settings
from doc_assistant.domain.models import SearchResult
from doc_assistant.evaluation.dataset import DEFAULT_MIN_OVERLAP, EvalQuestion, load_dataset
from doc_assistant.evaluation.runner import (
    EvaluationRun,
    QuestionResult,
    build_report,
    format_summary,
    index_dataset,
)
from doc_assistant.indexing.embedders import E5Embedder
from doc_assistant.indexing.vector_store import ChromaVectorStore
from doc_assistant.ingestion.chunking import RecursiveChunker
from doc_assistant.ingestion.pdf_loader import PypdfLoader
from doc_assistant.retrieval.marian_translator import MarianTranslator
from doc_assistant.retrieval.query_translation import TranslatingEmbedder

RRF_K = 60


def rrf(rankings: Sequence[Sequence[SearchResult]], k: int = RRF_K) -> list[SearchResult]:
    """Fusionne des classements complets ; score de sortie = meilleur cosinus du passage.

    Égalité de score RRF : meilleur rang individuel, puis identifiant (ordre déterministe).
    """
    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    best: dict[str, SearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            cid = result.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
            if cid not in best or result.score > best[cid].score:
                best[cid] = result
    order = sorted(fused, key=lambda cid: (-fused[cid], best_rank[cid], cid))
    return [best[cid] for cid in order]


def question_result(
    question: EvalQuestion, ranking: Sequence[SearchResult], top_k: int
) -> QuestionResult:
    retrieved = tuple(ranking[:top_k])
    ranks = tuple(
        next(
            (i for i, r in enumerate(retrieved, 1) if ref.matches(r.chunk, DEFAULT_MIN_OVERLAP)),
            None,
        )
        for ref in question.references
    )
    return QuestionResult(question, retrieved, ranks)


def full_rank(question: EvalQuestion, ranking: Sequence[SearchResult]) -> int | None:
    ranks = [
        next((i for i, r in enumerate(ranking, 1) if ref.matches(r.chunk)), None)
        for ref in question.references
    ]
    found = [r for r in ranks if r is not None]
    return min(found) if found else None


def same_retrieval(report: dict[str, Any], saved_path: Path) -> bool:
    saved = json.loads(saved_path.read_text("utf-8"))
    if saved["dataset"]["sha256"] != report["dataset"]["sha256"]:
        return False
    return [q["retrieved"] for q in saved["questions"]] == [
        q["retrieved"] for q in report["questions"]
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--baseline", type=Path, required=True, help="rapport baseline existant")
    parser.add_argument(
        "--translation", type=Path, required=True, help="rapport traduction existant"
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="nouveau rapport RRF")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"Refus d'écraser un rapport existant : {args.output}", file=sys.stderr)
        return 2

    settings = replace(Settings.from_env(), query_translation="fr-en")
    dataset = load_dataset(args.dataset, loader=PypdfLoader())
    embedder = E5Embedder.from_settings(settings)
    translating = TranslatingEmbedder(embedder, MarianTranslator.from_settings(settings))
    chunker = RecursiveChunker.from_settings(settings)

    with tempfile.TemporaryDirectory(prefix="docassist-rrf-", ignore_cleanup_errors=True) as tmp:
        store = ChromaVectorStore(Path(tmp), "rrf", embedder.model_name, embedder.dimension)
        retriever = index_dataset(dataset, embedder, chunker, store)  # index vide et dédié
        n = store.count()
        runs: dict[str, list[QuestionResult]] = {"baseline": [], "translation": [], "rrf": []}
        details: dict[str, dict[str, Any]] = {}
        for question in dataset.questions:
            query = translating.retrieval_query(question.question)  # = question si non française
            original = retriever.search(question.question, top_k=n)
            translated = (
                original if query == question.question else retriever.search(query, top_k=n)
            )
            fused = rrf([original]) if translated is original else rrf([original, translated])
            for name, ranking in (
                ("baseline", original),
                ("translation", translated),
                ("rrf", fused),
            ):
                runs[name].append(question_result(question, ranking, args.top_k))
            details[question.id] = {
                "lang": question.lang,
                "translated_query": None if query == question.question else query,
                "full_rank": {
                    "baseline": full_rank(question, original),
                    "translation": full_rank(question, translated),
                    "rrf": full_rank(question, fused),
                },
            }
        chunk_count = store.count()

    base_config = {
        "embedding_model": embedder.model_name,
        "embedding_model_revision": settings.embedding_model_revision,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    translation_config = {
        "query_translation": settings.query_translation,
        "translation_model": settings.translation_model,
        "translation_model_revision": settings.translation_model_revision,
    }
    reports = {
        name: build_report(
            EvaluationRun(dataset, tuple(results), chunk_count, args.top_k, DEFAULT_MIN_OVERLAP),
            base_config | ({} if name == "baseline" else translation_config),
        )
        for name, results in runs.items()
    }
    checks = {
        "baseline": same_retrieval(reports["baseline"], args.baseline),
        "translation": same_retrieval(reports["translation"], args.translation),
    }
    print(f"Reproduction des rapports existants : {checks}")
    if not all(checks.values()):
        print("Écart avec un rapport existant : aucun rapport écrit.", file=sys.stderr)
        return 1

    report = reports["rrf"]
    report["config"] |= {
        "fusion": "reciprocal_rank_fusion",
        "rrf_k": RRF_K,
        "fusion_inputs": ["original_query", "translated_query (questions françaises seulement)"],
        "absence_score": "cosinus maximal du passage en tête (requête d'origine ou traduite)",
    }
    report["experiment"] = {
        "script": "evaluation/experiments/rrf_fusion.py",
        "compared_reports": {
            "baseline": str(args.baseline.as_posix()),
            "translation": str(args.translation.as_posix()),
        },
        "questions": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(format_summary(report))
    print(f"\nRapport écrit : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
